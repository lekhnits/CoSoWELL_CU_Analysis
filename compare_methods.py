"""
compare_methods.py  –  Side-by-side comparison of three CU extraction methods
==============================================================================
Runs one transcript through all three pipelines and prints a unified report:

    Method 1 – Quantitext-style (Josephy-Hernandez et al., 2023)
               dictionary + lemmatisation, general language metrics
    Method 2 – SemR  (Stegmann et al., 2022)
               unique picture elements / total words
    Method 3 – BERT CIU  (Ng et al., 2025)
               fine-tuned BERT, sentence-level multi-label classification

Usage
-----
    # Default built-in example transcripts:
    python compare_methods.py

    # Plain-text file (one transcript):
    python compare_methods.py path/to/transcript.txt

    # CoSoWELL corpus mode (deduped on `id`, writes CSV + box plots):
    python compare_methods.py --corpus
    python compare_methods.py --corpus path/to/other_corpus.csv --limit 100
    python compare_methods.py --corpus --output-dir my_results

    # BERT checkpoint (optional; without one, backbone is pre-trained but
    # the classifier head is random – useful for pipeline testing only):
    python compare_methods.py path/to/transcript.txt --checkpoint path/to/ciu_bert.pt

    # Suppress the BERT method (e.g. no GPU / no checkpoint available):
    python compare_methods.py --no-bert

Dependencies
------------
    pip install spacy nltk torch transformers scikit-learn numpy tabulate \
                pandas matplotlib
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('wordnet')"

The three method scripts must live in the same directory as this file:
    1_quantitext_cu_extractor.py
    2_semr_semantic_relevance.py
    3_bert_ciu_extractor.py
"""

import sys
import os
import argparse
import json
import textwrap
import importlib.util
from pathlib import Path

# ── Make the sibling scripts importable without a package structure ──────────
_HERE = Path(__file__).parent

# Default location of the CoSoWELL corpus CSV (token-level; deduped by `id`
# in load_corpus() to produce one narrative per participant).
DEFAULT_CORPUS_PATH = (
    _HERE.parent / "osfstorage-archive" / "cosowell_corpus_v2.csv"
)

def _import_script(filename: str):
    """Import a .py file by path and return the module."""
    path = _HERE / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found: {path}\n"
            f"Make sure all four scripts are in the same directory."
        )
    spec   = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Example transcripts ──────────────────────────────────────────────────────
EXAMPLE_TRANSCRIPTS = {
    "healthy_control": (
        "I can see a woman standing at the kitchen sink washing and drying dishes. "
        "The water from the tap is overflowing onto the floor but she appears "
        "completely unaware of it and is not bothered by it at all. "
        "Behind her, a boy is standing on a wobbly stool and is reaching up to steal "
        "cookies from a cookie jar on a shelf in the cupboard. "
        "The stool looks like it is tipping over and he might fall. "
        "A girl is standing right next to him with her hand outstretched, "
        "asking or waiting for a cookie. "
        "There is a window with curtains above the sink and you can see outside "
        "into what looks like a garden."
    ),
    "mild_impairment": (
        "There is a woman. She is doing something at the, um, the kitchen thing. "
        "And there is a boy. He is getting something from up there. "
        "The floor is all wet. There is a girl too. "
        "She wants something I think."
    ),
    "severe_impairment": (
        "The woman. A boy. Something is falling. Wet."
    ),
}


# ── Formatting helpers ───────────────────────────────────────────────────────

def _section(title: str, width: int = 72) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def _wrap(text: str, indent: int = 4, width: int = 72) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix,
                         subsequent_indent=prefix)


def _fmt_list(items: list, indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}• {item}" for item in items) if items else f"{prefix}(none)"


# ── Method runners ───────────────────────────────────────────────────────────

def run_quantitext(mod, text: str) -> dict:
    """Call analyse_language_sample and return its result dict."""
    return mod.analyse_language_sample(text)


def run_semr(mod, text: str) -> dict:
    """Call compute_semr and return its result dict."""
    return mod.compute_semr(text, return_details=True)


_BERT_CACHE: dict = {}

def run_bert(mod, text: str, checkpoint: str | None, device: str) -> dict | None:
    """
    Instantiate (and cache) the BERT model + tokenizer, optionally load a
    checkpoint, and run predict_transcript. Returns None if torch is missing.

    The model is loaded once per (checkpoint, device) and reused across calls;
    this matters in corpus mode where the function is invoked per narrative.
    """
    try:
        import torch
        from transformers import BertTokenizerFast
    except ImportError:
        print("  [BERT] torch/transformers not installed – skipping.")
        return None

    key = (checkpoint, device)
    if key not in _BERT_CACHE:
        MODEL_NAME = "bert-base-uncased"
        tokenizer  = BertTokenizerFast.from_pretrained(MODEL_NAME)
        model      = mod.BertCIUClassifier(MODEL_NAME, dropout=0.2)
        if checkpoint:
            state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state)
            print(f"  [BERT] Loaded checkpoint: {checkpoint}")
        else:
            print("  [BERT] No checkpoint provided – classifier head is random.")
            print("         Results are for pipeline demonstration only.")
        _BERT_CACHE[key] = (tokenizer, model)

    tokenizer, model = _BERT_CACHE[key]
    return mod.predict_transcript(model, tokenizer, text, device=device)


# ── Comparison report ────────────────────────────────────────────────────────

def print_report(
    label:     str,
    text:      str,
    q_result:  dict | None,
    s_result:  dict | None,
    b_result:  dict | None,
) -> None:
    """Print the side-by-side comparison report for one transcript."""
    W = 72
    print("\n" + "═" * W)
    print(f"  TRANSCRIPT: {label}")
    print("═" * W)
    print(_wrap(text, indent=2))

    # ── Shared summary table ─────────────────────────────────────────────────
    print(_section("SUMMARY TABLE"))

    col_w = [36, 10, 10, 10]
    hdr   = f"{'Metric':<{col_w[0]}} {'Quantitext':>{col_w[1]}} {'SemR':>{col_w[2]}} {'BERT':>{col_w[3]}}"
    sep   = "  ".join("─" * w for w in col_w)
    print(f"  {hdr}")
    print(f"  {sep}")

    def row(metric, q_val, s_val, b_val):
        print(f"  {metric:<{col_w[0]}} {str(q_val):>{col_w[1]}} {str(s_val):>{col_w[2]}} {str(b_val):>{col_w[3]}}")

    q_cu   = q_result["content_unit_count"] if q_result else "—"
    s_cu   = s_result["unique_elements"]    if s_result else "—"
    b_cu   = b_result["total_ciu_count"]    if b_result else "—"
    row("CU / elements detected (out of 23)", q_cu, s_cu, b_cu)

    q_tot  = q_result["total_words"]  if q_result else "—"
    s_tot  = s_result["total_words"]  if s_result else "—"
    b_tot  = "—"  # BERT operates sentence-level
    row("Total words",                       q_tot, s_tot, b_tot)

    q_eff  = q_result["cu_efficiency"]  if q_result else "—"
    s_semr = s_result["semr"]           if s_result else "—"
    b_eff  = "—"
    row("Efficiency / SemR score",           q_eff, s_semr, b_eff)

    q_msl  = q_result["mean_sentence_length"] if q_result else "—"
    row("Mean sentence length (words)",      q_msl, "—", "—")

    q_ttr  = q_result["type_token_ratio"]  if q_result else "—"
    row("Type-token ratio",                 q_ttr, "—", "—")

    # ── Method 1 detail ──────────────────────────────────────────────────────
    print(_section("METHOD 1 – Quantitext  (Josephy-Hernandez et al., 2023)"))
    if q_result:
        print(f"    Words          : {q_result['total_words']}")
        print(f"    Sentences      : {q_result['total_sentences']}")
        print(f"    Mean sent. len : {q_result['mean_sentence_length']}")
        print(f"    CU count       : {q_result['content_unit_count']} / 23")
        print(f"    CU efficiency  : {q_result['cu_efficiency']}")
        print(f"    Type-token TTR : {q_result['type_token_ratio']}")
        print(f"    Mean log freq  : {q_result['mean_log_word_frequency']}")
        print(f"    POS counts     : {q_result['pos_counts']}")
        print(f"\n    Identified content units:")
        for cu in q_result["content_units"]:
            print(f"      [{cu['canonical']:25s}]  token='{cu['token']}'  POS={cu['pos']}")
    else:
        print("    (not run)")

    # ── Method 2 detail ──────────────────────────────────────────────────────
    print(_section("METHOD 2 – SemR  (Stegmann et al., 2022)"))
    if s_result:
        print(f"    SemR score       : {s_result['semr']:.6f}")
        print(f"    Unique elements  : {s_result['unique_elements']} / 23")
        print(f"    Total words      : {s_result['total_words']}")
        print(f"\n    Matched elements:")
        for m in s_result.get("matched", []):
            print(f"      [{m['element']:30s}]  via '{m['matched_form']}'")
        print(f"\n    Not matched ({len(s_result['not_matched'])}):")
        print(_fmt_list(s_result["not_matched"]))
    else:
        print("    (not run)")

    # ── Method 3 detail ──────────────────────────────────────────────────────
    print(_section("METHOD 3 – BERT CIU  (Ng et al., 2025)"))
    if b_result:
        print(f"    CIUs detected : {b_result['total_ciu_count']} / 23")
        print(f"    Temporal order: {b_result['unique_cius']}")
        print(f"\n    Per-sentence breakdown:")
        for r in b_result["sentence_results"]:
            sent_short = r["sentence"][:60] + ("…" if len(r["sentence"]) > 60 else "")
            print(f"      '{sent_short}'")
            if r["ordered_cius"]:
                print(f"        → {r['ordered_cius']}")
            else:
                print(f"        → (no CIUs above threshold)")
    elif b_result is None:
        print("    (skipped – no torch or --no-bert flag)")
    else:
        print("    (not run)")

    # ── Cross-method CIU overlap ─────────────────────────────────────────────
    print(_section("CIU OVERLAP ACROSS METHODS"))
    sets: dict[str, set] = {}
    if q_result:
        sets["Quantitext"] = {cu["canonical"] for cu in q_result["content_units"]}
    if s_result:
        sets["SemR"] = {m["element"] for m in s_result.get("matched", [])}
    if b_result:
        sets["BERT"] = set(b_result["unique_cius"])

    if len(sets) >= 2:
        names = list(sets.keys())
        # Pairwise Jaccard
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b_ = sets[names[i]], sets[names[j]]
                jaccard = len(a & b_) / max(len(a | b_), 1)
                shared  = sorted(a & b_)
                only_a  = sorted(a - b_)
                only_b  = sorted(b_ - a)
                print(f"\n  {names[i]}  ∩  {names[j]}  (Jaccard={jaccard:.2f})")
                print(f"    Shared   : {shared}")
                print(f"    Only {names[i]:<10}: {only_a}")
                print(f"    Only {names[j]:<10}: {only_b}")
    else:
        print("    (need at least 2 methods to compare)")

    print("\n" + "═" * W + "\n")


# ── Corpus mode ──────────────────────────────────────────────────────────────

def load_corpus(path: str | Path) -> "pd.DataFrame":
    """
    Load the CoSoWELL token-level CSV and return one row per unique `id`,
    keeping the `narrative`, `type`, and `test_session` columns.
    """
    import pandas as pd
    cols = ["id", "narrative", "type", "test_session"]
    df = pd.read_csv(path, usecols=cols)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    df = df[df["narrative"].astype(str).str.strip().ne("")]
    return df


def _flatten_result(
    row_id: str,
    type_: str | None,
    test_session: str | None,
    q: dict | None,
    s: dict | None,
    b: dict | None,
) -> dict:
    """Flatten the three method results into one CSV row."""
    out = {
        "id":            row_id,
        "type":          type_,
        "test_session":  test_session,
    }
    if q:
        out.update({
            "quantitext_cu_count":      q["content_unit_count"],
            "quantitext_total_words":   q["total_words"],
            "quantitext_total_sents":   q["total_sentences"],
            "quantitext_mean_sent_len": q["mean_sentence_length"],
            "quantitext_cu_efficiency": q["cu_efficiency"],
            "quantitext_ttr":           q["type_token_ratio"],
            "quantitext_mean_log_wf":   q["mean_log_word_frequency"],
        })
    if s:
        out.update({
            "semr_unique_elements":     s["unique_elements"],
            "semr_score":               s["semr"],
            "semr_total_words":         s["total_words"],
        })
    if b:
        out["bert_ciu_count"] = b.get("total_ciu_count")
    return out


def run_corpus(
    corpus_path: str | Path,
    mod1, mod2, mod3,
    output_dir: Path,
    checkpoint: str | None,
    device: str,
    limit: int | None,
    verbose: bool,
) -> "pd.DataFrame":
    """Run the three methods over every unique narrative in the corpus."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus: {corpus_path}")
    df = load_corpus(corpus_path)
    print(f"  {len(df)} unique narratives (deduped on `id`).")
    if limit:
        df = df.head(limit)
        print(f"  --limit {limit}: processing first {len(df)} narratives.")

    rows: list[dict] = []
    n = len(df)
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        text = rec.narrative
        if not isinstance(text, str) or not text.strip():
            continue

        if verbose:
            print(f"\n[{i}/{n}] id={rec.id}  type={rec.type}  session={rec.test_session}")
        elif i == 1 or i % 25 == 0 or i == n:
            print(f"  [{i}/{n}] processing …", flush=True)

        try:
            q = run_quantitext(mod1, text)
        except Exception as e:
            print(f"    Quantitext error on {rec.id}: {e}")
            q = None
        try:
            s = run_semr(mod2, text)
        except Exception as e:
            print(f"    SemR error on {rec.id}: {e}")
            s = None
        b = None
        if mod3 is not None:
            try:
                b = run_bert(mod3, text, checkpoint, device)
            except Exception as e:
                print(f"    BERT error on {rec.id}: {e}")
                b = None

        rows.append(_flatten_result(rec.id, rec.type, rec.test_session, q, s, b))

        if verbose:
            print_report(rec.id, text, q, s, b)

    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "compare_methods_results.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\nWrote results CSV: {csv_path}  ({len(out_df)} rows)")

    make_plots(out_df, output_dir)
    return out_df


def _significance_stars(p: float) -> str:
    """Conventional star annotation for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def make_plots(df: "pd.DataFrame", output_dir: Path) -> None:
    """
    Seaborn box plot of CU/element counts across all three methods, with
    pairwise Wilcoxon signed-rank tests (Bonferroni-corrected) shown as
    significance brackets above the boxes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import wilcoxon, friedmanchisquare

    method_cols = [
        ("quantitext_cu_count",   "Quantitext"),
        ("semr_unique_elements",  "SemR"),
        ("bert_ciu_count",        "BERT"),
    ]
    method_cols = [
        (c, lbl) for c, lbl in method_cols
        if c in df.columns and df[c].notna().any()
    ]
    if not method_cols:
        print("  No method columns to plot.")
        return

    # Long-format frame for seaborn, keeping only rows where every included
    # method produced a value (paired observations for the Wilcoxon test).
    paired = df.dropna(subset=[c for c, _ in method_cols])
    long = paired.melt(
        value_vars=[c for c, _ in method_cols],
        var_name="method",
        value_name="cu_count",
    )
    rename = {c: lbl for c, lbl in method_cols}
    long["method"] = long["method"].map(rename)
    order = [lbl for _, lbl in method_cols]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.boxplot(
        data=long, x="method", y="cu_count", order=order,
        hue="method", hue_order=order, palette="Set2", legend=False,
        ax=ax, width=0.55, fliersize=3, linewidth=1.2,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 8},
    )
    sns.stripplot(
        data=long, x="method", y="cu_count", order=order,
        ax=ax, color="black", alpha=0.15, size=2.5, jitter=0.18,
    )

    ax.set_ylabel("Content units detected (out of 23)")
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)

    # Friedman omnibus (non-parametric repeated-measures ANOVA) on the paired
    # method scores. Run when 3+ methods are present.
    friedman_txt = ""
    if len(method_cols) >= 3:
        arrs = [paired[c].to_numpy() for c, _ in method_cols]
        chi2, p_friedman = friedmanchisquare(*arrs)
        friedman_txt = (f"Friedman χ²({len(method_cols) - 1}) = {chi2:.2f},  "
                        f"p = {p_friedman:.3g} ({_significance_stars(p_friedman)})")
        print(f"  Friedman omnibus: {friedman_txt}")

    ax.set_title(
        f"CU detection by method  (n = {len(paired)})"
        + (f"\n{friedman_txt}" if friedman_txt else "")
    )

    # Pairwise Wilcoxon signed-rank with Bonferroni correction (post-hoc).
    pairs   = [(i, j) for i in range(len(order)) for j in range(i + 1, len(order))]
    n_pairs = max(len(pairs), 1)
    results = []
    for i, j in pairs:
        a = paired[method_cols[i][0]].to_numpy()
        b = paired[method_cols[j][0]].to_numpy()
        try:
            stat, p = wilcoxon(a, b, zero_method="wilcox", correction=False)
        except ValueError:
            stat, p = float("nan"), 1.0
        p_corr = min(p * n_pairs, 1.0)
        results.append((i, j, p, p_corr))
        print(f"  Wilcoxon {order[i]} vs {order[j]}: "
              f"W={stat:.1f}, p={p:.3g}, p_bonf={p_corr:.3g} ({_significance_stars(p_corr)})")

    # Draw significance brackets above the boxes.
    y_max  = long["cu_count"].max()
    y_min  = long["cu_count"].min()
    span   = max(y_max - y_min, 1.0)
    step   = span * 0.09
    base_y = y_max + step * 0.6
    for k, (i, j, _p_raw, p_corr) in enumerate(results):
        y = base_y + step * k
        ax.plot([i, i, j, j], [y, y + step * 0.25, y + step * 0.25, y],
                lw=1.2, color="black")
        ax.text((i + j) / 2, y + step * 0.30, _significance_stars(p_corr),
                ha="center", va="bottom", fontsize=14)

    ax.set_ylim(0, base_y + step * (len(results) + 0.5))

    fig.text(
        0.5, 0.005,
        "Post-hoc: pairwise Wilcoxon signed-rank, Bonferroni-corrected  "
        "(*** p<0.001,  ** p<0.01,  * p<0.05,  ns = not significant)",
        ha="center", va="bottom", fontsize=9, color="#555",
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    path = output_dir / "boxplot_cu_count.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare three CU extraction methods on a transcript."
    )
    parser.add_argument(
        "transcript_file",
        nargs="?",
        help="Path to a plain-text transcript file. "
             "If omitted, built-in example transcripts are used.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a fine-tuned BERT checkpoint (.pt). "
             "Without this the BERT head is random (demo only).",
    )
    parser.add_argument(
        "--no-bert",
        action="store_true",
        help="Skip the BERT method (useful if torch is not installed).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: 'cpu' or 'cuda'. Default: cpu.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also dump full results as JSON to stdout after the report.",
    )
    parser.add_argument(
        "--corpus",
        nargs="?",
        const=str(DEFAULT_CORPUS_PATH),
        default=None,
        help=(
            "Run on the CoSoWELL corpus CSV (one narrative per unique `id`). "
            f"Pass --corpus alone to use the default path "
            f"({DEFAULT_CORPUS_PATH.name}), or --corpus PATH for a different "
            f"CSV with `id` and `narrative` columns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_HERE.parent / "results_compare_methods"),
        help="Where to write the results CSV and plots in corpus mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N unique narratives (corpus mode).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the full per-narrative report in corpus mode "
             "(default: progress lines only).",
    )
    args = parser.parse_args()

    # ── Import the three method modules ──────────────────────────────────────
    print("Loading method modules …")
    mod1 = _import_script("1_quantitext_cu_extractor.py")
    mod2 = _import_script("2_semr_semantic_relevance.py")

    if args.corpus and not args.no_bert and not args.checkpoint:
        print("  [BERT] No --checkpoint provided — running with a random "
              "classifier head (slow; results are demo-only). "
              "Pass --no-bert to skip BERT entirely.")
    mod3 = None if args.no_bert else _import_script("3_bert_ciu_extractor.py")

    # ── Corpus mode short-circuits the single-file pipeline ──────────────────
    if args.corpus:
        run_corpus(
            corpus_path=args.corpus,
            mod1=mod1, mod2=mod2, mod3=mod3,
            output_dir=Path(args.output_dir),
            checkpoint=args.checkpoint,
            device=args.device,
            limit=args.limit,
            verbose=args.verbose,
        )
        return

    # ── Build transcript dict ─────────────────────────────────────────────────
    if args.transcript_file:
        with open(args.transcript_file, encoding="utf-8") as fh:
            content = fh.read().strip()
        transcripts = {Path(args.transcript_file).name: content}
    else:
        transcripts = EXAMPLE_TRANSCRIPTS

    # ── Run all methods and collect results ───────────────────────────────────
    all_results = {}

    for label, text in transcripts.items():
        print(f"\nProcessing: {label} …")

        print("  Running Method 1 (Quantitext) …")
        q = run_quantitext(mod1, text)

        print("  Running Method 2 (SemR) …")
        s = run_semr(mod2, text)

        print("  Running Method 3 (BERT CIU) …" if mod3 else "  Skipping Method 3 (BERT).")
        b = run_bert(mod3, text, args.checkpoint, args.device) if mod3 else None

        all_results[label] = {"quantitext": q, "semr": s, "bert": b}
        print_report(label, text, q, s, b)

    # ── Optional JSON dump ───────────────────────────────────────────────────
    if args.json:
        # Make BERT result JSON-serialisable (strip non-serialisable keys)
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(i) for i in obj]
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return str(obj)

        print(json.dumps(_clean(all_results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()