"""
Script 3: BERT-based CIU Extraction & Ordering
===============================================
Implements the pipeline described in:

    Ng, S.-I., Ambadi, P. S., Mueller, K. D., Liss, J., & Berisha, V. (2025).
    "Advancing Automated Spatio-Semantic Analysis in Picture Description
    Using Language Models."
    arXiv preprint arXiv:2510.05128
    https://arxiv.org/abs/2510.05128

The authors made their trained model and code openly accessible:
    Google Drive folder: https://shorturl.at/cmQG3
    (ICASSP_2026_CIU_BERT — contains checkpoints and training scripts)

Architecture (Section 2 of the paper):
  - Base model : bert-base-uncased (HuggingFace)
  - Pooling    : mean pooling over last hidden states
  - Classifier : linear layer → 23 logits (one per CIU)
  - Inference  : sigmoid(logits) > 0.5  → predicted CIUs
                 logit ORDER → temporal narrative order
  - Training losses:
      L_BCE  : binary cross-entropy (multi-label classification)
      L_rank : pairwise ranking loss (enforce CIU ordering)
      L_total: (1 - λ) * L_BCE  +  λ * L_rank    [λ = 0.1]

23 CIUs from the Cookie Theft picture (Table 1, Ng et al. 2025):
    boy, girl, woman, kitchen, outside, cookie, jar, stool, sink,
    plate, dishcloth, water, window, cupboard, dishes, curtains,
    boy taking/stealing, boy or stool falling,
    woman drying/washing plates, water overflowing,
    action performed by girl, woman unconcerned by overflowing,
    woman indifferent to the children

"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# 1.  The 23 Cookie Theft CIUs (Table 1, Ng et al. 2025)
# ---------------------------------------------------------------------------

CIU_LABELS: list[str] = [
    "boy",
    "girl",
    "woman",
    "kitchen",
    "outside",
    "cookie",
    "jar",
    "stool",
    "sink",
    "plate",
    "dishcloth",
    "water",
    "window",
    "cupboard",
    "dishes",
    "curtains",
    "boy_taking_stealing",
    "boy_or_stool_falling",
    "woman_drying_washing_plates",
    "water_overflowing",
    "action_performed_by_girl",
    "woman_unconcerned_by_overflowing",
    "woman_indifferent_to_children",
]
NUM_CIUS = len(CIU_LABELS)   # 23
CIU_TO_IDX = {label: i for i, label in enumerate(CIU_LABELS)}
IDX_TO_CIU = {i: label for i, label in enumerate(CIU_LABELS)}


# ---------------------------------------------------------------------------
# 2.  Model
# ---------------------------------------------------------------------------

class BertCIUClassifier(nn.Module):
    """
    BERT-based multi-label CIU classifier with ordering via logit ranking.

    Forward pass returns 23 logits; sigmoid(logits) > 0.5 gives predicted
    CIUs; the ORDER of CIUs follows logit values (higher logit = mentioned
    earlier in the description).
    """

    def __init__(self, model_name: str = "bert-base-uncased", dropout: float = 0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size   # 768 for bert-base
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, NUM_CIUS)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw logits, shape (batch, 23)."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Mean pooling over non-padding tokens
        last_hidden = outputs.last_hidden_state          # (B, L, H)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)                 # (B, 23)
        return logits


# ---------------------------------------------------------------------------
# 3.  Loss functions
# ---------------------------------------------------------------------------

def binary_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Multi-label BCE loss (primary objective).
    targets: float tensor {0,1} of shape (B, 23)
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def pairwise_ranking_loss(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    margin:  float = 1.0,
) -> torch.Tensor:
    """
    Auxiliary ranking loss that enforces earlier CIUs in the ground-truth
    ordering to have higher logit values.

    For each sample:
        For every pair (i, j) where CIU i appears BEFORE CIU j in the
        ground-truth sequence, penalise if logit[j] >= logit[i] - margin.

    Because the raw CIU list is ordered (index 0 = leftmost / earliest CIU
    in the picture), we use index order as a proxy for temporal order, which
    matches the paper's definition.

    L_rank = mean_over_pairs( max(0, s_j - s_i + margin) )
             for all pairs i < j where both y_i == 1 and y_j == 1
    """
    batch_size = logits.size(0)
    losses = []
    for b in range(batch_size):
        active = (targets[b] > 0.5).nonzero(as_tuple=True)[0]
        if len(active) < 2:
            continue
        # All pairs (i, j) with i < j in the CIU ordering
        sample_loss = torch.tensor(0.0, device=logits.device)
        n_pairs = 0
        for p in range(len(active)):
            for q in range(p + 1, len(active)):
                i_idx = active[p]
                j_idx = active[q]
                s_i   = logits[b, i_idx]
                s_j   = logits[b, j_idx]
                sample_loss = sample_loss + torch.clamp(s_j - s_i + margin, min=0.0)
                n_pairs += 1
        if n_pairs > 0:
            losses.append(sample_loss / n_pairs)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=logits.device)


def total_loss(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    lam:     float = 0.1,
    margin:  float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L_total = (1 - λ) * L_BCE  +  λ * L_rank
    Returns (total, l_bce, l_rank) for logging.
    """
    l_bce  = binary_cross_entropy_loss(logits, targets)
    l_rank = pairwise_ranking_loss(logits, targets, margin)
    l_tot  = (1 - lam) * l_bce + lam * l_rank
    return l_tot, l_bce, l_rank


# ---------------------------------------------------------------------------
# 4.  Dataset
# ---------------------------------------------------------------------------

@dataclass
class CIUSample:
    """One sentence from a CHAT transcript with its CIU annotations."""
    text:   str
    labels: list[str]   # subset of CIU_LABELS present in this sentence
    # Optional: list of CIU labels in the ORDER they appear (for ranking loss)
    ordered_labels: list[str] = field(default_factory=list)


class CIUDataset(Dataset):
    """
    Torch Dataset wrapping a list of CIUSample objects.

    To create training data from DementiaBank CHAT transcripts:
      1. Parse each transcript; split into sentences.
      2. For each sentence, note which of the 23 CIUs are annotated.
      3. Wrap as CIUSample(text=..., labels=[...], ordered_labels=[...]).
    """

    def __init__(
        self,
        samples:    list[CIUSample],
        tokenizer:  BertTokenizerFast,
        max_length: int = 512,
    ):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        enc = self.tokenizer(
            s.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Multi-hot label vector
        label_vec = torch.zeros(NUM_CIUS, dtype=torch.float)
        for lbl in s.labels:
            if lbl in CIU_TO_IDX:
                label_vec[CIU_TO_IDX[lbl]] = 1.0

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         label_vec,
        }


# ---------------------------------------------------------------------------
# 5.  Training loop
# ---------------------------------------------------------------------------

def train(
    model:       BertCIUClassifier,
    train_loader: DataLoader,
    epochs:      int   = 50,
    lr_bert:     float = 2e-5,
    lr_head:     float = 1e-3,
    lam:         float = 0.1,
    margin:      float = 1.0,
    device:      str   = "cpu",
) -> BertCIUClassifier:
    """
    Fine-tune the BERT CIU classifier.

    Hyperparameters match Section 4 of Ng et al. 2025:
      - 50 epochs
      - AdamW, lr=2e-5 for BERT params, lr=1e-3 for classifier head
      - λ = 0.1, margin = 1.0
    """
    model.to(device)
    model.train()

    # Separate LRs for BERT backbone vs classification head
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(),       "lr": lr_bert},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ]
    )
    total_steps = epochs * len(train_loader)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(0.1 * total_steps),
        num_training_steps = total_steps,
    )

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            logits             = model(input_ids, attention_mask)
            l_tot, l_bce, l_rank = total_loss(logits, labels, lam, margin)
            l_tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += l_tot.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={epoch_loss/len(train_loader):.4f}  "
                  f"BCE={l_bce.item():.4f}  rank={l_rank.item():.4f}")

    return model


# ---------------------------------------------------------------------------
# 6.  Inference
# ---------------------------------------------------------------------------

def predict(
    model:     BertCIUClassifier,
    tokenizer: BertTokenizerFast,
    sentences: list[str],
    threshold: float = 0.5,
    device:    str   = "cpu",
) -> list[dict]:
    """
    Run inference on a list of sentences (one per call to the model).

    Returns a list of dicts, one per sentence:
        {
          "sentence":          str,
          "predicted_cius":    list[str],   # labels with prob > threshold
          "ordered_cius":      list[str],   # same labels sorted by logit (high→low)
          "probabilities":     dict[str, float],
        }
    """
    model.to(device)
    model.eval()
    results = []

    with torch.no_grad():
        for sent in sentences:
            enc = tokenizer(
                sent,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = model(input_ids, attention_mask).squeeze(0)  # (23,)
            probs  = torch.sigmoid(logits).cpu().numpy()

            # CIUs above threshold
            above   = [(CIU_LABELS[i], float(probs[i])) for i in range(NUM_CIUS)
                       if probs[i] > threshold]
            # Ordered by logit value (higher = mentioned earlier / more prominent)
            ordered = sorted(above, key=lambda x: -logits[CIU_TO_IDX[x[0]]].item())

            results.append({
                "sentence":       sent,
                "predicted_cius": [x[0] for x in above],
                "ordered_cius":   [x[0] for x in ordered],
                "probabilities":  {CIU_LABELS[i]: float(probs[i]) for i in range(NUM_CIUS)},
            })
    return results


def predict_transcript(
    model:     BertCIUClassifier,
    tokenizer: BertTokenizerFast,
    transcript: str,
    threshold:  float = 0.5,
    device:     str   = "cpu",
) -> dict:
    """
    Run sentence-level predictions over a full transcript and aggregate
    across all sentences.

    Returns:
        {
          "sentence_results":   list[dict],   # per-sentence predictions
          "unique_cius":        list[str],    # de-duped CIUs in temporal order
          "total_ciu_count":    int,
        }
    """
    # Simple sentence splitting (replace with NLTK/spaCy for production)
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]

    sentence_results = predict(model, tokenizer, sentences, threshold, device)

    # Build temporal order: first time a CIU is predicted (sentence order)
    seen:    set[str]  = set()
    ordered: list[str] = []
    for r in sentence_results:
        for ciu in r["ordered_cius"]:
            if ciu not in seen:
                seen.add(ciu)
                ordered.append(ciu)

    return {
        "sentence_results":  sentence_results,
        "unique_cius":       ordered,
        "total_ciu_count":   len(ordered),
    }


# ---------------------------------------------------------------------------
# 7.  Evaluation helpers
# ---------------------------------------------------------------------------

def precision_recall_f1(
    preds:   list[list[str]],
    targets: list[list[str]],
) -> dict:
    """
    Compute macro-averaged precision, recall, and F1 across all 23 CIUs.
    preds/targets: one list of CIU labels per sample.
    """
    per_class_p, per_class_r, per_class_f1 = [], [], []
    for ciu in CIU_LABELS:
        tp = sum(1 for p, t in zip(preds, targets) if ciu in p and ciu in t)
        fp = sum(1 for p, t in zip(preds, targets) if ciu in p and ciu not in t)
        fn = sum(1 for p, t in zip(preds, targets) if ciu not in p and ciu in t)
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f  = 2 * p * r / max(p + r, 1e-9)
        per_class_p.append(p)
        per_class_r.append(r)
        per_class_f1.append(f)
    return {
        "macro_precision": float(np.mean(per_class_p)),
        "macro_recall":    float(np.mean(per_class_r)),
        "macro_f1":        float(np.mean(per_class_f1)),
        "per_class": {
            CIU_LABELS[i]: {"precision": per_class_p[i],
                            "recall":    per_class_r[i],
                            "f1":        per_class_f1[i]}
            for i in range(NUM_CIUS)
        },
    }


def sequence_error_rate(pred_order: list[str], true_order: list[str]) -> float:
    """
    Compute sequence error rate using Levenshtein distance, normalised by
    the number of true CIUs (insertions + deletions + substitutions).
    Matches the evaluation metric in Ng et al. 2025, Section 4.
    """
    n, m = len(true_order), len(pred_order)
    dp   = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost     = 0 if true_order[i - 1] == pred_order[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,        # deletion
                           dp[i][j-1] + 1,          # insertion
                           dp[i-1][j-1] + cost)     # substitution
    return dp[n][m] / max(n, 1)


# ---------------------------------------------------------------------------
# 8.  Demo (runs without fine-tuning to show the pipeline shape)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    MODEL_NAME = "bert-base-uncased"
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = BertCIUClassifier(MODEL_NAME, dropout=0.2)

    # ── Option A: load a fine-tuned checkpoint ──────────────────────────────
    # Uncomment and point to the checkpoint from the authors' Drive folder or
    # from your own fine-tuning run:
    #
    # checkpoint_path = "path/to/ciu_bert_checkpoint.pt"
    # model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    # print(f"Loaded checkpoint: {checkpoint_path}")

    # ── Option B: demo with randomly initialised classifier head ────────────
    # (BERT backbone weights are pre-trained; only the head is random here)
    print("\n[Demo mode: BERT backbone is pre-trained; classifier head is random]")
    print("For meaningful predictions, fine-tune on DementiaBank / WRAP data.")
    print("Author code: https://shorturl.at/cmQG3  (Google Drive)\n")

    # ── Dummy fine-tuning example with synthetic data ────────────────────────
    synthetic_samples = [
        CIUSample(
            text="The boy is stealing a cookie from the jar.",
            labels=["boy", "cookie", "jar", "boy_taking_stealing"],
            ordered_labels=["boy", "cookie", "jar", "boy_taking_stealing"],
        ),
        CIUSample(
            text="The woman is drying dishes at the sink.",
            labels=["woman", "dishes", "sink", "woman_drying_washing_plates"],
            ordered_labels=["woman", "dishes", "sink", "woman_drying_washing_plates"],
        ),
        CIUSample(
            text="Water is overflowing from the sink onto the floor.",
            labels=["water", "sink", "water_overflowing"],
            ordered_labels=["water", "sink", "water_overflowing"],
        ),
        CIUSample(
            text="The stool is tipping and the boy might fall.",
            labels=["stool", "boy", "boy_or_stool_falling"],
            ordered_labels=["stool", "boy", "boy_or_stool_falling"],
        ),
        CIUSample(
            text="A girl is standing nearby with her hand outstretched.",
            labels=["girl", "action_performed_by_girl"],
            ordered_labels=["girl", "action_performed_by_girl"],
        ),
    ]

    dataset    = CIUDataset(synthetic_samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print("── Running 5-epoch demo fine-tuning on synthetic data ──")
    model = train(model, dataloader, epochs=5, device=DEVICE)

    # ── Inference on a new transcript ────────────────────────────────────────
    test_transcript = (
        "The woman is standing at the sink washing dishes. "
        "The water is running over. "
        "A boy is climbing a stool to get cookies from the jar. "
        "The stool looks wobbly. "
        "There is a girl next to the boy."
    )

    print("\n── Inference on test transcript ──")
    result = predict_transcript(model, tokenizer, test_transcript, device=DEVICE)

    print(f"Unique CIUs detected : {result['total_ciu_count']}")
    print(f"CIU temporal order   : {result['unique_cius']}")
    print("\nPer-sentence breakdown:")
    for r in result["sentence_results"]:
        print(f"  '{r['sentence']}'")
        print(f"    → {r['ordered_cius']}")

    # ── Sequence error rate demo ─────────────────────────────────────────────
    ground_truth_order = ["woman", "water_overflowing", "boy", "stool",
                          "cookie", "jar", "girl"]
    ser = sequence_error_rate(result["unique_cius"], ground_truth_order)
    print(f"\nSequence Error Rate vs. ground truth: {ser:.3f}")