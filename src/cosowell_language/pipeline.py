"""
The whole analysis, in the order of the report.

`run_whole_analysis` reads like the report's table of contents. Each step reads what earlier
steps saved in results/ and saves its own output there, so any step can also be rerun alone
(python run_pipeline.py --steps <step name>).
"""

from __future__ import annotations

import os
import time

import pandas as pd

from .config import CORPUS_PATH, FEATURES_DIR, FIGURES_DIR, KEYS, NARRATIVE_TYPES, RESULTS_DIR
from .data_loading import load_narratives, load_survey
from .data_overview import (data_by_session, data_kinds_by_session, data_per_person_and_session, participant_demographics,
                            people_by_combination_of_sessions, people_in_each_pair_of_sessions, read_word_counts,
                            retention_between_sessions, session_dates)
from .metrics.all_language_metrics import measure_gaze, measure_language
from .plots.beta_comparison_figures import draw_isolation_vs_loneliness_figure
from .plots.box_plot_figure import draw_box_plots
from .plots.marker_figures import draw_markers_of_question
from .plots.classifier_explanation_figures import draw_metric_importance_figure
from .plots.model_comparison_figure import (draw_distance_from_line_groups_figure, draw_model_comparison_figure,
                                            draw_roc_curves_figure, draw_scores_by_group_figure)
from .plots.data_overview_figure import draw_data_overview_figure
from .plots.survey_exploration_figures import (draw_loneliness_vs_isolation_figure, draw_mixed_model_figure,
                                               draw_session_correlation_figure)
from .plots.fdr_hit_heatmap_figure import draw_fdr_hit_heatmap
from .plots.figure_labels import ISOLATION_MARKERS
from .plots.figure_style import BLUE_RAMP, DOUBLE_COLUMN, ORANGE_RAMP
from .plots.text_type_forest_figure import draw_text_type_forest_figure
from .plots.volcano_figure import draw_volcano_figure
from .stats.analysis_data import load_analysis_data
from .stats.box_plot_data import (MARKER_GROUPINGS, SOCIAL_METRICS, box_plot_data_and_effect_sizes,
                                  marker_values_and_effect_sizes)
from .stats.box_plot_effect_sizes import effect_label
from .stats.hit_counts_and_sample import count_tests_and_hits, sample_summary
from .stats.hit_replication import replicate_fdr_hits
from .stats.hypothesis_tests import run_hypothesis_tests
from .stats.metric_regression_model import scopes
from .stats.classifier_explanation import classification_report_table, confusion_matrices, metric_importance
from .stats.model_comparison_per_text import (DISTANCE_FROM_LINE, LABEL_AGREEMENT, TARGETS,
                                               classify_per_text_type, distance_from_line_groups,
                                               label_agreement_groups, rank_models, scores_by_group, summarise_folds)
from .survey.survey_exploration import (SCORES_USED, correlations_at_each_session, loneliness_and_isolation,
                                        mixed_model_across_sessions, survey_scores_used)


def run_whole_analysis(recompute_metrics: bool = False, run_transformer_models: bool = False) -> None:
    """Every step, data overview to classifiers. The metrics take hours, so by default they are read from results/."""
    run_step(describe_the_data)
    run_step(describe_the_participants)
    run_step(explore_survey_scores)
    if recompute_metrics:
        run_step(measure_language_of_every_narrative)
        run_step(measure_gaze_in_picture_descriptions)
        run_step(measure_creativity_of_picture_descriptions, run_transformer_models)
    run_step(test_hypotheses_in_all_texts)
    run_step(test_hypotheses_in_picture_texts)
    run_step(draw_hypothesis_figures)
    run_step(draw_marker_figures)
    run_step(compare_prediction_models_per_text)


def run_step(step, *args) -> None:
    print(f"\n=== {step.__name__} ===")
    start = time.time()
    step(*args)
    print(f"--- {step.__name__}: {time.time() - start:.0f} s")


def save(table: pd.DataFrame, path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, **kwargs)


def show(table, digits: int = 3) -> None:
    """Print a result table (without the row numbers of a DataFrame)."""
    rounded = table.round(digits)
    if isinstance(table, pd.DataFrame):
        print(rounded.to_string(index=False))
    else:
        print(rounded.to_string())


def describe_the_data() -> None:
    """results/data_overview/ and figure 0: what each person gave at each session."""
    if not CORPUS_PATH.exists():
        print(f"skipped: {CORPUS_PATH} is not there")
        return
    narratives = load_narratives()
    survey = load_survey()
    writers = narratives[KEYS].drop_duplicates()
    analysed = writers.merge(survey[KEYS], on=KEYS)

    by_session = data_by_session(narratives, survey)
    writers_in_both = people_in_each_pair_of_sessions(writers)
    analysed_in_both = people_in_each_pair_of_sessions(analysed)

    folder = RESULTS_DIR / "data_overview"
    save(by_session, folder / "data_by_session.csv")
    save(writers_in_both, folder / "writers_in_both_sessions.csv")
    save(analysed_in_both, folder / "analysed_people_in_both_sessions.csv")
    save(people_by_combination_of_sessions(writers), folder / "writers_by_combination_of_sessions.csv", index=False)
    save(people_by_combination_of_sessions(analysed), folder / "analysed_people_by_combination_of_sessions.csv",
         index=False)
    retention = retention_between_sessions(writers)
    save(retention, folder / "writer_retention_between_sessions.csv")
    data_kinds = data_kinds_by_session(read_word_counts(FEATURES_DIR), survey)
    save(data_kinds, folder / "data_kinds_by_session.csv")
    draw_data_overview_figure(data_per_person_and_session(writers, survey), session_dates(),
                              FIGURES_DIR / "fig0_data_overview.png")
    show(by_session.reset_index())


def describe_the_participants() -> None:
    """results/data_overview/participant_demographics.csv (report Table 4): the analysed people at their first
    session, and the people with two sessions."""
    sessions = load_analysis_data(NARRATIVE_TYPES)["session_scores"]
    demographics = participant_demographics(load_survey(), sessions)
    save(demographics, RESULTS_DIR / "data_overview" / "participant_demographics.csv", index=False)
    show(demographics)


def explore_survey_scores() -> None:
    """
    Questionnaire only, no hypothesis tests: results/survey_exploration/ and figures S2-S3.
    Associations between the survey scores the language models use (over all sessions and at each session);
    loneliness against objective isolation.
    """
    scores = survey_scores_used(load_survey())
    over_all_sessions = mixed_model_across_sessions(scores)
    at_each_session = correlations_at_each_session(scores)
    associations = pd.concat([at_each_session, over_all_sessions], ignore_index=True)
    loneliness_isolation = loneliness_and_isolation(scores)

    folder = RESULTS_DIR / "survey_exploration"
    save(associations, folder / "score_associations.csv", index=False)
    save(loneliness_isolation.to_frame("value"), folder / "loneliness_vs_isolation.csv")
    draw_mixed_model_figure(over_all_sessions, SCORES_USED, FIGURES_DIR / "figS2a_survey_mixed_model.png")
    draw_session_correlation_figure(at_each_session, SCORES_USED,
                                    FIGURES_DIR / "figS2b_survey_correlations_by_session.png")
    draw_loneliness_vs_isolation_figure(scores, loneliness_isolation,
                                        FIGURES_DIR / "figS3_loneliness_vs_isolation.png")
    show(loneliness_isolation)


def measure_language_of_every_narrative(types=NARRATIVE_TYPES) -> None:
    """results/features/<type>.csv: every metric family that applies to the text type."""
    narratives = load_narratives(types=types)
    for ntype in types:
        of_type = narratives[narratives["ntype"].eq(ntype)].reset_index(drop=True)
        metrics = measure_language(of_type, ntype)
        save(metrics, FEATURES_DIR / f"{ntype}.csv", index=False)
        print(f"{ntype}: {len(metrics)} narratives x {metrics.shape[1] - 2} metrics")


def measure_gaze_in_picture_descriptions() -> None:
    """results/features/cookie_gaze.csv: the order in which picture content is mentioned, as a scanpath."""
    metrics = measure_gaze(load_narratives(types=["cookie"]))
    save(metrics, FEATURES_DIR / "cookie_gaze.csv", index=False)
    print(f"{len(metrics)} narratives x {metrics.shape[1] - 2} metrics")


def measure_creativity_of_picture_descriptions(run_transformer_models: bool = False) -> None:
    """results/features/cookie_creativity.csv. Without run_transformer_models, texts missing from the
    transformer caches get NaN for those metrics."""
    # One thread per library: the models run text by text and several thread pools at once slow each other down.
    # These must be set before torch and scikit-learn are loaded, which is why the import is here.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(variable, "1")
    from .metrics.creativity.all_creativity_metrics import measure_creativity

    metrics = measure_creativity(CORPUS_PATH, run_models=run_transformer_models)
    save(metrics, FEATURES_DIR / "cookie_creativity.csv", index=False)
    print(f"{len(metrics)} narratives x {metrics.shape[1] - 2} metrics")


def test_hypotheses_in_all_texts() -> None:
    """RQ1, RQ2 and RQ1-2 in every text type: results/all/."""
    test_hypotheses(NARRATIVE_TYPES, RESULTS_DIR / "all")


def test_hypotheses_in_picture_texts() -> None:
    """The same with the picture descriptions only: results/cookie/."""
    test_hypotheses(["cookie"], RESULTS_DIR / "cookie")


def test_hypotheses(types: list[str], out) -> None:
    """Every research-question test, FDR hits with their replication, and counts per research question."""
    data = load_analysis_data(types)
    text_metrics, features, used_metrics = data["text_metrics"], data["features"], data["used_metrics"]
    session_scores, scored_texts = data["session_scores"], data["scored_texts"]
    save(used_metrics, out / "features_used.csv", index_label="feature")
    save(sample_summary(session_scores, scored_texts), out / "sample.csv", header=["value"])
    print(f"{len(features)} metrics; per text type: {used_metrics.sum().to_dict()}")

    tests = run_hypothesis_tests(scored_texts, features, types)
    hits = replicate_fdr_hits(tests, scored_texts, text_metrics, used_metrics, types)
    counts = count_tests_and_hits(tests, hits)
    save(tests, out / "screen.csv", index=False)
    save(hits, out / "hits.csv", index=False)
    save(counts, out / "overview.csv", index=False)
    show(counts)


def draw_hypothesis_figures() -> None:
    """results/figures/fig1-fig4 and figS1, and the box-plot effect sizes (results/all/group_effect_sizes.csv)."""
    results = RESULTS_DIR / "all"
    tests = pd.read_csv(results / "screen.csv")
    hits = pd.read_csv(results / "hits.csv")
    data = load_analysis_data(NARRATIVE_TYPES)
    sessions, effects = box_plot_data_and_effect_sizes(data["raw"], data["session_scores"])
    save(effects, results / "group_effect_sizes.csv", index=False)
    show(effects)
    effect_labels = {(e.grouping, e.metric): effect_label(e) for e in effects.itertuples()}

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    draw_volcano_figure(tests, hits, FIGURES_DIR / "fig1_all_research_questions_volcano.png")
    draw_isolation_and_loneliness_figures(tests, hits, sessions, effect_labels)
    draw_fdr_hit_heatmap(tests, hits, "pooled", FIGURES_DIR / "figS1_heatmap.png")


def draw_isolation_and_loneliness_figures(tests: pd.DataFrame, hits: pd.DataFrame, sessions: pd.DataFrame,
                                          effect_labels: dict) -> None:
    """Figures 2-4 (RQ1, RQ2): isolation vs loneliness in one model, raw values by group, and the isolation
    markers in each text type."""
    draw_isolation_vs_loneliness_figure(tests, ISOLATION_MARKERS,
                                        FIGURES_DIR / "fig2_rq1_2_isolation_vs_loneliness.png")
    groups = [("isolation_group", "Objective isolation (thirds)", ["Low", "Middle", "High"], BLUE_RAMP[1:]),
              ("loneliness_group", "Loneliness score", ["3", "4–5", "6–9"], ORANGE_RAMP[1:])]
    draw_box_plots(sessions, SOCIAL_METRICS, groups, FIGURES_DIR / "fig3_rq1_rq2_boxes.png", DOUBLE_COLUMN,
                   effect_labels)
    draw_text_type_forest_figure(hits, tests, scopes(NARRATIVE_TYPES), FIGURES_DIR / "fig4_rq1_by_text_type.png",
                                 max_panels=8, finding="RQ1 isolation", term="isolation_index")



# Supplementary figure of each research question's robust markers, and the colours of its groups.
MARKER_FIGURES = {
    ("RQ1 isolation", "isolation_index"): ("figS4_rq1_markers.png", BLUE_RAMP[1:]),
    ("RQ2 loneliness", "loneliness"): ("figS5_rq2_markers.png", ORANGE_RAMP[1:]),
}


def draw_marker_figures() -> None:
    """
    Figures S4-S5 and results/all/marker_group_effect_sizes.csv: every robust marker of each research question by
    groups of its survey score, in the text type where it is a marker (unadjusted, descriptive). Box plots of
    the raw values for up to three markers, a PCA of the markers for more.
    """
    hits = pd.read_csv(RESULTS_DIR / "all" / "hits.csv")
    data = load_analysis_data(NARRATIVE_TYPES)
    sessions, effects = marker_values_and_effect_sizes(data["raw"], data["session_scores"], hits)
    save(effects, RESULTS_DIR / "all" / "marker_group_effect_sizes.csv", index=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for (finding, term), (file_name, colors) in MARKER_FIGURES.items():
        of_question = effects[effects["finding"].eq(finding) & effects["term"].eq(term)]
        if of_question.empty:
            continue
        group, xlabel, levels = MARKER_GROUPINGS[(finding, term)]
        draw_markers_of_question(sessions, of_question, group, xlabel, levels, colors, FIGURES_DIR / file_name)
    show(effects[["finding", "metric", "cliffs_delta", "delta_size", "spearman_rho", "n"]])



def compare_prediction_models_per_text() -> None:
    """
    Machine-learning classifiers (serve RQ1, RQ2): results/model_comparison/ and figures 5-6. Eight classifiers tell lonely from not lonely and isolated
    from not isolated people, from the earlier FDR hits for loneliness and isolation, one text type at a time.
    Nested cross-validation: 5 outer folds by person for testing; FDR feature selection, scaling and a grid search
    over 5 inner folds within each training set. The four models with the best inner AUC are drawn (mean and SD
    over the outer folds), and the ROC curves of the top-ranked one. Pilots (figures 7a-8c): the same
    predictions scored separately for people whose loneliness and isolation labels agree or disagree, and for
    people near or far from the mixed-model line of loneliness on isolation.
    """
    out = RESULTS_DIR / "model_comparison"
    data = load_analysis_data(NARRATIVE_TYPES)
    hits = pd.read_csv(RESULTS_DIR / "all" / "hits.csv")
    folds, predictions = classify_per_text_type(data["scored_texts"], data["session_scores"], data["used_metrics"],
                                                hits)
    summary = summarise_folds(folds)
    ranking = rank_models(folds)
    save(folds, out / "classifier_fold_scores.csv", index=False)
    save(summary, out / "classifier_summary.csv", index=False)
    save(predictions, out / "classifier_out_of_fold_predictions.csv", index=False)
    save(ranking, out / "classifier_ranking.csv", index=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    draw_model_comparison_figure(summary, ranking, TARGETS, FIGURES_DIR / "fig5_classifier_scores.png")
    # One model, the top-ranked (chosen on training folds only), for the ROC curves and both pilots.
    top_model = ranking.loc[0, "model"]
    draw_roc_curves_figure(predictions, top_model, TARGETS, FIGURES_DIR / "fig6_roc_curves.png")
    explain_top_classifier(data, hits, predictions, top_model, out)
    agreement = label_agreement_groups(data["scored_texts"], data["session_scores"])
    by_agreement = scores_by_group(predictions, agreement)
    distance_from_line = distance_from_line_groups(data["session_scores"])
    by_distance = scores_by_group(predictions, distance_from_line)
    save(by_agreement, out / "classifier_scores_by_label_agreement.csv", index=False)
    save(by_distance, out / "classifier_scores_by_distance_from_line.csv", index=False)
    draw_scores_by_group_figure(by_agreement, top_model, LABEL_AGREEMENT, TARGETS,
                                FIGURES_DIR / "fig7a_label_agreement.png")
    draw_roc_curves_figure(predictions, top_model, TARGETS, FIGURES_DIR / "fig7b_label_agreement_roc_curves.png",
                           agreement, LABEL_AGREEMENT)
    draw_distance_from_line_groups_figure(distance_from_line, DISTANCE_FROM_LINE,
                                          FIGURES_DIR / "fig8a_on_and_off_the_line.png")
    draw_scores_by_group_figure(by_distance, top_model, DISTANCE_FROM_LINE, TARGETS,
                                FIGURES_DIR / "fig8b_distance_from_line.png")
    draw_roc_curves_figure(predictions, top_model, TARGETS, FIGURES_DIR / "fig8c_distance_from_line_roc_curves.png",
                           distance_from_line, DISTANCE_FROM_LINE)
    show(ranking)
    shown = summary[summary["model"].isin(ranking.loc[ranking["shown_in_figure"], "model"])]
    show(shown[["target", "ntype", "model", "balanced_accuracy_mean", "balanced_accuracy_std", "roc_auc_mean",
                "roc_auc_std", "n_selected_metrics_mean"]])
    for by_group in (by_agreement, by_distance):
        show(by_group[by_group["model"].eq(top_model)].drop(columns=["model", "share_positive_std"]))


def explain_top_classifier(data: dict, hits: pd.DataFrame, predictions: pd.DataFrame, top_model: str, out) -> None:
    """
    The top-ranked classifier, a closer look: classification report and confusion matrices of its out-of-fold
    predictions, and permutation importance of each metric on the test folds (figure S6).
    """
    report = classification_report_table(predictions, top_model)
    confusion = confusion_matrices(predictions, top_model)
    importance = metric_importance(data["scored_texts"], data["session_scores"], data["used_metrics"], hits,
                                   top_model)
    save(report, out / "classifier_classification_report.csv", index=False)
    save(confusion, out / "classifier_confusion_matrices.csv", index=False)
    save(importance, out / "classifier_metric_importance.csv", index=False)
    draw_metric_importance_figure(importance, top_model, TARGETS,
                                  FIGURES_DIR / "figS6_classifier_metric_importance.png")
    show(report)


STEPS = {step.__name__: step for step in [
    describe_the_data, describe_the_participants, explore_survey_scores,
    measure_language_of_every_narrative, measure_gaze_in_picture_descriptions,
    measure_creativity_of_picture_descriptions,
    test_hypotheses_in_all_texts, test_hypotheses_in_picture_texts,
    draw_hypothesis_figures, draw_marker_figures, compare_prediction_models_per_text,
]}
