"""
Run the whole analysis (src/cosowell_language/pipeline.py), from the data overview to the classifiers.

    python run_pipeline.py                                # every step; metrics are read from results/features/
    python run_pipeline.py --with-metrics                 # recompute the metrics first (several hours)
    python run_pipeline.py --with-metrics --run-models    # ... including transformer outputs missing from the cache
    python run_pipeline.py --steps draw_hypothesis_figures draw_marker_figures   # only some steps
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cosowell_language import pipeline  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=list(pipeline.STEPS), help="run only these steps, in this order")
    parser.add_argument("--with-metrics", action="store_true", help="recompute results/features/ first")
    parser.add_argument("--run-models", action="store_true", help="creativity: compute transformer outputs not cached")
    args = parser.parse_args()

    if args.steps:
        for name in args.steps:
            pipeline.run_step(pipeline.STEPS[name])
    else:
        pipeline.run_whole_analysis(recompute_metrics=args.with_metrics, run_transformer_models=args.run_models)
