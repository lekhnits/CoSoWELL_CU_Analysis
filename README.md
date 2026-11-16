# Language markers of loneliness and social isolation in older adults' narratives

Code for the analysis *"Do the words people write show how lonely or isolated they are? Language markers in the
CoSoWELL narratives"*.

## Installation

Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4 vader_lexicon
```

The Hugging Face models, GloVe vectors and other resources are downloaded on first use.

## Data

The CoSoWELL release 2 survey and corpus (Kyröläinen, Paul & Kuperman) are distributed by the dataset authors
[add OSF link]. Put these two files in `data/`:

- `cosowell_corpus_matched_v2.txt`
- `cosowell_survey_v2_numericVar.csv`

The metrics per narrative are already in `results/features/`, so the analysis runs without recomputing them.

## Running

```bash
python run_pipeline.py                                # every step; metrics read from results/features/ (~4 min)
python run_pipeline.py --with-metrics                 # recompute the metrics first (several hours)
python run_pipeline.py --with-metrics --run-models    # also run the transformer models missing from the cache
python run_pipeline.py --steps draw_hypothesis_figures draw_marker_figures   # only some steps
```

Results and figures are written to `results/`.

Notebooks (run from the repository folder):

- `analysis_walkthrough.ipynb`: the analysis question by question, with the results inline
- `new_picture_descriptions.ipynb`: the metrics for new picture descriptions

## Tests

```bash
python -m pytest tests    # skipped when the data files are missing
```
