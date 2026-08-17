# Applied Machine Learning — Summer 2026

Two tasks: sentiment classification of movie-review snippets contaminated with
spam email (Task 1), and 5-point face-landmark alignment (Task 2).

## Layout

```
task1_nlp/     run_task1.py     end-to-end pipeline -> submission/results_task1.csv
               make_figures.py  method diagrams
               tests_sanity.py  28 checks; run before trusting any result
               src/             structural features, spam gate, sentiment models
task2_cv/      run_task2.py     end-to-end pipeline -> submission/results_task2.csv
               make_figures.py  method diagrams
               tests_sanity.py  28 checks; run before trusting any result
               src/             dataset, augmentation, heatmaps, CNN, shape model,
                                evaluation, robustness
make_submission.py builds submission.zip and validates both CSVs first
```

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import nltk; [nltk.download(r) for r in ('movie_reviews','stopwords','punkt','wordnet')]"
```

The datasets are not in this repository (the brief says not to redistribute
them). Put them at:

```
task1_nlp/data/sentiment_analysis_{training,validation,test}_data.csv
task2_cv/data/{train,val,test}.npz          # or set AML_T2_TRAIN / AML_T2_VAL / AML_T2_TEST
task1_nlp/data/glove.6B.100d.txt            # optional; the BiLSTM falls back without it
```

## Running

```sh
.venv/bin/python task1_nlp/tests_sanity.py     # 28/28 before trusting anything
.venv/bin/python task2_cv/tests_sanity.py      # 28/28
.venv/bin/python task1_nlp/run_task1.py        # ~2 min
.venv/bin/python task2_cv/run_task2.py         # ~1-2 h on an M4 (MPS)
.venv/bin/python make_submission.py
```

Both prediction files are written by the worksheets' own `save_as_csv`
functions, copied verbatim, and neither pipeline ever reorders the test set.
