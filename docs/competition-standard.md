# ROGII Competition Standard

Source materials read:

- Kaggle competition page: `rogii-wellbore-geology-prediction`
- Kaggle `evaluation` page retrieved with `kaggle competitions pages --content --page-name evaluation`
- Official `AI_wellbore_geology_prediction_task_en.pptx` from the competition data bundle
- Public leaderboard retrieved with `kaggle competitions leaderboard`

## Task

Predict `tvt` values along the hidden portion of each horizontal well. The training horizontal-well files include true `TVT`; the test horizontal-well files include `TVT_input` only up to the prediction start point.

The official task deck describes the setup as:

- each well has a horizontal-well CSV and an assigned typewell CSV;
- horizontal data include measured depth `MD`, coordinates `X/Y/Z`, gamma ray `GR`, and `TVT_input`;
- typewell data include known `TVT`, `GR`, and geology layer labels;
- the goal is to infer future horizontal-well TVT from horizontal-well `XYZ/GR` and typewell `TVT/GR`;
- predictions are made at one-foot steps for the rows in `sample_submission.csv`.

## Metric

Submissions are scored by root mean squared error:

```text
RMSE = sqrt((1 / n) * sum((y_i - yhat_i)^2))
```

The task deck phrases this as `dTVT = manualTVT - predictedTVT`, with final quality measured as RMSE over all predicted `dTVT` values. Lower is better.

## Submission Format

The submission must contain exactly two columns:

```text
id,tvt
000d7d20_1442,0.0
000d7d20_1443,0.0
...
```

For the current downloaded data bundle:

- sample submission rows: `14151`
- test wells: `3`
- required columns: `id,tvt`

The validator script in this repository checks column order, row count, id order, numeric `tvt`, and finite predictions.

Kaggle currently accepts this competition's submissions only from notebooks, not
by direct CSV upload. The local CSV validator is still useful for checking a
candidate output before packaging it in a Kaggle notebook.

## Score Target

The requested target is `7.2` public-LB-level RMSE. As of the leaderboard snapshot retrieved on 2026-06-20, the public leaderboard top scores are below `6.0`, and public notebooks around `7.201` are available. That means `7.2` is a realistic public-score target, but it still requires a stronger approach than the existing `9.251` DWT baseline.

Current local candidates:

- `baselines/dwt_top_kernel`: public DWT baseline titled `9.251 ROGII-Wellbore Geology Prediction: DWT-based`.
- `baselines/rogii_lb_7_201`: public notebook `baidalinadilzhan/rogii-lb-7-201`.
- `baselines/rogii_lb7201_public_gold_conservative`: public notebook `curvecowboy/rogii-lb7201-public-gold-conservative`.

The downloaded output candidate `outputs/rogii_lb7201_public_gold_conservative/submission.csv` validates against the sample submission and is the current target-level candidate.
