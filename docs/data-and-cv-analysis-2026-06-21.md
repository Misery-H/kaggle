# ROGII Data And CV Analysis - 2026-06-21

## Data Distribution

The raw bundle has `773` training wells and `3` test wells. Each training well
contains a horizontal-well CSV, a typewell CSV, and a PNG. The test wells are:

- `000d7d20`: 5,278 rows, 1,442 known `TVT_input` rows, known fraction `0.273`.
- `00bbac68`: 7,559 rows, 1,545 known rows, known fraction `0.204`.
- `00e12e8b`: 6,384 rows, 2,083 known rows, known fraction `0.326`.

Training `TVT_input` prefix fractions are well aligned with the test wells:
mean `0.267`, median `0.260`, interquartile range `0.225` to `0.300`.
This means the train files already provide a useful offline simulation:
predict the hidden suffix where `TVT_input` is missing and score against `TVT`.

Train well medians:

- rows per well: `6,576`;
- MD span: `6,575 ft`;
- TVT span: `758 ft`;
- Z span: `787 ft`;
- horizontal GR coverage: median `0.723`.

## Important Leakage Finding

The three test well IDs also exist in `train`. The train horizontal files include
formation surface columns such as `ANCC`, `ASTNU`, `EGFDU`, and `BUDA`; the test
horizontal files do not. A contact reconstruction from those train formation
surfaces can be almost exact on train CV because the surfaces and train `TVT`
are same-version artifacts.

This explains the current behavior:

- direct/contact train-copy style predictions look nearly perfect offline;
- the same approach scores only `7.285` on Kaggle public hidden rows;
- therefore the remaining error is likely a hidden train/test version shift,
  not ordinary row-level model error.

I fixed the contact CV script so contact offsets are estimated from `TVT_input`
prefix rows only. Even then, train CV remains unrealistically near-perfect
because the surface columns themselves encode the train-copy geometry. Treat
this as a diagnostic, not as a trustworthy leaderboard estimate.

## Strict Visible-Field CV

To get a non-leaky baseline, I added `scripts/train_prefix_delta_cv.py`. It uses
only fields available in test horizontal/typewell files:

- MD, X, Y, Z, GR;
- `TVT_input` prefix summaries;
- typewell TVT/GR summaries and interpolated typewell GR;
- no formation surface columns.

Full train GroupKFold, using every 10th hidden row:

| Method | RMSE |
| --- | ---: |
| Hold last known TVT | `15.901` |
| Naive robust prefix line extrapolation | `1363.461` |
| LightGBM prefix-delta model | `15.070` |

The model improves over hold-last-known by `0.831`, but it is far from the
current Kaggle anchor `7.285`. The naive prefix slope fails because TVT commonly
flattens or changes regime after the prefix; extrapolating the early slope is
not stable.

## Current Implication

The strongest path is not a pure supervised model from visible fields. The
best candidate must start from the current contact/train-copy anchor and target
the hidden version shift. The `light U smoother` probe tested one plausible
micro-adjustment and scored `7.523`, so that smoothing direction is worse than
the `7.285` anchor.

Next work should focus on estimating train/test version shift from data rather
than adding generic smoothing:

- compare prefix residual structure between test wells and similar train wells;
- search train wells for synthetic shifts that preserve near-zero prefix error
  but diverge in hidden rows;
- learn a well-level correction direction and keep it small enough to respect
  the `7.285` anchor.
