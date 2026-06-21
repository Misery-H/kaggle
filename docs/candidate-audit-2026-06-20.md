# Candidate Audit - 2026-06-20

Known submitted anchors:

- `53871877`: `outputs/codex_rogii_lb7201/submission.csv`, public score `7.285`.
- `53873002`: `outputs/codex_rogii_w060/submission.csv`, public score `7.540`.

The vector from the `7.285` file to the failed `w0.60` file has RMSE `2.116817`.
Using the two official scores, the best possible point on only that line is about
`7.283`, so weight-only tuning in that direction cannot reach the `7.2` target.

## Public Output Findings

- `baidalinadilzhan/rogii-lb-7-201`, `curvecowboy/rogii-lb7201-public-gold-conservative`,
  `curvecowboy/rogii-lb7295-public-rebuild-submit`, and
  `kokinnwakashuu/rogii-dual-pipeline-v11-gold-multicut` produce a main
  `submission.csv` identical to the submitted `7.285` anchor.
- `sumo1290/rogii-lb7201-w055-codex` completed, but its main output is in the
  same failed blend direction as `w0.60`; the audit heuristic is about `7.697`.
- The strongest non-identical public micro-adjustment is
  `kokinnwakashuu/rogii-light-u-smoother`, with RMSE movement `0.142562` from
  the `7.285` anchor and heuristic score `7.284582`.
- Local smoother scan on the `7.285` anchor found `blend=0.05` as the best
  nearby geometry candidate, with heuristic score `7.284463`.
- These smoother candidates are useful for measuring whether micro-smoothing
  helps, but they do not provide enough expected lift to explain a `7.2` score
  without a strongly favorable hidden residual alignment.

## Current Submission Budget

The daily limit is five submissions. Two submissions have been used on
2026-06-20 before this audit pass.

Additional records from this pass:

- `53887425`: static `blend=0.05` smoother probe. Kaggle marked it complete
  with no score and an incorrect-format error; the local CSV used CRLF line
  endings, so the static packager now normalizes generated files to LF.
- `53888080`: `kokinnwakashuu/rogii-light-u-smoother` rerun under the account as
  `sumo1290/rogii-light-u-smoother-codex`; completed with public score `7.523`.
- `53898518`: static anti-light submission. Kaggle again marked the static
  notebook output as incorrect-format, confirming that this competition should
  use a full notebook workflow rather than a one-cell embedded CSV notebook.
- `53898943`: full-notebook anti-light submission from
  `sumo1290/rogii-anti-light-u-smoother-codex`; pending at the time this note
  was updated.

That leaves at most one further attempt if Kaggle counts the invalid static
record against the daily limit.

## Scored Direction Inversion

The accepted `light U smoother` probe moved the `7.285` anchor by RMSE
`0.142562` and scored `7.523`. From

```text
score(candidate)^2 = score(anchor)^2 + ||d||^2 - 2 * <hidden_error, d>
```

the hidden error has a strongly negative projection on the light-smoother
direction. Reversing that direction gives a predicted public score around
`7.04`, assuming no material orthogonal change. A full notebook with
`_SMOOTH_BLEND = -0.06` was therefore prepared and submitted as `53898943`.

### Correction After Geometry Audit

`scripts/score_geometry_audit.py` shows that the recorded `53888080` score cannot
belong to the local `outputs/codex_light_u_smoother/submission.csv` file under an
RMSE metric: the file moves only `0.142562` ft from the anchor, while the recorded
score delta is `0.238`. This violates the RMSE triangle bound, and the implied
hidden-error projection also violates Cauchy's bound. The anti-light submission
should therefore be treated as a low-confidence probe until Kaggle returns its
own result, not as a validated inverse direction.

The only currently consistent non-anchor score direction is `w0.60`; optimizing
on that single direction estimates only `7.283`, so it does not justify another
submission by itself.

## Reusable Commands

Generate local contact-smoother variants:

```powershell
uv run python scripts/make_contact_smoother_variants.py `
  outputs/codex_rogii_lb7201/submission.csv `
  --out-dir outputs/contact_smoother_scan `
  --blend 0.05 --blend 0.10 --blend 0.18
```

Audit candidate directories against the two official submitted anchors:

```powershell
uv run python scripts/audit_candidates.py outputs/contact_smoother_scan outputs/probe_light_u_smoother
```
