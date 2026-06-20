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
2026-06-20, leaving three attempts. No additional submission was made during
this audit pass.

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
