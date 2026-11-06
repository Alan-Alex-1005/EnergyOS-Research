# RESEARCH_CLAIMS.md

Purpose: a concise, honest boundary between what this project's
implementation and experiments actually demonstrate, and what they do
not. Written for viva/defense preparation — read this before answering
"what does your system prove?"

## Demonstrated by implementation and experiment

- **A working, end-to-end pipeline** that ingests unfamiliar CSV/XLSX
  energy data (WIDE or LONG format), automatically profiles and maps it
  to a canonical model, cleans it conservatively, scores its own data
  quality, and reports which downstream analyses it can honestly
  support — all without a fixed schema.
- **Chronological, leakage-guarded forecasting**: a TRAIN/VALIDATION/TEST
  split (never shuffled) where model *selection* uses only validation
  RMSE and final reported metrics (MAE/RMSE/MAPE/R2) come from a TEST
  period never seen during selection. Verified by regression tests that
  corrupt the test segment and confirm model selection is unaffected.
- **Robust, training-only anomaly thresholds**: median/MAD statistics
  fit exclusively on training residuals, applied to test residuals.
  Verified by regression tests that vary test data while confirming
  fitted thresholds don't change.
- **Specialized agents that wrap, not replace, that analysis** —
  Forecast/Anomaly/Equipment/Cost/Carbon agents each report a
  capability-aware status (DONE/LIMITED/UNAVAILABLE/INSUFFICIENT_DATA)
  and never fabricate equipment, schedules, tariffs, or carbon factors
  that weren't actually present in the data or config.
- **A collaboration layer that measurably adds framework capability**
  over isolated analytics or independent agents: cross-agent evidence
  correlation, a documented Agent Agreement Score (excluding agents
  that can't independently vote), a documented Evidence Completeness
  Score, and a Decision-Support Confidence score — all explicitly
  labeled as decision-support tooling, not statistical probability.
- **A deterministic AI Energy Engineer** that assembles WHAT/WHERE/
  WHEN/WHY/impact/evidence/missing-context/confidence/root-cause/action
  from already-computed results — no LLM, fully reproducible, and it
  explicitly states "Root cause is not confirmed by available evidence"
  whenever (which is always, in this system) no schedule model exists
  to confirm one.
- **A three-level framework-coverage evaluation** (Isolated Analytics vs
  Independent Agents vs Collaborative Framework) measuring Explanation
  Coverage, Evidence Diversity, and Recommendation Support from actual
  runtime capability — with an identical denominator across all three
  levels per scenario, so the comparison can't be tilted by definition.

## NOT demonstrated — do not claim these

- **Universal forecasting superiority.** The baselines implemented
  (persistence, seasonal-naive, single-lag AR(1)) are deliberately
  simple and transparent. No comparison against state-of-the-art
  forecasting models (ARIMA, gradient boosting, deep learning) has been
  performed.
- **Statistically significant accuracy improvement from collaboration.**
  The Day-5 evaluation measures *framework coverage and evidence
  diversity*, not prediction accuracy. Collaboration does not change a
  single forecast or anomaly number — it only correlates, scores, and
  explains numbers Day 2 already computed. No claim is made that
  Level 3 forecasts more accurately than Level 1.
- **Guaranteed root-cause diagnosis.** No schedule, occupancy, or
  causal model exists anywhere in this system. The Equipment Agent and
  AI Energy Engineer report identifiers/context as, at most, a labeled,
  unconfirmed hypothesis — never a proven cause.
- **Real-world energy savings, cost savings, or carbon reduction.**
  Excess-energy, cost, and carbon figures are mathematical estimates
  from a forecast baseline (kW x hours, or excess kWh x a supplied
  rate/factor) — they are not measured savings from an intervention,
  and no intervention has been tested.
- **Validation against real-world operational data at scale.** As of
  this writing, all runs use either the bundled synthetic demo dataset
  or small, explicitly-labeled CONTROLLED/SYNTHETIC test fixtures (see
  `docs/CHANGELOG.md` for exactly which). Real public dataset
  validation (e.g. Building Data Genome Project 2) was attempted but
  blocked by this development environment's network restrictions (Git
  LFS / Zenodo / Figshare / Kaggle are all unreachable from this
  sandbox) — see the final CHANGELOG entry for the exact blocker. The
  real-data evaluation runner (`src/evaluation/real_data_runner.py`) is
  built and tested, but has not yet been exercised on genuine
  real-world measurements.
- **Superiority over any existing Building Energy Management System
  (BEMS).** No existing commercial or research BEMS has been run
  side-by-side with this system for comparison.
- **Production-grade robustness at scale.** This is a research
  demonstrator (single-process, file-based, no database, no
  concurrency, no authentication) — not a deployment-ready product.

## The honest one-sentence summary

This project demonstrates a working, leakage-aware, capability-honest
multi-agent framework that measurably increases *framework coverage and
explanation completeness* over isolated analytics or independent
agents — it does not demonstrate improved prediction accuracy,
confirmed root causes, or real-world energy/cost/carbon outcomes.
