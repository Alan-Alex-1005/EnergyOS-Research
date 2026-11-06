# CHANGELOG.md

## Session 2 — Long-Format Semantic Mapping Fix

**Fixed:** LONG-format datasets (`timestamp | meter_id | metric | value
| unit`) with recognizable power/energy metric values (e.g. `power_kw`,
`energy_kwh`) incorrectly reported Forecasting, Anomaly Detection, and
Equipment Analysis as `UNAVAILABLE`. Root cause: capability assessment
only inspected physical-column roles, and LONG format's `metric`/
`value` columns correctly have no fixed column-level role by design —
so the dataset's real semantics (carried in the *values* of the metric
column) were invisible to `capabilities.py`.

**Added:**
- `src/long_format.py` — new module. Classifies each metric *value*
  into a role (reusing `role_detection`'s pattern table via the new
  `match_role_by_label()` function), grouped into logical series by
  metric + meter/device/equipment identifier where available. Builds
  matching canonical rows for resolved series; never fabricates a role
  or unit for an unrecognized metric (left `UNKNOWN`/unresolved).
- Optional `metric_mappings` config block for explicit per-metric
  overrides, mirroring the existing `mappings` override for physical
  columns.
- `profiler.py`: LONG-format detection now also identifies an optional
  unit column (`unit`/`units`/`uom`/`unit_of_measure`).
- `capabilities.assess_capabilities()`: new optional `extra_roles`
  parameter (backward-compatible) so capability verdicts reflect
  LONG-format logical-series roles, not just physical-column roles.
- `terminal.py`: new "Long-Format Metric Mapping" CLI report section.
- `tests/test_long_format.py` — 12 new tests: POWER+ENERGY metrics make
  Forecasting/Anomaly Detection available; mixed recognized/
  unrecognized metrics; multiple meters each get independent series;
  optional unit-column resolution; config override + invalid-config
  rejection; WIDE-format regression guard.
- Fixtures: `long_format_mixed_metrics.csv`,
  `long_format_with_unit_column.csv`.

**Changed (non-breaking):**
- `pipeline.py`: computes `LongFormatMappingResult`, merges its
  canonical rows into `canonical_df`, passes resolved roles to
  `capabilities.assess_capabilities()`. New `PipelineResult.
  long_format_mapping` field (defaulted, doesn't break existing
  callers).
- `run_demo.py`: passes `long_format_mapping` through to
  `render_report()`.
- `docs/RESEARCH_SPEC.md`, `docs/ARCHITECTURE.md`: updated to describe
  the corrected LONG-format behavior.

**Verified:**
- Full test suite: 101/101 passing (89 pre-existing + 12 new), no
  regressions to WIDE-format behavior.
- CLI re-run end-to-end for wide CSV, long CSV (plain, mixed-metric,
  and unit-column variants), XLSX (single + multi-sheet), and messy
  CSVs (missing values, duplicate rows) — all correct.

**Not changed (explicitly out of scope for this fix, see
`PROJECT_STATE.md` → Known Issues):** `quality.py`'s mapping-coverage/
completeness factors remain column-based (cosmetic only, doesn't affect
correctness of canonical data or capabilities); duplicate-timestamp
handling for LONG data is unchanged; per-series row-count thresholds
for capability checks are unchanged (whole-dataset row count only).

---

## Session 1 — Day 1 Initial Build

**Added:** Full Day 1 pipeline per the master prompt:
`loader.py`, `profiler.py`, `schema.py` (timestamp detection),
`role_detection.py`, `mapping.py`, `units.py`, `cleaning.py`,
`canonical.py`, `quality.py`, `capabilities.py`, `terminal.py`,
`pipeline.py` (orchestrator), `run_demo.py` (CLI).

Test suite: 89 tests across loader, profiler, schema, role detection,
mapping, units, cleaning, quality, capabilities, canonical, and full
end-to-end pipeline scenarios — covering all 15 required test-fixture
scenarios (simple/wide/long CSV, XLSX single/multi-sheet, weird
timestamp names, missing values, duplicate rows, duplicate timestamps,
ambiguous numeric columns, status columns, unrecognized column names,
no-timestamp files, no-energy-data files, mixed known/unknown columns,
explicit config overrides).

Demo dataset `data/example.csv` (963 rows, 15-minute wide-format
building energy data with realistic messiness) and
`configs/example.json` created for hand-testing.

Initial `docs/RESEARCH_SPEC.md` and `docs/ARCHITECTURE.md` written.

## Session 3 — Day 2: Forecasting, Anomaly Detection, Excess Energy

**Added:** `analysis_series.py` (AnalysisSeries abstraction, per-series sufficiency), `timeseries.py` (chronological prep + TRAIN/VALIDATION/TEST 70/15/15 split, never shuffled), `forecasting/` (Persistence, Seasonal Naive, optional Linear-Lag AR1, MAE/RMSE/MAPE/R2 metrics with explicit zero/constant handling, model-selection engine), `anomaly/` (median/MAD robust z-score, documented severity ladder, event grouping with isolated-spike/sustained classification), `impact/excess_energy.py` (kW×h integration, POWER-only, strict preconditions), `visualization/research_plots.py` (8 matplotlib PNGs, honest skips), `analysis_pipeline.py` (orchestrator + results.json), `analysis_terminal.py` (CLI report). `run_demo.py` gained `--analyze`, `--series`, `--no-figures`.

**Fixed (Part A):** LONG-format quality scoring (mapping coverage/completeness/unit confidence/temporal consistency now series-aware, not column-based); cross-series duplicate timestamps (different meters/metrics) no longer flagged as corruption — true within-series duplicates still are; forecasting/anomaly sufficiency now assessed per logical series, never whole-dataset row count.

**Critical correction applied mid-build:** replaced the initial 80/20 train/test split with a proper chronological TRAIN(70%)/VALIDATION(15%)/TEST(15%) split. Model selection uses VALIDATION RMSE only; final reported MAE/RMSE/MAPE/R2 come from the untouched TEST set. Regression tests confirm selection is unaffected by corrupting the test segment.

**Verified:** 210/210 tests passing. End-to-end `--analyze` run on `data/example.csv`: Seasonal Naive selected (validation RMSE 4.07 vs Persistence 11.67, Linear Lag 7.30), test MAE 3.18/RMSE 3.87/MAPE 6.61%/R2 0.73, 0 anomalies (peak z=2.19, below threshold), excess energy 57.7 kWh. All 8 PNGs generated, non-empty, real image content. `results.json` values cross-checked identical to terminal output. Sparse (5-row) and LONG-format multi-series datasets both handled honestly (INSUFFICIENT_DATA reported without crash; 4 independent series listed and individually selectable via `--series`).

**Known limitations:** seasonal period only assumed for 15-min/30-min/hourly cadences; per-series MIN_ROWS check is a flat threshold (20), not adaptive; Linear-Lag is a single-lag AR(1), not a general ML baseline; excess energy restricted to POWER role with canonical kW unit only.

## Session 4 — Day 3: Specialized Energy Agents

**Added:** `agents/` package — `base.py` (common AgentResult contract: agent_id/status/finding/evidence/metrics/provenance/limitations/warnings; status one of DONE/LIMITED/UNAVAILABLE/INSUFFICIENT_DATA/ERROR), `forecast_agent.py` and `anomaly_agent.py` (wrap existing Day-2 results, zero recomputation), `equipment_agent.py` (reports only identifier/location/status/occupancy context genuinely present; never infers schedule deviation since no schedule definition exists anywhere in the system), `cost_agent.py` and `carbon_agent.py` (excess kWh x configured tariff/emission factor; UNAVAILABLE without one), `impact_config.py` (optional config `impact` block: tariff_per_kwh, tariff_currency, carbon_factor_kg_per_kwh), `runner.py` (capability-aware execution of all five). `run_demo.py --analyze` now prints a "SPECIALIZED ENERGY AGENTS" section after Day-2 analytics.

**Verified:** 224/224 tests passing (14 new agent tests). Rich-data demo (`data/example.csv` + configured tariff/carbon factor): Forecast Agent DONE (Seasonal Naive, RMSE 3.87), Anomaly Agent DONE (0 anomalies, peak z=2.19), Equipment Agent LIMITED (no meter/device column — single asset), Cost Agent DONE ($10.39 = 57.7 kWh x $0.18), Carbon Agent DONE (24.24 kg CO2e = 57.7 kWh x 0.42 kg/kWh). Sparse 5-row demo: Forecast/Anomaly INSUFFICIENT_DATA, Equipment LIMITED, Cost/Carbon UNAVAILABLE — no crashes, no fabricated findings. LONG-format dataset: Equipment Agent DONE (meter_id present).

**Known limitations:** no agent-level confidence scoring yet (explicitly deferred); Cost/Carbon agents only read tariff/carbon factor from config, not from a mapped TARIFF/CARBON_FACTOR data column even when Day 1 detects one exists; Equipment Agent never attempts schedule-deviation detection since no schedule model exists yet.

**Day 4 (not started):** multi-agent orchestrator, cross-agent evidence correlation/agreement, confidence scoring, incident intelligence, AI Energy Engineer.

## Session 5 — Day 4: Collaborative Multi-Agent Intelligence

**Added:** `orchestration/` package — `evidence.py` (SharedEvidence model extracted from existing AgentResult objects, zero new computation), `correlation.py` (incidents formed only when anomalies_detected>0 for a series; everything else attaches as supporting evidence only within an overlapping time window), `agreement.py` (Agent Agreement Score: forecast_agent + anomaly_agent only, documented exclusion of equipment/cost/carbon agents since they're contextual or mechanically derived from the same excess-energy number; unavailable agents dropped from both numerator and denominator, never counted as disagreement), `completeness.py` (6-category weighted score, core data-dependent categories at 85% combined weight vs optional config-dependent cost/carbon at 15%, so a missing tariff never misleadingly tanks the score), `confidence.py` (Decision-Support Confidence — explicitly labeled not a probability — 45% data quality / 30% agreement / 25% completeness, with agreement's weight redistributed when not computable), `recommendations.py` (deterministic, finding/possible_explanation/recommended_action kept strictly separate; hypotheses only offered with genuine equipment-context grounding, always labeled unconfirmed), `energy_engineer.py` (deterministic WHAT/WHERE/WHEN/WHY/impact/evidence/missing-context/confidence/root-cause/action narrative — root cause is always "not confirmed by available evidence" since no schedule model exists anywhere in this system), `orchestrator.py` (single CollaborationResult assembling all of the above from one `run_all_agents()` call), `terminal.py` (MULTI-AGENT COLLABORATION + AI ENERGY ENGINEER sections), `results.py` (results.json "collaboration" section + PNG generation, both reading the same CollaborationResult). 5 new research PNGs (09-13). `run_demo.py --analyze` now computes one `CollaborationResult` that drives the specialized-agent report, collaboration report, AI Energy Engineer, PNGs, and results.json — no independent recomputation anywhere.

**Verified:** 264/264 tests passing (40 new Day-4 tests). Three end-to-end scenarios run: (A) Normal operation on `data/example.csv` — 0 incidents, 100% agreement, 75% completeness (no tariff configured), 92.8/100 confidence, AI Energy Engineer correctly reports normal operation. (B) Unexplained spike (synthetic, injected sustained anomaly, no equipment context) — 1 incident correlated, CRITICAL severity, 93.8/100 confidence, root cause explicitly unconfirmed, recommendation stays generic (no equipment to point to). (C) Rich-context CONTROLLED/SYNTHETIC incident (explicit meter_id + configured tariff/carbon factor) — all 5 agents DONE, 100% completeness, 100% confidence, equipment identifier surfaced as a labeled unconfirmed HYPOTHESIS (never a stated cause), cost ($169.69) and carbon (293.1 kg) impacts computed. All 13 PNGs (8 Day 1/2 + 5 Day 4) verified non-empty; results.json cross-checked identical to terminal output in every scenario.

**Known limitations:** Agent Agreement Score covers only 2 of 5 agents by design (documented exclusion rationale above) — a richer agreement model across all agents is a candidate for Day 5. No cross-incident correlation (each series' incident is independent; no attempt to link incidents across series/time). AI Energy Engineer text generation is deterministic/templated, not LLM-based, per spec. Incident correlation is single-trigger (anomaly only) — forecast deviation alone never creates an incident.

**Day 5 (not started):** anything beyond Day 4 scope — this project's specification did not define Day 5.
