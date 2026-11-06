# PROJECT_STATE.md

_Last updated: end of Day 4 (collaborative multi-agent intelligence)._

## Current Milestone

**Day 1 — Universal Data Understanding: COMPLETE.**
**Day 2 — Forecasting, Anomaly Detection, Excess Energy: COMPLETE.**
**Day 3 — Specialized Energy Agents: COMPLETE.**
**Day 4 — Collaborative Multi-Agent Intelligence: COMPLETE.**

`run_demo.py <file> --analyze` now runs the full stack in one command:
Day 1 (understand) -> Day 2 (forecast/anomaly/excess energy) -> Day 3
(5 specialized agents) -> Day 4 (evidence correlation, incidents, agent
agreement, evidence completeness, decision-support confidence,
recommendations, deterministic AI Energy Engineer) -> PNGs (13) +
results.json, all from one computed `CollaborationResult`.

264/264 tests passing. See `docs/CHANGELOG.md` "Session 5" for full
Day 4 detail and the three verified end-to-end scenarios. No Day 5 was
specified for this project.

## Completed Items

- All 13 `src/` modules implemented: `loader.py`, `profiler.py`,
  `schema.py`, `role_detection.py`, `mapping.py`, `units.py`,
  `cleaning.py`, `canonical.py`, `long_format.py`, `quality.py`,
  `capabilities.py`, `terminal.py`, `pipeline.py`.
- `run_demo.py` CLI works end-to-end for CSV and XLSX, with `--sheet`
  and `--config` options, and never leaks a raw traceback.
- **Long-format semantic mapping** (this session's fix): LONG-format
  datasets (`timestamp | meter_id | metric | value | unit`) now get
  real role/unit inference **per metric value**, grouped into logical
  series by metric + meter/device/equipment identifier. This makes
  Forecasting, Anomaly Detection, and Equipment Analysis correctly
  report `AVAILABLE` for LONG data containing recognizable
  power/energy metrics — previously they incorrectly reported
  `UNAVAILABLE` because capability assessment only looked at physical
  column roles, and LONG format's `metric`/`value` columns have no
  fixed role by design. See `docs/ARCHITECTURE.md` → "Format Handling"
  and `docs/RESEARCH_SPEC.md` → principle 8 for the full explanation.
- An optional `metric_mappings` block in the JSON config lets a user
  override a specific metric value's role/unit, mirroring the existing
  `mappings` override for physical columns.
- 15/15 required test scenarios covered, plus dedicated long-format
  regression tests (mixed recognized/unrecognized metrics, multiple
  meters, optional unit column, config override, invalid config
  rejection).
- Full docs: `RESEARCH_SPEC.md`, `ARCHITECTURE.md` (both current),
  `PROJECT_STATE.md` (this file), `NEXT_TASKS.md`, `CHANGELOG.md`,
  top-level `README.md`.

## Test Status

**101 / 101 tests passing** (pytest, `tests/` directory).

```
python3 -m pytest -q
........................................................................ [ 71%]
.............................                                            [100%]
101 passed in ~1.3s
```

Breakdown by file: `test_loader.py` (9), `test_profiler.py` (8),
`test_schema.py` (7), `test_role_detection.py` (8), `test_mapping.py`
(7), `test_units.py` (9), `test_cleaning.py` (7), `test_quality.py` (5),
`test_capabilities.py` (4), `test_canonical.py` (5), `test_pipeline.py`
(18), `test_long_format.py` (12, new this session).

## CLI Verification (this session)

Run manually end-to-end and output inspected for correctness:

| Dataset | Command | Result |
|---|---|---|
| WIDE CSV | `python run_demo.py data/example.csv` | 963→960 rows, 3 duplicates removed, 3 missing flagged, Quality 97.9/100 GOOD, Forecasting/Anomaly/Weather/Schedule AVAILABLE |
| LONG CSV | `python run_demo.py tests/fixtures/long_format_meter.csv` | 4 logical series resolved (power_kw/energy_kwh × M1/M2), Forecasting/Anomaly/Equipment now AVAILABLE (previously the gap this session fixed) |
| LONG CSV (mixed) | `python run_demo.py tests/fixtures/long_format_mixed_metrics.csv` | active_power→POWER, consumption→ENERGY resolved per meter (3 meters); widget_index correctly left UNKNOWN/unresolved for every meter |
| LONG CSV (unit column) | `python run_demo.py tests/fixtures/long_format_with_unit_column.csv` | metric names "power"/"energy" carry no unit hint; optional `unit` column correctly supplies kW/kWh |
| XLSX (single sheet) | `python run_demo.py tests/fixtures/sample_single_sheet.xlsx` | Quality 100/100 GOOD |
| XLSX (multi-sheet, no `--sheet`) | `python run_demo.py tests/fixtures/sample_multi_sheet.xlsx` | Clean `ERROR: ... multiple sheets ...` message, exit code 1, no traceback |
| XLSX (multi-sheet, `--sheet SiteB`) | `python run_demo.py tests/fixtures/sample_multi_sheet.xlsx --sheet SiteB` | Loads SiteB correctly |
| Messy CSV (missing values) | `python run_demo.py tests/fixtures/missing_values.csv` | 4 MISSING_VALUE flagged, never filled, Completeness 87% |
| Messy CSV (duplicate rows) | `python run_demo.py tests/fixtures/duplicate_rows.csv` | 2 REMOVED_DUPLICATE, Forecasting UNAVAILABLE (< 20 rows after cleaning — correct, honest verdict) |
| Missing file | `python run_demo.py data/does_not_exist.csv` | `ERROR: File not found: ...`, exit code 1 |
| Unsupported extension | `python run_demo.py /tmp/bad.txt` | `ERROR: Unsupported file extension '.txt' ...`, exit code 1 |

## Files Added (this session, on top of prior Day 1 work)

- `src/long_format.py` (new module)
- `role_detection.match_role_by_label()` (new function, additive —
  existing `detect_role()` untouched)
- `profiler._detect_format()` extended to also detect an optional unit
  column for LONG format
- `capabilities.assess_capabilities()` gained an optional `extra_roles`
  parameter (backward-compatible — existing call sites unaffected)
- `pipeline.py`: wires `long_format.build_long_format_mapping()` in,
  merges its canonical rows, passes `extra_roles` to capabilities
- `terminal.py`: new "Long-Format Metric Mapping" report section
- `tests/test_long_format.py` (new, 12 tests)
- `tests/fixtures/long_format_mixed_metrics.csv`,
  `tests/fixtures/long_format_with_unit_column.csv` (new fixtures)
- Docs updated: `RESEARCH_SPEC.md`, `ARCHITECTURE.md`

## Known Issues / Limitations (honest, not swept under the rug)

- **LONG-format quality scoring is column-based, not series-aware.**
  `quality.py`'s "Mapping Coverage" factor still counts physical columns
  (`metric`/`value` remain "unresolved" columns even though their
  content is largely resolved at the metric-value level). A LONG
  dataset can show `Mapping Coverage: 50%` and `Status: FAIR` even when
  every metric is successfully classified. This is a scoring cosmetic
  issue, not a correctness issue — capabilities and canonical data are
  already correct. Flagged as a Day 2 cleanup item, not fixed now to
  keep this session's change scoped to the one requested gap.
- **LONG-format duplicate-timestamp handling can look alarming.**
  Multiple metrics/meters sharing one timestamp (by design, in LONG
  format) are counted as `UNRESOLVED` in the cleaning report and drag
  down the "Temporal Consistency" quality factor (e.g. 5% in the demo
  above). This is the pre-existing, deliberately conservative behavior
  from before this session (rows are flagged, never dropped) — it just
  becomes more visible now that LONG datasets are otherwise scored
  higher. Not a bug; a candidate for a LONG-aware duplicate-timestamp
  rule in Day 2.
- **Per-series row-count thresholds aren't checked.** Forecasting/
  Anomaly Detection's `MIN_ROWS_FOR_TIME_SERIES` (20) is checked against
  the whole dataset's row count, not each logical series' row count. A
  LONG dataset with one well-populated series and one nearly-empty
  series will still show both capabilities as blanket AVAILABLE. Not
  addressed this session (out of scope for the requested fix).
- **LONG format detection itself is a heuristic** (profiler.py): it
  requires an exact column-name match (`metric`/`parameter`/... and
  `value`/`reading`/...). A LONG dataset using different column names
  (e.g. `sensor_type`/`reading_value`) would currently profile as
  UNKNOWN/WIDE rather than LONG, and long_format.py would not engage.
  This was a pre-existing limitation, unchanged this session.
- Equipment/site/building/floor/zone/room hierarchy is only ever
  populated from columns that are actually present — by design, never
  fabricated (spec section 9).

## Design Decisions

See `docs/ARCHITECTURE.md` for full detail. Headline decisions from
this session:

- Long-format semantic mapping lives in its own module
  (`long_format.py`) rather than being folded into `role_detection.py`
  or `mapping.py`, because it operates on a fundamentally different
  unit (metric *values*, grouped into logical series) than either
  existing module (which both operate on physical *columns*).
- `long_format.py` reuses `role_detection`'s exact same `_NAME_RULES`
  pattern table via the new `match_role_by_label()` function, so a
  metric value like `"power_kw"` is judged with the identical evidence
  a column named `power_kw` would be — no separate, drifting rule set.
- `long_format.canonical_df` uses the exact same column layout as
  `canonical.build_canonical()`'s output, so `pipeline.py` can
  `pd.concat` them directly with no reconciliation step.
- Capability assessment takes an additive `extra_roles` set rather than
  requiring `mapping_result` itself to somehow encode per-metric roles
  — keeps `MappingResult` meaning "physical column mapping" and
  `LongFormatMappingResult` meaning "logical series mapping" as two
  clearly separate, individually-testable concerns.

## Exact Run Commands

```bash
# Run the full test suite
cd energyos_research_demo
python3 -m pytest -q

# Regenerate test fixtures (if ever needed)
python3 tests/fixtures/generate_fixtures.py

# Run the CLI
python3 run_demo.py data/example.csv
python3 run_demo.py data/example.xlsx --sheet Sheet1
python3 run_demo.py data/example.csv --config configs/example.json
python3 run_demo.py tests/fixtures/long_format_meter.csv
```

## Next Task

See `docs/NEXT_TASKS.md` for the full Day 2 scope (Forecasting Agent +
Anomaly Detection Agent). **Day 2 has not been implemented** — only
described, per instruction.
