# Architecture — Day 1

## Pipeline

```
ANY CSV / XLSX
      |
loader.py            -> LoadResult (DataFrame + metadata + provenance)
      |
profiler.py           -> Profile (per-column stats, WIDE/LONG format hint)
      |
schema.py              -> TimestampInfo (detected column, confidence, sampling interval)
      |
role_detection.py      -> list[RoleSuggestion] (role, confidence, evidence, unit, per non-timestamp column)
      |
mapping.py              -> MappingResult (auto suggestions + optional JSON config overrides, merged)
      |
cleaning.py              -> (cleaned DataFrame, CleaningReport) (duplicates removed, invalid timestamps
      |                       excluded, missing/out-of-range values flagged not fixed)
      |
canonical.py              -> canonical_df (long-format: timestamp, source_column, role, raw/normalized
      |                       value+unit, equipment/location context, quality, provenance)
      |
long_format.py            -> LongFormatMappingResult (LONG-format metric/value semantic mapping — see
      |                       "Format Handling" below; merges additional canonical rows when applicable)
      |
quality.py                -> QualityResult (0-100 overall + 6-factor breakdown, documented formula)
      |
capabilities.py            -> CapabilityReport (which Day 2+ analyses are possible, with reasons)
      |
terminal.py                 -> CLI report string
```

`src/pipeline.py` wires every stage together (`run_pipeline()`) so both
`run_demo.py` and the test suite exercise the exact same code path — the
CLI and the tests can never drift apart.

## Module Responsibilities

| Module | Responsibility | Never does |
|---|---|---|
| `loader.py` | Read CSV/XLSX into a DataFrame; handle missing/empty/malformed files and multi-sheet XLSX ambiguity | Guess a sheet when ambiguous; leak raw pandas tracebacks |
| `profiler.py` | Per-column statistics (nulls, cardinality, numeric ranges, suspected shapes); WIDE vs LONG format hint | Assign a semantic role |
| `schema.py` | Timestamp column detection, sampling interval, monotonicity, tz handling | Fallback-parse numeric columns as epoch timestamps |
| `role_detection.py` | Assign one of the fixed role taxonomy values per column, with confidence + evidence | Auto-map below the 0.5 confidence threshold |
| `mapping.py` | Merge automatic suggestions with an optional JSON config; config always wins | Accept an invalid role/column silently |
| `units.py` | Deterministic unit conversion to canonical units (kW, kWh, C, V, A, Hz) | Guess an unrecognized unit |
| `cleaning.py` | Remove exact duplicate rows, exclude unparseable-timestamp rows, flag missing/out-of-range values, sort | Fill missing values; drop out-of-range values; silently remove duplicate-timestamp rows |
| `canonical.py` | Build the long-format canonical dataset with raw+normalized value/unit and equipment/location context, from WIDE-mapped physical columns | Fabricate missing equipment/location context |
| `long_format.py` | Semantic mapping for LONG-format data: assign a role/unit per **metric value** (not per column), grouped into logical series by metric + meter/device/equipment identifier; builds the matching canonical rows | Fabricate a role/unit for a metric value that matches no pattern (leaves it `UNKNOWN`, unresolved) |
| `quality.py` | 6-factor weighted 0-100 quality score, fully documented formula | Use an opaque/AI-generated score |
| `capabilities.py` | Determine which Day 2+ analyses (forecasting, cost, carbon, etc.) are possible, with a reason — grounded in both physical-column roles AND resolved LONG-format logical-series roles | Claim a capability without grounding it in the actual mapping |
| `terminal.py` | Render the CLI report, including a "Long-Format Metric Mapping" section when applicable | Print fabricated values |

## Canonical Data Model

Suggested by the master prompt as `CanonicalRecord` / `CanonicalSeries`.
Implemented as:

- **`canonical_df`** (pandas DataFrame, long format) — the primary,
  vectorized representation used throughout the pipeline. One row per
  `(timestamp, source_column)` for every mapped measured role.
- **`CanonicalRecord`** (dataclass in `canonical.py`) — a row-object view
  of `canonical_df`, produced on demand via `to_canonical_records()` for
  inspection/tests/samples. Not materialized for the full dataset by
  default (a 50k-row x 5-measure-column dataset would be 250k Python
  objects — the DataFrame form scales, the dataclass form is for
  human-readable samples).

Fields: `timestamp, source_column, role, raw_value, raw_unit,
normalized_value, normalized_unit, equipment_id, device_id, meter_id,
site, building, floor, zone, room, quality, provenance`.

Context fields (`equipment_id` … `room`) are populated only when the
source dataset actually contains a column mapped to that role — never
fabricated.

## Format Handling (WIDE vs LONG)

`profiler.py` detects LONG format when it finds a low-cardinality
"metric"-like categorical column paired with a single generic "value"
numeric column (e.g. `timestamp | meter_id | metric | value`), and
additionally looks for an optional `unit`/`units`/`uom` column.

Per spec section 6 ("do not force conversion until mapping logic
understands the structure"), `role_detection.py` still correctly leaves
the physical `metric` and `value` **columns** unresolved — neither
column *name* means anything on its own; the meaning lives in the
`metric` column's individual *values*. That column-level conservatism is
intentional and unchanged.

What Day 1 does add (`long_format.py`) is a second, metric-VALUE-level
mapping pass that runs only when `profile.format_hint == "LONG"`:

1. Group rows by `(metric value, identifier columns already resolved at
   the physical-column level — meter/device/equipment id)`. Each group is
   one **logical series** — e.g. `metric="power_kw", meter_id="M1"`.
2. Classify each metric value's text (`"power_kw"`, `"active_power"`,
   `"consumption"`, …) using the exact same name-pattern table
   `role_detection._NAME_RULES` uses for column names
   (`role_detection.match_role_by_label`), so a metric value is judged
   by the same evidence a column name would be.
3. Resolve the series' unit from, in order of trust: an explicit
   `metric_mappings` config override → a consistent value in the
   optional unit column → a unit embedded in the metric text itself
   (e.g. `"power_kw"` → `kW`). If none of these apply, the unit is left
   `None` — never guessed.
4. A metric value that matches no pattern (and has no config override)
   stays `role="UNKNOWN"` and is recorded in `unresolved_metrics` —
   exactly like an unmapped physical column. It **never** produces a
   canonical row.
5. Resolved series feed into two places: (a) `canonical_df`, via
   canonical rows built directly from that group's timestamp/value/
   identifier data (concatenated onto the WIDE-derived canonical rows
   in `pipeline.py`), and (b) `capabilities.assess_capabilities(...,
   extra_roles=...)`, so Forecasting/Anomaly Detection/Equipment
   Analysis are evaluated against the roles the dataset actually
   contains — not just what happened to be resolvable one physical
   column at a time.

An optional `metric_mappings` block in the JSON config
(`{"metric_mappings": {"widget_index": {"role": "OCCUPANCY"}}}`)
overrides automatic classification for a specific metric value, mirroring
how `mappings` overrides a physical column — validated against the same
role taxonomy and rejected with a clear `MappingConfigError` otherwise.

## Error Handling Philosophy

Every module that can fail raises its own typed exception
(`LoaderError`, `MappingConfigError`, `PipelineError`) with a
human-readable message. `run_demo.py` catches `PipelineError` and any
unexpected exception at the top level and prints `ERROR: ...` /
`UNEXPECTED ERROR: ...` to stderr with exit code 1 — a raw pandas
traceback should never reach the terminal in normal CLI usage.

## Extensibility for Day 2+

- New agents consume `canonical_df` + `CapabilityReport` — they check
  `capabilities` before attempting an analysis rather than assuming data
  is present.
- New roles can be added to `role_detection.ROLES` and a new rule row in
  `_NAME_RULES` without touching any other module.
- New unit families extend the tables in `units.py` (`CANONICAL_UNITS` +
  a `_..._TO_...` conversion table).
