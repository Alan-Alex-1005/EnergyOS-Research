# EnergyOS Research Demo v2 — Research Specification

## Research Theme

**"A Collaborative Multi-Agent AI Framework for Explainable Energy
Intelligence and Decision Support."**

This repository is a **research demonstrator**, not the commercial
EnergyOS product. It exists to prove out, in isolation and with full
transparency, the data-understanding layer that every later agent
(forecasting, anomaly detection, cost, carbon, etc.) will depend on.

## 5-Day Plan

| Day | Scope |
|-----|-------|
| **1 (this repo, DONE)** | Universal Data Understanding: load any CSV/XLSX, profile it, detect timestamps, detect column roles (WIDE format) AND metric-value roles (LONG format), map to a canonical energy model, clean conservatively, normalize units, score data quality, report dataset capabilities. |
| 2 | Forecasting Agent + Anomaly Detection Agent (built on top of the Day 1 canonical dataset). |
| 3 | Equipment Agent, Cost Agent, Carbon Agent. |
| 4 | Multi-agent orchestrator, evidence correlation, confidence scoring across agents. |
| 5 | LLM-based "AI Energy Engineer" explanation layer, PNG chart generation, terminal research narration. |

## Day 1 Design Principles (non-negotiable, carried into every later day)

1. **Never assume column names.** Role detection uses name hints, dtype,
   parseability, numeric distributions, and cardinality — never a fixed
   schema.
2. **Never fabricate.** Missing equipment, location, tariff, or carbon
   data is left `null`, not guessed. Every mapped role carries an
   explicit confidence and evidence trail.
3. **Never silently transform.** Cleaning actions are classified
   (`REMOVED_DUPLICATE`, `INVALID_TIMESTAMP`, `MISSING_VALUE`,
   `OUT_OF_RANGE`, `UNRESOLVED`) and counted — nothing is fixed invisibly.
4. **Numeric columns are never fallback-parsed as timestamps.** This was
   flagged explicitly in the master prompt and is enforced with a
   dedicated rule in `src/schema.py` plus a regression test.
5. **Low-confidence role suggestions are left `UNKNOWN`**, not auto-mapped
   — a human (via `--config`) must confirm anything under the 0.5
   confidence threshold.
6. **The Data Quality Score formula is fully documented** (see
   `src/quality.py` module docstring) — not an opaque model output.
7. **Every derived field is classifiable as** `MEASURED`, `CALCULATED`,
   `ESTIMATED`, `INFERRED`, or `CONFIGURED`. Day 1 canonical records are
   tagged `MEASURED` (raw pass-through) or carry `mapping_source`
   (`AUTO` vs `CONFIG`) as provenance.
8. **Column-level conservatism and metric-value-level semantics are two
   different jobs.** For LONG-format data, `role_detection.py` correctly
   leaves the physical `metric`/`value` columns unresolved (a column
   name like `"value"` is genuinely meaningless on its own) — but
   `long_format.py` still recovers the real semantics by classifying
   each **metric value** (`"power_kw"`, `"active_power"`, …) with the
   same evidence-based approach, grouped into logical series by
   meter/device/equipment identifier where available. A metric value
   that matches nothing stays `UNKNOWN`, exactly like principle 5.

**Regression example (from real testing during development):** a
LONG-format dataset with columns `timestamp | meter_id | metric | value`
and metric values `power_kw` / `energy_kwh` initially reported
Forecasting and Anomaly Detection as `UNAVAILABLE`, because capability
assessment looked only at physical-column roles — and `metric`/`value`
have no fixed role. This was fixed by having `long_format.py` classify
each metric value into a logical series (with its own confidence,
evidence, and unit) and folding those resolved roles into capability
assessment. `metric_mappings` in the JSON config can also override a
specific metric value's role, mirroring physical-column overrides.

## Why This Matters for the M.Tech Research Contribution

The multi-agent framework's explainability claim only holds if every
downstream agent can point back to *why* it trusts a given data field.
Day 1's confidence/evidence/provenance trail is the foundation that
claim rests on — later agents will attach their own
`CALCULATED`/`ESTIMATED`/`INFERRED` provenance on top of this layer
rather than replacing it.
