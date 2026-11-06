# EnergyOS Research Demo v2

### A Collaborative Multi-Agent AI Framework for Explainable Energy Intelligence and Decision Support

---

## 1. What this project does

This is a research demonstrator that takes an unfamiliar energy dataset
(CSV or XLSX — any building, any meter layout) and turns it into an
explainable, end-to-end analysis:

1. **Data Intelligence** — automatically profiles the file, detects the
   timestamp column, maps columns/values to roles (power, energy,
   temperature, status, meter ID, ...), cleans it conservatively, and
   scores its own data quality — no fixed schema required.
2. **Forecast + Anomaly** — a chronological TRAIN/VALIDATION/TEST split
   (never shuffled), transparent baseline forecast models compared
   honestly, and a robust statistics-based anomaly detector fit only on
   training data.
3. **Specialized Agents** — Forecast, Anomaly, Equipment/Context, Cost,
   and Carbon agents, each capability-aware (they say `UNAVAILABLE`
   rather than guess).
4. **Multi-Agent Collaboration** — shared evidence, incident
   correlation, an Agent Agreement Score, an Evidence Completeness
   Score, and a Decision-Support Confidence score, all documented and
   testable.
5. **AI Energy Engineer** — a deterministic (no LLM) narrative
   answering WHAT/WHERE/WHEN/WHY/impact/evidence/missing
   context/confidence/root cause/recommended action.

Every number in the output is computed from your actual input file.
Nothing is hard-coded, and nothing is fabricated when data is missing.

## 2. Workflow

```
CSV / XLSX
   -> Data Intelligence (profiling, mapping, cleaning, quality score)
   -> Forecast + Anomaly Detection (chronological, leakage-guarded)
   -> Specialized Agents (Forecast / Anomaly / Equipment / Cost / Carbon)
   -> Multi-Agent Collaboration (evidence, incidents, agreement, completeness)
   -> Decision-Support Confidence
   -> AI Energy Engineer (unified explanation)
   -> Terminal report + PNGs + results.json
```

## 3. Requirements

- **Python 3.10 or newer** (tested on 3.12)
- Packages (installed via `requirements.txt`): `pandas`, `numpy`,
  `matplotlib`, `openpyxl`, `pytest`

No database, no external API keys, no internet access needed to run
the demo or your own files.

## 4. Installation — Windows

Open **PowerShell** or **Command Prompt** in the project folder (the
folder containing this README), then run:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` isn't recognized, try `py` instead (`py -m venv venv`, then
`venv\Scripts\activate`).

## 5. Installation — Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Quick Start

The single easiest command, using the bundled demo dataset:

```bash
python run_demo.py data/example.csv --analyze
```

This prints the full report (Data Intelligence -> Forecast -> Anomaly ->
Agents -> Collaboration -> AI Energy Engineer) and writes PNGs +
`results.json` under `outputs/<run_id>/`.

## 7. Running your own CSV or XLSX

```bash
python run_demo.py path\to\your_file.csv --analyze
python run_demo.py path\to\your_file.xlsx --sheet Sheet1 --analyze
```

You don't need to tell it which column is what — it automatically
profiles the schema (timestamp, power/energy columns, meter IDs,
status columns, etc.) and reports exactly which analyses it can
support for your data (`ANALYTICAL SERIES AVAILABLE` in the terminal
output). If your file has more than one usable series (e.g. multiple
meters), list them and pick one explicitly:

```bash
python run_demo.py your_file.csv --analyze --series "POWER:Main_Power_kW"
```

To override automatic mapping, control the train/validation/test
split, or set a real tariff/carbon factor, pass a JSON config (see
`configs/example.json`):

```bash
python run_demo.py your_file.csv --analyze --config configs/example.json
```

## 8. Running with dataset provenance (recommended for real data)

`run_real_demo.py` is a thin wrapper around `run_demo.py --analyze` — it
adds no new analysis logic, it just records where your data came from
and prints it in the report:

```bash
python run_real_demo.py --input "path\to\energy.csv" --name "My Dataset" --provider "My Utility" --source "https://example.org/dataset"
```

With a real, documented tariff and/or carbon factor (only pass these if
you actually have a documented value — the Cost/Carbon agents honestly
report `UNAVAILABLE` if you don't):

```bash
python run_real_demo.py --input data.csv --name "X" --provider "Y" --source "Z" --tariff-per-kwh 0.18 --tariff-currency USD --carbon-factor-kg-per-kwh 0.42
```

Run `python run_real_demo.py --help` for the full option list.

### Reproducing the Building Data Genome Project 2 (BDG2) experiment

This demo's evaluation framework was designed to be tested against the
real, public **Building Data Genome Project 2** dataset (Miller et al.,
*Scientific Data*, 2020 — see citation in section 16). BDG2's actual
measurement files are distributed via Git LFS / Zenodo / Kaggle, which
this project's development sandbox could not reach (see section 13 and
`docs/RESEARCH_CLAIMS.md`). To reproduce the experiment yourself once
you have downloaded a BDG2 electricity meter CSV locally:

```bash
python run_real_demo.py --input "path\to\bdg2_electricity_raw.csv" \
    --name "Building Data Genome Project 2 (raw electricity)" \
    --provider "Miller et al. / Building and Urban Data Science (BUDS) Lab" \
    --source "https://github.com/buds-lab/building-data-genome-project-2" \
    --series "POWER:<meter_column_name>"
```

If the file has one timestamp column and many per-building columns
(BDG2's typical wide layout), point `--series` at the specific building
column you want analyzed once the automatic profiler lists the
available series (run once without `--series` first to see the list).

## 9. Understanding the terminal output

| Section | What it means |
|---|---|
| **Data Quality** | A 0-100 score from 6 documented, weighted factors (timestamp validity, mapping coverage, completeness, duplicate quality, unit confidence, temporal consistency) — see `src/quality.py`. |
| **Forecast** | The model selected by lowest **validation** RMSE; MAE/RMSE/MAPE/R2 reported are from the **held-out test period**, never seen during selection. |
| **Anomaly** | Robust z-scores (median/MAD) fit on **training residuals only**, applied to test residuals. Severity bands (NORMAL/LOW/MEDIUM/HIGH/CRITICAL) are documented prototype thresholds, not universal constants. |
| **Equipment Agent** | Reports only identifier/location context that actually exists in your file — `LIMITED` if no meter/device column was found, never fabricated. |
| **Cost Agent / Carbon Agent** | Only run (`DONE`) if you supplied a real tariff / carbon factor via config; otherwise `UNAVAILABLE`. |
| **Agent Agreement** | Whether Forecast and Anomaly agents' independent directional reads of the test period agree — the only two agents used, by design (see `docs/ARCHITECTURE.md` / `RESEARCH_CLAIMS.md` for why equipment/cost/carbon are excluded). |
| **Evidence Completeness** | How much of six evidence categories (forecast, anomaly, equipment, energy impact, cost, carbon) are actually present, weighted so a missing tariff/carbon config isn't misleadingly punished. |
| **Decision-Support Confidence** | A weighted blend of Data Quality + Agent Agreement + Evidence Completeness. **This is explicitly NOT a statistical probability that any diagnosis is correct** — it's a transparency tool showing why the system is or isn't confident in what it found. |
| **AI Energy Engineer** | The single unified explanation. If root cause can't be confirmed (it never can be, in this system — there's no schedule model), it says so explicitly rather than guessing. |

## 10. Understanding the outputs

- `outputs/<run_id>/` — one folder per run (named from your input
  filename + a timestamp).
- `01_dataset_profile.png` ... `13_decision_confidence.png` — up to 13
  research PNGs (some are honestly skipped with a reason if the data
  doesn't support them — e.g. no incident means no incident-summary
  chart).
- `results.json` — every number shown in the terminal, in machine-
  readable form, including full agent/evidence/collaboration detail.

## 11. Research methodology (brief)

- Time-series split is always **chronological**: TRAIN (70%) ->
  VALIDATION (15%) -> TEST (15%) by default, configurable, **never
  shuffled**.
- Model **selection** uses validation RMSE only. Final reported metrics
  come from the test period, which is never consulted during selection
  (verified by regression tests — see `docs/RESEARCH_CLAIMS.md`).
- Anomaly thresholds (median/MAD) are fit on training residuals only,
  never on the data being classified.
- Full methodology and what is/isn't demonstrated: see
  `docs/RESEARCH_SPEC.md` and `docs/RESEARCH_CLAIMS.md`.

## 12. Level 1 vs Level 2 vs Level 3 (research evaluation)

`src/evaluation/` implements a three-level comparison, all derived from
the exact same underlying analysis (no model is re-run to make one
level look better):

- **Level 1 — Isolated Analytics**: raw forecast + anomaly + excess-
  energy output, no agents.
- **Level 2 — Independent Specialized Agents**: the 5 agents run
  independently, no cross-agent correlation.
- **Level 3 — Collaborative Multi-Agent Framework**: the complete
  system (evidence correlation, agreement, completeness, confidence,
  recommendations, AI Energy Engineer).

The comparison measures **Explanation Coverage**, **Evidence
Diversity**, and **Recommendation Support** — framework/evidence
metrics, never prediction accuracy. `src/evaluation/real_data_runner.py`
runs this comparison against real, provenance-documented datasets (see
section 8); `src/evaluation/scenarios.py` provides synthetic fixtures
for automated software testing only — never presented as research
results.

## 13. Limitations

See `docs/RESEARCH_CLAIMS.md` for the full, explicit list of what this
project does **not** demonstrate (forecasting superiority, statistically
significant accuracy improvement, guaranteed root-cause diagnosis,
real-world savings, or superiority over any existing BEMS). In brief:
this is a research demonstrator with transparent baseline models, not a
production system, its confidence score is a decision-support aid, not
a probability, and real public-dataset validation (attempted with
BDG2) was blocked by this development environment's network
restrictions, not by any failure of the framework itself.

## 14. Running tests

```bash
python -m pytest -q
```

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `py` not found | Reinstall Python from python.org and check "Add to PATH" during install (Windows), or use `python3` (Linux/macOS). |
| `ModuleNotFoundError` | Make sure the virtual environment is activated and `pip install -r requirements.txt` completed without errors. |
| Nothing happens / wrong file used | Run commands from the project's root folder (where this README lives) — file paths are relative to your current directory. |
| `ERROR: Unsupported file extension` | Only `.csv` and `.xlsx` are supported. |
| `Forecasting: UNAVAILABLE` / `INSUFFICIENT_DATA` | Your series needs a detected timestamp column and at least ~20 non-null observations of a POWER/ENERGY/DEMAND series. Check the `ANALYTICAL SERIES AVAILABLE` listing in the terminal output. |
| `No POWER/ENERGY/DEMAND analytical series were found` | The file needs at least one column recognized as a power/energy/demand measurement — check column names against the role hints in `src/role_detection.py`, or use `--config` to map a column explicitly. |

## 16. Real-data provenance and citation

This demo ships with a synthetic `data/example.csv` for a self-contained
Quick Start. The evaluation framework is designed for real public
datasets — the one identified for validation during development was:

> **Building Data Genome Project 2** — Miller, C., Kathirgamanathan, A.,
> Picchetti, B. et al. *The Building Data Genome Project 2, energy meter
> data from the ASHRAE Great Energy Predictor III competition.* Sci Data
> 7, 368 (2020). https://doi.org/10.1038/s41597-020-00712-x
> Repository: https://github.com/buds-lab/building-data-genome-project-2

Retrieval of the actual measurement data (stored via Git LFS, Zenodo,
and Kaggle) was blocked by this development sandbox's network
restrictions — see `docs/RESEARCH_CLAIMS.md` for the exact detail. No
BDG2 measurements were used in this repository's results; section 8
above documents exactly how to reproduce the experiment once you have
downloaded the real file yourself.
