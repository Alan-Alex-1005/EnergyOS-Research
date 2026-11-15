
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "dfab_thermal_meter.csv"
OUTPUT = ROOT / "dfab_hourly_power.csv"

if not INPUT.is_file():
    raise SystemExit(
        f"Dataset not found: {INPUT}"
    )

# Read only the timestamp and measured main thermal power.
data = pd.read_csv(
    INPUT,
    usecols=[0, 1],
)

timestamp = pd.to_datetime(
    data.iloc[:, 0],
    errors="coerce",
    utc=True,
)

power = pd.to_numeric(
    data.iloc[:, 1],
    errors="coerce",
)

series = pd.Series(
    power.to_numpy(),
    index=timestamp,
    name="Main_Power_kW",
)

series = series[
    series.index.notna()
].sort_index()

if series.empty:
    raise SystemExit("No valid measurements found.")

if series.index.has_duplicates:
    raise SystemExit(
        "Duplicate timestamps detected; "
        "review source before aggregation."
    )

# Compute actual hourly average thermal power.
hourly = series.resample("1h").agg(
    ["mean", "count"]
)

# Require at least 45 valid minute readings per hour.
# Incomplete hours remain missing, not invented.
hourly["Main_Power_kW"] = hourly["mean"].where(
    hourly["count"] >= 45
)

result = hourly[["Main_Power_kW"]].reset_index()
result.columns = ["timestamp", "Main_Power_kW"]

# Preserve timezone information as UTC timestamps.
result["timestamp"] = result["timestamp"].dt.strftime(
    "%Y-%m-%dT%H:%M:%S+00:00"
)

result.to_csv(
    OUTPUT,
    index=False,
)

print("SUCCESS: DFAB hourly dataset prepared.")
print("Output:", OUTPUT)
print("Total hourly intervals:", len(result))
print(
    "Valid hourly observations:",
    result["Main_Power_kW"].notna().sum(),
)
print(
    "Missing or incomplete hours:",
    result["Main_Power_kW"].isna().sum(),
)
