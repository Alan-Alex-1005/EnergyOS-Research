from pathlib import Path
import math
import textwrap

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "benchmark_results"
SUMMARY_CSV = BENCHMARK_DIR / "benchmark_summary.csv"
ALL_MODELS_CSV = BENCHMARK_DIR / "benchmark_all_models.csv"
OUT = ROOT / "presentation_outputs"


def wrap_label(text, width=20):
    return "\n".join(textwrap.wrap(str(text), width=width))


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def ensure_files():
    if not SUMMARY_CSV.is_file():
        raise SystemExit(
            f"Missing file: {SUMMARY_CSV}\n"
            "Run python benchmark_models.py first."
        )
    if not ALL_MODELS_CSV.is_file():
        raise SystemExit(
            f"Missing file: {ALL_MODELS_CSV}\n"
            "Run python benchmark_models.py first."
        )


def load_data():
    summary = pd.read_csv(SUMMARY_CSV)
    all_models = pd.read_csv(ALL_MODELS_CSV)

    for col in [
        "validation_rmse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "original_rows",
        "usable_rows",
        "missing_values",
        "duplicate_timestamps",
        "gap_count",
        "train_n",
        "validation_n",
        "test_n",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    for col in [
        "validation_rmse",
        "test_rmse",
        "test_mae",
        "test_r2",
    ]:
        if col in all_models.columns:
            all_models[col] = pd.to_numeric(all_models[col], errors="coerce")

    return summary, all_models


def prettify_style():
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["font.size"] = 11


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_test_rmse(summary):
    df = summary.copy()
    df = df.sort_values("test_rmse", ascending=True)

    x = [wrap_label(v, 18) for v in df["dataset"]]
    y = df["test_rmse"]
    labels = df["selected_model"]

    plt.figure()
    bars = plt.barh(x, y)
    plt.xlabel("Test RMSE")
    plt.ylabel("Dataset")
    plt.title("EnergyOS Benchmark — Test RMSE by Dataset")

    for bar, model, rmse in zip(bars, labels, y):
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {model} | {rmse:.2f}",
            va="center",
        )

    save_fig(OUT / "01_test_rmse_by_dataset.png")


def plot_selected_models(summary):
    counts = summary["selected_model"].value_counts()

    plt.figure()
    bars = plt.bar(counts.index, counts.values)
    plt.ylabel("Number of datasets selected")
    plt.xlabel("Model")
    plt.title("Which Forecasting Models Were Selected?")

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{int(h)}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=15, ha="right")
    save_fig(OUT / "02_selected_models_count.png")


def plot_trustai_status(summary):
    status_counts = summary["trustai"].fillna("UNKNOWN").value_counts()

    plt.figure()
    bars = plt.bar(status_counts.index, status_counts.values)
    plt.ylabel("Number of datasets")
    plt.xlabel("TrustAI status")
    plt.title("TrustAI Reliability Decisions Across Datasets")

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{int(h)}",
            ha="center",
            va="bottom",
        )

    save_fig(OUT / "03_trustai_status.png")


def plot_r2(summary):
    df = summary.copy().sort_values("test_r2", ascending=True)

    x = [wrap_label(v, 18) for v in df["dataset"]]
    y = df["test_r2"]

    plt.figure()
    bars = plt.barh(x, y)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Test R²")
    plt.ylabel("Dataset")
    plt.title("Forecast Generalization Quality (Test R²)")

    for bar, r2 in zip(bars, y):
        xpos = bar.get_width()
        txt = f"{r2:.3f}"
        offset = 0.02 if xpos >= 0 else -0.25
        plt.text(
            xpos + offset,
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
        )

    save_fig(OUT / "04_test_r2_by_dataset.png")


def plot_validation_vs_test(summary):
    df = summary.copy()

    indices = list(range(len(df)))
    width = 0.38

    plt.figure()
    plt.bar(
        [i - width / 2 for i in indices],
        df["validation_rmse"],
        width=width,
        label="Validation RMSE",
    )
    plt.bar(
        [i + width / 2 for i in indices],
        df["test_rmse"],
        width=width,
        label="Test RMSE",
    )

    plt.xticks(indices, [wrap_label(v, 18) for v in df["dataset"]], rotation=0)
    plt.ylabel("RMSE")
    plt.xlabel("Dataset")
    plt.title("Validation vs Test RMSE")
    plt.legend()

    save_fig(OUT / "05_validation_vs_test_rmse.png")


def plot_model_comparison_grid(all_models):
    datasets = list(all_models["dataset"].dropna().unique())
    models = [
        "Persistence",
        "Seasonal Naive",
        "Linear Lag (AR1)",
        "Calendar Profile (Hour x Weekday)",
    ]

    n = len(datasets)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.8 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, dataset in zip(axes, datasets):
        subset = all_models[all_models["dataset"] == dataset].copy()

        vals = []
        selected_flags = []
        for m in models:
            row = subset[subset["model"] == m]
            if row.empty:
                vals.append(float("nan"))
                selected_flags.append(False)
            else:
                vals.append(row["validation_rmse"].iloc[0])
                selected_flags.append(bool(row["selected"].iloc[0]))

        bars = ax.bar(range(len(models)), vals)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([wrap_label(m, 16) for m in models], rotation=0)
        ax.set_ylabel("Validation RMSE")
        ax.set_title(wrap_label(dataset, 28))

        for i, (bar, is_sel, val) in enumerate(zip(bars, selected_flags, vals)):
            if pd.notna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            if is_sel:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0,
                    "SELECTED",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8,
                )

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle("Per-Dataset Model Comparison (Validation RMSE)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "06_model_comparison_grid.png"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    plt.close()


def plot_data_quality(summary):
    df = summary.copy()

    completeness_pct = []
    for _, row in df.iterrows():
        usable = safe_float(row.get("usable_rows"))
        missing = safe_float(row.get("missing_values"))
        if usable is None or missing is None or (usable + missing) == 0:
            completeness_pct.append(float("nan"))
        else:
            completeness_pct.append(100.0 * usable / (usable + missing))

    plot_df = pd.DataFrame(
        {
            "dataset": df["dataset"],
            "completeness_pct": completeness_pct,
            "gap_count": df.get("gap_count", pd.Series([float("nan")] * len(df))),
        }
    )

    indices = list(range(len(plot_df)))
    width = 0.38

    plt.figure()
    plt.bar(
        [i - width / 2 for i in indices],
        plot_df["completeness_pct"],
        width=width,
        label="Completeness %",
    )
    plt.bar(
        [i + width / 2 for i in indices],
        plot_df["gap_count"],
        width=width,
        label="Gap count",
    )

    plt.xticks(indices, [wrap_label(v, 18) for v in plot_df["dataset"]])
    plt.ylabel("Value")
    plt.xlabel("Dataset")
    plt.title("Dataset Completeness and Gaps")
    plt.legend()

    save_fig(OUT / "07_dataset_quality.png")


def write_summary_slide_text(summary):
    out = OUT / "presentation_summary.txt"

    lines = []
    lines.append("ENERGYOS RESEARCH DEMO v2 — PRESENTATION SUMMARY")
    lines.append("=" * 55)
    lines.append("")

    for _, row in summary.iterrows():
        lines.append(f"Dataset: {row['dataset']}")
        lines.append(f"  Selected model : {row['selected_model']}")
        lines.append(f"  Validation RMSE: {row['validation_rmse']}")
        lines.append(f"  Test RMSE      : {row['test_rmse']}")
        lines.append(f"  Test R²        : {row['test_r2']}")
        lines.append(f"  TrustAI        : {row['trustai']}")
        lines.append(f"  Notes          : {row['trustai_reasons']}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")


def main():
    ensure_files()
    OUT.mkdir(exist_ok=True)
    prettify_style()

    summary, all_models = load_data()

    plot_test_rmse(summary)
    plot_selected_models(summary)
    plot_trustai_status(summary)
    plot_r2(summary)
    plot_validation_vs_test(summary)
    plot_model_comparison_grid(all_models)
    plot_data_quality(summary)
    write_summary_slide_text(summary)

    print("\nAll presentation figures saved in:")
    print(OUT)


if __name__ == "__main__":
    main()