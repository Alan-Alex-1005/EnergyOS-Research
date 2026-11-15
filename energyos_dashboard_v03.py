
"""EnergyOS Adaptive Dashboard v0.3.

Uses existing EnergyOS results and exported forecast evidence.
Does not refit or modify forecasting models.

Requires: pandas, plotly
"""

import argparse
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


BG = "#080F1D"
PANEL = "#111E32"
CYAN = "#37D5F3"
ORANGE = "#FFB36B"
RED = "#FF647B"
GREEN = "#57E2B3"
TEXT = "#EAF2FF"
MUTED = "#9BAEC8"


def esc(value):
    return html.escape(str(value), quote=True)


def figure_html(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            family="Segoe UI, Arial",
            color=MUTED,
            size=12,
        ),
        margin=dict(l=65, r=30, t=35, b=50),
        height=390,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.15),
    )

    fig.update_xaxes(
        gridcolor="#26364E",
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#26364E",
        zeroline=False,
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "energyos_evidence",
            },
        },
    )


def panel(title, subtitle, content, wide=False):
    cls = "panel wide" if wide else "panel"
    return (
        f'<section class="{cls}">'
        '<header>'
        f'<h2>{esc(title)}</h2>'
        f'<span>{esc(subtitle)}</span>'
        '</header>'
        f"{content}</section>"
    )


def kpi(title, value, subtitle):
    return (
        '<div class="kpi">'
        f"<small>{esc(title)}</small>"
        f"<strong>{esc(value)}</strong>"
        f"<span>{esc(subtitle)}</span>"
        "</div>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--output",
        default="presentation_outputs/energyos_v03.html",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    evidence_path = Path(args.evidence)

    for path in [results_path, evidence_path]:
        if not path.is_file():
            parser.error(f"File not found: {path}")

    report = json.loads(
        results_path.read_text(encoding="utf-8")
    )

    df = pd.read_csv(evidence_path)

    required = [
        "timestamp",
        "actual",
        "predicted",
        "residual",
    ]

    missing = [c for c in required if c not in df]
    if missing:
        parser.error(f"Missing evidence columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    for col in ["actual", "predicted", "residual"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if df["timestamp"].isna().any():
        parser.error("Evidence contains invalid timestamps")

    if df["timestamp"].duplicated().any():
        parser.error("Evidence has duplicate timestamps")

    df = df.sort_values("timestamp").reset_index(drop=True)

    if df[["actual", "predicted"]].dropna().empty:
        parser.error("No valid actual/predicted observations")

    valid = df[["actual", "predicted", "residual"]].notna().all(axis=1)
    difference = (
        df.loc[valid, "actual"]
        - df.loc[valid, "predicted"]
        - df.loc[valid, "residual"]
    )

    if not difference.empty and difference.abs().max() > 1e-5:
        parser.error(
            "Residuals disagree with actual minus predicted"
        )

    if "is_anomaly" in df:
        anomaly_flags = (
            df["is_anomaly"]
            .astype(str)
            .str.lower()
            .isin(["1", "1.0", "true", "yes"])
        )
    else:
        anomaly_flags = pd.Series(
            False, index=df.index
        )

    analysis = report.get("analysis") or {}
    forecast = analysis.get("forecast") or {}
    anomaly = analysis.get("anomaly") or {}
    collab = report.get("collaboration") or {}
    agents = collab.get("agents") or {}
    quality = report.get("dataset") or {}
    metrics = forecast.get("metrics") or {}

    unit = analysis.get("unit") or ""
    model = forecast.get("selected_model") or "Unavailable"

    trust = "NOT REPORTED"
    for reason in (
        agents.get("forecast_agent") or {}
    ).get("limitations") or []:
        if "Forecast reliability:" in reason:
            trust = reason.split(
                "Forecast reliability:", 1
            )[1].strip()

    # Full test-period forecast overview.
    overview = go.Figure()

    overview.add_trace(go.Scattergl(
        x=df["timestamp"],
        y=df["actual"],
        name="Actual",
        mode="lines",
        connectgaps=False,
        line=dict(color=CYAN, width=1.8),
    ))

    overview.add_trace(go.Scattergl(
        x=df["timestamp"],
        y=df["predicted"],
        name="Predicted",
        mode="lines",
        connectgaps=False,
        line=dict(color=ORANGE, width=1.8),
    ))

    overview.update_yaxes(title=f"Measurement ({unit})")

    # Mark the actual flagged observations.
    flagged = df.loc[anomaly_flags].dropna(
        subset=["actual"]
    )

    if not flagged.empty:
        overview.add_trace(go.Scatter(
            x=flagged["timestamp"],
            y=flagged["actual"],
            mode="markers",
            name="Flagged anomalies",
            marker=dict(
                color=RED,
                size=9,
                symbol="circle-open",
                line=dict(width=2),
            ),
        ))

    overview_html = figure_html(overview)

    # Focused view: first reported anomaly, otherwise
    # the beginning of the test period.
    focus = df.copy()
    if not flagged.empty:
        center = flagged["timestamp"].iloc[0]
        focus = df.loc[
            (df["timestamp"] >= center - pd.Timedelta(days=3))
            & (df["timestamp"] <= center + pd.Timedelta(days=4))
        ]
        focus_description = "Seven-day anomaly-centered window"
    else:
        start = df["timestamp"].min()
        focus = df.loc[
            df["timestamp"] <= start + pd.Timedelta(days=7)
        ]
        focus_description = "First seven days of the test period"

    focused = go.Figure()

    for name, column, color in [
        ("Actual", "actual", CYAN),
        ("Predicted", "predicted", ORANGE),
    ]:
        focused.add_trace(go.Scatter(
            x=focus["timestamp"],
            y=focus[column],
            mode="lines",
            name=name,
            connectgaps=False,
            line=dict(color=color, width=2.5),
        ))

    focus_flags = anomaly_flags.loc[focus.index]
    focus_anomalies = focus.loc[focus_flags].dropna(
        subset=["actual"]
    )

    if not focus_anomalies.empty:
        focused.add_trace(go.Scatter(
            x=focus_anomalies["timestamp"],
            y=focus_anomalies["actual"],
            mode="markers",
            name="Anomaly",
            marker=dict(
                color=RED,
                size=10,
                symbol="diamond",
            ),
        ))

    focused.update_yaxes(title=f"Measurement ({unit})")

    # Residual timeline.
    residuals = go.Figure()
    residuals.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["residual"],
        mode="lines",
        name="Residual",
        line=dict(color=ORANGE, width=1.6),
        connectgaps=False,
    ))
    residuals.add_hline(
        y=0, line_color=MUTED, line_dash="dash"
    )
    residuals.update_yaxes(title=f"Residual ({unit})")

    # Model comparison from original validation metrics.
    candidates = [
        item for item in forecast.get("comparison") or []
        if item.get("validation_rmse") is not None
    ]

    model_chart = None
    if candidates:
        names = [c["model"] for c in candidates]
        scores = [c["validation_rmse"] for c in candidates]

        comparison = go.Figure(go.Bar(
            y=names,
            x=scores,
            orientation="h",
            marker_color=[
                GREEN if name == model else "#566B89"
                for name in names
            ],
            text=[f"{x:.2f}" for x in scores],
            textposition="outside",
        ))

        comparison.update_layout(
            margin=dict(l=185, r=50, t=35, b=50)
        )
        comparison.update_xaxes(
            title=f"Validation RMSE ({unit})",
            range=[0, max(scores) * 1.2],
        )
        model_chart = figure_html(comparison)

    # Specialized agents.
    agent_cards = []
    for name, data in agents.items():
        status = data.get("status", "UNKNOWN")
        finding = data.get("finding", "No finding available")

        agent_cards.append(
            '<div class="agent">'
            f"<h3>{esc(name.replace('_', ' ').title())}</h3>"
            f'<small>{esc(status)}</small>'
            f"<p>{esc(finding)}</p>"
            "</div>"
        )

    engineer = collab.get("ai_energy_engineer") or {}
    conclusion = (
        f"<p>{esc(engineer.get('why', 'Not available'))}</p>"
        "<h3>Recommended action</h3>"
        f"<p>{esc(engineer.get('recommended_action', 'Not available'))}</p>"
        "<h3>Physical root cause</h3>"
        f"<p>{esc(engineer.get('root_cause', 'Not established'))}</p>"
    )

    evidence_completeness = (
        collab.get("evidence_completeness") or {}
    ).get("score", "N/A")

    agreement = (
        collab.get("agreement") or {}
    ).get("score", "N/A")

    confidence = (
        collab.get("decision_confidence") or {}
    ).get("score", "N/A")

    kpis = "".join([
        kpi("DATA QUALITY",
            f"{quality.get('quality_score', 'N/A')}/100",
            quality.get("quality_status", "")),
        kpi("TEST RMSE",
            f"{metrics.get('rmse', 0):.2f} {unit}",
            model),
        kpi("ANOMALIES",
            str(anomaly.get("anomalies_detected", "N/A")),
            f"{len(anomaly.get('events') or [])} events"),
        kpi("TRUSTAI",
            trust.replace("_", " "),
            "Forecast reliability"),
    ])

    blocks = [
        panel(
            "Actual vs Predicted",
            "Full test period · original EnergyOS predictions",
            overview_html,
            wide=True,
        ),
        panel(
            "Forecast Investigation",
            focus_description,
            figure_html(focused),
            wide=True,
        ),
        panel(
            "Forecast Residuals",
            "Actual minus predicted",
            figure_html(residuals),
        ),
    ]

    if model_chart:
        blocks.append(panel(
            "Forecast Model Selection",
            "Validation RMSE · lower is better",
            model_chart,
        ))

    blocks.extend([
        panel(
            "Specialized Energy Agents",
            "Actual agent findings",
            '<div class="agents">'
            + "".join(agent_cards)
            + "</div>",
            wide=True,
        ),
        panel(
            "TrustAI & Evidence",
            "Reliability and collaboration",
            '<div class="stats">'
            + kpi("EVIDENCE COMPLETENESS",
                  f"{evidence_completeness}/100", "")
            + kpi("AGENT AGREEMENT",
                  f"{agreement}%", "")
            + kpi("DECISION SUPPORT",
                  f"{confidence}/100",
                  "Internal weighted score")
            + "</div>"
            + '<div class="conclusion">'
            + conclusion
            + "</div>",
            wide=True,
        ),
    ])

    css = """
    *{box-sizing:border-box}
    body{background:#080F1D;color:#EAF2FF;margin:0;
      font-family:Segoe UI,Arial,sans-serif}
    main{max-width:1480px;margin:auto;padding:30px 28px 70px}
    nav{display:flex;justify-content:space-between;align-items:center;
      border-bottom:1px solid #26364E;padding-bottom:22px}
    .brand{font-size:26px;font-weight:800}
    .brand span{color:#37D5F3}
    .meta{color:#9BAEC8;font-size:12px}
    .eyebrow{color:#37D5F3;letter-spacing:.2em;
      font-size:11px;font-weight:800;margin-top:30px}
    h1{font-size:clamp(32px,4vw,54px);
      letter-spacing:-.05em;margin:8px 0}
    .subtitle{color:#9BAEC8;margin-bottom:30px}
    .kpis,.stats{display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}
    .kpi{background:#111E32;border:1px solid #26364E;
      border-radius:16px;padding:20px;min-width:0;
      display:flex;flex-direction:column;gap:9px}
    .kpi small{color:#9BAEC8;font-size:11px}
    .kpi strong{font-size:clamp(17px,2.1vw,27px);
      overflow-wrap:anywhere}
    .kpi span{color:#9BAEC8;font-size:11px}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
      gap:18px;margin-top:20px}
    .panel{background:#111E32;border:1px solid #26364E;
      border-radius:18px;overflow:hidden;min-width:0}
    .wide{grid-column:1/-1}
    .panel header{display:flex;align-items:center;
      justify-content:space-between;flex-wrap:wrap;gap:8px;
      padding:22px 25px 10px}
    .panel h2{font-size:18px;margin:0}
    .panel header span{color:#9BAEC8;font-size:12px}
    .agents{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
      gap:13px;padding:20px}
    .agent{background:#192B42;border-radius:13px;padding:18px}
    .agent h3{margin:0 0 9px;font-size:15px}
    .agent small{color:#57E2B3;font-weight:700}
    .agent p{color:#9BAEC8;font-size:12px;line-height:1.6}
    .stats{grid-template-columns:repeat(3,minmax(0,1fr));
      padding:20px}
    .conclusion{background:#192B42;border-radius:13px;
      padding:20px;margin:0 20px 20px}
    .conclusion h3{font-size:14px}
    .conclusion p{color:#9BAEC8;font-size:13px}
    footer{color:#9BAEC8;font-size:12px;margin-top:28px}
    @media(max-width:900px){
      .grid,.agents{grid-template-columns:1fr}
      .wide{grid-column:auto}
      .kpis,.stats{grid-template-columns:repeat(2,minmax(0,1fr))}
    }
    @media(max-width:520px){
      main{padding:18px}
      .kpis,.stats{grid-template-columns:1fr}
    }
    """

    page = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>EnergyOS v0.3</title>'
        f"<style>{css}</style>"
        f"<script>{get_plotlyjs()}</script>"
        "</head><body><main>"
        '<nav><div class="brand">Energy<span>OS</span></div>'
        '<div class="meta">ADAPTIVE VISUALIZATION v0.3</div></nav>'
        '<div class="eyebrow">EVIDENCE-DRIVEN ENERGY INTELLIGENCE</div>'
        "<h1>See the energy. Understand the evidence.</h1>"
        f'<p class="subtitle">{esc(report.get("run_id", ""))}'
        f" · {esc(analysis.get('series_id', ''))}</p>"
        f'<div class="kpis">{kpis}</div>'
        '<div class="grid">'
        + "".join(blocks)
        + "</div>"
        "<footer>Real analytical outputs. "
        "Unreliable forecasts remain subject to TrustAI review. "
        "Charts do not prove physical root causes.</footer>"
        "</main></body></html>"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")

    print("Dashboard created:", output.resolve())
    print("Evidence rows:", len(df))
    print("Flagged anomalies:", len(flagged))
    print("Selected model:", model)
    print("TrustAI:", trust)


if __name__ == "__main__":
    main()
