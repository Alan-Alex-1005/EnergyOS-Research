
"""EnergyOS Adaptive Visualization v0.2 — read-only, source-driven.

Example:
    python energyos_adaptive_viz.py \
        --results outputs/B01_meters_20261111T152659/results.json \
        --input B01_meters.csv \
        --output presentation_outputs/energyos_dashboard.html

Dependencies: pandas, plotly. Never modifies forecasting results.
"""
import argparse
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

BG = "#07111f"
PANEL = "#101f34"
GRID = "#26384f"
INK = "#e9f2ff"
MUTED = "#9db0c9"
CYAN = "#48d6f6"
ORANGE = "#ffb066"
RED = "#ff6a80"
GREEN = "#53e3b0"


def esc(value):
    return html.escape(str(value), quote=True)


def nice(value, digits=1):
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def badge(value):
    status = str(value).upper()
    kind = (
        "ok"
        if status in {
            "DONE", "GOOD",
            "PROVISIONALLY_ACCEPTABLE", "GENERATED"
        }
        else (
            "warn"
            if status in {
                "LIMITED", "FAIR", "REVIEW_REQUIRED"
            }
            else "muted"
        )
    )
    return (
        f'<span class="badge {kind}">'
        f'{esc(str(value).replace("_", " "))}'
        f'</span>'
    )


def section(title, subtitle, contents, wide=False):
    extra = " wide" if wide else ""
    return (
        f'<section class="panel{extra}">'
        f'<div class="panel-head">'
        f'<h2>{esc(title)}</h2>'
        f'<span>{esc(subtitle)}</span>'
        f'</div>{contents}</section>'
    )


def plot_style(fig, title, unit=None, height=370):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=INK),
            x=0.03
        ),
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            family="Inter, Segoe UI, Arial, sans-serif",
            color=MUTED,
            size=11
        ),
        margin=dict(l=65, r=28, t=74, b=55),
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            x=0,
            y=1.12
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        title=unit if unit else None
    )

    return fig


def chart(fig):
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "energyos_chart"
            }
        }
    )


def read_measurements(
    path,
    series_id,
    timestamp_col=None
):
    df = pd.read_csv(
        path,
        low_memory=False
    )

    measure_col = (
        series_id.split(":", 1)[1]
        if ":" in series_id
        else series_id
    )

    if measure_col not in df.columns:
        raise ValueError(
            f"Measurement column {measure_col!r} "
            f"not found. Available: {list(df.columns)}"
        )

    if timestamp_col is None:
        choices = [
            c for c in df.columns
            if c != measure_col
            and (
                "time" in c.lower()
                or "date" in c.lower()
            )
        ]

        if not choices:
            raise ValueError(
                "Timestamp column not identifiable; "
                "supply --timestamp-column"
            )

        timestamp_col = choices[0]

    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column "
            f"{timestamp_col!r} not found"
        )

    ts = pd.to_datetime(
        df[timestamp_col],
        errors="coerce",
        utc=True
    )

    if ts.notna().mean() < 0.90:
        raise ValueError(
            "Too few parseable timestamps "
            "for a reliable chart"
        )

    raw = pd.DataFrame({
        "timestamp": ts,
        "value": pd.to_numeric(
            df[measure_col],
            errors="coerce"
        )
    })

    raw = (
        raw.loc[raw.timestamp.notna()]
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )

    if raw.timestamp.duplicated().any():
        raise ValueError(
            "Duplicate timestamps: cannot infer "
            "aggregation semantics"
        )

    if raw.value.notna().sum() < 2:
        raise ValueError(
            "Too few valid numeric measurements"
        )

    return raw, measure_col, timestamp_col


def break_long_gaps(raw):
    """
    Insert NaN separators at missing time intervals.

    Existing missing values are preserved.
    No interpolation is performed.
    """

    diffs = (
        raw.timestamp
        .diff()
        .dt.total_seconds()
    )

    positive = diffs[diffs > 0]

    if positive.empty:
        return raw.copy(), 0

    expected = positive.round(6).mode().iloc[0]

    if expected <= 0:
        return raw.copy(), 0

    indices = diffs[
        diffs > expected * 1.5
    ].index

    separators = []

    for i in indices:
        t = (
            raw.loc[i - 1, "timestamp"]
            + pd.Timedelta(
                seconds=float(expected)
            )
        )

        if t < raw.loc[i, "timestamp"]:
            separators.append({
                "timestamp": t,
                "value": float("nan")
            })

    if not separators:
        return raw.copy(), 0

    rendered = pd.concat(
        [
            raw,
            pd.DataFrame(separators)
        ],
        ignore_index=True
    )

    rendered = (
        rendered
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )

    return rendered, len(separators)


def kpi(label, value, note=""):
    return (
        '<div class="kpi">'
        f'<small>{esc(label)}</small>'
        f'<strong>{esc(value)}</strong>'
        f'<span>{esc(note)}</span>'
        '</div>'
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "EnergyOS adaptive read-only "
            "evidence dashboard"
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "presentation_outputs/"
            "energyos_dashboard.html"
        )
    )

    parser.add_argument(
        "--timestamp-column",
        default=None
    )

    args = parser.parse_args()

    for item in (args.results, args.input):
        if not item.is_file():
            parser.error(
                f"File not found: {item}"
            )

    report = json.loads(
        args.results.read_text(
            encoding="utf-8"
        )
    )

    analysis = report.get("analysis") or {}

    agents_info = (
        report.get("collaboration") or {}
    )

    forecast = (
        analysis.get("forecast") or {}
    )

    anomaly = (
        analysis.get("anomaly") or {}
    )

    agents = (
        agents_info.get("agents") or {}
    )

    series_id = (
        report.get("selected_series")
        or analysis.get("series_id")
        or ""
    )

    if not series_id:
        parser.error(
            "No selected series in results.json"
        )

    raw, value_col, ts_col = read_measurements(
        args.input,
        series_id,
        args.timestamp_column
    )

    rendered, gap_count = break_long_gaps(
        raw
    )

    role = (
        analysis.get("role")
        or "Measurement"
    )

    unit = (
        analysis.get("unit")
        or ""
    )

    selected_model = (
        forecast.get("selected_model")
        or "Not selected"
    )

    quality = (
        report.get("dataset") or {}
    )

    metrics = (
        forecast.get("metrics") or {}
    )

    events = (
        anomaly.get("events") or []
    )

    trust = "NOT REPORTED"

    for item in (
        agents.get("forecast_agent") or {}
    ).get("limitations") or []:

        if "Forecast reliability:" in item:
            trust = item.split(
                "Forecast reliability:",
                1
            )[1].strip()

    panels = []
    manifest = []

    # -----------------------------------------
    # 1. Historical measurements
    # -----------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=rendered.timestamp,
            y=rendered.value,
            name="Measured",
            mode="lines",
            line=dict(
                color=CYAN,
                width=1.5
            ),
            connectgaps=False,
            hovertemplate=(
                "%{x}<br>%{y:.2f} "
                + esc(unit)
                + "<extra></extra>"
            )
        )
    )

    plot_style(
        fig,
        "Measured series · full history",
        f"{role} ({unit})",
        height=410
    )

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.09
            )
        )
    )

    panels.append(
        section(
            "Measurement timeline",
            "Measured data · interactive",
            chart(fig),
            wide=True
        )
    )

    manifest.append((
        "Measurement history",
        "GENERATED",
        (
            "Missing intervals separated "
            f"({gap_count} long gaps)"
        )
    ))

    # -----------------------------------------
    # 2. Hourly operating pattern
    # -----------------------------------------

    intervals = (
        raw.timestamp
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    typical = (
        intervals[intervals > 0].median()
        if not intervals.empty
        else float("inf")
    )

    if typical <= 12 * 3600:

        valid = (
            raw.dropna(
                subset=["value"]
            ).copy()
        )

        valid["hour_utc"] = (
            valid.timestamp.dt.hour
        )

        hourly = (
            valid.groupby("hour_utc")["value"]
            .agg(["mean", "count"])
        )

        if len(hourly) >= 8:

            fig = go.Figure(
                go.Scatter(
                    x=hourly.index,
                    y=hourly["mean"],
                    mode="lines+markers",
                    line=dict(
                        color=ORANGE,
                        width=3
                    ),
                    marker=dict(size=6),
                    name="Hourly mean"
                )
            )

            plot_style(
                fig,
                "Mean by hour of day",
                f"Mean {role} ({unit})"
            )

            fig.update_xaxes(
                title="Hour of day · UTC",
                dtick=2,
                range=[0, 23]
            )

            panels.append(
                section(
                    "Daily operating pattern",
                    "UTC · observed averages",
                    chart(fig)
                )
            )

            manifest.append((
                "Hourly profile",
                "GENERATED",
                "Subdaily measurements available"
            ))

        else:
            manifest.append((
                "Hourly profile",
                "SKIPPED",
                "Insufficient distinct hourly observations"
            ))

    else:
        manifest.append((
            "Hourly profile",
            "SKIPPED",
            "Sampling interval not subdaily"
        ))

    # -----------------------------------------
    # 3. Forecast model comparison
    # -----------------------------------------

    candidates = [
        c for c in forecast.get("comparison") or []
        if c.get("validation_rmse") is not None
    ]

    if candidates:

        candidates.sort(
            key=lambda c: float(
                c["validation_rmse"]
            ),
            reverse=True
        )

        names = [
            c.get("model", "Unknown")
            for c in candidates
        ]

        scores = [
            float(c["validation_rmse"])
            for c in candidates
        ]

        colors = [
            CYAN if name == selected_model
            else "#547092"
            for name in names
        ]

        fig = go.Figure(
            go.Bar(
                x=scores,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[
                    nice(v, 2)
                    for v in scores
                ],
                textposition="outside",
                cliponaxis=False
            )
        )

        plot_style(
            fig,
            "Validation RMSE · lower is better",
            f"RMSE ({unit})"
        )

        fig.update_layout(
            margin=dict(
                l=185,
                r=45,
                t=72,
                b=55
            )
        )

        fig.update_xaxes(
            title=f"Validation RMSE ({unit})",
            range=[
                0,
                max(scores) * 1.2
            ]
        )

        panels.append(
            section(
                "Forecast-model selection",
                "No test-set selection leakage",
                chart(fig)
            )
        )

        manifest.append((
            "Model comparison",
            "GENERATED",
            "Validation metrics available"
        ))

    else:
        manifest.append((
            "Model comparison",
            "SKIPPED",
            "No model validation metrics"
        ))

    # -----------------------------------------
    # 4. Prediction evidence availability
    # -----------------------------------------

    manifest.append((
        "Actual vs predicted",
        "SKIPPED",
        (
            "Point-level predictions "
            "not in results.json"
        )
    ))

    # -----------------------------------------
    # 5. Anomaly investigations
    # -----------------------------------------

    if events:

        fig = go.Figure(
            go.Scattergl(
                x=rendered.timestamp,
                y=rendered.value,
                mode="lines",
                name="Measured",
                line=dict(
                    color=CYAN,
                    width=1.7
                ),
                connectgaps=False
            )
        )

        parsed_events = []

        for ev in events:

            start = pd.to_datetime(
                ev.get("start_time"),
                errors="coerce",
                utc=True
            )

            end = pd.to_datetime(
                ev.get("end_time"),
                errors="coerce",
                utc=True
            )

            if (
                pd.isna(start)
                or pd.isna(end)
            ):
                continue

            parsed_events.append(
                (start, end, ev)
            )

            if end > start:

                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=RED,
                    opacity=0.21,
                    line_width=0
                )

            else:

                fig.add_vline(
                    x=start.timestamp() * 1000,
                    line_color=RED,
                    line_dash="dot"
                )

        plot_style(
            fig,
            "Measured series with verified event windows",
            f"{role} ({unit})"
        )

        if parsed_events:

            start = parsed_events[0][0]

            fig.update_xaxes(
                range=[
                    start - pd.Timedelta(days=2),
                    start + pd.Timedelta(days=4)
                ]
            )

        panels.append(
            section(
                "Anomaly investigation",
                (
                    f"{len(events)} reported events "
                    "· zoom to inspect"
                ),
                chart(fig)
            )
        )

        manifest.append((
            "Anomaly event timeline",
            "GENERATED",
            "Using reported event timestamps"
        ))

    else:
        manifest.append((
            "Anomaly event timeline",
            "SKIPPED",
            "No reported anomaly events"
        ))

    # -----------------------------------------
    # 6. Specialized agent findings
    # -----------------------------------------

    if agents:

        agent_rows = []

        for name, info in agents.items():

            info = info or {}

            finding = (
                info.get("finding")
                or "No finding reported"
            )

            agent_rows.append(
                '<div class="agent">'
                '<div>'
                f'<h3>{esc(name.replace("_", " ").title())}</h3>'
                f'<p>{esc(finding)}</p>'
                '</div>'
                f'{badge(info.get("status", "UNKNOWN"))}'
                '</div>'
            )

        panels.append(
            section(
                "Specialist agent findings",
                "Evidence-driven · actual run outputs",
                (
                    '<div class="agents">'
                    + "".join(agent_rows)
                    + '</div>'
                ),
                wide=True
            )
        )

        manifest.append((
            "Agent findings",
            "GENERATED",
            "Agent evidence available"
        ))

    # -----------------------------------------
    # 7. TrustAI and evidence
    # -----------------------------------------

    complete = (
        agents_info.get(
            "evidence_completeness"
        ) or {}
    ).get("score")

    agreement = (
        agents_info.get(
            "agreement"
        ) or {}
    ).get("score")

    decision = (
        agents_info.get(
            "decision_confidence"
        ) or {}
    ).get("score")

    ai_engineer = (
        agents_info.get(
            "ai_energy_engineer"
        ) or {}
    )

    trust_html = (
        '<div class="trust-grid">'
        + kpi(
            "Forecast reliability",
            trust.replace("_", " ")
        )
        + kpi(
            "Evidence completeness",
            f"{nice(complete)} / 100"
        )
        + kpi(
            "Agent agreement",
            f"{nice(agreement)}%"
        )
        + kpi(
            "Decision support",
            f"{nice(decision)} / 100",
            "Not a statistical probability"
        )
        + '</div>'
        + '<div class="finding">'
        + '<h3>AI Energy Engineer conclusion</h3>'
        + (
            f'<p>{esc(ai_engineer.get("why", "No conclusion available"))}</p>'
        )
        + (
            '<p><strong>Recommended action:</strong> '
            + esc(
                ai_engineer.get(
                    "recommended_action",
                    "Not available"
                )
            )
            + '</p>'
        )
        + (
            '<small>Forecast reliability is not '
            'equivalent to verified physical root cause.'
            '</small>'
        )
        + '</div>'
    )

    panels.append(
        section(
            "TrustAI and evidence",
            "Uncertainty remains visible",
            trust_html,
            wide=True
        )
    )

    manifest.append((
        "TrustAI evidence",
        "GENERATED",
        "Run metadata and agent conclusions"
    ))

    # -----------------------------------------
    # 8. Unsupported impact charts
    # -----------------------------------------

    for agent_name in (
        "cost_agent",
        "carbon_agent"
    ):

        info = (
            agents.get(agent_name) or {}
        )

        manifest.append((
            (
                agent_name.replace(
                    "_", " "
                ).title()
                + " chart"
            ),
            "SKIPPED",
            (
                info.get("finding")
                or "No validated impact evidence"
            )
        ))

    table_rows = "".join(
        (
            f"<tr><td>{esc(name)}</td>"
            f"<td>{badge(status)}</td>"
            f"<td>{esc(reason)}</td></tr>"
        )
        for name, status, reason in manifest
    )

    panels.append(
        section(
            "Visualization applicability",
            "Honest about unsupported charts",
            (
                '<div class="table-wrap">'
                '<table>'
                '<thead><tr>'
                '<th>Visualization</th>'
                '<th>Status</th>'
                '<th>Explanation</th>'
                '</tr></thead>'
                '<tbody>'
                + table_rows
                + '</tbody>'
                '</table>'
                '</div>'
            ),
            wide=True
        )
    )

    # -----------------------------------------
    # 9. Dashboard KPI cards
    # -----------------------------------------

    kpis = "".join([
        kpi(
            "DATA QUALITY",
            (
                f"{nice(quality.get('quality_score'))}"
                " / 100"
            ),
            str(
                quality.get(
                    "quality_status",
                    ""
                )
            )
        ),
        kpi(
            "TEST RMSE",
            (
                f"{nice(metrics.get('rmse'), 2)} "
                f"{unit}"
            ),
            selected_model
        ),
        kpi(
            "ANOMALIES",
            str(
                anomaly.get(
                    "anomalies_detected",
                    "N/A"
                )
            ),
            f"{len(events)} event(s)"
        ),
        kpi(
            "DECISION SUPPORT",
            f"{nice(decision)} / 100",
            "Internal weighted score"
        )
    ])

    # -----------------------------------------
    # 10. Responsive dashboard layout
    # -----------------------------------------

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EnergyOS | Adaptive Evidence Dashboard</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: {BG};
    color: {INK};
    font: 14px/1.55 Inter, Segoe UI, Arial, sans-serif;
}}

.wrapper {{
    max-width: 1480px;
    margin: auto;
    padding: 24px 26px 70px;
}}

nav {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid {GRID};
    padding: 4px 0 21px;
    gap: 10px;
    flex-wrap: wrap;
}}

.brand {{
    font-weight: 800;
    letter-spacing: -0.05em;
    font-size: 25px;
}}

.brand b {{
    color: {CYAN};
}}

.nav-meta {{
    color: {MUTED};
    font-size: 12px;
}}

.eyebrow {{
    font-weight: 800;
    font-size: 11px;
    letter-spacing: 0.21em;
    color: {CYAN};
    margin-top: 26px;
}}

h1 {{
    font-size: clamp(29px, 3.3vw, 46px);
    letter-spacing: -0.05em;
    margin: 5px 0 8px;
}}

.subtitle {{
    color: {MUTED};
    margin: 0 0 24px;
    overflow-wrap: anywhere;
}}

.kpis,
.trust-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 13px;
}}

.kpi {{
    border: 1px solid {GRID};
    background: {PANEL};
    border-radius: 15px;
    padding: 18px;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
}}

.kpi small {{
    font-size: 11px;
    color: {MUTED};
    letter-spacing: 0.04em;
}}

.kpi strong {{
    font-size: clamp(16px, 2.2vw, 27px);
    line-height: 1.2;
    overflow-wrap: anywhere;
}}

.kpi span {{
    color: {MUTED};
    font-size: 11px;
    overflow-wrap: anywhere;
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
    margin-top: 20px;
    align-items: start;
}}

.panel {{
    border: 1px solid {GRID};
    background: {PANEL};
    border-radius: 18px;
    min-width: 0;
    overflow: hidden;
}}

.panel.wide {{
    grid-column: 1 / -1;
}}

.panel-head {{
    padding: 19px 22px 0;
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: baseline;
    flex-wrap: wrap;
}}

.panel h2 {{
    font-size: 17px;
    margin: 0;
}}

.panel-head span {{
    color: {MUTED};
    font-size: 12px;
}}

.agents {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    padding: 20px;
}}

.agent {{
    background: #172a43;
    border-radius: 12px;
    padding: 15px;
    display: flex;
    justify-content: space-between;
    align-items: start;
    gap: 14px;
}}

.agent h3,
.finding h3 {{
    font-size: 14px;
    margin: 0 0 7px;
    text-transform: capitalize;
}}

.agent p,
.finding p {{
    color: {MUTED};
    font-size: 12px;
    margin: 0;
}}

.badge {{
    display: inline-block;
    white-space: nowrap;
    border: 1px solid #4c5c70;
    border-radius: 999px;
    color: #bfcce0;
    padding: 4px 9px;
    font-size: 10px;
    font-weight: 800;
}}

.badge.ok {{
    color: #60eac1;
    border-color: #26715f;
    background: #153d37;
}}

.badge.warn {{
    color: #ffd38e;
    border-color: #80623a;
    background: #453625;
}}

.trust-grid {{
    padding: 22px;
}}

.finding {{
    margin: 0 22px 23px;
    padding: 16px 18px;
    border-radius: 12px;
    background: #172a43;
}}

.finding p + p {{
    margin-top: 9px;
}}

.finding small {{
    display: block;
    color: {MUTED};
    margin-top: 12px;
}}

.table-wrap {{
    overflow: auto;
    padding: 16px 22px 24px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    min-width: 580px;
}}

td,
th {{
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid {GRID};
    font-size: 12px;
}}

th {{
    color: {MUTED};
}}

td:last-child {{
    color: {MUTED};
}}

footer {{
    color: {MUTED};
    font-size: 12px;
    padding-top: 25px;
}}

@media(max-width: 900px) {{
    .kpis,
    .trust-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    .grid,
    .agents {{
        grid-template-columns: 1fr;
    }}

    .panel.wide {{
        grid-column: auto;
    }}
}}

@media(max-width: 520px) {{
    .wrapper {{
        padding: 14px;
    }}

    .kpis,
    .trust-grid {{
        grid-template-columns: 1fr;
    }}

    .panel-head {{
        padding: 16px 15px 0;
    }}
}}
</style>

<script>{get_plotlyjs()}</script>
</head>

<body>
<div class="wrapper">

<nav>
    <div class="brand">⚡ Energy<b>OS</b></div>
    <div class="nav-meta">
        RESEARCH DEMO V2 · ADAPTIVE VISUALIZATION V0.2
    </div>
</nav>

<div class="eyebrow">
    EVIDENCE-AWARE ENERGY INTELLIGENCE
</div>

<h1>Understand energy. Verify decisions.</h1>

<p class="subtitle">
    {esc(args.input.name)} ·
    {esc(series_id)} ·
    {len(raw):,} timestamped observations ·
    {esc(selected_model)}
</p>

<div class="kpis">
    {kpis}
</div>

<div class="grid">
    {''.join(panels)}
</div>

<footer>
    Run: {esc(report.get('run_id', 'Unknown'))}
    · This visualization uses existing measurements and
    analytical results. No forecasts or impact values
    are fabricated.
</footer>

</div>
</body>
</html>'''

    # -----------------------------------------
    # 11. Export HTML report
    # -----------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    args.output.write_text(
        page,
        encoding="utf-8"
    )

    print(
        "Created:",
        args.output.resolve()
    )

    print(
        "Generated charts:",
        sum(
            st == "GENERATED"
            for _, st, _ in manifest
        )
    )

    print(
        "Skipped charts:",
        sum(
            st == "SKIPPED"
            for _, st, _ in manifest
        )
    )

    print(
        "Time-gap separators:",
        gap_count
    )

    for name, status, reason in manifest:
        print(
            f"  {status:9} "
            f"{name}: {reason}"
        )


if __name__ == "__main__":
    main()
