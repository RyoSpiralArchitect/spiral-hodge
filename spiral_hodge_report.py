"""
Generate a static interactive HTML report from Spiral Hodge layer metrics.

The report is intentionally dependency-free: it reads layer_metrics.csv and
emits a standalone HTML file with embedded data, CSS, and vanilla JavaScript.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ENERGY_METRICS = [
    ("spectral_curl_ratio", "Spectral curl"),
    ("spectral_curl_low_ratio", "Spectral low curl"),
    ("spectral_curl_high_ratio", "Spectral high curl"),
    ("hodge_curl_ratio", "Hodge curl"),
    ("hltd_exact_ratio", "HLTD exact"),
    ("hltd_coexact_ratio", "HLTD coexact"),
    ("hltd_harmonic_ratio", "HLTD harmonic"),
    ("hltd_semantic_flow_ratio", "HLTD semantic flow"),
    ("local_abs_vorticity_mean", "Local vorticity"),
    ("turning_mean_abs_angle", "Path turning"),
    ("graph_high_freq_ratio", "Graph high freq"),
]

SIGNED_METRICS = [
    ("trajectory_signed_circulation_alignment", "Trajectory circulation"),
    ("turning_alignment", "Path turning"),
    ("spectral_signed_curl_alignment", "Spectral curl circulation"),
    ("local_signed_vorticity_ratio", "Local Jacobian vorticity"),
    ("hodge_signed_curl_alignment", "Hodge face curl"),
    ("spectral_signed_vorticity_ratio", "Spectral vorticity"),
]

DEFAULT_NULL_METRICS = [
    ("spectral_curl_ratio", "Spectral curl ratio"),
    ("spectral_curl_low_ratio", "Spectral low curl ratio"),
    ("spectral_curl_mid_ratio", "Spectral mid curl ratio"),
    ("spectral_curl_high_ratio", "Spectral high curl ratio"),
    ("spectral_curl_high_band_ratio", "High curl share"),
    ("hodge_curl_ratio", "Hodge curl ratio"),
    ("local_abs_vorticity_mean", "Local abs vorticity"),
    ("local_signed_vorticity_ratio", "Local signed vorticity"),
    ("turning_mean_abs_angle", "Trajectory mean abs turning"),
    ("turning_alignment", "Trajectory turning alignment"),
    ("trajectory_signed_circulation_alignment", "Trajectory signed circulation"),
    ("spectral_signed_curl_alignment", "Spectral signed curl"),
    ("spectral_signed_vorticity_ratio", "Spectral signed vorticity"),
    ("hodge_signed_curl_alignment", "Hodge signed curl"),
    ("graph_high_freq_ratio", "Graph high-frequency ratio"),
    ("hltd_exact_ratio", "HLTD exact/presence ratio"),
    ("hltd_coexact_ratio", "HLTD coexact/local swirl ratio"),
    ("hltd_harmonic_ratio", "HLTD harmonic/global loop ratio"),
    ("hltd_semantic_flow_ratio", "HLTD semantic-flow ratio"),
    ("hltd_exact_coexact_alignment", "HLTD exact/coexact alignment"),
    ("hltd_coexact_harmonic_alignment", "HLTD coexact/harmonic alignment"),
]


def _coerce_cell(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer() and abs(number) < 2**53:
        return int(number)
    return number


def read_layer_metrics(path: Path) -> List[Dict[str, Any]]:
    """Read a Spiral Hodge layer_metrics.csv file."""

    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({key: _coerce_cell(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _variants(rows: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for row in rows:
        variant = str(row.get("variant", ""))
        if variant and variant not in out:
            out.append(variant)
    return out


def _layers(rows: Sequence[Dict[str, Any]]) -> List[int]:
    layers = {
        int(layer)
        for layer in (_finite_float(row.get("layer")) for row in rows)
        if layer is not None
    }
    return sorted(layers)


def rows_for_variant(rows: Sequence[Dict[str, Any]], variant: str) -> List[Dict[str, Any]]:
    return sorted(
        [row for row in rows if str(row.get("variant")) == variant],
        key=lambda row: int(_finite_float(row.get("layer")) or 0),
    )


def peak_for_metric(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    *,
    absolute: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return the row/value/layer for the strongest finite metric."""

    best: Optional[Tuple[float, Dict[str, Any], float]] = None
    for row in rows:
        value = _finite_float(row.get(metric))
        if value is None:
            continue
        score = abs(value) if absolute else value
        if best is None or score > best[0]:
            best = (score, row, value)
    if best is None:
        return None
    _, row, value = best
    return {
        "variant": row.get("variant"),
        "layer": int(_finite_float(row.get("layer")) or 0),
        "metric": metric,
        "value": value,
        "absValue": abs(value),
    }


def build_variant_summaries(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for variant in _variants(rows):
        rr = rows_for_variant(rows, variant)
        summary: Dict[str, Any] = {"variant": variant, "peaks": {}}
        for metric, _ in ENERGY_METRICS:
            peak = peak_for_metric(rr, metric)
            if peak is not None:
                summary["peaks"][metric] = peak
        for metric, _ in SIGNED_METRICS:
            peak = peak_for_metric(rr, metric, absolute=True)
            if peak is not None:
                summary["peaks"][metric] = peak
        summaries.append(summary)
    return summaries


def build_reverse_diagnostics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    real = {int(_finite_float(row.get("layer")) or 0): row for row in rows_for_variant(rows, "real")}
    reverse = {int(_finite_float(row.get("layer")) or 0): row for row in rows_for_variant(rows, "reverse_tokens")}
    diagnostics: List[Dict[str, Any]] = []
    if not real or not reverse:
        return diagnostics

    for metric, label in SIGNED_METRICS:
        max_abs_sum = 0.0
        compared = 0
        for layer, real_row in real.items():
            rev_row = reverse.get(layer)
            if rev_row is None:
                continue
            a = _finite_float(real_row.get(metric))
            b = _finite_float(rev_row.get(metric))
            if a is None or b is None:
                continue
            max_abs_sum = max(max_abs_sum, abs(a + b))
            compared += 1
        if compared:
            diagnostics.append(
                {
                    "metric": metric,
                    "label": label,
                    "layersCompared": compared,
                    "maxAbsRealPlusReverse": max_abs_sum,
                }
            )
    return diagnostics


def _dataset_meta(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first = rows[0]
    return {
        "variants": _variants(rows),
        "layers": _layers(rows),
        "tokens": first.get("tokens"),
        "dim": first.get("dim"),
        "layerCount": first.get("layers"),
        "rowCount": len(rows),
    }


def _json_for_html(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_report_html(
    rows: Sequence[Dict[str, Any]],
    *,
    title: str,
    csv_path: Path,
) -> str:
    """Build a standalone HTML report string."""

    meta = _dataset_meta(rows)
    payload = {
        "title": title,
        "csvPath": str(csv_path),
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "meta": meta,
        "rows": list(rows),
        "variantSummaries": build_variant_summaries(rows),
        "reverseDiagnostics": build_reverse_diagnostics(rows),
        "energyMetrics": [{"key": key, "label": label} for key, label in ENERGY_METRICS],
        "signedMetrics": [{"key": key, "label": label} for key, label in SIGNED_METRICS],
        "nullMetrics": [{"key": key, "label": label} for key, label in DEFAULT_NULL_METRICS],
    }
    payload_json = _json_for_html(payload)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --ink: #17202a;
      --muted: #5f6b7a;
      --panel: #ffffff;
      --line: #d9dee7;
      --blue: #2563eb;
      --red: #dc2626;
      --green: #059669;
      --amber: #d97706;
      --violet: #7c3aed;
      --cyan: #0891b2;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.08), 0 8px 24px rgba(15, 23, 42, 0.07);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    .wrap {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }}
    .topbar {{
      padding: 28px 0 22px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      align-items: end;
    }}
    h1 {{
      font-size: clamp(1.6rem, 3vw, 2.4rem);
      line-height: 1.05;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.94rem;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    button, select {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      min-height: 36px;
      padding: 7px 10px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
    }}
    button.active {{
      border-color: var(--blue);
      background: #eef4ff;
      color: #123a8c;
      font-weight: 650;
    }}
    main {{
      padding: 22px 0 40px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .stats {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-bottom: 16px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: var(--shadow);
      min-height: 96px;
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .stat .value {{
      font-size: 1.45rem;
      font-weight: 760;
      line-height: 1.15;
    }}
    .stat .note {{
      color: var(--muted);
      margin-top: 6px;
      font-size: 0.9rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: var(--shadow);
      min-width: 0;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 1rem;
      letter-spacing: 0;
    }}
    .two {{
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }}
    .chart {{
      width: 100%;
      min-height: 310px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .axis text {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .axis line, .axis path, .grid-line {{
      stroke: var(--line);
      stroke-width: 1;
      vector-effect: non-scaling-stroke;
    }}
    .zero-line {{
      stroke: #101828;
      stroke-width: 1;
      opacity: 0.55;
      vector-effect: non-scaling-stroke;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      display: inline-block;
    }}
    .readout {{
      color: var(--muted);
      min-height: 22px;
      margin-top: 8px;
      font-size: 0.93rem;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.92rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 650;
      background: #fbfcff;
      position: sticky;
      top: 0;
    }}
    .table-scroll {{
      overflow: auto;
      max-height: 420px;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .section {{
      margin-top: 16px;
    }}
    .caption {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-top: -6px;
      margin-bottom: 12px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      background: #fff;
      color: var(--muted);
      font-size: 0.84rem;
      margin-right: 6px;
      margin-top: 6px;
    }}
    @media (max-width: 860px) {{
      .topbar, .two, .stats {{
        grid-template-columns: 1fr;
      }}
      .toolbar {{
        justify-content: flex-start;
      }}
      .wrap {{
        width: min(100vw - 20px, 1180px);
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <div>
        <h1>{title}</h1>
        <div class="meta" id="reportMeta"></div>
      </div>
      <div class="toolbar">
        <div id="variantButtons"></div>
        <select id="nullMetricSelect" aria-label="Null model metric"></select>
      </div>
    </div>
  </header>
  <main class="wrap">
    <section class="grid stats" id="stats"></section>
    <section class="grid two">
      <div class="panel">
        <h2>Selected Variant: Curl Energy</h2>
        <div class="caption">Unsigned ratios by layer. These can remain unchanged under token reversal.</div>
        <div class="chart" id="energyChart"></div>
        <div class="legend" id="energyLegend"></div>
        <div class="readout" id="energyReadout"></div>
      </div>
      <div class="panel">
        <h2>Selected Variant: Signed Orientation</h2>
        <div class="caption">Signed handedness metrics. Reversing the token path should flip the sign.</div>
        <div class="chart" id="signedChart"></div>
        <div class="legend" id="signedLegend"></div>
        <div class="readout" id="signedReadout"></div>
      </div>
    </section>
    <section class="panel section">
      <h2>Real vs Null Models</h2>
      <div class="caption">Choose a metric in the top-right menu to compare all variants.</div>
      <div class="chart" id="nullChart"></div>
      <div class="legend" id="nullLegend"></div>
      <div class="readout" id="nullReadout"></div>
    </section>
    <section class="grid two section">
      <div class="panel">
        <h2>Peak Summary</h2>
        <div class="table-scroll">
          <table id="peakTable"></table>
        </div>
      </div>
      <div class="panel">
        <h2>Reverse-Direction Check</h2>
        <div class="caption">Small values mean real + reverse nearly cancels, as expected for signed metrics.</div>
        <div id="reverseBadges"></div>
      </div>
    </section>
    <section class="panel section">
      <h2>Layer Table</h2>
      <div class="caption">Rows for the selected variant with the most useful energy and orientation metrics.</div>
      <div class="table-scroll">
        <table id="layerTable"></table>
      </div>
    </section>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const report = JSON.parse(document.getElementById("payload").textContent);
    const rows = report.rows;
    const variants = report.meta.variants;
    const layers = report.meta.layers;
    const colors = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2", "#4f46e5"];
    let selectedVariant = variants.includes("real") ? "real" : variants[0];
    let selectedNullMetric = "spectral_curl_ratio";

    function numberValue(value) {{
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }}

    function fmt(value, digits = 4) {{
      const n = numberValue(value);
      if (n === null) return "NA";
      const sign = n > 0 && Math.abs(n) < 1 ? "+" : "";
      return sign + n.toFixed(digits);
    }}

    function rowsFor(variant) {{
      return rows
        .filter((row) => row.variant === variant)
        .sort((a, b) => Number(a.layer) - Number(b.layer));
    }}

    function metricLabel(metric) {{
      const all = [...report.energyMetrics, ...report.signedMetrics, ...report.nullMetrics];
      const found = all.find((item) => item.key === metric);
      return found ? found.label : metric;
    }}

    function metricDomain(metric, series) {{
      const signed = metric.includes("signed") || metric.includes("vorticity") || metric.includes("alignment");
      if (signed) return [-1, 1];
      const values = series.flatMap((line) => line.points.map((p) => p.y).filter((v) => Number.isFinite(v)));
      const maxValue = Math.max(1, ...values);
      return [0, Math.min(1, maxValue) === 1 ? 1 : maxValue * 1.08];
    }}

    function makeSeries(variant, specs) {{
      const rr = rowsFor(variant);
      return specs.map((spec, index) => ({{
        label: spec.label,
        color: colors[index % colors.length],
        metric: spec.key,
        points: rr.map((row) => ({{
          x: Number(row.layer),
          y: numberValue(row[spec.key]),
          variant,
          metric: spec.key
        }})).filter((point) => point.y !== null)
      }}));
    }}

    function makeNullSeries(metric) {{
      return variants.map((variant, index) => ({{
        label: variant,
        color: colors[index % colors.length],
        metric,
        points: rowsFor(variant).map((row) => ({{
          x: Number(row.layer),
          y: numberValue(row[metric]),
          variant,
          metric
        }})).filter((point) => point.y !== null)
      }}));
    }}

    function renderLegend(id, series) {{
      const el = document.getElementById(id);
      el.replaceChildren();
      for (const line of series) {{
        const item = document.createElement("span");
        const swatch = document.createElement("i");
        swatch.className = "swatch";
        swatch.style.background = line.color;
        item.appendChild(swatch);
        item.append(document.createTextNode(line.label));
        el.appendChild(item);
      }}
    }}

    function renderChart(id, series, options = {{}}) {{
      const el = document.getElementById(id);
      el.replaceChildren();
      const width = 900;
      const height = 320;
      const margin = {{ top: 18, right: 24, bottom: 38, left: 54 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const xMin = Math.min(...layers);
      const xMax = Math.max(...layers);
      const yDomain = options.yDomain || metricDomain(options.metric || "", series);
      const yMin = yDomain[0];
      const yMax = yDomain[1];
      const sx = (x) => margin.left + ((x - xMin) / Math.max(xMax - xMin, 1)) * innerW;
      const sy = (y) => margin.top + (1 - ((y - yMin) / Math.max(yMax - yMin, 1e-12))) * innerH;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", options.title || "metric chart");

      function line(x1, y1, x2, y2, cls) {{
        const node = document.createElementNS(svg.namespaceURI, "line");
        node.setAttribute("x1", x1);
        node.setAttribute("y1", y1);
        node.setAttribute("x2", x2);
        node.setAttribute("y2", y2);
        node.setAttribute("class", cls);
        svg.appendChild(node);
      }}

      const tickCount = 5;
      for (let i = 0; i <= tickCount; i++) {{
        const y = yMin + (i / tickCount) * (yMax - yMin);
        const py = sy(y);
        line(margin.left, py, width - margin.right, py, "grid-line");
        const text = document.createElementNS(svg.namespaceURI, "text");
        text.setAttribute("x", margin.left - 10);
        text.setAttribute("y", py + 4);
        text.setAttribute("text-anchor", "end");
        text.setAttribute("class", "axis");
        text.textContent = y.toFixed(yMax <= 1 && yMin >= -1 ? 2 : 1);
        svg.appendChild(text);
      }}
      if (yMin < 0 && yMax > 0) {{
        line(margin.left, sy(0), width - margin.right, sy(0), "zero-line");
      }}
      for (const layer of layers) {{
        const px = sx(layer);
        line(px, height - margin.bottom, px, height - margin.bottom + 5, "axis");
        const text = document.createElementNS(svg.namespaceURI, "text");
        text.setAttribute("x", px);
        text.setAttribute("y", height - 12);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("class", "axis");
        text.textContent = layer;
        svg.appendChild(text);
      }}

      for (const lineSpec of series) {{
        if (!lineSpec.points.length) continue;
        const path = document.createElementNS(svg.namespaceURI, "path");
        const d = lineSpec.points.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{sx(point.x).toFixed(2)}} ${{sy(point.y).toFixed(2)}}`).join(" ");
        path.setAttribute("d", d);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", lineSpec.color);
        path.setAttribute("stroke-width", "2.4");
        path.setAttribute("vector-effect", "non-scaling-stroke");
        svg.appendChild(path);
        for (const point of lineSpec.points) {{
          const circle = document.createElementNS(svg.namespaceURI, "circle");
          circle.setAttribute("cx", sx(point.x));
          circle.setAttribute("cy", sy(point.y));
          circle.setAttribute("r", 4.5);
          circle.setAttribute("fill", lineSpec.color);
          circle.tabIndex = 0;
          const title = document.createElementNS(svg.namespaceURI, "title");
          title.textContent = `${{point.variant}} layer ${{point.x}} · ${{lineSpec.label}} = ${{fmt(point.y)}}`;
          circle.appendChild(title);
          circle.addEventListener("mouseenter", () => {{
            const readout = document.getElementById(options.readoutId);
            if (readout) readout.textContent = title.textContent;
          }});
          svg.appendChild(circle);
        }}
      }}
      el.appendChild(svg);
    }}

    function renderStats() {{
      const el = document.getElementById("stats");
      el.replaceChildren();
      const selectedRows = rowsFor(selectedVariant);
      const spectralPeak = peak(selectedRows, "spectral_curl_ratio", false);
      const signedPeak = peak(selectedRows, "spectral_signed_vorticity_ratio", true);
      const statData = [
        ["Rows", report.meta.rowCount, `${{variants.length}} variants · ${{layers.length}} layers`],
        ["Token Window", report.meta.tokens, `${{report.meta.dim}} dimensions`],
        ["Spectral Curl Peak", spectralPeak ? `L${{spectralPeak.layer}} · ${{fmt(spectralPeak.value)}}` : "NA", selectedVariant],
        ["Signed Vorticity Peak", signedPeak ? `L${{signedPeak.layer}} · ${{fmt(signedPeak.value)}}` : "NA", selectedVariant]
      ];
      for (const [label, value, note] of statData) {{
        const node = document.createElement("div");
        node.className = "stat";
        node.innerHTML = `<div class="label"></div><div class="value"></div><div class="note"></div>`;
        node.children[0].textContent = label;
        node.children[1].textContent = value;
        node.children[2].textContent = note;
        el.appendChild(node);
      }}
    }}

    function peak(rr, metric, absolute) {{
      let best = null;
      for (const row of rr) {{
        const value = numberValue(row[metric]);
        if (value === null) continue;
        const score = absolute ? Math.abs(value) : value;
        if (!best || score > best.score) best = {{ layer: Number(row.layer), value, score }};
      }}
      return best;
    }}

    function renderVariantButtons() {{
      const el = document.getElementById("variantButtons");
      el.replaceChildren();
      for (const variant of variants) {{
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = variant;
        button.className = variant === selectedVariant ? "active" : "";
        button.addEventListener("click", () => {{
          selectedVariant = variant;
          renderAll();
        }});
        el.appendChild(button);
      }}
    }}

    function renderMetricSelect() {{
      const select = document.getElementById("nullMetricSelect");
      if (select.children.length === 0) {{
        for (const item of report.nullMetrics) {{
          const option = document.createElement("option");
          option.value = item.key;
          option.textContent = item.label;
          select.appendChild(option);
        }}
        select.value = selectedNullMetric;
        select.addEventListener("change", () => {{
          selectedNullMetric = select.value;
          renderAll();
        }});
      }}
    }}

    function renderTables() {{
      const peakTable = document.getElementById("peakTable");
      peakTable.replaceChildren();
      const peakHeader = peakTable.insertRow();
      ["Variant", "Metric", "Layer", "Value"].forEach((text) => {{
        const th = document.createElement("th");
        th.textContent = text;
        peakHeader.appendChild(th);
      }});
      for (const summary of report.variantSummaries) {{
        for (const item of [...report.energyMetrics, ...report.signedMetrics]) {{
          const peak = summary.peaks[item.key];
          if (!peak) continue;
          const row = peakTable.insertRow();
          [summary.variant, item.label, peak.layer, fmt(peak.value)].forEach((value) => {{
            const cell = row.insertCell();
            cell.textContent = value;
          }});
        }}
      }}

      const layerTable = document.getElementById("layerTable");
      layerTable.replaceChildren();
      const columns = [
        ["layer", "Layer"],
        ["spectral_curl_ratio", "Spectral curl"],
        ["spectral_curl_high_ratio", "High curl"],
        ["hodge_curl_ratio", "Hodge curl"],
        ["hltd_exact_ratio", "HLTD exact"],
        ["hltd_coexact_ratio", "HLTD coexact"],
        ["hltd_harmonic_ratio", "HLTD harmonic"],
        ["hltd_semantic_flow_ratio", "HLTD flow"],
        ["local_signed_vorticity_ratio", "Local signed"],
        ["turning_alignment", "Turning"],
        ["graph_high_freq_ratio", "Graph high"],
        ["trajectory_signed_circulation_alignment", "Trajectory signed"],
        ["spectral_signed_curl_alignment", "Spectral signed"],
        ["spectral_signed_vorticity_ratio", "Vorticity signed"],
        ["hodge_signed_curl_alignment", "Hodge signed"]
      ];
      const header = layerTable.insertRow();
      columns.forEach(([, label]) => {{
        const th = document.createElement("th");
        th.textContent = label;
        header.appendChild(th);
      }});
      for (const rowData of rowsFor(selectedVariant)) {{
        const row = layerTable.insertRow();
        for (const [key] of columns) {{
          const cell = row.insertCell();
          cell.textContent = key === "layer" ? rowData[key] : fmt(rowData[key]);
        }}
      }}
    }}

    function renderReverseBadges() {{
      const el = document.getElementById("reverseBadges");
      el.replaceChildren();
      if (!report.reverseDiagnostics.length) {{
        el.textContent = "No real/reverse_tokens pair found.";
        return;
      }}
      for (const item of report.reverseDiagnostics) {{
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = `${{item.label}}: max |real + reverse| = ${{item.maxAbsRealPlusReverse.toExponential(2)}}`;
        el.appendChild(badge);
      }}
    }}

    function renderCharts() {{
      const energySeries = makeSeries(selectedVariant, report.energyMetrics);
      renderChart("energyChart", energySeries, {{ yDomain: [0, 1], readoutId: "energyReadout", title: "curl energy" }});
      renderLegend("energyLegend", energySeries);

      const signedSeries = makeSeries(selectedVariant, report.signedMetrics);
      renderChart("signedChart", signedSeries, {{ yDomain: [-1, 1], readoutId: "signedReadout", title: "signed orientation" }});
      renderLegend("signedLegend", signedSeries);

      const nullSeries = makeNullSeries(selectedNullMetric);
      renderChart("nullChart", nullSeries, {{
        yDomain: metricDomain(selectedNullMetric, nullSeries),
        metric: selectedNullMetric,
        readoutId: "nullReadout",
        title: metricLabel(selectedNullMetric)
      }});
      renderLegend("nullLegend", nullSeries);
    }}

    function renderMeta() {{
      const el = document.getElementById("reportMeta");
      el.textContent = `${{report.csvPath}} · generated ${{report.generatedAt}}`;
    }}

    function renderAll() {{
      renderMeta();
      renderMetricSelect();
      renderVariantButtons();
      renderStats();
      renderCharts();
      renderTables();
      renderReverseBadges();
    }}

    renderAll();
  </script>
</body>
</html>
"""


def resolve_paths(
    *,
    run_dir: Optional[str],
    csv_path: Optional[str],
    output: Optional[str],
) -> Tuple[Path, Path]:
    """Resolve input CSV and output HTML paths."""

    if run_dir is None and csv_path is None:
        run_dir = "."
    base_dir = Path(run_dir).expanduser() if run_dir else None

    if csv_path:
        metrics_path = Path(csv_path).expanduser()
        if not metrics_path.is_absolute() and base_dir is not None:
            metrics_path = base_dir / metrics_path
    else:
        assert base_dir is not None
        metrics_path = base_dir / "layer_metrics.csv"
    metrics_path = metrics_path.resolve()

    if output:
        output_path = Path(output).expanduser()
        if not output_path.is_absolute() and base_dir is not None:
            output_path = base_dir / output_path
    else:
        output_path = metrics_path.parent / "report.html"
    output_path = output_path.resolve()
    return metrics_path, output_path


def write_report(
    *,
    metrics_path: Path,
    output_path: Path,
    title: str,
) -> Path:
    rows = read_layer_metrics(metrics_path)
    html = build_report_html(rows, title=title, csv_path=metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a static HTML report from Spiral Hodge layer metrics")
    parser.add_argument("--run-dir", default=None, help="Directory containing layer_metrics.csv")
    parser.add_argument("--csv", dest="csv_path", default=None, help="Path to layer_metrics.csv")
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults to report.html next to the CSV.")
    parser.add_argument("--title", default="Spiral Hodge Report", help="Report title")
    args = parser.parse_args(argv)

    metrics_path, output_path = resolve_paths(run_dir=args.run_dir, csv_path=args.csv_path, output=args.output)
    if not metrics_path.is_file():
        parser.error(f"layer metrics CSV not found: {metrics_path}")
    write_report(metrics_path=metrics_path, output_path=output_path, title=args.title)
    print(f"saved report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
