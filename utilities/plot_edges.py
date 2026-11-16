"""Interactive view of a learned graph (edges.csv) as a single self-contained HTML file.

    python -m utilities.plot_edges --edges results/final/edges.csv --columns prepared/columns.csv \
        --out results/final/graph.html

Layout: a timeline. The x-axis is the step number (time runs left to right), with two lanes: machine columns on top,
presence columns below (names that don't follow the ``N_stage:step`` pattern get a middle lane). Edges are arcs whose
height grows with the distance they span; with the time order every edge points left to right. Stages are shaded
bands.

In the page: hover a column or an edge's apex for details; click a column to highlight its parents and children;
type in the search box (Enter) to find a column; Clear or Esc resets. Drag to zoom along time; double-click to reset.
Works with any edges.csv that has ``source,target`` columns (``weight``, ``kind``, ``detail`` are used if present).
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .presence import parse_columns

log = logging.getLogger(__name__)

# Reference data-viz palette: categorical slots 1-3 (validated all-pairs in both modes), surfaces and ink.
LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
         "band": "rgba(137,135,129,0.10)", "machine": "#2a78d6", "presence": "#eb6834", "other": "#1baf7a"}
DARK = {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "band": "rgba(137,135,129,0.14)", "machine": "#3987e5", "presence": "#d95926", "other": "#199e70"}
KINDS = [("machine", "machine column"), ("presence", "presence column"), ("other", "other column name")]
LANE = {"machine": 1.0, "presence": 0.0, "other": 0.5}
ARC_POINTS = 13


def load_nodes(edges: pd.DataFrame, columns_csv: str | None) -> pd.DataFrame:
    """One row per column that has at least one edge, with its step metadata and degrees."""
    names = pd.unique(pd.concat([edges["source"].astype(str), edges["target"].astype(str)], ignore_index=True))
    meta = {}
    if columns_csv:
        cols = pd.read_csv(columns_csv, keep_default_na=False, na_values={"step_number": [""]})
        meta = cols.set_index("name").to_dict("index")
    parsed = {p.name: p for p in parse_columns(list(names), id_column="\0")[0]}
    rows = []
    for n in names:
        if n in meta:
            m = meta[n]
            rows.append((n, float(m["step_number"]), str(m["stage"]), str(m["step"]), m["kind"],
                         float(m["lots_visited"]) if str(m.get("lots_visited", "")) != "" else np.nan))
        elif n in parsed:
            p = parsed[n]
            rows.append((n, float(p.step_number), p.stage, p.step, p.kind, np.nan))
        else:
            rows.append((n, np.nan, "", "", "other", np.nan))
    nodes = pd.DataFrame(rows, columns=["name", "step_number", "stage", "step", "kind", "lots_visited"])
    nodes["in_degree"] = nodes["name"].map(edges["target"].astype(str).value_counts()).fillna(0).astype(int)
    nodes["out_degree"] = nodes["name"].map(edges["source"].astype(str).value_counts()).fillna(0).astype(int)
    return nodes


def _positions(nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = nodes["step_number"].to_numpy(dtype=float).copy()
    keys = nodes["name"].str.replace(r"::mach$", "", regex=True)
    known = ~np.isnan(x)
    for num in np.unique(x[known]):                       # different steps sharing a number: spread them a little
        at = known & (x == num)
        uniq = sorted(set(keys[at]))
        if len(uniq) > 1:
            width = min(0.8 / len(uniq), 0.25)
            offset = {k: (i - (len(uniq) - 1) / 2) * width for i, k in enumerate(uniq)}
            x[at] = num + keys[at].map(offset).to_numpy()
    start = (np.nanmax(x) + 1) if known.any() else 0.0
    x[~known] = start + np.arange((~known).sum())       # names without a step number: in order of appearance
    y = nodes["kind"].map(LANE).fillna(0.5).to_numpy(dtype=float)
    return x, y


def _controls(x0, y0, x1, y1) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic Bezier control points: arcs above the lanes from machine columns, below from presence columns."""
    span = max(float(np.nanmax(np.abs(x1 - x0))) if len(x0) else 1.0, 1.0)
    sign = np.where(y0 >= 0.75, 1.0, np.where(y0 <= 0.25, -1.0, 1.0))
    h = np.where(x0 == x1, 0.0, 0.2 + 1.2 * np.sqrt(np.abs(x1 - x0) / span))
    return (x0 + x1) / 2, (y0 + y1) / 2 + sign * h


def _curves(x0, y0, x1, y1, cx, cy) -> tuple[list, list]:
    t = np.linspace(0, 1, ARC_POINTS)[None, :]
    a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t ** 2
    X = a * x0[:, None] + b * cx[:, None] + c * x1[:, None]
    Y = a * y0[:, None] + b * cy[:, None] + c * y1[:, None]
    gap = np.full((len(x0), 1), np.nan)
    return (np.hstack([X, gap]).ravel().tolist(), np.hstack([Y, gap]).ravel().tolist())


def _edge_frame(edges: pd.DataFrame) -> pd.DataFrame:
    e = edges.copy()
    e["source"], e["target"] = e["source"].astype(str), e["target"].astype(str)
    for col, default in (("kind", "learned"), ("weight", np.nan), ("detail", "")):
        if col not in e:
            e[col] = default
    e["detail"] = e["detail"].fillna("").astype(str)
    return e.reset_index(drop=True)


def _geometry(edges: pd.DataFrame, nodes: pd.DataFrame):
    x, y = _positions(nodes)
    idx = {n: i for i, n in enumerate(nodes["name"])}
    s = edges["source"].map(idx).to_numpy()
    t = edges["target"].map(idx).to_numpy()
    cx, cy = _controls(x[s], y[s], x[t], y[t])
    return x, y, s, t, cx, cy


def build_figure(edges: pd.DataFrame, nodes: pd.DataFrame):
    import plotly.graph_objects as go
    C = LIGHT
    e = _edge_frame(edges)
    x, y, s, t, cx, cy = _geometry(e, nodes)
    fig = go.Figure()
    learned = (e["kind"] != "identical").to_numpy()
    for mask, name, dash, visible in ((learned, "learned edge", "solid", True),
                                      (~learned, "identical (same lots)", "dot", "legendonly")):
        xs, ys = _curves(x[s[mask]], y[s[mask]], x[t[mask]], y[t[mask]], cx[mask], cy[mask])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=name, meta="edges", hoverinfo="skip",
                                   visible=visible, opacity=0.45, line={"color": C["muted"], "width": 1, "dash": dash}))
    # edge hover targets at each arc's apex (the curve's midpoint)
    mx, my = 0.25 * x[s] + 0.5 * cx + 0.25 * x[t], 0.25 * y[s] + 0.5 * cy + 0.25 * y[t]
    w = e["weight"].to_numpy(dtype=float)
    texts = [f"{html.escape(a)} → {html.escape(b)}<br>{k}" + (f" · weight {wt:+.2f}" if not math.isnan(wt) else "")
             + (f"<br>{html.escape(d[:240])}" if d else "")
             for a, b, k, wt, d in zip(e["source"], e["target"], e["kind"], w, e["detail"])]
    fig.add_trace(go.Scattergl(x=mx, y=my, mode="markers", name="edge details", meta="edge-hover", showlegend=False,
                               marker={"size": 10, "color": "rgba(0,0,0,0)"}, hovertext=texts, hoverinfo="text"))
    fig.add_trace(go.Scattergl(x=[], y=[], mode="lines", name="selected", meta="highlight", showlegend=False,
                               hoverinfo="skip", line={"color": C["text"], "width": 2.5}))
    degree = (nodes["in_degree"] + nodes["out_degree"]).to_numpy()
    size = np.clip(7 + 2 * np.sqrt(degree), 8, 22)
    for kind, label in KINDS:
        at = np.flatnonzero((nodes["kind"] == kind).to_numpy())
        if not len(at):
            continue
        sub = nodes.iloc[at]
        hover = [f"<b>{html.escape(r.name)}</b><br>"
                 + (f"stage {html.escape(r.stage)} · step {html.escape(r.step)} · N = {r.step_number:g}<br>"
                    if not math.isnan(r.step_number) else "")
                 + (f"lots visited: {r.lots_visited:,.0f}<br>" if not math.isnan(r.lots_visited) else "")
                 + f"parents: {r.in_degree} · children: {r.out_degree}"
                 for r in sub.itertuples(index=False)]
        fig.add_trace(go.Scattergl(x=x[at], y=y[at], mode="markers", name=label, meta="nodes", customdata=at.tolist(),
                                   hovertext=hover, hoverinfo="text",
                                   marker={"size": size[at], "color": C[kind], "line": {"width": 1, "color": C["surface"]}}))
    apex = 0.25 * y[s] + 0.5 * cy + 0.25 * y[t]           # the furthest point of each arc from the lanes
    y_lo = min(float(apex.min()) if len(apex) else 0.0, 0.0) - 0.35
    y_hi = max(float(apex.max()) if len(apex) else 1.0, 1.0) + 0.35
    top = y_hi + 0.2                                        # stage markers sit above the highest arc
    shapes, stage_x, stage_text, stage_name = [], [], [], []
    known = nodes[~nodes["step_number"].isna()]
    if len(known) and known["stage"].ne("").any():
        order = known.sort_values("step_number")
        runs = (order["stage"] != order["stage"].shift()).cumsum()
        for k, (_, g) in enumerate(order.groupby(runs, sort=False)):
            lo, hi = g["step_number"].min() - 0.5, g["step_number"].max() + 0.5
            if k % 2 == 0:
                shapes.append({"type": "rect", "xref": "x", "yref": "paper", "x0": lo, "x1": hi, "y0": 0, "y1": 1,
                               "fillcolor": C["band"], "line": {"width": 0}, "layer": "below"})
            stage_x.append((lo + hi) / 2)
            stage_name.append(g["stage"].iloc[0])
            stage_text.append(f"stage {html.escape(g['stage'].iloc[0])}<br>steps {lo + 0.5:g}–{hi - 0.5:g}")
        labelled = len(stage_x) <= 30                       # names on the page when they fit; always on hover
        fig.add_trace(go.Scatter(x=stage_x, y=[top] * len(stage_x), mode="markers+text" if labelled else "markers",
                                 name="stages", meta="stages", showlegend=False, hovertext=stage_text,
                                 hoverinfo="text", text=[n[:18] for n in stage_name] if labelled else None,
                                 textposition="top center", textfont={"color": C["text2"], "size": 11},
                                 marker={"size": 12, "symbol": "line-ew", "color": C["muted"],
                                         "line": {"width": 2, "color": C["muted"]}}))
    lanes = [(LANE[k], k) for k, _ in KINDS if (nodes["kind"] == k).any()]
    n_learned, n_ident = int(learned.sum()), int((~learned).sum())
    fig.update_layout(
        title={"text": f"{n_learned:,} learned edges among {len(nodes):,} columns"
                       + (f" · {n_ident:,} identical pairs (toggle in legend)" if n_ident else ""),
               "font": {"color": C["text"], "size": 16}, "x": 0.01},
        shapes=shapes, height=720, hovermode="closest", dragmode="zoom",
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"], font={"color": C["text2"], "size": 12},
        legend={"orientation": "h", "x": 0.01, "y": 1.06, "bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 80, "r": 20, "t": 90, "b": 50},
        xaxis={"title": "step number (time →)", "showgrid": False, "zeroline": False, "linecolor": C["grid"]},
        yaxis={"tickvals": [v for v, _ in lanes], "ticktext": [k for _, k in lanes], "range": [y_lo, top + 0.45],
               "fixedrange": True, "showgrid": False, "zeroline": False},
        hoverlabel={"bgcolor": C["surface"], "font": {"color": C["text"]}, "bordercolor": C["grid"]})
    return fig


_CONTROLS = """<style>
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #fcfcfb; color: #0b0b0b; }
  .stx-bar { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; padding: 10px 16px;
             border-bottom: 1px solid rgba(11,11,11,0.10); font-size: 14px; }
  .stx-bar input { min-width: 260px; padding: 5px 8px; font: inherit; color: inherit; background: transparent;
                   border: 1px solid #c3c2b7; border-radius: 6px; }
  .stx-bar button { padding: 5px 10px; font: inherit; color: inherit; background: transparent;
                    border: 1px solid #c3c2b7; border-radius: 6px; cursor: pointer; }
  #stx-info { color: #52514e; }
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a19; color: #f0efec; }
    .stx-bar { border-bottom-color: rgba(255,255,255,0.10); }
    .stx-bar input, .stx-bar button { border-color: #383835; }
    #stx-info { color: #c3c2b7; }
  }
</style>
<div class="stx-bar">
  <label for="stx-search">Find a column</label>
  <input id="stx-search" type="search" placeholder="part of a name, then Enter" autocomplete="off">
  <button id="stx-clear" type="button">Clear</button>
  <span id="stx-info" aria-live="polite">Click a column to highlight its parents and children.</span>
</div>
"""

_SCRIPT = """
var gd = document.getElementById('{plot_id}');
var P = __PAYLOAD__;
var dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
var C = dark ? P.dark : P.light;
var NODE = [], HL = -1, EDGE = [], STAGE = -1;
gd.data.forEach(function (t, k) {
  if (t.meta === 'nodes') NODE.push(k);
  if (t.meta === 'highlight') HL = k;
  if (t.meta === 'edges') EDGE.push(k);
  if (t.meta === 'stages') STAGE = k;
});
if (dark) {
  Plotly.relayout(gd, {paper_bgcolor: C.surface, plot_bgcolor: C.surface, 'font.color': C.text2,
                       'title.font.color': C.text, 'xaxis.linecolor': C.grid, 'hoverlabel.bgcolor': C.surface,
                       'hoverlabel.font.color': C.text, 'hoverlabel.bordercolor': C.grid,
                       shapes: (gd.layout.shapes || []).map(function (s) { s.fillcolor = C.band; return s; })});
  NODE.forEach(function (k) {
    Plotly.restyle(gd, {'marker.color': C[P.kinds[gd.data[k].name]], 'marker.line.color': C.surface}, [k]);
  });
  Plotly.restyle(gd, {'line.color': C.text}, [HL]);
  if (STAGE >= 0) Plotly.restyle(gd, {'textfont.color': C.text2}, [STAGE]);
}
var info = document.getElementById('stx-info');
function curve(e, xs, ys) {
  var x0 = P.x[e[0]], y0 = P.y[e[0]], x1 = P.x[e[1]], y1 = P.y[e[1]], cx = e[2], cy = e[3];
  for (var i = 0; i < __ARC_POINTS__; i++) {
    var t = i / (__ARC_POINTS__ - 1), a = (1 - t) * (1 - t), b = 2 * (1 - t) * t, c = t * t;
    xs.push(a * x0 + b * cx + c * x1); ys.push(a * y0 + b * cy + c * y1);
  }
  xs.push(null); ys.push(null);
}
function stxHighlight(i) {
  i = Number(i);
  var keep = {}, xs = [], ys = [];
  keep[i] = true;
  P.parents[i].forEach(function (k) { var e = P.edges[k]; keep[e[0]] = true; curve(e, xs, ys); });
  P.children[i].forEach(function (k) { var e = P.edges[k]; keep[e[1]] = true; curve(e, xs, ys); });
  NODE.forEach(function (k) {
    var op = gd.data[k].customdata.map(function (g) { return keep[g] ? 1 : 0.12; });
    Plotly.restyle(gd, {'marker.opacity': [op]}, [k]);
  });
  Plotly.restyle(gd, {opacity: 0.12}, EDGE);
  Plotly.restyle(gd, {x: [xs], y: [ys]}, [HL]);
  info.textContent = P.names[i] + ': ' + P.parents[i].length + ' parent edge(s), ' +
                     P.children[i].length + ' child edge(s). Esc or Clear to reset.';
}
function stxClear() {
  NODE.forEach(function (k) { Plotly.restyle(gd, {'marker.opacity': 1}, [k]); });
  Plotly.restyle(gd, {opacity: 0.45}, EDGE);
  Plotly.restyle(gd, {x: [[]], y: [[]]}, [HL]);
  info.textContent = 'Click a column to highlight its parents and children.';
}
window.stxHighlight = stxHighlight;
window.stxClear = stxClear;
gd.on('plotly_click', function (ev) {
  var p = ev.points[0];
  if (NODE.indexOf(p.curveNumber) >= 0) stxHighlight(p.customdata);
});
document.getElementById('stx-clear').addEventListener('click', stxClear);
document.addEventListener('keydown', function (ev) { if (ev.key === 'Escape') stxClear(); });
document.getElementById('stx-search').addEventListener('keydown', function (ev) {
  if (ev.key !== 'Enter') return;
  var q = ev.target.value.trim().toLowerCase();
  if (!q) return;
  for (var i = 0; i < P.names.length; i++) {
    if (P.names[i].toLowerCase().indexOf(q) >= 0) {
      stxHighlight(i);
      var r = gd.layout.xaxis.range, half = Math.max(10, (r[1] - r[0]) / 8);
      Plotly.relayout(gd, {'xaxis.range': [P.x[i] - half, P.x[i] + half]});
      return;
    }
  }
  info.textContent = 'No column name contains "' + ev.target.value + '".';
});
"""


def write_viewer(fig, nodes: pd.DataFrame, edges: pd.DataFrame, path: str | Path, cdn: bool = False) -> None:
    e = _edge_frame(edges)
    x, y, s, t, cx, cy = _geometry(e, nodes)
    parents = [[] for _ in range(len(nodes))]
    children = [[] for _ in range(len(nodes))]
    for k, (a, b) in enumerate(zip(s, t)):
        children[a].append(k)
        parents[b].append(k)
    payload = {"names": nodes["name"].tolist(), "x": np.round(x, 4).tolist(), "y": y.tolist(),
               "edges": [[int(a), int(b), round(float(c), 4), round(float(d), 4)] for a, b, c, d in zip(s, t, cx, cy)],
               "parents": parents, "children": children, "light": LIGHT, "dark": DARK,
               "kinds": {label: kind for kind, label in KINDS}}
    script = (_SCRIPT.replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/"))
              .replace("__ARC_POINTS__", str(ARC_POINTS)))
    page = fig.to_html(full_html=True, include_plotlyjs="cdn" if cdn else True, post_script=script,
                       config={"displaylogo": False, "scrollZoom": True, "responsive": True})
    page = page.replace("<body>", "<body>\n" + _CONTROLS, 1)
    page = page.replace("<head>", '<head>\n<meta charset="utf-8"><title>Learned graph</title>', 1)
    Path(path).write_text(page, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m utilities.plot_edges", description=__doc__.split("\n\n")[0])
    ap.add_argument("--edges", required=True, help="edges.csv with source,target (weight, kind, detail optional)")
    ap.add_argument("--columns", default=None, help="columns.csv from utilities.preprocess (stage names, lots visited)")
    ap.add_argument("--out", required=True, help="output .html")
    ap.add_argument("--min-weight", type=float, default=0.0, help="hide learned edges with |weight| below this")
    ap.add_argument("--max-edges", type=int, default=30_000, help="keep at most this many edges (strongest first)")
    ap.add_argument("--cdn", action="store_true", help="load plotly.js from the web (smaller file, needs internet)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    edges = pd.read_csv(args.edges)
    missing = {"source", "target"} - set(edges.columns)
    if missing:
        raise SystemExit(f"{args.edges} needs columns 'source' and 'target' (missing: {sorted(missing)})")
    edges = _edge_frame(edges)
    if args.min_weight > 0:
        weak = (edges["kind"] != "identical") & ~(edges["weight"].abs() >= args.min_weight)
        edges = edges[~weak]
    if len(edges) > args.max_edges:
        n = len(edges)
        order = edges["weight"].abs().sort_values(ascending=False, na_position="last").index
        edges = edges.loc[order].head(args.max_edges)
        print(f"WARNING: kept {args.max_edges} of {n} edges (--max-edges {args.max_edges}), strongest |weight| first")
    nodes = load_nodes(edges, args.columns)
    fig = build_figure(edges, nodes)
    write_viewer(fig, nodes, edges, args.out, cdn=args.cdn)
    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out} ({len(nodes):,} columns, {len(edges):,} edges, {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
