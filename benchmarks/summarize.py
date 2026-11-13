"""Collect benchmarks/*/results.json into one Markdown table and a speed-vs-accuracy figure.

Usage (from the repository root, after benchmarks/run_simulated.sh):
    python benchmarks/summarize.py
Writes benchmarks/summary.md and benchmarks/speed_vs_accuracy.png. Needs matplotlib (pip install matplotlib).
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = [  # folder, dataset label, suffix for the method name
    ("cont100", "Continuous, 100 variables, 20,000 rows", ""),
    ("cont100_tabu", "Continuous, 100 variables, 20,000 rows", " (tabu 10)"),
    ("cont100_fges_bound", "Continuous, 100 variables, 20,000 rows", ""),
    ("cont500", "Continuous, 500 variables, 50,000 rows", ""),
    ("cont500_tabu", "Continuous, 500 variables, 50,000 rows", " (tabu 10)"),
    ("cont500_fges_unbounded", "Continuous, 500 variables, 50,000 rows", ""),
    ("cont2500", "Continuous, 2,500 variables, 100,000 rows", ""),
    ("cat50", "Categorical, 50 variables, 20,000 rows", ""),
]
NAMES = {"dagma": "DAGMA", "dagma_hc": "DAGMA + hill climbing", "fges": "fGES", "empty graph": "Empty graph"}
ORDER = ["DAGMA", "DAGMA + hill climbing", "DAGMA + hill climbing (tabu 10)", "fGES", "fGES (bound 20)"]
# Okabe-Ito colour-blind-safe colours; one per method family, the second variant hollow and dashed.
STYLE = {"DAGMA": ("#D55E00", "o", "-", True), "DAGMA + hill climbing": ("#0072B2", "s", "-", True),
         "DAGMA + hill climbing (tabu 10)": ("#0072B2", "s", "--", False), "fGES": ("#009E73", "^", "-", True),
         "fGES (bound 20)": ("#009E73", "^", "--", False)}


def load() -> pd.DataFrame:
    rows = []
    for folder, label, suffix in RUNS:
        f = HERE / folder / "results.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        for r in res["methods"]:
            if r["method"] == "empty graph":
                continue
            name = NAMES[r["method"]] + suffix
            bound = res.get("details", {}).get(r["method"], {}).get("bound_k", 0)
            if r["method"] == "fges" and bound:
                name = f"fGES (bound {bound})"
            rows.append({**r, "dataset": label, "variables": res["n_variables"], "data_type": res["data_type"],
                         "method": name})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> str:
    out = []
    for label, g in df.groupby("dataset", sort=False):
        g = g.set_index("method").reindex([m for m in ORDER if m in set(g.method)]).reset_index()
        t = pd.DataFrame({
            "Method": g.method,
            "Time (s)": g.seconds.round(0).astype(int),
            "Edges": g.edges.astype(int),
            "Skeleton F1": g.skeleton_f1.round(2),
            "Directed F1": g.directed_f1.round(2),
            "SHD": g.shd.astype(int),
            "Held-out gain / row": g.gain_vs_empty_per_row.round(1),
        })
        out += [f"**{label}** (true graph: {int(g.true_edges.iloc[0])} edges)", "", t.to_markdown(index=False), ""]
    return "\n".join(out)


def figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cont = df[df.data_type == "continuous"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    panels = [("skeleton_f1", "Skeleton F1 (higher is better)"), ("directed_f1", "Directed F1 (higher is better)"),
              ("shd_per_true_edge", "SHD / true edges (lower is better)")]
    cont = cont.assign(shd_per_true_edge=cont.shd / cont.true_edges)
    for ax, (col, title) in zip(axes, panels):
        for m in ORDER:
            g = cont[cont.method == m].sort_values("variables")
            if g.empty:
                continue
            color, marker, ls, filled = STYLE[m]
            ax.plot(g.seconds, g[col], ls=ls, color=color, lw=2, marker=marker, ms=8,
                    mfc=color if filled else "white", mec=color, label=m)
            if "tabu" in m:                              # sits on top of plain hill climbing; label that one only
                continue
            for _, r in g.iterrows():
                ax.annotate(f"{int(r.variables):,}", (r.seconds, r[col]), textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color="#444444")
        ax.set_xscale("log")
        ax.set_xlabel("Runtime in seconds (log scale)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy against the true graph")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Speed vs accuracy on simulated continuous data (point labels: number of variables)", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main() -> None:
    df = load()
    md = table(df)
    (HERE / "summary.md").write_text(md + "\n")
    print(md)
    figure(df, HERE / "speed_vs_accuracy.png")


if __name__ == "__main__":
    main()
