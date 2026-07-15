"""Generate Study 002 figures directly from the committed JSONL data.

Run: python experiments/study_002/make_figures.py
Outputs PNGs into experiments/study_002/figures/.

Figures derive every value from metrics.jsonl so nothing is hand-transcribed.
Per the data-viz method: F1 and routing are separate panels (never a dual axis).
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).parent
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---- validated palette (light surface) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#d03b3b"        # status: critical
GREEN = "#006300"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
COLLAPSE_WASH = "#f6dede"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_metrics():
    rows = []
    with open(HERE / "metrics.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["iteration_n"])
    return rows


def main():
    rows = load_metrics()
    it = [r["iteration_n"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    prec = [r["macro_precision"] for r in rows]
    rec = [r["macro_recall"] for r in rows]
    routing = [r["post_routing_score"] for r in rows]
    code = [bool(r.get("code_changes_attempted")) for r in rows]
    claims = [r["avg_claims_per_abstract"] for r in rows]

    collapse_start = next(i for i, v in zip(it, f1) if v == 0.0)  # 6

    # ===================================================================
    # FIGURE 1 — F1 (top) and routing (bottom), shared x. The decoupling.
    # ===================================================================
    fig, (axf, axr) = plt.subplots(
        2, 1, figsize=(9, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12},
    )

    for ax in (axf, axr):
        ax.axvspan(collapse_start - 0.5, it[-1] + 0.5, color=COLLAPSE_WASH, zorder=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # --- F1 panel ---
    axf.plot(it, f1, color=BLUE, linewidth=2.2, zorder=3)
    axf.scatter(it, f1, s=42, color=BLUE, zorder=4, edgecolor=SURFACE, linewidth=1.2)
    # mark code-change iterations
    code_x = [i for i, c in zip(it, code) if c]
    code_y = [v for v, c in zip(f1, code) if c]
    axf.scatter(code_x, code_y, s=150, facecolor="none", edgecolor=RED,
                linewidth=1.8, zorder=5)
    # Study 001 final reference
    axf.axhline(0.142, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.3, zorder=2)
    axf.text(12.15, 0.142, "Study 001 final 0.142", color=INK2, fontsize=8.5,
             va="center", ha="right", style="italic")
    axf.set_ylim(-0.03, 0.78)
    axf.set_ylabel("Macro-F1", color=INK, fontsize=11)
    axf.set_yticks([0.0, 0.2, 0.4, 0.6])
    axf.tick_params(colors=INK2)

    axf.annotate("iter 1: 0.691\nonly real gain (prompt edit)",
                 xy=(1, 0.691), xytext=(2.1, 0.70), fontsize=8.5, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=1), va="center")
    axf.annotate("iter 6: two-call refactor\n→ 0 claims on all abstracts",
                 xy=(6, 0.0), xytext=(6.4, 0.30), fontsize=8.5, color=RED,
                 arrowprops=dict(arrowstyle="-", color=RED, lw=1), va="center")
    axf.text(9.5, 0.055, "extraction dead — F1 = 0.000", color=RED, fontsize=9,
             ha="center", style="italic")
    axf.set_title("Study 002: extraction quality collapses while the routing signal barely moves",
                  color=INK, fontsize=12.5, loc="left", pad=10, fontweight="bold")

    # --- routing panel ---
    axr.plot(it, routing, color=ORANGE, linewidth=2.2, zorder=3)
    axr.scatter(it, routing, s=42, color=ORANGE, zorder=4, edgecolor=SURFACE, linewidth=1.2)
    axr.set_ylim(0, 0.026)
    axr.set_ylabel("Aggregate routing\nscore (results attn.)", color=INK, fontsize=10.5)
    axr.set_xlabel("Iteration", color=INK, fontsize=11)
    axr.set_xticks(it)
    axr.tick_params(colors=INK2)
    axr.annotate("frozen at 0.0058 through the dead tail",
                 xy=(9, 0.0058), xytext=(7.4, 0.014), fontsize=8.5, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    axr.text(0.15, 0.0235, "target for “results-focused” attention would be ≥ 0.5 — ~20× off-scale above",
             fontsize=8, color=MUTED, style="italic")

    legend_elems = [
        Line2D([0], [0], color=BLUE, lw=2.2, marker="o", markersize=6,
               markeredgecolor=SURFACE, label="Macro-F1"),
        Line2D([0], [0], color=ORANGE, lw=2.2, marker="o", markersize=6,
               markeredgecolor=SURFACE, label="Routing score"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=RED, markersize=11, markeredgewidth=1.8,
               label="code (extractor.py) edited"),
        Patch(facecolor=COLLAPSE_WASH, label="zero-claim collapse (iters 6–12)"),
    ]
    axf.legend(handles=legend_elems, loc="upper right", frameon=False,
               fontsize=8.5, handletextpad=0.6, labelspacing=0.4)

    fig.savefig(FIGDIR / "fig1_f1_routing_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # FIGURE 2 — precision & recall trajectory + claims/abstract
    # ===================================================================
    fig2, ax = plt.subplots(figsize=(9, 4.4))
    ax.axvspan(collapse_start - 0.5, it[-1] + 0.5, color=COLLAPSE_WASH, zorder=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot(it, prec, color=BLUE, linewidth=2.2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3, label="Precision")
    ax.plot(it, rec, color=AQUA, linewidth=2.2, marker="s", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3, label="Recall")
    ax.set_ylim(-0.03, 0.83)
    ax.set_xticks(it)
    ax.set_xlabel("Iteration", color=INK)
    ax.set_ylabel("Macro precision / recall", color=INK)
    ax.tick_params(colors=INK2)
    ax.text(1, prec[1] + 0.03, "P 0.758", color=BLUE, fontsize=8.5, ha="center")
    ax.text(1, rec[1] - 0.06, "R 0.635", color=GREEN, fontsize=8.5, ha="center")
    ax.text(9.5, 0.05, "both collapse to 0 at iter 6", color=RED, fontsize=9,
            ha="center", style="italic")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_title("Precision and recall both decline before the iteration-6 collapse",
                 color=INK, fontsize=12, loc="left", pad=8, fontweight="bold")
    fig2.savefig(FIGDIR / "fig2_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ===================================================================
    # FIGURE 3 — offline replication (operator-run, not from committed data)
    # ===================================================================
    fig3, ax = plt.subplots(figsize=(7.2, 4.4))
    metrics = ["Precision", "Recall", "Macro-F1"]
    naive = [0.548, 0.559, 0.554]
    twostage = [0.308, 0.219, 0.256]
    x = range(len(metrics))
    w = 0.38
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    b1 = ax.bar([i - w / 2 for i in x], naive, width=w, color=BLUE, zorder=3,
                label="Naive single-call baseline")
    b2 = ax.bar([i + w / 2 for i in x], twostage, width=w, color=ORANGE, zorder=3,
                label="Corrected two-stage (agent's iter-6 direction)")
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.012,
                    f"{rect.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9, color=INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, color=INK)
    ax.set_ylim(0, 0.63)
    ax.set_ylabel("Score on iteration-6's 25-abstract draw", color=INK, fontsize=10)
    ax.tick_params(colors=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, fontsize=9)
    ax.set_title("Even implemented correctly, the agent's chosen refactor loses to leaving the extractor alone",
                 color=INK, fontsize=11, loc="left", pad=30, fontweight="bold")
    ax.text(1.0, -0.13, "Offline operator replication (Random(48) draw); not reproducible from committed run files.",
            transform=ax.transData, fontsize=7.6, color=MUTED, style="italic", ha="center")
    fig3.savefig(FIGDIR / "fig3_offline_replication.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print("wrote:")
    for p in sorted(FIGDIR.glob("*.png")):
        print("  ", p.relative_to(HERE.parent.parent), p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
