"""Build the three headline figures from results/summary.csv.

Usage:  python harness/make_plots.py
Writes: plots/fig1_collapse.png, plots/fig2_pareto.png, plots/fig3_failures.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "summary.csv"
PLOTS = ROOT / "plots"

QUANT_ORDER = ["bf16", "Q8_0", "Q4_K_M", "Q3_K_M"]
MODEL_ORDER = ["qwen3-1.7b", "qwen3-4b", "llama-3.1-8b"]
MODEL_LABEL = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B-Instruct",
               "llama-3.1-8b": "Llama-3.1-8B-Instruct"}

# colorblind-safe, readable in print
C_T1 = "#4C72B0"      # single-shot
C_T2 = "#C44E52"      # trajectory
C_OFF = "#8C8C8C"
C_ON = "#2E7D32"
FAIL_COLORS = {
    "invalid-output": "#C44E52",
    "wrong-tool": "#DD8452",
    "bad-args": "#CCB974",
    "loop": "#64B5CD",
    "premature-stop": "#8172B3",
    "server-error": "#555555",
}

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False,
})


def num(v):
    if v in ("", None, "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def load() -> list[dict]:
    with open(SUMMARY, encoding="utf-8") as f:
        rows = [{k: num(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["model"] = str(r["model"])
        r["quant"] = str(r["quant"])
    return rows


def present_quants(rows: list[dict], model: str) -> list[str]:
    have = {r["quant"] for r in rows if r["model"] == model}
    return [q for q in QUANT_ORDER if q in have]


def pick(rows, model, quant, constrained):
    for r in rows:
        if (r["model"] == model and r["quant"] == quant
                and int(r["constrained"]) == int(constrained)):
            return r
    return None


# --------------------------------------------------------------- figure 1

def fig_collapse(rows: list[dict]) -> None:
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.5),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        quants = present_quants(rows, model)
        xs = range(len(quants))
        t1 = [pick(rows, model, q, 0) for q in quants]
        t1y = [(r["t1_acc"] * 100 if r and r["t1_acc"] is not None else None) for r in t1]
        t2 = [pick(rows, model, q, 0) for q in quants]
        t2y = [(r["t2_success"] * 100 if r and r["t2_success"] is not None else None)
               for r in t2]

        ax.plot(xs, t1y, "o-", color=C_T1, lw=2, ms=6, label="single tool call (BFCL)")
        ax.plot(xs, t2y, "s-", color=C_T2, lw=2, ms=6, label="multi-step trajectory")

        # shade the gap between the two curves — the paper's core claim
        pairs = [(x, a, b) for x, a, b in zip(xs, t1y, t2y)
                 if a is not None and b is not None]
        if pairs:
            ax.fill_between([p[0] for p in pairs], [p[1] for p in pairs],
                            [p[2] for p in pairs], color=C_T2, alpha=0.08)

        for x, q in zip(xs, quants):
            r = pick(rows, model, q, 0)
            if r and not int(r["fits_8gb"]):
                ax.annotate("weights\n> 8 GB", (x, 3), ha="center", fontsize=7,
                            color="#888888")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(quants, rotation=20)
        ax.set_ylim(0, 100)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=10)
        ax.set_xlabel("weight quantization")

    axes[0].set_ylabel("success rate (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Single-call accuracy is flat across the ladder; multi-step success "
                 "is not", fontsize=11, y=1.0)
    fig.text(0.5, -0.04, "unconstrained decoding · RTX 5050 Laptop 8 GB · "
                         "llama.cpp b10502 · temperature 0",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig1_collapse.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 2

def fig_pareto(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    tps = [r["decode_tps_median"] for r in rows if r["decode_tps_median"]]
    lo, hi = (min(tps), max(tps)) if tps else (1, 1)

    def size_of(v):
        if not v:
            return 40
        frac = 0 if hi == lo else (v - lo) / (hi - lo)
        return 40 + 320 * frac

    for r in rows:
        if r["peak_vram_mib"] is None or r["t2_success"] is None:
            continue
        on = int(r["constrained"])
        ax.scatter(r["peak_vram_mib"] / 1024, r["t2_success"] * 100,
                   s=size_of(r["decode_tps_median"]),
                   color=C_ON if on else C_OFF, alpha=0.75,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"{MODEL_LABEL.get(r['model'], r['model']).split('-')[0]} {r['quant']}",
                    (r["peak_vram_mib"] / 1024, r["t2_success"] * 100),
                    textcoords="offset points", xytext=(7, 4), fontsize=6.5,
                    color="#444444")

    ax.axvline(8.0, color="#B00020", ls="--", lw=1.2, zorder=2)
    ax.annotate("8 GB VRAM limit", (8.0, ax.get_ylim()[1]), rotation=90,
                va="top", ha="right", fontsize=8, color="#B00020",
                textcoords="offset points", xytext=(-4, -6))

    ax.set_xlabel("peak VRAM (GiB)")
    ax.set_ylabel("multi-step trajectory success (%)")
    ax.set_title("Cost of reliability on a consumer GPU", fontsize=11)
    handles = [
        Line2D([], [], marker="o", ls="", color=C_OFF, ms=8, label="unconstrained"),
        Line2D([], [], marker="o", ls="", color=C_ON, ms=8, label="grammar-constrained"),
        Line2D([], [], marker="o", ls="", color="#BBBBBB", ms=5,
               label=f"marker size = decode tok/s ({lo:.0f}–{hi:.0f})"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig2_pareto.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 3

def fig_failures(rows: list[dict]) -> None:
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.6),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    used: list[str] = []
    for ax, model in zip(axes, models):
        quants = present_quants(rows, model)
        xs = list(range(len(quants)))
        bottoms = [0.0] * len(quants)
        for ft, color in FAIL_COLORS.items():
            vals = []
            for q in quants:
                r = pick(rows, model, q, 0)
                n = r["t2_n"] if r and r["t2_n"] else 0
                cnt = r.get(f"t2_primary_{ft}") if r else 0
                vals.append((cnt or 0) / n * 100 if n else 0.0)
            if any(vals):
                ax.bar(xs, vals, bottom=bottoms, color=color, width=0.62,
                       label=ft if ft not in used else None)
                if ft not in used:
                    used.append(ft)
                bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(xs)
        ax.set_xticklabels(quants, rotation=20)
        ax.set_ylim(0, 100)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=10)
        ax.set_xlabel("weight quantization")

    axes[0].set_ylabel("failed trajectories (% of runs)")
    handles = [Line2D([], [], marker="s", ls="", color=FAIL_COLORS[f], ms=8, label=f)
               for f in used]
    axes[-1].legend(handles=handles, loc="upper left", fontsize=7.5,
                    title="primary failure", title_fontsize=8)
    fig.suptitle("How agents fail, by quantization level", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig3_failures.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 4

def fig_masking(rows: list[dict]) -> None:
    """Success rate vs step-level error volume: the score can hide the damage."""
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.9 * len(models), 3.6))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        quants = present_quants(rows, model)
        xs = list(range(len(quants)))
        base = [pick(rows, model, q, 0) for q in quants]
        succ = [(r["t2_success"] * 100 if r and r["t2_success"] is not None else None)
                for r in base]
        b1 = [(r["t2_success_budget1"] * 100
               if r and r.get("t2_success_budget1") is not None else None) for r in base]
        vol = [(r["t2_err_volume_mean"] if r else None) for r in base]

        ax.plot(xs, succ, "s-", color=C_T2, lw=2, ms=6, label="success (as preregistered)")
        ax.plot(xs, b1, "^--", color="#8172B3", lw=1.6, ms=6,
                label="success, ≤1 step error allowed")
        ax.set_ylim(0, 100)
        ax.set_ylabel("trajectory success (%)")
        ax.set_xticks(xs)
        ax.set_xticklabels(quants, rotation=20)
        ax.set_xlabel("weight quantization")

        ax2 = ax.twinx()
        ax2.bar(xs, vol, color="#DD8452", alpha=0.30, width=0.55, zorder=0,
                label="step errors per trajectory")
        ax2.set_ylabel("step errors per trajectory", color="#B35B29")
        ax2.tick_params(axis="y", colors="#B35B29")
        ax2.set_ylim(0, max(v for v in vol if v is not None) * 1.6)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        ax.set_title(MODEL_LABEL.get(model, model), fontsize=10)

    h1, l1 = axes[0].get_legend_handles_labels()
    axes[0].legend(h1 + [Line2D([], [], marker="s", ls="", color="#DD8452",
                                alpha=0.5, ms=8, label="step errors per trajectory")],
                   l1 + ["step errors per trajectory"], loc="lower left", fontsize=7.5)
    fig.suptitle("A flat score can hide rising damage: error volume grows as the "
                 "ladder descends", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig4_masking.png", bbox_inches="tight")
    plt.close(fig)


def fig_substitution(rows: list[dict]) -> None:
    """Where the errors went when the grammar was switched on.

    Only configurations in which `invalid-output` actually occurred are shown:
    those are the cells where the constraint binds at all.
    """
    cells = []
    for r in rows:
        if int(r["constrained"]) or not r.get("t2_step_invalid-output"):
            continue
        on = pick(rows, r["model"], r["quant"], 1)
        if on:
            cells.append((r, on))
    if not cells:
        return

    order = ["invalid-output", "wrong-tool", "bad-args", "loop"]
    fig, ax = plt.subplots(figsize=(1.9 * len(cells) + 2.4, 4.0))
    width, gap = 0.36, 0.20
    xticks, xlabels = [], []

    for i, (off, on) in enumerate(cells):
        base = i * (2 * width + gap + 0.34)
        for j, (col, cfg) in enumerate(((base, off), (base + width, on))):
            bottom = 0.0
            for ft in order:
                v = cfg.get(f"t2_step_{ft}") or 0
                if v:
                    ax.bar(col, v, bottom=bottom, width=width,
                           color=FAIL_COLORS[ft], edgecolor="white", linewidth=0.6)
                    bottom += v
            ax.annotate("off" if j == 0 else "on", (col, bottom + 1.5),
                        ha="center", fontsize=8,
                        color="#333333" if j == 0 else C_ON)
        xticks.append(base + width / 2)
        xlabels.append(f"{MODEL_LABEL.get(off['model'], off['model']).split('-Instruct')[0]}\n{off['quant']}")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("step-level errors (2 reps × 20 scenarios)")
    ax.set_title("Constrained decoding removes invalid output entirely —\n"
                 "and the errors reappear as wrong tools and wrong arguments",
                 fontsize=10.5)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=FAIL_COLORS[f],
                              ms=8, label=f) for f in order],
              loc="upper left", fontsize=8, title="step error type", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig5_substitution.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SUMMARY.exists():
        print("results/summary.csv missing -- run harness/aggregate.py first")
        return
    rows = load()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig_collapse(rows)
    fig_pareto(rows)
    fig_failures(rows)
    fig_masking(rows)
    fig_substitution(rows)
    print("wrote plots/fig1_collapse.png, fig2_pareto.png, fig3_failures.png, "
          "fig4_masking.png, fig5_substitution.png")


if __name__ == "__main__":
    main()
