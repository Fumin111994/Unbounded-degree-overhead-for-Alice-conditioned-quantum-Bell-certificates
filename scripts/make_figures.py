#!/usr/bin/env python3
"""M7: publication figures 1-3 with certified/numerical separation.

Figure 1 (level-2 closure transition), Figure 2 (level-1 family separation
and covering argument), Figure 3 (filtration schematic).  Solver-generated
curves are numerical; certified statements are shaded intervals and marked
points.  Data is exported to artifacts/figures/ for the reproduction
package.

Outputs: figures/fig{1,2,3}.pdf, artifacts/figures/fig{1,2}_data.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402

FIGDIR = ROOT / "figures"
ARTDIR = ROOT / "artifacts" / "figures"

# colorblind-friendly palette
C_EXACT = "#1f5fb4"
C_CERT = "#2e8b57"
C_NUM = "#555555"
C_PIN = "#d95f02"
C_THRESH = "#7570b3"
C_BAND = "#f4c7c3"
C_SHADE = "#cfe0f4"

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.4,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "figure.dpi": 200,
    }
)


def os_values(alphas, level):
    values = []
    for a in alphas:
        fn = tilted_chsh(float(a))
        summary, _ = solve_problem(
            build_onesided_problem(fn, level, enforce_probability_positivity=False),
            "CLARABEL",
        )
        values.append(summary["value"])
    return np.array(values)


def q_of(a):
    return np.sqrt(8.0 + 2.0 * np.asarray(a) ** 2)


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------


def figure1() -> None:
    fan = json.loads((ARTDIR.parent / "m65" / "tangent_fan.json").read_text())
    a_cov, a_hi = fan["covered_interval_float"]
    root = json.loads((ARTDIR.parent / "m6" / "rootlocus.json").read_text())
    slacks = {}
    for rec in root["results"]:
        slacks[rec["alpha"]] = max(
            info["slack_above_q_float"] for info in rec["blocks"].values()
        )

    alphas = np.linspace(1.0, 1.96, 49)
    os2 = os_values(alphas, 2)
    q = q_of(alphas)
    gap = os2 - q

    fig, (ax_top, ax) = plt.subplots(
        2, 1, figsize=(6.3, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.5], "hspace": 0.12},
    )

    # top: value curves
    ax_top.plot(alphas, q, color=C_EXACT, lw=1.6,
                label=r"$q(\alpha)=\sqrt{8+2\alpha^2}$ (exact); "
                      r"$\omega_2^{\rm std}=q$ overlaps it")
    ax_top.plot(alphas, os2, color=C_NUM, lw=1.2, ls=(0, (4, 2)),
                label=r"$\omega_2^{\rm os}(\alpha)$, level 2 (numerical)")
    ax_top.axvline(1.28427, color=C_THRESH, lw=1.0, ls=":")
    ax_top.annotate(r"$\alpha^{*}\approx1.28427$" + "\n(numerical only)",
                    xy=(1.28427, 3.18), xytext=(1.06, 3.62),
                    fontsize=7.5, color=C_THRESH,
                    arrowprops=dict(arrowstyle="->", color=C_THRESH, lw=0.7))
    ax_top.set_ylabel("value")
    ax_top.set_ylim(2.8, 3.95)
    ax_top.legend(loc="lower right", frameon=False)
    ax_top.grid(alpha=0.25)

    # bottom: gap on log scale
    ax.axvspan(a_cov, a_hi, color=C_SHADE, alpha=0.85, zorder=0,
               label="exactly certified non-closure\n(tangent fan, algebraic endpoints)")
    ax.axvspan(1.28427, a_cov, color=C_BAND, alpha=0.55, zorder=0)
    ax.axvspan(a_hi, 2.0, color=C_BAND, alpha=0.55, zorder=0,
               label="not covered by certificates")
    ax.plot(alphas, np.maximum(gap, 3e-9), color=C_NUM, lw=1.2,
            label=r"$\omega_2^{\rm os}(\alpha)-q(\alpha)$ (numerical)")
    ax.axhspan(3e-9, 3e-8, color="#dddddd", alpha=0.8, zorder=0)
    ax.annotate("solver noise floor", xy=(1.7, 7e-9), fontsize=7,
                color="#666666", ha="left")
    ax.axvline(1.28427, color=C_THRESH, lw=1.0, ls=":")
    for x_pin, slack, name in [(1.25, slacks.get("5/4", 1.8e-8), r"$5/4$"),
                               (41/32, slacks.get("41/32", 9e-9), r"$41/32$")]:
        ax.plot([x_pin], [slack], marker="D", color=C_PIN, markersize=5,
                zorder=5)
    ax.annotate("pins: exact closure interval\n"
                r"$\omega_2^{\rm os}\in[q,q+s^{*}]$, $s^{*}<2\times10^{-8}$",
                xy=(41/32, 2.2e-8), xytext=(1.42, 4.5e-8),
                fontsize=7.5, color=C_PIN,
                arrowprops=dict(arrowstyle="->", color=C_PIN, lw=0.7))
    ax.set_yscale("log")
    ax.set_ylim(2.5e-9, 3e-1)
    ax.set_xlim(1.0, 2.0)
    ax.set_xlabel(r"tilt $\alpha$")
    ax.set_ylabel(r"$\omega_2^{\rm os}(\alpha)-q(\alpha)$")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25, which="both")

    fig.savefig(FIGDIR / "fig1.pdf", bbox_inches="tight")
    plt.close(fig)

    (ARTDIR / "fig1_data.json").write_text(json.dumps({
        "alphas": alphas.tolist(),
        "omega_os2_numerical": os2.tolist(),
        "quantum_exact": q.tolist(),
        "gap": gap.tolist(),
        "certified": {
            "nonclosure_interval_algebraic": fan["covered_interval"],
            "nonclosure_interval_float": [a_cov, a_hi],
            "alpha_star_numerical": 1.28427,
            "pins": {"5/4": slacks.get("5/4"), "41/32": slacks.get("41/32")},
        },
        "note": "solver-generated curves are numerical; certified statements "
                "are the shaded interval and marked pins",
    }, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------


def figure2() -> None:
    cov = json.loads((ARTDIR.parent / "m26" / "anchors" / "coverage.json").read_text())
    anchors = [(float(__import__("fractions").Fraction(r["alpha"])),
                float(__import__("fractions").Fraction(r["bound"])))
               for r in cov["anchors"]]
    anchors.sort()

    alphas = np.linspace(0.0, 1.96, 79)
    os1 = os_values(alphas, 1)
    line = 2.0**1.5 + alphas

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(alphas, line, color=C_EXACT, lw=1.6,
            label=r"standard level 1: $2\sqrt{2}+\alpha$ (exact)")
    ax.plot(alphas, os1, color=C_NUM, lw=1.2, ls=(0, (4, 2)),
            label=r"$\omega_1^{\rm os}(\alpha)$ (numerical)")
    # certified chord from the two endpoints (Thm: separation with margin)
    chord = 2.0**1.5 + (2.0 - 2.0**0.5) * alphas
    ax.plot(alphas, chord, color="#555555", lw=1.2, ls="--",
            label=r"certified chord $2\sqrt{2}+(2-\sqrt{2})\alpha$")
    # exact rational anchor bounds: cross-checks, not load-bearing
    xs = [p[0] for p in anchors]
    ys = [p[1] for p in anchors]
    ax.plot(xs, ys, color=C_CERT, lw=0, marker="v", markersize=4.5,
            label="exact anchor bounds $U_i$ (cross-checks)")
    # nonsignaling ceiling
    ax.axhline(4.0, color="#b3541e", lw=1.0, ls="--",
               label=r"$\omega_{\rm ns}(F_\alpha)=4$ (nonsignaling ceiling)")
    ax.set_xlim(0, 2)
    ax.set_ylim(2.7, 4.8)
    ax.set_xlabel(r"tilt $\alpha$")
    ax.set_ylabel("value")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # inset: derivative-0 vs linear departure near alpha=0
    ax_in = ax.inset_axes([0.16, 0.56, 0.34, 0.38])
    a2 = np.linspace(0, 0.26, 53)
    os1_in = os_values(a2, 1)
    ax_in.plot(a2, 2.0**1.5 + a2, color=C_EXACT, lw=1.2)
    ax_in.plot(a2, os1_in, color=C_NUM, lw=1.1, ls=(0, (3, 2)))
    ax_in.plot(a2, 2.0**1.5 + a2**2 / np.sqrt(2), color=C_THRESH, lw=1.0,
               ls=":")
    ax_in.text(0.5, 0.04,
               r"near $\alpha=0$: $\omega_1^{\rm os}\approx"
               r"2\sqrt{2}+\alpha^2/\sqrt{2}$ (numerical)",
               fontsize=6.5, transform=ax_in.transAxes, ha="center")
    ax_in.tick_params(labelsize=6.5)
    ax_in.grid(alpha=0.25)

    fig.savefig(FIGDIR / "fig2.pdf", bbox_inches="tight")
    plt.close(fig)

    (ARTDIR / "fig2_data.json").write_text(json.dumps({
        "alphas": alphas.tolist(),
        "omega_os1_numerical": os1.tolist(),
        "standard_exact_line": line.tolist(),
        "certified_chord": chord.tolist(),
        "anchors": [{"alpha": a, "bound": u} for a, u in anchors],
        "note": "solver-generated curves are numerical; certified statements "
                "are the endpoint (0, 2 sqrt(2)), the nonsignaling ceiling 4, "
                "the chord 2 sqrt(2) + (2 - sqrt(2)) alpha, and the anchor "
                "points (cross-checks)",
    }, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 3: filtration schematic
# ---------------------------------------------------------------------------


def _matrix(ax, origin, n, cell, filled, highlight=None, labels=None,
            fontsize=6.0):
    x0, y0 = origin
    for i in range(n):
        for j in range(n):
            face = "#dce9f7" if (i, j) in filled else "white"
            if highlight and (i, j) in highlight:
                face = "#f7b267"
            ax.add_patch(Rectangle((x0 + j * cell, y0 - i * cell), cell, cell,
                                   facecolor=face, edgecolor="#3a3a3a",
                                   lw=0.7, zorder=2))
    if labels:
        for i, lab in enumerate(labels):
            ax.text(x0 - 0.06, y0 - i * cell + cell / 2, lab, ha="right",
                    va="center", fontsize=fontsize)
            ax.text(x0 + i * cell + cell / 2, y0 + cell + 0.10, lab,
                    ha="center", va="bottom", fontsize=fontsize)


def figure3() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    std_labels = ["I", "$P_{A,0}$", "$P_{A,1}$", "$P_{B,0}$", "$P_{B,1}$"]
    _matrix(ax, (1.15, 3.6), 5, 0.62,
            filled={(i, j) for i in range(5) for j in range(5)},
            labels=std_labels)
    ax.text(2.9, 4.55, "standard level-1 body:\nall entries exist (total degree $\\leq 2$)",
            fontsize=7.5, ha="center")

    os_labels = ["I", "$P_{B,0}$", "$P_{B,1}$"]
    hl = {(1, 2), (2, 1)}
    _matrix(ax, (5.6, 3.6), 3, 0.62,
            filled={(i, j) for i in range(3) for j in range(3)},
            highlight=hl, labels=os_labels)
    ax.text(6.75, 4.55, "Alice-conditioned level-1 block $\\Gamma_{a|x}$\n(Bob words $\\leq 1$)",
            fontsize=7.5, ha="center")
    ax.annotate(
        "$\\varphi_{a|x}(P_{B,0}P_{B,1})$:\ntotal degree 3,\nABSENT from the left",
        xy=(5.6 + 2 * 0.62 + 0.31, 3.6 - 1 * 0.62 - 0.31),
        xytext=(7.5, 2.35), fontsize=7.5, color="#b3541e", ha="left",
        arrowprops=dict(arrowstyle="->", color="#b3541e", lw=0.9),
    )
    ax.text(6.75, 0.75, "Alice's $(x,a)$ is a block label;\nBob words carry the moments",
            fontsize=7, ha="center", color="#555555")
    ax.text(2.9, 0.75, "moment body over words\n$[I, P_{A,0}, P_{A,1}, P_{B,0}, P_{B,1}]$",
            fontsize=7, ha="center", color="#555555")

    fig.savefig(FIGDIR / "fig3.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGDIR.mkdir(exist_ok=True)
    ARTDIR.mkdir(parents=True, exist_ok=True)
    figure1()
    print("fig1 done", flush=True)
    figure2()
    print("fig2 done", flush=True)
    figure3()
    print("fig3 done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
