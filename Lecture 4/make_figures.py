#!/usr/bin/env python3
"""Figures for Lecture 4 (interpolation). Run from this directory; writes figures/*.png.
The root-finding figures (newton, newton_runaway, secant) are copied from Lecture 3."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "#FAFAF7"
INK = "#182430"
MUTED = "#55636F"
V = ["#440154", "#365C8D", "#1FA187", "#A0DA39"]   # viridis picks, dark to light

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.2,
    # Real LaTeX for all text, so figures match the MathJax on the slides.
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

# 1. interpolation: known values at nodes, a question mark in between
def func(x):
    return np.sin(x) + 0.35 * x

nodes = np.linspace(0.5, 5.5, 6)
x = np.linspace(0.5, 5.5, 400)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(x, func(x), color=MUTED, lw=1.4, ls="--", label="the function")
ax.plot(nodes, func(nodes), color=V[2], lw=2.0, label="linear interpolation")
ax.plot(nodes, func(nodes), "o", color=V[0], ms=8, zorder=5, label="known values")
xq = 0.5 * (nodes[2] + nodes[3])
yq = np.interp(xq, nodes, func(nodes))
ax.plot(xq, yq, "o", ms=9, mfc=PAPER, mec=V[1], mew=2, zorder=6)
ax.annotate(r"$f(x)$ here?", (xq + 0.06, yq + 0.03), (3.7, 1.42), fontsize=15,
            arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
ax.legend(frameon=False, fontsize=13, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.0))
save(fig, "interp")

# 2. Runge's phenomenon: degree-10 polynomial through 11 equally spaced nodes
def runge(x):
    return 1.0 / (1.0 + 25.0 * x * x)

nodes = np.linspace(-1, 1, 11)
x = np.linspace(-1, 1, 800)
# Lagrange interpolant evaluated directly from the formula
def lagrange(xq, xi, yi):
    total = np.zeros_like(xq)
    for i in range(len(xi)):
        term = np.full_like(xq, yi[i])
        for j in range(len(xi)):
            if j != i:
                term *= (xq - xi[j]) / (xi[i] - xi[j])
        total += term
    return total

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(x, runge(x), color=V[0], label=r"$f(x) = 1/(1 + 25x^2)$")
ax.plot(x, lagrange(x, nodes, runge(nodes)), color=V[2],
        label="degree-10 interpolation")
ax.plot(nodes, runge(nodes), "o", color=V[0], ms=7, zorder=5)
ax.set_xlabel("$x$")
ax.legend(frameon=False, loc="upper center", fontsize=13)
save(fig, "runge")
