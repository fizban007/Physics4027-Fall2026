#!/usr/bin/env python3
"""Figures for Lecture 5 (numerical integration). Run from this directory;
writes figures/*.png and prints the numbers quoted on the slides."""
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

# Demo function for the rule pictures
def f(x):
    return 1.0 / (1.0 + x * x) + 0.1 * x

A, B, NPANEL = 0.0, 2.0, 4
xf = np.linspace(A, B, 400)

# 1. left-endpoint rectangles (the naive Riemann rule)
fig, ax = plt.subplots(figsize=(7.5, 2.6))
nodes = np.linspace(A, B, NPANEL + 1)
for i in range(NPANEL):
    ax.fill_between([nodes[i], nodes[i+1]], 0, f(nodes[i]),
                    color=V[2], alpha=0.35, edgecolor=V[2], lw=1.5)
ax.plot(xf, f(xf), color=V[0])
ax.plot(nodes[:-1], f(nodes[:-1]), "o", color=V[0], ms=6, zorder=5)
ax.set_xlabel("$x$"); ax.set_ylim(0, 1.25)
save(fig, "rectangles")

# 2. trapezoids: the linear interpolant, integrated
fig, ax = plt.subplots(figsize=(7.5, 2.6))
for i in range(NPANEL):
    xs = [nodes[i], nodes[i+1]]
    ax.fill_between(xs, 0, f(np.array(xs)), color=V[2], alpha=0.35,
                    edgecolor=V[2], lw=1.5)
ax.plot(xf, f(xf), color=V[0])
ax.plot(nodes, f(nodes), "o", color=V[0], ms=6, zorder=5)
ax.set_xlabel("$x$"); ax.set_ylim(0, 1.25)
save(fig, "trapezoids")

# 3. Simpson: parabola through each pair of intervals
fig, ax = plt.subplots(figsize=(7.5, 2.6))
for i in range(0, NPANEL, 2):
    x0, x1, x2 = nodes[i], nodes[i+1], nodes[i+2]
    coef = np.polyfit([x0, x1, x2], f(np.array([x0, x1, x2])), 2)
    xs = np.linspace(x0, x2, 100)
    ax.fill_between(xs, 0, np.polyval(coef, xs), color=V[2], alpha=0.35)
    ax.plot(xs, np.polyval(coef, xs), color=V[2], lw=1.8)
ax.plot(xf, f(xf), color=V[0])
ax.plot(nodes, f(nodes), "o", color=V[0], ms=6, zorder=5)
ax.set_xlabel("$x$"); ax.set_ylim(0, 1.25)
save(fig, "simpson")

# Test integral for the convergence plots (same one as the 2025 slides):
# I = int_0^2 x^4 asinh(x) dx, smooth on [0, 2]
def g(x):
    return x**4 * np.arcsinh(x)

def trap(n):
    x = np.linspace(0.0, 2.0, n + 1)
    y = g(x)
    h = 2.0 / n
    return h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])

def simpson(n):                     # n must be even
    x = np.linspace(0.0, 2.0, n + 1)
    y = g(x)
    h = 2.0 / n
    return h / 3.0 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-2:2].sum() + y[-1])

I_REF = simpson(2**20)              # error ~ h^4 ~ 1e-24: converged in double
print(f"reference I = {I_REF:.15f}")

# 4. trapezoid vs Simpson convergence (log-log, slopes 2 and 4)
ns = np.array([2**k for k in range(1, 14)])
err_t = np.array([abs(trap(n) - I_REF) for n in ns])
err_s = np.array([abs(simpson(n) - I_REF) for n in ns])
print("trap slope:", np.polyfit(np.log(ns[:8]), np.log(err_t[:8]), 1)[0])
print("simpson slope:", np.polyfit(np.log(ns[:6]), np.log(err_s[:6]), 1)[0])

fig, ax = plt.subplots(figsize=(7.5, 3.9))
ax.loglog(ns, err_t, "o-", color=V[0], label="trapezoidal")
ax.loglog(ns, err_s, "s-", color=V[2], label="Simpson")
ax.loglog(ns, 0.8 * err_t[0] * (ns / ns[0]) ** -2.0, ls="--", lw=1.2,
          color=MUTED, label=r"$N^{-2}$, $N^{-4}$")
ax.loglog(ns, 0.8 * err_s[0] * (ns / ns[0]) ** -4.0, ls="--", lw=1.2, color=MUTED)
ax.set_xlabel("number of intervals $N$"); ax.set_ylabel("error")
ax.legend(frameon=False, fontsize=13)
save(fig, "convergence")

# 5. Legendre polynomials P1..P5
from numpy.polynomial import legendre as L
fig, ax = plt.subplots(figsize=(7.5, 3.9))
x = np.linspace(-1, 1, 400)
for n in range(1, 6):
    c = np.zeros(n + 1); c[n] = 1
    color = plt.cm.viridis(0.15 + 0.17 * (n - 1))
    ax.plot(x, L.legval(x, c), color=color, label=f"$P_{n}$")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("$x$"); ax.set_ylim(-1.1, 1.1)
ax.legend(frameon=False, fontsize=13, ncol=5, loc="lower right")
save(fig, "legendre")

# 6. Gauss-Legendre vs Simpson at equal evaluation counts
def gauss(n):
    xi, wi = L.leggauss(n)
    # map [-1, 1] -> [0, 2]
    return (2.0 - 0.0) / 2.0 * np.sum(wi * g((2.0 - 0.0) / 2.0 * xi + 1.0))

ns_g = np.arange(2, 13)
err_g = np.array([abs(gauss(n) - I_REF) for n in ns_g])
eps = 2.0**-52 * abs(I_REF)
err_g = np.maximum(err_g, 1e-18)    # keep zeros plottable on the log axis
for n, e in zip(ns_g, err_g):
    if n in (4, 6, 8, 10): print(f"gauss n={n}: err = {e:.2e}")
evals_s = np.array([3, 5, 9, 17, 33, 65, 129])          # Simpson: N+1 evaluations
err_s2 = np.array([abs(simpson(n - 1) - I_REF) for n in evals_s])

fig, ax = plt.subplots(figsize=(7.5, 3.9))
ax.semilogy(evals_s, err_s2, "s-", color=V[2], label="Simpson")
ax.semilogy(ns_g, err_g, "o-", color=V[0], label="Gauss--Legendre")
ax.axhline(eps, color=MUTED, lw=1.0, ls=":")
ax.text(80, eps * 2.2, r"machine precision", fontsize=12, color=MUTED)
ax.set_xlabel("number of function evaluations")
ax.set_ylabel("error")
ax.set_xscale("log")
ax.legend(frameon=False, fontsize=13)
save(fig, "gauss_convergence")
print(f"eps*I = {eps:.1e}")
