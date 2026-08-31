#!/usr/bin/env python3
"""Figures for Lecture 3 (root finding). Run from this directory; writes figures/*.png."""
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

def f(x):  return x - 2.0 + np.exp(-x)
def fp(x): return 1.0 - np.exp(-x)
def g(x):  return 2.0 - np.exp(-x)
ROOT = 1.8414056604369606

def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

# 1. f(x) crossing zero
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.linspace(0, 3, 400)
ax.axhline(0, color=MUTED, lw=1)
ax.plot(x, f(x), color=V[0], label=r"$f(x) = x - 2 + e^{-x}$")
ax.plot(ROOT, 0, "o", color=V[2], ms=9, zorder=5)
ax.annotate(r"$x_* \approx 1.8414$", (ROOT, 0), (2.05, -0.55), fontsize=15,
            arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$"); ax.legend(frameon=False, loc="upper left")
save(fig, "fx")

# 2. cobweb: converging and diverging fixed-point iteration
def cobweb(ax, gfun, x0, n, lo, hi, label, color):
    xs = np.linspace(lo, hi, 400)
    ax.plot(xs, gfun(xs), color=color, label=label)
    ax.plot(xs, xs, color=MUTED, lw=1, ls="--", label="$y = x$")
    x = x0
    px, py = [x0], [0.0]
    for _ in range(n):
        y = gfun(x)
        px += [x, y]; py += [y, y]
        x = y
        if not (lo <= x <= hi): break
    ax.plot(px, py, color=V[3], lw=1.6, marker="o", ms=4, mfc=PAPER, mec=V[3], alpha=0.95)
    ax.plot(x0, lo, "v", color=INK, ms=9, clip_on=False, zorder=6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("$x$")
    ax.legend(frameon=False, loc="upper left", fontsize=13)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.9))
cobweb(a1, g, 1.0, 8, 0.8, 2.2, r"$g(x) = 2 - e^{-x}$,  $|g'(x_*)| = 0.16$", V[0])
a1.set_title("converges", fontsize=15)
cobweb(a2, lambda x: -np.log(2.0 - x), 1.80, 6, 0.8, 1.99, r"$g(x) = -\ln(2-x)$,  $|g'(x_*)| = 6.3$", V[1])
a2.set_title("diverges", fontsize=15)
save(fig, "cobweb")

# 3. bisection on [1, 2]
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.linspace(0.9, 2.1, 400)
ax.axhline(0, color=MUTED, lw=1)
ax.plot(x, f(x), color=V[0])
a, b = 1.0, 2.0
for k in range(5):
    y = -0.18 - 0.12 * k
    ax.plot([a, b], [y, y], color=V[2], lw=3, solid_capstyle="butt")
    ax.text(2.06, y, f"step {k}", va="center", fontsize=12, color=MUTED)
    mid = 0.5 * (a + b)
    ax.plot(mid, 0, "|", color=V[1], ms=14, mew=2)
    if f(a) * f(mid) < 0: b = mid
    else: a = mid
ax.plot(ROOT, 0, "o", color=V[2], ms=8, zorder=5)
ax.set_xlim(0.9, 2.25); ax.set_ylim(-0.85, 0.35)
ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
ax.text(1.02, 0.22, r"$f(a) < 0$", color=V[0], fontsize=13)
ax.text(1.86, 0.22, r"$f(b) > 0$", color=V[0], fontsize=13)
save(fig, "bisection")

# 4. Newton: two tangent steps from x0 = 1
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.linspace(0.8, 2.3, 400)
ax.axhline(0, color=MUTED, lw=1)
ax.plot(x, f(x), color=V[0])
xn = 1.0
for k, c in zip(range(2), [V[1], V[2]]):
    fx, d = f(xn), fp(xn)
    xnext = xn - fx / d
    xs = np.array([xn - 0.15, xnext + 0.12])
    ax.plot(xs, fx + d * (xs - xn), color=c, lw=1.6, ls="--")
    ax.plot([xn, xn], [0, fx], color=c, lw=1, ls=":")
    ax.plot(xn, fx, "o", color=c, ms=7); ax.plot(xnext, 0, "o", color=c, ms=7, mfc=PAPER)
    ax.text(xn, 0.06, f"$x_{k}$", ha="center", fontsize=14, color=c)
    xn = xnext
ax.text(xn, 0.06, "$x_2$", ha="center", fontsize=14, color=V[2])
ax.plot(ROOT, 0, "o", color=V[0], ms=6, zorder=5)
ax.set_xlim(0.8, 2.3); ax.set_ylim(-0.75, 0.35); ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
save(fig, "newton")

# 5. Newton runaway on arctan from x0 = 1.5
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.linspace(-4, 4, 600)
ax.axhline(0, color=MUTED, lw=1)
ax.plot(x, np.arctan(x), color=V[0], label=r"$f(x) = \arctan x$")
xn = 1.5
for k, c in zip(range(3), [V[1], V[2], V[3]]):
    fx, d = np.arctan(xn), 1.0 / (1.0 + xn * xn)
    xnext = xn - fx / d
    xs = np.array(sorted([xn, xnext]))
    xs = np.array([xs[0] - 0.1, xs[1] + 0.1])
    ax.plot(xs, fx + d * (xs - xn), color=c, lw=1.5, ls="--")
    ax.plot(xn, fx, "o", color=c, ms=7)
    ax.text(xn, fx + (0.18 if fx > 0 else -0.28), f"$x_{k}$", ha="center", fontsize=14, color=c)
    xn = xnext
ax.set_xlim(-4, 4); ax.set_ylim(-1.9, 1.9); ax.set_xlabel("$x$")
ax.legend(frameon=False, loc="upper left")
save(fig, "newton_runaway")

# 6. secant step through (x0, f0), (x1, f1)
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.linspace(0.8, 2.3, 400)
ax.axhline(0, color=MUTED, lw=1)
ax.plot(x, f(x), color=V[0])
x0, x1 = 1.0, 2.0
f0, f1 = f(x0), f(x1)
x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
xs = np.array([0.85, 2.2])
ax.plot(xs, f0 + (f1 - f0) / (x1 - x0) * (xs - x0), color=V[1], lw=1.6, ls="--")
for xx, yy, lab in [(x0, f0, "$x_0$"), (x1, f1, "$x_1$")]:
    ax.plot(xx, yy, "o", color=V[1], ms=7); ax.plot([xx, xx], [0, yy], color=V[1], lw=1, ls=":")
    ax.text(xx, 0.06 if yy < 0 else -0.12, lab, ha="center", fontsize=14, color=V[1])
ax.plot(x2, 0, "o", color=V[2], ms=8, mfc=PAPER); ax.text(x2, 0.06, "$x_2$", ha="center", fontsize=14, color=V[2])
ax.plot(ROOT, 0, "o", color=V[0], ms=6, zorder=5)
ax.set_xlim(0.8, 2.3); ax.set_ylim(-0.75, 0.35); ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
save(fig, "secant")
print("figures written")
