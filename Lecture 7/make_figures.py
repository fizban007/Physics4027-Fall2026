#!/usr/bin/env python3
"""Figures for Lecture 7 (Runge-Kutta coefficients, stiffness, stability).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides."""
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

# The stiff system: du/dx = 998 u + 1998 v, dv/dx = -999 u - 1999 v
# u(0) = 1, v(0) = 0; exact u = 2 e^-x - e^-1000x, eigenvalues -1 and -1000.
A = np.array([[998.0, 1998.0], [-999.0, -1999.0]])
y0 = np.array([1.0, 0.0])

def rk4_sys(h, n):
    f = lambda y: A @ y
    ys = [y0]
    y = y0.copy()
    for _ in range(n):
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        ys.append(y.copy())
    return np.array(ys)

def u_exact(x):
    return 2.0 * np.exp(-x) - np.exp(-1000.0 * x)

# 1. RK4 at a stable and an unstable step size
X = 2.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
xx = np.linspace(0, X, 800)
# left: stable step over the whole range
n = int(round(X / 0.0025))
ys = rk4_sys(0.0025, n)
xs = np.arange(n + 1) * 0.0025
a1.plot(xx, u_exact(xx), color=V[0], label=r"exact $u(x)$")
a1.plot(xs[::8], ys[::8, 0], "o", ms=3, color=V[2], label=r"RK4, $h = 0.0025$")
a1.set_xlabel("$x$")
a1.legend(frameon=False, fontsize=12, loc="upper right")
# right: unstable step; everything happens in the first 0.05
X2 = 0.05
n2 = int(round(X2 / 0.0029))
ys2 = rk4_sys(0.0029, n2)
xs2 = np.arange(n2 + 1) * 0.0029
xz = np.linspace(0, X2, 400)
a2.plot(xz, u_exact(xz), color=V[0], label=r"exact $u(x)$")
a2.plot(xs2, ys2[:, 0], "o-", ms=4, lw=1.2, color=V[2],
        label=r"RK4, $h = 0.0029$")
a2.set_xlabel("$x$")
a2.set_ylim(-30, 6)
a2.legend(frameon=False, fontsize=12, loc="lower left")
save(fig, "stiff_rk4")
n1 = int(round(X / 0.0025)); n2 = int(round(X / 0.0029))
print(f"rk4 h=0.0025: final u err = {abs(rk4_sys(0.0025, n1)[-1,0] - u_exact(X)):.2e}")
print(f"rk4 h=0.0029: final u = {rk4_sys(0.0029, n2)[-1,0]:.3e} (exploding)")

# 2. the fast transient: exact u on a log-x zoom near 0
fig, ax = plt.subplots(figsize=(7.5, 3.0))
xz = np.linspace(0, 0.01, 500)
ax.plot(xz, u_exact(xz), color=V[0])
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x)$")
ax.annotate(r"the $e^{-1000x}$ transient", (0.002, 1.55), fontsize=14, color=MUTED)
save(fig, "transient")

# 3. implicit Euler with a large step on the same system
def implicit_euler_sys(h, n):
    M = np.linalg.inv(np.eye(2) - h * A)
    ys = [y0]
    y = y0.copy()
    for _ in range(n):
        y = M @ y
        ys.append(y.copy())
    return np.array(ys)

fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(xx, u_exact(xx), color=V[0], label=r"exact $u(x)$")
h = 0.1
ys = implicit_euler_sys(h, int(round(X / h)))
xs = np.arange(len(ys)) * h
ax.plot(xs, ys[:, 0], "s-", ms=5, lw=1.4, color=V[2],
        label=r"implicit Euler, $h = 0.1$")
ax.set_xlabel("$x$")
ax.legend(frameon=False, fontsize=13, loc="upper right")
save(fig, "implicit_euler")
print(f"implicit euler h=0.1 (40x the RK4 limit): final u err = "
      f"{abs(ys[-1,0] - u_exact(X)):.2e}")
