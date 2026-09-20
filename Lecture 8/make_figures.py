#!/usr/bin/env python3
"""Figures for Lecture 8 (adaptive step size control).
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

# The example problem: y(t) = sin(e^{-t^2/a^2} omega t), a chirped pulse.
# Its ODE is dy/dt = omega e^{-t^2/a^2} (1 - 2 t^2/a^2) cos(e^{-t^2/a^2} omega t),
# a pure quadrature (f does not depend on y), so the exact answer is known.
A_ENV = 3.0
OMEGA = 20.0
T0, T1 = -20.0, 20.0

def y_exact(t):
    return np.sin(np.exp(-t * t / (A_ENV * A_ENV)) * OMEGA * t)

def f(t, y):
    env = np.exp(-t * t / (A_ENV * A_ENV))
    phase = env * OMEGA * t
    return OMEGA * env * (1.0 - 2.0 * t * t / (A_ENV * A_ENV)) * np.cos(phase)

def rk4_step(t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

# Step-doubling adaptive RK4, exactly the algorithm on the slides:
# one full step and two half steps, err from their difference,
# h' = S h err^{-1/5}, accept iff err < 1.
def adaptive_rk4(atol, rtol, h_init=0.1, safety=0.9):
    t, y, h = T0, y_exact(T0), h_init
    ts, ys, hs = [t], [y], []
    n_eval = 0
    n_reject = 0
    while t < T1:
        if t + h > T1:
            h = T1 - t
        y1 = rk4_step(t, y, h)                       # one full step
        y_half = rk4_step(t, y, 0.5 * h)             # two half steps
        y2 = rk4_step(t + 0.5 * h, y_half, 0.5 * h)
        n_eval += 11                                 # k1 is shared: 4 + 4 + 4 - 1
        eps = abs(y1 - y2)
        target = atol + rtol * abs(y1)
        err = eps / target
        if err < 1.0:                                # accept
            t += h
            y = y2
            ts.append(t); ys.append(y); hs.append(h)
        else:
            n_reject += 1
        h = safety * h * err ** (-0.2)
    return np.array(ts), np.array(ys), np.array(hs), n_eval, n_reject

# Fixed-step RK4 over the same range with n steps (4 evaluations per step).
def fixed_rk4(n):
    ts = np.linspace(T0, T1, n + 1)
    h = (T1 - T0) / n
    ys = [y_exact(T0)]
    y = ys[0]
    for t in ts[:-1]:
        y = rk4_step(t, y, h)
        ys.append(y)
    return ts, np.array(ys)

# 1. the problem: a chirped pulse
fig, ax = plt.subplots(figsize=(9.5, 3.2))
tt = np.linspace(T0, T1, 4000)
ax.plot(tt, y_exact(tt), color=V[0], lw=1.6)
ax.set_xlabel("$t$")
ax.set_ylabel("$y(t)$")
save(fig, "example_y")

# 2. the adaptive run
TOL = 1e-6
ts, ys, hs, n_eval, n_reject = adaptive_rk4(atol=TOL, rtol=TOL)
n_acc = len(hs)
err_adaptive = np.max(np.abs(ys - y_exact(ts)))
print(f"adaptive rk4, atol = rtol = {TOL:g}:")
print(f"  accepted steps = {n_acc}, rejected = {n_reject}, "
      f"f evaluations = {n_eval}")
print(f"  max error = {err_adaptive:.2e}")
print(f"  h range: {hs.min():.2e} .. {hs.max():.2f} "
      f"(ratio {hs.max()/hs.min():.0f})")

# fixed-step RK4 at the same cost in evaluations
n_fixed = n_eval // 4
tf, yf = fixed_rk4(n_fixed)
err_fixed = np.max(np.abs(yf - y_exact(tf)))
print(f"fixed rk4 at the same cost ({n_fixed} steps, {4*n_fixed} evals): "
      f"max error = {err_fixed:.2e}")

# how many fixed steps to match the adaptive error?
n = n_fixed
while True:
    tf2, yf2 = fixed_rk4(n)
    e = np.max(np.abs(yf2 - y_exact(tf2)))
    if e <= err_adaptive:
        break
    n = int(n * 1.25)
print(f"fixed rk4 needs ~{n} steps ({4*n} evals) to match the adaptive error "
      f"({e:.2e}); cost ratio {4*n/n_eval:.0f}x")

# 3. comparison figure: error along the way, at equal cost in evaluations
fig, ax = plt.subplots(figsize=(8.0, 3.4))
ax.semilogy(ts, np.abs(ys - y_exact(ts)) + 1e-18, color=V[2],
            lw=1.4, label=f"adaptive, {n_eval} evaluations")
ax.semilogy(tf, np.abs(yf - y_exact(tf)) + 1e-18, color=V[1],
            lw=1.4, label=f"fixed $h$, {4*n_fixed} evaluations")
ax.set_xlabel("$t$")
ax.set_ylabel("$|y - y_\\mathrm{exact}|$")
ax.set_ylim(1e-12, 1e-1)
ax.legend(frameon=False, fontsize=12, loc="upper right")
save(fig, "example_compare")

# 4. the step size history
fig, ax = plt.subplots(figsize=(8.0, 3.4))
ax.semilogy(ts[1:], hs, color=V[0], lw=1.4)
ax.set_xlabel("$t$")
ax.set_ylabel("accepted $h$")
save(fig, "example_h")
