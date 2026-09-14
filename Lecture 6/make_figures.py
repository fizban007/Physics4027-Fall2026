#!/usr/bin/env python3
"""Figures for Lecture 6 (ODEs: Euler and Runge-Kutta). Run from this
directory; writes figures/*.png and prints the numbers quoted on the slides."""
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

# Solvers for y' = f(t, y), scalar
def euler(f, t0, y0, h, n):
    ts, ys = [t0], [y0]
    t, y = t0, y0
    for _ in range(n):
        y = y + h * f(t, y)
        t = t + h
        ts.append(t); ys.append(y)
    return np.array(ts), np.array(ys)

def rk4(f, t0, y0, h, n):
    ts, ys = [t0], [y0]
    t, y = t0, y0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t = t + h
        ts.append(t); ys.append(y)
    return np.array(ts), np.array(ys)

# 1. Euler's idea: tangent-line steps on y' = y, drifting off the true curve
f_exp = lambda t, y: y
fig, ax = plt.subplots(figsize=(7, 3.2))
tt = np.linspace(0, 2, 300)
ax.plot(tt, np.exp(tt), color=V[0], label=r"true solution $y = e^t$")
h = 0.5
ts, ys = euler(f_exp, 0.0, 1.0, h, 4)
ax.plot(ts, ys, "o-", color=V[2], label=r"Euler steps, $h = 0.5$")
ax.set_xlabel("$t$")
ax.legend(frameon=False, fontsize=13, loc="upper left")
save(fig, "euler_idea")
print(f"euler on y'=y, h=0.5: y(2) = {ys[-1]:.4f} vs e^2 = {np.exp(2):.4f}")

# 2. Euler on y' = cos t at h = 0.2: visible drift from sin t
f_cos = lambda t, y: np.cos(t)
fig, ax = plt.subplots(figsize=(7.5, 3.2))
tt = np.linspace(0, 10, 500)
ax.plot(tt, np.sin(tt), color=V[0], label=r"$\sin t$")
ts, ys = euler(f_cos, 0.0, 0.0, 0.2, 50)
ax.plot(ts, ys, "o-", color=V[2], ms=3.5, lw=1.4, label=r"Euler, $h = 0.2$")
ax.set_xlabel("$t$")
ax.legend(frameon=False, fontsize=13, loc="lower left")
save(fig, "euler_cos")
print(f"euler on y'=cos t, h=0.2: max |err| = {np.max(np.abs(ys - np.sin(ts))):.3f}")

# 3. RK4 vs Euler at the same large step h = 1
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(tt, np.sin(tt), color=V[0], label=r"$\sin t$")
ts_e, ys_e = euler(f_cos, 0.0, 0.0, 1.0, 10)
ts_r, ys_r = rk4(f_cos, 0.0, 0.0, 1.0, 10)
ax.plot(ts_e, ys_e, "o-", color=V[2], lw=1.4, label=r"Euler, $h = 1$")
ax.plot(ts_r, ys_r, "s-", color=V[1], lw=1.4, label=r"RK4, $h = 1$")
ax.set_xlabel("$t$")
ax.legend(frameon=False, fontsize=13, loc="lower left")
save(fig, "rk4_vs_euler")
print(f"h=1: euler max err = {np.max(np.abs(ys_e - np.sin(ts_e))):.3f}, "
      f"rk4 max err = {np.max(np.abs(ys_r - np.sin(ts_r))):.2e}")

# 4. convergence: global error at t = 10 vs h, slopes 1 and 4
hs = np.array([1.0 / 2**k for k in range(0, 14)])
err_e, err_r = [], []
for h in hs:
    n = int(round(10.0 / h))
    _, ye = euler(f_cos, 0.0, 0.0, h, n)
    _, yr = rk4(f_cos, 0.0, 0.0, h, n)
    err_e.append(abs(ye[-1] - np.sin(10.0)))
    err_r.append(abs(yr[-1] - np.sin(10.0)))
err_e, err_r = np.array(err_e), np.array(err_r)
print("euler slope:", np.polyfit(np.log(hs), np.log(err_e), 1)[0])
print("rk4 slope:", np.polyfit(np.log(hs[:7]), np.log(err_r[:7]), 1)[0])

fig, ax = plt.subplots(figsize=(7.5, 3.9))
ax.loglog(hs, err_e, "o-", color=V[2], label="Euler")
ax.loglog(hs, err_r, "s-", color=V[1], label="RK4")
ax.loglog(hs, 0.8 * err_e[0] * (hs / hs[0]) ** 1.0, ls="--", lw=1.2,
          color=MUTED, label=r"$h^1$, $h^4$")
ax.loglog(hs, 0.8 * err_r[0] * (hs / hs[0]) ** 4.0, ls="--", lw=1.2, color=MUTED)
ax.set_xlabel("step size $h$")
ax.set_ylabel(r"error at $t = 10$")
ax.legend(frameon=False, fontsize=13, loc="lower right")
save(fig, "convergence")

# 5. RK4 stages on one step: the four sampled slopes
fig, ax = plt.subplots(figsize=(7, 3.4))
f_demo = lambda t, y: y          # y' = y again, one step from t=0
t0, y0, h = 0.0, 1.0, 1.0
tt = np.linspace(-0.05, 1.15, 200)
ax.plot(tt, np.exp(tt), color=V[0], lw=1.8, label=r"true solution")
k1 = f_demo(t0, y0)
k2 = f_demo(t0 + h / 2, y0 + h / 2 * k1)
k3 = f_demo(t0 + h / 2, y0 + h / 2 * k2)
k4 = f_demo(t0 + h, y0 + h * k3)
stages = [(t0, y0, k1, "$k_1$"), (t0 + h/2, y0 + h/2*k1, k2, "$k_2$"),
          (t0 + h/2, y0 + h/2*k2, k3, "$k_3$"), (t0 + h, y0 + h*k3, k4, "$k_4$")]
for i, (ts_, ys_, k, lab) in enumerate(stages):
    d = 0.22
    color = plt.cm.viridis(0.15 + 0.22 * i)
    ax.plot([ts_ - d, ts_ + d], [ys_ - d * k, ys_ + d * k], color=color, lw=2.4)
    ax.plot(ts_, ys_, "o", color=color, ms=6)
    ax.annotate(lab, (ts_, ys_), (ts_ - 0.16, ys_ + 0.16), fontsize=14, color=color)
y1 = y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
ax.plot(t0 + h, y1, "*", color=V[0], ms=16, zorder=6)
ax.annotate(r"$y_{n+1}$", (t0 + h, y1), (t0 + h - 0.32, y1 + 0.05), fontsize=14)
ax.set_xlabel("$t$"); ax.set_ylim(0.7, 3.1)
ax.legend(frameon=False, fontsize=13, loc="upper left")
save(fig, "rk4_stages")
print(f"one rk4 step on y'=y, h=1: y1 = {y1:.5f} vs e = {np.e:.5f}")
