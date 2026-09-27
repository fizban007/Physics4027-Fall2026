#!/usr/bin/env python3
"""Figures for Lecture 10 (multistep methods, energy conservation).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides.

Part I uses the pendulum theta'' = -sin(theta) at theta_0 = 170 degrees, as
in Lecture 9; after exactly one period the exact solution is back at its
initial state, which is what the errors are measured against. Part II uses
the pendulum at theta_0 = 90 degrees for long runs."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.special import ellipk

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

def period(theta0):
    return 4 * ellipk(np.sin(0.5 * theta0) ** 2)

# ============================================================ Part I
# ------------------------------------------- 1. where Adams-Bashforth comes from
g = lambda t: 1.2 + 0.5 * np.sin(1.3 * t) + 0.15 * t     # a stand-in for f(t, y(t))
tn = np.array([0.0, 1.0, 2.0, 3.0])                       # t_{n-3} .. t_n
fig, ax = plt.subplots(figsize=(6.4, 3.0))
tt = np.linspace(-0.3, 4.3, 400)
ax.plot(tt, g(tt), color=MUTED, lw=1.4, ls="--", label=r"true $f(t, y(t))$")
coef = np.polyfit(tn, g(tn), 3)
tp = np.linspace(0.0, 4.0, 300)
ax.plot(tp, np.polyval(coef, tp), color=V[1], lw=2.0, label="cubic through the stored slopes")
ts = np.linspace(3.0, 4.0, 100)
ax.fill_between(ts, 0, np.polyval(coef, ts), color=V[2], alpha=0.35, lw=0)
ax.plot(tn, g(tn), "o", ms=8, color=V[0], zorder=5)
for i, lab in enumerate([r"$f_{n-3}$", r"$f_{n-2}$", r"$f_{n-1}$", r"$f_n$"]):
    ax.annotate(lab, (tn[i], g(tn[i])), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=14)
ax.text(3.5, 0.35, r"$\displaystyle\int_{t_n}^{t_{n+1}}$", ha="center", fontsize=15)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels([r"$t_{n-3}$", r"$t_{n-2}$", r"$t_{n-1}$", r"$t_n$", r"$t_{n+1}$"])
ax.set_yticks([])
ax.set_ylim(0, 3.2)
ax.legend(frameon=False, fontsize=12, loc="upper left")
save(fig, "adams")

# ------------------------------------------- 2. cost: RK4 vs AB4 vs ABM4
TH0 = np.radians(170.0)
T170 = period(TH0)
Y0 = np.array([TH0, 0.0])
f = lambda y: np.array([y[1], -np.sin(y[0])])

def rk4_step(y, h):
    k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def run_rk4(n):
    h = T170 / n; y = Y0.copy()
    for _ in range(n):
        y = rk4_step(y, h)
    return y, 4 * n

def run_adams(n, corrector):
    """AB4 (corrector=False) or AB4-AM4 predictor-corrector, PECE.
    Start-up: three RK4 steps. Returns the state after one period and the
    number of evaluations of f."""
    h = T170 / n
    y = Y0.copy(); fs = [f(y)]; n_eval = 1
    for _ in range(3):
        y = rk4_step(y, h); n_eval += 4
        fs.append(f(y)); n_eval += 1
    for _ in range(3, n):
        y_pred = y + h / 24 * (55 * fs[-1] - 59 * fs[-2] + 37 * fs[-3] - 9 * fs[-4])
        if corrector:
            f_pred = f(y_pred); n_eval += 1
            y = y + h / 24 * (9 * f_pred + 19 * fs[-1] - 5 * fs[-2] + fs[-3])
        else:
            y = y_pred
        fs.append(f(y)); n_eval += 1
    return y, n_eval

ns = [50, 100, 200, 400, 800, 1600]
methods = [("RK4", run_rk4, V[0]),
           ("Adams-Bashforth 4", lambda n: run_adams(n, False), V[2]),
           ("predictor-corrector (AB4 + AM4)", lambda n: run_adams(n, True), V[1])]
fig, ax = plt.subplots(figsize=(5.6, 3.9))
print("one period at 170 degrees, error vs evaluations of f:")
for name, run, col in methods:
    evals, errs = [], []
    for n in ns:
        y, ne = run(n)
        evals.append(ne); errs.append(np.linalg.norm(y - Y0))
    slope = np.polyfit(np.log(evals[2:]), np.log(errs[2:]), 1)[0]
    print(f"  {name:32s} " + "  ".join(f"{e}:{r:.1e}" for e, r in zip(evals, errs))
          + f"   slope {slope:.2f}")
    ax.loglog(evals, errs, "o-", ms=5, color=col, label=name)
ax.set_xlabel(r"evaluations of $f$")
ax.set_ylabel("error after one period")
ax.legend(frameon=False, fontsize=12, loc="lower left")
save(fig, "multistep_cost")

# ------------------------------------------- 3. Adams-Bashforth stability intervals
print("Adams-Bashforth: stable for real h*lambda down to")
for name, c in (("AB1 (Euler)", [1.0]), ("AB2", [3 / 2, -1 / 2]),
                ("AB3", [23 / 12, -16 / 12, 5 / 12]),
                ("AB4", [55 / 24, -59 / 24, 37 / 24, -9 / 24])):
    k = len(c)
    def max_root(z):
        p = np.zeros(k + 1); p[0] = 1.0; p[1] = -1.0 - z * c[0]
        for j in range(1, k):
            p[j + 1] = -z * c[j]
        return max(abs(np.roots(p)))
    zs = np.linspace(-2.5, 0, 25001)
    print(f"  {name:12s} {min(z for z in zs if max_root(z) <= 1 + 1e-9):.3f}")

# ============================================================ Part II
acc = lambda x: -np.sin(x)
energy = lambda x, v: 0.5 * v * v - np.cos(x)

def rk2(x, v, h):
    xm = x + h / 2 * v; vm = v + h / 2 * acc(x)
    return x + h * vm, v + h * acc(xm)

def rk4(x, v, h):
    k1x, k1v = v, acc(x)
    k2x, k2v = v + h / 2 * k1v, acc(x + h / 2 * k1x)
    k3x, k3v = v + h / 2 * k2v, acc(x + h / 2 * k2x)
    k4x, k4v = v + h * k3v, acc(x + h * k3x)
    return (x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x),
            v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))

def leapfrog(x, v, h):
    v = v + h / 2 * acc(x)
    x = x + h * v
    v = v + h / 2 * acc(x)
    return x, v

def euler(x, v, h):
    return x + h * v, v + h * acc(x)

# ------------------------------------------- 4. energy over 1000 periods
TH90 = np.radians(90.0)
T90 = period(TH90)
N_PER = 50                      # RK steps per period
H = T90 / N_PER
E0 = energy(TH90, 0.0)
PERIODS = 1000
runs = [("RK2 (midpoint)", rk2, H, V[3]),
        ("RK4", rk4, H, V[0]),
        (r"leapfrog, same $h$", leapfrog, H, V[1]),
        (r"leapfrog, $h/4$ (same cost as RK4)", leapfrog, H / 4, V[2])]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
print(f"90 degrees, h = T/{N_PER} = {H:.4f}, largest energy error during period 1, 10, 100, 1000:")
for name, step, h, col in runs:
    per_period = int(round(T90 / h))
    x, v = TH90, 0.0
    # largest |E - E0| within each period: the envelope of the error
    env = np.zeros(PERIODS)
    for p_i in range(PERIODS):
        worst = 0.0
        for _ in range(per_period):
            x, v = step(x, v, h)
            worst = max(worst, abs(energy(x, v) - E0))
        env[p_i] = worst
    periods = np.arange(1, PERIODS + 1)
    print(f"  {name:36s} " + "  ".join(f"{env[p - 1]:.1e}" for p in (1, 10, 100, 1000)))
    ax.loglog(periods, env, color=col, lw=2.0, label=name)
ax.set_ylim(1e-7, 1e2)
ax.set_xlim(1, PERIODS)
ax.set_xlabel("time (periods)")
ax.set_ylabel(r"largest $|E - E_0|$ in the period")
ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=1)
save(fig, "energy_long")

# ------------------------------------------- 5. time reversal
print("run 1000 steps at h = 0.1, flip v, run 1000 back; distance from start:")
for name, step in (("RK4", rk4), ("leapfrog", leapfrog)):
    x, v = TH90, 0.3
    for _ in range(1000):
        x, v = step(x, v, 0.1)
    v = -v
    for _ in range(1000):
        x, v = step(x, v, 0.1)
    print(f"  {name:9s} {np.hypot(x - TH90, -v - 0.3):.1e}")

# ------------------------------------------- 6. area factor of one step, harmonic oscillator
print("determinant of one step on x'' = -x (area factor):")
def step_matrix(step, h):
    global acc
    saved = acc; acc = lambda x: -x
    cols = [step(1.0, 0.0, h), step(0.0, 1.0, h)]
    acc = saved
    return np.array(cols).T
for h in (0.1, 0.5):
    print(f"  h = {h}: " + "  ".join(
        f"{name} {np.linalg.det(step_matrix(s, h)):.10f}"
        for name, s in (("Euler", euler), ("RK2", rk2), ("RK4", rk4), ("leapfrog", leapfrog))))
print(f"  formulas at h = 0.1: 1 + h^2 = {1 + 0.01}, 1 + h^4/4 = {1 + 1e-4 / 4}, "
      f"1 - h^6/72 = {1 - 1e-6 / 72:.10f}")

# shadow energy of leapfrog on the harmonic oscillator
acc_saved = acc; acc = lambda x: -x
h = 0.5; x, v = 1.0, 0.0; e_true, e_shadow = [], []
for _ in range(1000):
    x, v = leapfrog(x, v, h)
    e_true.append(0.5 * v * v + 0.5 * x * x)
    e_shadow.append(0.5 * v * v + 0.5 * (1 - h * h / 4) * x * x)
acc = acc_saved
print(f"leapfrog on x'' = -x, h = 0.5, 1000 steps: E varies by {np.ptp(e_true):.2e}, "
      f"E - h^2 x^2/8 varies by {np.ptp(e_shadow):.1e}")

# ------------------------------------------- 7. phase-space area
def blob(n=400):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return 1.2 + 0.35 * np.cos(a), 0.35 * np.sin(a)

def area(x, v):
    return 0.5 * abs(np.dot(x, np.roll(v, -1)) - np.dot(v, np.roll(x, -1)))

HB, NB = 0.1, 60                # 6 time units, a bit under one period
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), sharex=True, sharey=True)
x0, v0 = blob()
A0 = area(x0, v0)
print(f"phase-space blob, {NB} steps of h = {HB}: area / initial area")
for ax, (name, step, h, n) in zip(axes, [("exact (RK4, tiny steps)", rk4, HB / 20, NB * 20),
                                         ("Euler", euler, HB, NB),
                                         ("leapfrog", leapfrog, HB, NB)]):
    x, v = x0.copy(), v0.copy()
    for _ in range(n):
        x, v = step(x, v, h)
    ratio = area(x, v) / A0
    print(f"  {name:24s} {ratio:.4f}")
    ax.add_patch(Polygon(np.c_[x0, v0], closed=True, fc=V[3], ec=MUTED, alpha=0.5, lw=0.8))
    ax.add_patch(Polygon(np.c_[x, v], closed=True, fc=V[1], ec=V[0], alpha=0.55, lw=1.0))
    ax.set_title(f"{name}: area $\\times$ {ratio:.2f}", fontsize=14)
    ax.set_xlabel(r"$\theta$")
th = np.linspace(-np.pi, np.pi, 300)
for ax in axes:
    for e in (-0.5, 0.0, 0.5):
        vv = np.sqrt(np.clip(2 * (e + np.cos(th)), 0, None))
        ax.plot(th, vv, color=MUTED, lw=0.5); ax.plot(th, -vv, color=MUTED, lw=0.5)
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.0, 2.0); ax.set_aspect("equal")
axes[0].set_ylabel(r"$\dot\theta$")
save(fig, "phase_area")

# ------------------------------------------- 8. the leapfrog staggering
fig, ax = plt.subplots(figsize=(7.0, 1.9))
for k in range(4):
    ax.plot(k, 1, "o", ms=12, color=V[1])
    ax.text(k, 1.28, [r"$x_0$", r"$x_1$", r"$x_2$", r"$x_3$"][k], ha="center", fontsize=15)
for k in range(3):
    ax.plot(k + 0.5, 0, "s", ms=11, color=V[2])
    ax.text(k + 0.5, -0.45, [r"$v_{1/2}$", r"$v_{3/2}$", r"$v_{5/2}$"][k], ha="center", fontsize=15)
    ax.annotate("", xy=(k + 1, 1), xytext=(k + 0.5, 0.05),
                arrowprops=dict(arrowstyle="->", color=V[1], lw=1.3))
    if k == 0:
        ax.annotate("", xy=(0.5, 0.05), xytext=(0, 0.95),
                    arrowprops=dict(arrowstyle="->", color=V[2], lw=1.3))
    if k < 2:
        ax.annotate("", xy=(k + 1.5, 0), xytext=(k + 1, 0.95),
                    arrowprops=dict(arrowstyle="->", color=V[2], lw=1.3))
ax.text(3.35, 1.0, "positions at whole steps", va="center", fontsize=15, color=V[1])
ax.text(2.85, 0.0, "velocities at half steps", va="center", fontsize=15, color=V[2])
ax.set_xlim(-0.3, 5.9); ax.set_ylim(-0.8, 1.6); ax.axis("off")
save(fig, "leapfrog_grid")
