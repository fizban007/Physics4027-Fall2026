#!/usr/bin/env python3
"""Figures for Lecture 9 (dense output, stop conditions, event location).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides.

Running example: the pendulum theta'' = -sin(theta) released from rest at
theta_0 = 170 degrees, which has an exact solution in Jacobi elliptic
functions: sin(theta/2) = k sn(K - t | k^2), k = sin(theta_0/2). The first
zero of theta is at t = K(k^2), a quarter period."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipj

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

# ---------------------------------------------------------------- the problem
THETA0 = np.radians(170.0)
K_MOD = np.sin(0.5 * THETA0)
M_PAR = K_MOD * K_MOD
QUARTER = ellipk(M_PAR)            # exact time of the first zero crossing

def theta_exact(t):
    sn, cn, dn, ph = ellipj(QUARTER - t, M_PAR)
    return 2.0 * np.arcsin(K_MOD * sn)

def f(t, y):
    return np.array([y[1], -np.sin(y[0])])

Y0 = np.array([THETA0, 0.0])

# ---------------------------------------------------------- Dormand-Prince 5(4)
C = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
A = [[],
     [1/5],
     [3/40, 9/40],
     [44/45, -56/15, 32/9],
     [19372/6561, -25360/2187, 64448/6561, -212/729],
     [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
     [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84]]
B = [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0]
BSTAR = [5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
# dense-output coefficients (Hairer, Norsett & Wanner; Numerical Recipes 17.2)
D = [-12715105075/11282082432, 0.0, 87487479700/32700410799,
     -10690763975/1880347072, 701980252875/199316789632,
     -1453857185/822651844, 69997945/29380423]

def dp45_step(t, y, h):
    k = []
    for i in range(7):
        yi = y + h * sum(A[i][j] * k[j] for j in range(i))
        k.append(f(t + C[i] * h, yi))
    y_new = y + h * sum(B[i] * k[i] for i in range(7))
    y_low = y + h * sum(BSTAR[i] * k[i] for i in range(7))
    return y_new, y_low, k

def interp_linear(y0, y1, k, h, s):
    return (1 - s) * y0 + s * y1

def interp_hermite(y0, y1, k, h, s):
    # cubic through both endpoints with both slopes: k_1 = f_n, k_7 = f_{n+1}
    return ((1 - s) * y0 + s * y1
            + s * (s - 1) * ((1 - 2 * s) * (y1 - y0) + (s - 1) * h * k[0] + s * h * k[6]))

def interp_dp(y0, y1, k, h, s):
    r1 = y0
    r2 = y1 - y0
    r3 = h * k[0] - r2
    r4 = r2 - h * k[6] - r3
    r5 = h * sum(D[i] * k[i] for i in range(7))
    return r1 + s * (r2 + (1 - s) * (r3 + s * (r4 + (1 - s) * r5)))

def adaptive_dp45(t0, t_end, y0, atol, rtol, h=0.1, safety=0.9, stop=None):
    """Returns the list of accepted steps as (t, h, y_n, y_{n+1}, k)."""
    t, y = t0, y0.copy()
    steps = []
    while t < t_end:
        h = min(h, t_end - t)
        y_new, y_low, k = dp45_step(t, y, h)
        target = atol + rtol * np.abs(y_new)
        err = np.sqrt(np.mean(((y_new - y_low) / target) ** 2))
        if err < 1.0:
            steps.append((t, h, y, y_new, k))
            t, y = t + h, y_new
            if stop is not None and stop(t, y):
                break
        h = safety * h * min(5.0, err ** (-0.2)) if err > 0 else 5 * h
    return steps

# ----------------------------------------------- 1. where the steps land
TOL = 1e-6
PERIOD = 4 * QUARTER
steps = adaptive_dp45(0.0, PERIOD, Y0, TOL, TOL)
ts = np.array([s[0] for s in steps] + [steps[-1][0] + steps[-1][1]])
hs = np.array([s[1] for s in steps])
print(f"theta0 = 170 deg, quarter period K = {QUARTER:.12f}, period = {PERIOD:.6f}")
print(f"one period at tol {TOL:g}: {len(steps)} accepted steps, "
      f"h from {hs.min():.3f} to {hs.max():.3f}")

fig, ax = plt.subplots(figsize=(9.0, 3.2))
tt = np.linspace(0, PERIOD, 2000)
ax.plot(tt, np.degrees(theta_exact(tt)), color=V[0], lw=1.6)
ax.plot(ts, np.degrees([Y0[0]] + [s[3][0] for s in steps]), "o",
        ms=5, color=V[2], label="accepted steps")
t_out = np.arange(0, PERIOD, 0.5)
ax.plot(t_out, np.full_like(t_out, -200.0), "|", ms=12, mew=1.6, color=V[1],
        label="where you want output")
ax.set_ylim(-215, 190)
ax.set_xlabel("$t$")
ax.set_ylabel(r"$\theta$ (degrees)")
ax.legend(frameon=False, fontsize=12, loc="upper center", ncol=2)
save(fig, "steps")

# ----------------------------------------------- 2. interpolants on one step
T_START = 1.2
def one_step(h):
    y0 = np.array([theta_exact(T_START), 0.0])
    # angular velocity from energy: omega^2 = 2(cos theta - cos theta0), omega < 0
    y0[1] = -np.sqrt(2.0 * (np.cos(y0[0]) - np.cos(THETA0)))
    y1, _, k = dp45_step(T_START, y0, h)
    return y0, y1, k

H_BIG = 1.6
y0s, y1s, ks = one_step(H_BIG)
ss = np.linspace(0, 1, 400)
tloc = T_START + ss * H_BIG
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
a1.plot(tloc, np.degrees(theta_exact(tloc)), color=V[0], lw=3.0, label="exact")
a1.plot(tloc, np.degrees([interp_linear(y0s, y1s, ks, H_BIG, s)[0] for s in ss]),
        "--", color=V[1], lw=1.6, label="linear")
a1.plot(tloc, np.degrees([interp_hermite(y0s, y1s, ks, H_BIG, s)[0] for s in ss]),
        color=V[2], lw=1.6, label="cubic Hermite")
a1.plot([T_START, T_START + H_BIG], np.degrees([y0s[0], y1s[0]]), "o",
        ms=7, color=INK)
a1.set_xlabel("$t$")
a1.set_ylabel(r"$\theta$ (degrees)")
a1.legend(frameon=False, fontsize=12, loc="lower left")

h_list = np.logspace(-2, np.log10(1.6), 14)
names = [("linear", interp_linear, V[1]),
         ("cubic Hermite", interp_hermite, V[2]),
         ("Dormand-Prince native", interp_dp, V[0])]
s_in = np.linspace(0.02, 0.98, 49)
print("interpolation error over one step (max over interior points):")
for name, fn, col in names:
    errs = []
    for h in h_list:
        y0, y1, k = one_step(h)
        e = max(abs(fn(y0, y1, k, h, s)[0] - theta_exact(T_START + s * h)) for s in s_in)
        errs.append(e)
    errs = np.array(errs)
    slope = np.polyfit(np.log(h_list[:5]), np.log(errs[:5]), 1)[0]
    print(f"  {name:22s} slope {slope:.2f}; err at h=0.1: "
          f"{np.interp(np.log(0.1), np.log(h_list), errs):.1e}")
    a2.loglog(h_list, errs, "o-", ms=4, lw=1.4, color=col, label=name)
a2.set_xlabel("step size $h$")
a2.set_ylabel("max interpolation error")
a2.legend(frameon=False, fontsize=11, loc="lower right")
save(fig, "interp")

# ----------------------------------------------- 3. stopping at theta = 0
def stop_theta(t, y):
    return y[0] <= 0.0

def locate(step, g):
    """bisection on the dense-output interpolant; no new f evaluations"""
    t, h, y0, y1, k = step
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(interp_dp(y0, y1, k, h, mid)) > 0:
            lo = mid
        else:
            hi = mid
    return t + 0.5 * (lo + hi) * h

print("quarter-period (first theta = 0) error:")
for tol in (1e-6, 1e-9):
    st = adaptive_dp45(0.0, 10.0, Y0, tol, tol, stop=stop_theta)
    last = st[-1]
    t_end = last[0] + last[1]
    th0, th1 = last[2][0], last[3][0]
    t_secant = last[0] + last[1] * th0 / (th0 - th1)
    t_event = locate(last, lambda y: y[0])
    print(f"  tol {tol:g}: {len(st)} steps, last h = {last[1]:.3f}")
    print(f"    stop at step end   {t_end:.12f}  err {abs(t_end - QUARTER):.1e}")
    print(f"    linear between ends {t_secant:.12f}  err {abs(t_secant - QUARTER):.1e}")
    print(f"    bisection on dense  {t_event:.12f}  err {abs(t_event - QUARTER):.1e}")
    if tol == 1e-6:
        fig_step, fig_ev = last, t_event

t, h, y0, y1, k = fig_step
pad = 0.25 * h
tt = np.linspace(t - pad, t + h + pad, 400)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.axhline(0, color=MUTED, lw=0.8)
ax.plot(tt, np.degrees(theta_exact(tt)), color=V[0], lw=1.8, label="exact")
ax.plot([t, t + h], np.degrees([y0[0], y1[0]]), "o", ms=8, color=V[2],
        label="last two step ends")
ax.axvline(t + h, color=V[2], lw=1.0, ls=":")
ax.annotate("naive stop", (t + h, -5), xytext=(-8, 0), textcoords="offset points",
            ha="right", fontsize=13, color=V[2])
ax.plot([fig_ev], [0], "*", ms=16, color=V[1], label="located event")
ax.set_xlabel("$t$")
ax.set_ylabel(r"$\theta$ (degrees)")
ax.legend(frameon=False, fontsize=12, loc="upper right")
save(fig, "event")
