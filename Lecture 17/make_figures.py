#!/usr/bin/env python3
"""Figures for Lecture 17 (PDEs: the advection equation, the method of
characteristics, and from characteristics to the upwind scheme). Run from this
directory; writes figures/*.png and prints the numbers quoted on the slides."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

PAPER = "#FAFAF7"
INK = "#182430"
MUTED = "#55636F"
V = ["#440154", "#365C8D", "#1FA187", "#A0DA39"]   # viridis picks, dark to light
ACCENT = "#C2410C"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.2,
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})
cmap = plt.get_cmap("viridis")

def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def arrow(ax, start, end, color, lw=2.0, ls="-"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=14))

gauss = lambda x: np.exp(-x**2 / (2 * 0.5**2))

# ------------------------------------------- 1. constant speed: lines, and the profile carried along
v = 1.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
x0s = np.linspace(-3, 3, 13)
for x0 in x0s:
    a1.plot([x0, x0 + v * 3], [0, 3], color=cmap(gauss(x0) * 0.85), lw=1.6)
a1.set_xlim(-3.2, 6.2); a1.set_ylim(0, 3.05)
a1.set_xlabel("$x$"); a1.set_ylabel("$t$")
a1.set_title(r"characteristics $x = x_0 + v t$, coloured by the value $u$ they carry", fontsize=13)
xx = np.linspace(-3.2, 6.2, 600)
for t, col in [(0.0, V[0]), (3.0, ACCENT)]:
    a2.plot(xx, gauss(xx - v * t), color=col, label=f"$t = {t:.0f}$")
arrow(a2, (0.0, 1.08), (3.0, 1.08), MUTED, lw=1.6)
a2.text(1.5, 1.14, "$v t$", ha="center", fontsize=14, color=MUTED)
a2.set_xlim(-3.2, 6.2); a2.set_ylim(-0.05, 1.3)
a2.set_xlabel("$x$"); a2.set_ylabel("$u$"); a2.legend(frameon=False, fontsize=13)
a2.set_title("the profile is carried along, unchanged", fontsize=13)
fig.tight_layout()
save(fig, "char_constant")

# ------------------------------------------- 2. variable speed v(x) = 1 + 0.5 sin x
vfun = lambda x: 1.0 + 0.5 * np.sin(x)
T = 3.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
x0s = np.linspace(-4, 4, 25)
for x0 in x0s:
    sol = solve_ivp(lambda t, x: [vfun(x[0])], [0, T], [x0], dense_output=True, rtol=1e-10, atol=1e-10)
    tt = np.linspace(0, T, 100)
    a1.plot(sol.sol(tt)[0], tt, color=cmap(gauss(x0) * 0.85), lw=1.4)
a1.set_xlim(-4.2, 8.2); a1.set_ylim(0, T + 0.05)
a1.set_xlabel("$x$"); a1.set_ylabel("$t$")
a1.set_title(r"characteristics $\dot x = v(x)$, $v(x) = 1 + \tfrac{1}{2}\sin x$", fontsize=13)
x0 = np.linspace(-6, 6, 2000)
sol = solve_ivp(lambda t, x: vfun(x), [0, T], x0, t_eval=[0, 1.5, T], rtol=1e-10, atol=1e-10)
for i, (t, col) in enumerate([(0.0, V[0]), (1.5, V[2]), (T, ACCENT)]):
    a2.plot(sol.y[:, i], gauss(x0), color=col, label=f"$t = {t:g}$")
a2.set_xlim(-4.2, 8.2); a2.set_ylim(-0.05, 1.3)
a2.set_xlabel("$x$"); a2.set_ylabel("$u$"); a2.legend(frameon=False, fontsize=13)
a2.set_title("steeper where the curves converge", fontsize=13)
fig.tight_layout()
save(fig, "char_variable")
# where do they converge? v' < 0 for cos x < 0: x in (pi/2, 3pi/2); the pulse at t = 3 sits near x ~ 3.5
print(f"[variable] v(x) = 1 + 0.5 sin x; at t = {T} the peak has moved from 0 to x = {solve_ivp(lambda t, x: vfun(x), [0, T], [0.0], rtol=1e-10).y[0, -1]:.2f};"
      f" max slope of profile: t=0 {np.max(np.abs(np.gradient(gauss(x0), x0))):.2f}, t={T} {np.max(np.abs(np.gradient(gauss(x0), sol.y[:, -1]))):.2f}")

# ------------------------------------------- 3. a source term: decay while advected
tau = 2.0
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for t, col in [(0.0, V[0]), (1.5, V[2]), (3.0, ACCENT)]:
    ax.plot(xx, gauss(xx - v * t) * np.exp(-t / tau), color=col, label=f"$t = {t:g}$")
ax.set_xlim(-3.2, 6.2); ax.set_ylim(-0.05, 1.2)
ax.set_xlabel("$x$"); ax.set_ylabel("$u$"); ax.legend(frameon=False, fontsize=13)
ax.set_title(r"$\partial_t u + v\,\partial_x u = -u/\tau$: carried along, and decaying", fontsize=13)
fig.tight_layout()
save(fig, "char_source")
print(f"[source] tau = {tau}: peak at t = 3 is exp(-3/tau) = {np.exp(-3/tau):.3f}")

# ------------------------------------------- 4. Burgers: characteristics cross
u0 = lambda x: np.exp(-x**2)
du0 = lambda x: -2 * x * np.exp(-x**2)
xs = np.linspace(-3, 3, 4001)
t_break = -1.0 / np.min(du0(xs))
x_break = xs[np.argmin(du0(xs))]
print(f"[burgers] u0 = exp(-x^2): min u0' = {np.min(du0(xs)):.4f} at x0 = {x_break:.3f}; breaking time t_b = {t_break:.3f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
Tmax = 2.0
for x0 in np.linspace(-2.5, 2.5, 26):
    a1.plot([x0, x0 + u0(x0) * Tmax], [0, Tmax], color=cmap(u0(x0) * 0.85), lw=1.4)
a1.axhline(t_break, color=ACCENT, lw=1.2, ls="--")
a1.text(-2.4, t_break + 0.06, r"$t_b = %.2f$" % t_break, color=ACCENT, fontsize=13)
a1.set_xlim(-2.6, 3.6); a1.set_ylim(0, Tmax + 0.05)
a1.set_xlabel("$x$"); a1.set_ylabel("$t$")
a1.set_title(r"characteristics $x = x_0 + u_0(x_0)\,t$: faster ones overtake", fontsize=13)
for t, col in [(0.0, V[0]), (0.6, V[1]), (t_break, V[2]), (2.0, ACCENT)]:
    a2.plot(xs + u0(xs) * t, u0(xs), color=col, label=f"$t = {t:.2f}$" if t != t_break else "$t = t_b$")
a2.set_xlim(-2.6, 3.6); a2.set_ylim(-0.05, 1.2)
a2.set_xlabel("$x$"); a2.set_ylabel("$u$"); a2.legend(frameon=False, fontsize=12, loc="upper left")
a2.set_title("the front steepens, then folds over", fontsize=13)
fig.tight_layout()
save(fig, "burgers")

# ------------------------------------------- 4b. Burgers after breaking: the shock path
from scipy.optimize import brentq
def branches(x, t):
    """All x0 whose characteristic x0 + u0(x0) t passes through x at time t."""
    grid = np.linspace(-4, 4, 1601)
    g = grid + u0(grid) * t - x
    roots = []
    for i in np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]:
        roots.append(brentq(lambda z: z + u0(z) * t - x, grid[i], grid[i + 1]))
    return roots
def shock_speed(x, t):
    r = branches(x, t)
    return 0.5 * (u0(min(r)) + u0(max(r)))       # Rankine-Hugoniot: average of the outer branches
x_b = x_break + u0(x_break) * t_break
T_end = 3.0
ts_shock = np.linspace(t_break, T_end, 200)
xs_shock = [x_b]
for t0, t1 in zip(ts_shock[:-1], ts_shock[1:]):   # RK4 along the shock path
    h = t1 - t0; x = xs_shock[-1]
    k1 = shock_speed(x, t0); k2 = shock_speed(x + 0.5 * h * k1, t0 + 0.5 * h)
    k3 = shock_speed(x + 0.5 * h * k2, t0 + 0.5 * h); k4 = shock_speed(x + h * k3, t1)
    xs_shock.append(x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
xs_shock = np.array(xs_shock)
print(f"[shock] starts at (x, t) = ({x_b:.3f}, {t_break:.3f}); at t = 2 it is at x = {np.interp(2.0, ts_shock, xs_shock):.3f},"
      f" at t = 3 at x = {xs_shock[-1]:.3f}; speed at t = 3: {shock_speed(xs_shock[-1], T_end):.3f}")
def shock_x(t):
    return np.interp(t, ts_shock, xs_shock) if t >= t_break else np.inf
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
tt = np.linspace(0, T_end, 400)
for x0 in np.linspace(-2.5, 2.5, 26):
    xline = x0 + u0(x0) * tt
    after = tt >= t_break                                           # the shock exists only from t_b on
    side = np.sign(xline[after] - np.array([shock_x(t) for t in tt[after]]))
    hit = np.where(side != side[0])[0]                             # first time the line changes side
    last = np.where(after)[0][0] + hit[0] if hit.size else tt.size
    a1.plot(xline[:last], tt[:last], color=cmap(u0(x0) * 0.85), lw=1.4)
a1.plot(xs_shock, ts_shock, color=ACCENT, lw=3.0)
a1.plot([x_b], [t_break], "o", color=ACCENT, ms=7)
a1.text(x_b + 0.15, t_break - 0.12, "$t_b$", color=ACCENT, fontsize=13)
a1.text(np.interp(2.4, ts_shock, xs_shock) + 0.15, 2.4, "shock", color=ACCENT, fontsize=13)
a1.set_xlim(-2.6, 3.6); a1.set_ylim(0, T_end + 0.05)
a1.set_xlabel("$x$"); a1.set_ylabel("$t$")
a1.set_title("characteristics end at the shock", fontsize=13)
xx_s = np.linspace(-2.6, 3.6, 1200)
for t, col in [(0.0, V[0]), (2.0, V[2]), (3.0, ACCENT)]:
    xsh = shock_x(t)
    prof = np.array([u0(min(branches(x, t))) if x < xsh else u0(max(branches(x, t))) for x in xx_s]) if t > 0 else u0(xx_s)
    a2.plot(xx_s, prof, color=col, label=f"$t = {t:.0f}$")
a2.set_xlim(-2.6, 3.6); a2.set_ylim(-0.05, 1.2)
a2.set_xlabel("$x$"); a2.set_ylabel("$u$"); a2.legend(frameon=False, fontsize=12, loc="upper left")
a2.set_title("the profile keeps one value: a jump", fontsize=13)
fig.tight_layout()
save(fig, "burgers_shock")

# ------------------------------------------- 5. wave equation: two families and the domain of dependence
c = 1.0
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
for x0 in np.arange(-6, 7, 1.0):
    a1.plot([x0, x0 + c * 3], [0, 3], color=MUTED, lw=0.8, alpha=0.5)
    a1.plot([x0, x0 - c * 3], [0, 3], color=MUTED, lw=0.8, alpha=0.5)
X, Tt = 1.0, 2.0
a1.fill([X - c * Tt, X + c * Tt, X], [0, 0, Tt], color=V[2], alpha=0.25, lw=0)
a1.plot([X - c * Tt, X], [0, Tt], color=V[2], lw=2.2); a1.plot([X + c * Tt, X], [0, Tt], color=V[2], lw=2.2)
a1.plot([X], [Tt], "o", color=ACCENT, ms=7)
a1.text(X + 0.15, Tt + 0.08, "$(x, t)$", color=ACCENT, fontsize=14)
a1.text(X, 0.25, "$[x - ct,\\ x + ct]$", ha="center", fontsize=13, color=V[2])
a1.set_xlim(-3.2, 5.2); a1.set_ylim(0, 3.05)
a1.set_xlabel("$x$"); a1.set_ylabel("$t$")
a1.set_title("two families, $x \\pm ct$, and the domain of dependence", fontsize=13)
for t, col in [(0.0, V[0]), (2.0, ACCENT)]:
    a2.plot(xx, 0.5 * gauss(xx - c * t) + 0.5 * gauss(xx + c * t), color=col, label=f"$t = {t:.0f}$")
a2.set_xlim(-4, 4); a2.set_ylim(-0.05, 1.2)
a2.set_xlabel("$x$"); a2.set_ylabel("$u$"); a2.legend(frameon=False, fontsize=13)
a2.set_title("a pulse released from rest splits in two", fontsize=13)
fig.tight_layout()
save(fig, "wave_cone")

# ------------------------------------------- 6. FTCS blows up; upwind as interpolation along the characteristic
L, N = 10.0, 200
dx = L / N
xg = -L / 2 + dx * np.arange(N)
def step_ftcs(u, C):
    return u - 0.5 * C * (np.roll(u, -1) - np.roll(u, 1))
def step_upwind(u, C):          # v > 0: the foot of the characteristic lies between x_{j-1} and x_j
    return (1 - C) * u + C * np.roll(u, 1)
def run(step, u, C, n):
    for _ in range(n):
        u = step(u, C)
    return u
C = 0.5
u_init = gauss(xg)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, n in zip(axs, [0, 240, 400]):
    u = run(step_ftcs, u_init.copy(), C, n)
    ax.plot(xg, gauss(xg - v * n * C * dx), color=MUTED, lw=1.2, ls="--", label="exact")
    ax.plot(xg, u, color=ACCENT, lw=1.6, label="FTCS")
    ax.set_title(f"after {n} steps", fontsize=13); ax.set_xlabel("$x$")
    print(f"[ftcs] C = {C}: after {n:3d} steps max |u| = {np.max(np.abs(u)):.3g}")
axs[0].set_ylabel("$u$"); axs[0].legend(frameon=False, fontsize=12)
axs[0].set_ylim(-3, 3)
fig.tight_layout()
save(fig, "ftcs")

# stencil diagram: the foot of the characteristic for three Courant numbers
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, Cn in zip(axs, [0.5, 1.0, 1.5]):
    for j, lab in zip([-1, 0, 1], ["$x_{j-1}$", "$x_j$", "$x_{j+1}$"]):
        ax.plot([j], [0], "o", color=INK, ms=7)
        ax.text(j, -0.22, lab, ha="center", fontsize=13)
    ax.plot([0], [1], "o", color=ACCENT, ms=8)
    ax.text(0.08, 1.05, "$u_j^{n+1}$", fontsize=13, color=ACCENT)
    foot = -Cn
    ax.plot([foot, 0], [0, 1], color=V[1], lw=2.0)
    ax.plot([foot], [0], "s", color=V[1], ms=7)
    ax.text(foot - 0.12, 0.16, "foot", ha="right", fontsize=12, color=V[1])
    ax.axhline(0, color=MUTED, lw=0.8); ax.axhline(1, color=MUTED, lw=0.8)
    ax.text(-1.95, 0.02, "$t_n$", fontsize=12, color=MUTED); ax.text(-1.95, 1.02, "$t_{n+1}$", fontsize=12, color=MUTED)
    ax.set_xlim(-2.1, 1.6); ax.set_ylim(-0.4, 1.3); ax.axis("off")
    ax.set_title(r"$v\Delta t/\Delta x = %.1f$" % Cn + ("" if Cn <= 1 else ": the foot is outside the stencil"), fontsize=13)
fig.tight_layout()
save(fig, "upwind_stencil")

# upwind transport of a pulse and a step, C = 0.5 and C = 1, after one crossing of the box
n_cross = int(round(L / (v * C * dx)))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, init, name in zip(axs, [gauss(xg), np.where(np.abs(xg) < 1, 1.0, 0.0)], ["Gaussian", "step"]):
    ax.plot(xg, init, color=MUTED, lw=1.2, ls="--", label="exact")
    for Cn, col in [(1.0, V[1]), (0.5, ACCENT)]:
        n = int(round(L / (v * Cn * dx)))
        u = run(step_upwind, init.copy(), Cn, n)
        ax.plot(xg, u, color=col, lw=1.6, label=r"upwind, $v\Delta t/\Delta x = %.1f$" % Cn)
        print(f"[upwind] {name}, C = {Cn}: {n} steps = one box crossing; max u = {u.max():.3f}, max |u - exact| = {np.max(np.abs(u - init)):.3f}")
    ax.set_xlabel("$x$"); ax.set_title(f"{name}, after one crossing of the box", fontsize=13)
axs[0].set_ylabel("$u$"); axs[0].legend(frameon=False, fontsize=11)
fig.tight_layout()
save(fig, "upwind_pulse")
print("done")
