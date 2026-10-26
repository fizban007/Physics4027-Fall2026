#!/usr/bin/env python3
"""Movies for Lecture 17. Run from this directory after make_figures.py; writes
figures/*.mp4 and a poster frame figures/*_poster.png for each (the PDF shows
the poster). Needs ffmpeg."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.integrate import solve_ivp

PAPER = "#FAFAF7"; INK = "#182430"; MUTED = "#55636F"
V = ["#440154", "#365C8D", "#1FA187", "#A0DA39"]; ACCENT = "#C2410C"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.2, "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})
cmap = plt.get_cmap("viridis")
FPS, SECONDS = 20, 5
NF = FPS * SECONDS
gauss = lambda x: np.exp(-x**2 / (2 * 0.5**2))

def write(fig, update, name, frames=NF):
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(f"figures/{name}.mp4", writer=FFMpegWriter(fps=FPS, codec="libx264",
              extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]), dpi=100)
    update(frames - 1)
    fig.savefig(f"figures/{name}_poster.png", dpi=100)
    plt.close(fig)
    print(f"[movie] {name}: {frames} frames at {FPS} fps")

def two_panel(title_left, title_right, xlim_left, xlim_right, ylim_right=(-0.05, 1.3)):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    a1.set_xlabel("$x$"); a1.set_ylabel("$t$"); a1.set_title(title_left, fontsize=13); a1.set_xlim(*xlim_left)
    a2.set_xlabel("$x$"); a2.set_ylabel("$u$"); a2.set_title(title_right, fontsize=13)
    a2.set_xlim(*xlim_right); a2.set_ylim(*ylim_right)
    fig.tight_layout()
    return fig, a1, a2

# ------------------------------------------- constant speed
v, T = 1.0, 3.0
fig, a1, a2 = two_panel(r"characteristics $x = x_0 + vt$", "the profile at time $t$", (-3.2, 6.2), (-3.2, 6.2))
for x0 in np.linspace(-3, 3, 13):
    a1.plot([x0, x0 + v * T], [0, T], color=cmap(gauss(x0) * 0.85), lw=1.6)
a1.set_ylim(0, T + 0.05)
tline = a1.axhline(0, color=ACCENT, lw=1.4, ls="--")
xx = np.linspace(-3.2, 6.2, 600)
a2.plot(xx, gauss(xx), color=V[0], lw=1.2, ls=":", label="$t = 0$")
(line,) = a2.plot(xx, gauss(xx), color=ACCENT, label="now")
a2.legend(frameon=False, fontsize=12, loc="upper right")
def upd(i):
    t = T * i / (NF - 1); tline.set_ydata([t, t]); line.set_ydata(gauss(xx - v * t))
    a2.set_title(f"the profile at $t = {t:.2f}$", fontsize=13)
write(fig, upd, "char_constant")

# ------------------------------------------- variable speed
vfun = lambda x: 1.0 + 0.5 * np.sin(x)
fig, a1, a2 = two_panel(r"characteristics $\dot x = v(x)$, $v = 1 + \tfrac{1}{2}\sin x$", "the profile at time $t$", (-4.2, 8.2), (-4.2, 8.2))
for x0 in np.linspace(-4, 4, 25):
    sol = solve_ivp(lambda t, x: [vfun(x[0])], [0, T], [x0], dense_output=True, rtol=1e-10, atol=1e-10)
    tt = np.linspace(0, T, 100); a1.plot(sol.sol(tt)[0], tt, color=cmap(gauss(x0) * 0.85), lw=1.4)
a1.set_ylim(0, T + 0.05)
tline = a1.axhline(0, color=ACCENT, lw=1.4, ls="--")
x0 = np.linspace(-6, 6, 2000)
sol = solve_ivp(lambda t, x: vfun(x), [0, T], x0, dense_output=True, rtol=1e-10, atol=1e-10)
a2.plot(x0, gauss(x0), color=V[0], lw=1.2, ls=":", label="$t = 0$")
(line,) = a2.plot(x0, gauss(x0), color=ACCENT, label="now")
a2.legend(frameon=False, fontsize=12, loc="upper right")
def upd(i):
    t = T * i / (NF - 1); tline.set_ydata([t, t]); line.set_xdata(sol.sol(t))
    a2.set_title(f"the profile at $t = {t:.2f}$", fontsize=13)
write(fig, upd, "char_variable")

# ------------------------------------------- Burgers
u0 = lambda x: np.exp(-x**2); du0 = lambda x: -2 * x * np.exp(-x**2)
xs = np.linspace(-3, 3, 4001); t_break = -1.0 / np.min(du0(xs)); Tb = 2.0
fig, a1, a2 = two_panel(r"characteristics $x = x_0 + u_0(x_0)\,t$", "the profile at time $t$", (-2.6, 3.6), (-2.6, 3.6), (-0.05, 1.2))
for x0 in np.linspace(-2.5, 2.5, 26):
    a1.plot([x0, x0 + u0(x0) * Tb], [0, Tb], color=cmap(u0(x0) * 0.85), lw=1.4)
a1.axhline(t_break, color=MUTED, lw=1.0, ls=":"); a1.text(-2.4, t_break + 0.06, r"$t_b$", color=MUTED, fontsize=13)
a1.set_ylim(0, Tb + 0.05)
tline = a1.axhline(0, color=ACCENT, lw=1.4, ls="--")
a2.plot(xs, u0(xs), color=V[0], lw=1.2, ls=":", label="$t = 0$")
(line,) = a2.plot(xs, u0(xs), color=ACCENT, label="now")
a2.legend(frameon=False, fontsize=12, loc="upper left")
def upd(i):
    t = Tb * i / (NF - 1); tline.set_ydata([t, t]); line.set_xdata(xs + u0(xs) * t)
    a2.set_title(f"the profile at $t = {t:.2f}$" + (" (past $t_b$)" if t > t_break else ""), fontsize=13)
write(fig, upd, "burgers")

# ------------------------------------------- wave equation: cone and the splitting pulse
c, Tw = 1.0, 2.5
fig, a1, a2 = two_panel("two families, $x \\pm ct$", "the profile at time $t$", (-3.2, 5.2), (-4, 4), (-0.05, 1.2))
for x0 in np.arange(-6, 7, 1.0):
    a1.plot([x0, x0 + c * 3], [0, 3], color=MUTED, lw=0.8, alpha=0.5); a1.plot([x0, x0 - c * 3], [0, 3], color=MUTED, lw=0.8, alpha=0.5)
X = 1.0
cone = a1.fill([X, X, X], [0, 0, 0], color=V[2], alpha=0.25, lw=0)[0]
(l1,) = a1.plot([X, X], [0, 0], color=V[2], lw=2.0); (l2,) = a1.plot([X, X], [0, 0], color=V[2], lw=2.0)
(pt,) = a1.plot([X], [0], "o", color=ACCENT, ms=7)
a1.set_ylim(0, 3.05)
tline = a1.axhline(0, color=ACCENT, lw=1.4, ls="--")
a2.plot(xx, gauss(xx), color=V[0], lw=1.2, ls=":", label="$t = 0$")
(line,) = a2.plot(xx, gauss(xx), color=ACCENT, label="now")
a2.legend(frameon=False, fontsize=12, loc="upper right")
def upd(i):
    t = Tw * i / (NF - 1); tline.set_ydata([t, t])
    cone.set_xy(np.array([[X - c * t, 0], [X + c * t, 0], [X, t]]))
    l1.set_data([X - c * t, X], [0, t]); l2.set_data([X + c * t, X], [0, t]); pt.set_data([X], [t])
    line.set_ydata(0.5 * gauss(xx - c * t) + 0.5 * gauss(xx + c * t))
    a2.set_title(f"the profile at $t = {t:.2f}$", fontsize=13)
write(fig, upd, "wave_cone")

# ------------------------------------------- FTCS blows up
L, N = 10.0, 200; dx = L / N; xg = -L / 2 + dx * np.arange(N); C = 0.5
step_ftcs = lambda u: u - 0.5 * C * (np.roll(u, -1) - np.roll(u, 1))
fig, ax = plt.subplots(figsize=(8, 4.0)); ax.set_xlim(-5, 5); ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
(ex,) = ax.plot(xg, gauss(xg), color=MUTED, lw=1.2, ls="--", label="exact")
(num,) = ax.plot(xg, gauss(xg), color=ACCENT, lw=1.6, label="FTCS")
ax.legend(frameon=False, fontsize=12, loc="upper left"); fig.tight_layout()
ftcs_states = [gauss(xg)]                # state after 4 i steps, precomputed so frames are pure
for i in range(100):
    u = ftcs_states[-1]
    for _ in range(4): u = step_ftcs(u)
    ftcs_states.append(u)
def upd(i):
    n = 4 * i; num.set_ydata(ftcs_states[i])
    xe = (xg - v * n * C * dx + L / 2) % L - L / 2; ex.set_ydata(gauss(xe))
    ax.set_title(f"FTCS, $v\\Delta t/\\Delta x = {C}$, after {n} steps", fontsize=13)
write(fig, upd, "ftcs", frames=101)      # 400 steps

# ------------------------------------------- upwind at C = 1 and C = 0.5, one crossing of the box
def step_up(u, Cn): return (1 - Cn) * u + Cn * np.roll(u, 1)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
inits = [gauss(xg), np.where(np.abs(xg) < 1, 1.0, 0.0)]
lines = []
for ax, init, name in zip(axs, inits, ["Gaussian", "step"]):
    ax.set_xlim(-5, 5); ax.set_ylim(-0.1, 1.2); ax.set_xlabel("$x$"); ax.set_title(name, fontsize=13)
    (e,) = ax.plot(xg, init, color=MUTED, lw=1.2, ls="--", label="exact")
    (l1,) = ax.plot(xg, init, color=V[1], lw=1.6, label=r"upwind, $v\Delta t/\Delta x = 1$")
    (l5,) = ax.plot(xg, init, color=ACCENT, lw=1.6, label=r"upwind, $v\Delta t/\Delta x = 0.5$")
    lines.append((e, l1, l5))
axs[0].set_ylabel("$u$"); axs[0].legend(frameon=False, fontsize=10, loc="upper left"); fig.tight_layout()
frames_up = 101                          # 200 steps at C = 1, 400 at C = 0.5: one crossing
up_states = []                           # per profile: lists of (u at C=1, u at C=0.5) after frame i
for init in inits:
    u1, u5, seq = init.copy(), init.copy(), [(init.copy(), init.copy())]
    for i in range(frames_up - 1):
        for _ in range(2): u1 = step_up(u1, 1.0)
        for _ in range(4): u5 = step_up(u5, 0.5)
        seq.append((u1, u5))
    up_states.append(seq)
def upd(i):
    for k, init in enumerate(inits):
        u1, u5 = up_states[k][i]
        shift = (xg - v * 2 * i * dx + L / 2) % L - L / 2
        e, l1, l5 = lines[k]
        e.set_ydata(np.interp(shift, xg, init, period=L)); l1.set_ydata(u1); l5.set_ydata(u5)
    fig.suptitle(f"after a distance $vt = {2 * i * dx:.1f}$ of a box of length 10", fontsize=13, y=0.99)
write(fig, upd, "upwind", frames=frames_up)
