#!/usr/bin/env python3
"""Movies for Lecture 18: FTCS blowing up, then upwind, Lax-Friedrichs and
Lax-Wendroff carrying a Gaussian and a step once around the box. Writes
figures/*.mp4 and a poster frame for each. Needs ffmpeg."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

PAPER = "#FAFAF7"; INK = "#182430"; MUTED = "#55636F"
V = ["#440154", "#365C8D", "#1FA187", "#A0DA39"]; ACCENT = "#C2410C"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.2, "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})
FPS = 20
L, N = 10.0, 200; dx = L/N; xg = -L/2 + dx*np.arange(N); v = 1.0
gauss = lambda x: np.exp(-x**2/(2*0.5**2))
inits = [gauss(xg), np.where(np.abs(xg) < 1, 1.0, 0.0)]

def step(scheme, u, C):
    um, up = np.roll(u, 1), np.roll(u, -1)
    if scheme == "FTCS":   return u - 0.5*C*(up - um)
    if scheme == "upwind": return (1 - C)*u + C*um
    if scheme == "Lax":    return 0.5*(up + um) - 0.5*C*(up - um)
    if scheme == "LW":     return u - 0.5*C*(up - um) + 0.5*C**2*(up - 2*u + um)

def write(fig, update, name, frames):
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(f"figures/{name}.mp4", writer=FFMpegWriter(fps=FPS, codec="libx264",
              extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]), dpi=100)
    update(frames - 1); fig.savefig(f"figures/{name}_poster.png", dpi=100); plt.close(fig)
    print(f"[movie] {name}: {frames} frames")

# ------------------------------------------- FTCS, Gaussian, C = 0.5, 400 steps
C = 0.5
states = [gauss(xg)]
for i in range(100):
    u = states[-1]
    for _ in range(4): u = step("FTCS", u, C)
    states.append(u)
fig, ax = plt.subplots(figsize=(8, 4.0)); ax.set_xlim(-5, 5); ax.set_ylim(-2.2, 2.2); ax.set_xlabel("$x$"); ax.set_ylabel("$u$")
(ex,) = ax.plot(xg, gauss(xg), color=MUTED, lw=1.2, ls="--", label="exact")
(num,) = ax.plot(xg, gauss(xg), color=ACCENT, lw=1.6, label="FTCS")
ax.legend(frameon=False, fontsize=12, loc="upper left"); fig.tight_layout()
def upd(i):
    n = 4*i; num.set_ydata(states[i])
    ex.set_ydata(gauss((xg - v*n*C*dx + L/2) % L - L/2))
    ax.set_title(f"FTCS, $C = {C}$, after {n} steps", fontsize=13)
write(fig, upd, "ftcs", 101)

# ------------------------------------------- the three stable schemes: two Courant numbers each, one crossing
def two_courant_movie(scheme, name, C_hi, C_lo, label_hi, label_lo):
    """C_hi needs n_hi steps per crossing, C_lo needs n_lo; frames advance both by the same distance."""
    frames = 101
    n_hi, n_lo = int(round(L/(v*C_hi*dx))), int(round(L/(v*C_lo*dx)))
    per_hi, per_lo = n_hi//(frames - 1), n_lo//(frames - 1)
    hist = []
    for init in inits:
        uh, ul, seq = init.copy(), init.copy(), [(init.copy(), init.copy())]
        for i in range(frames - 1):
            for _ in range(per_hi): uh = step(scheme, uh, C_hi)
            for _ in range(per_lo): ul = step(scheme, ul, C_lo)
            seq.append((uh, ul))
        hist.append(seq)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0)); lines = []
    for ax, init, nm in zip(axs, inits, ["Gaussian", "step"]):
        ax.set_xlim(-5, 5); ax.set_ylim(-0.4, 1.4); ax.set_xlabel("$x$"); ax.set_title(nm, fontsize=13)
        (e,) = ax.plot(xg, init, color=MUTED, lw=1.2, ls="--", label="exact")
        (lh,) = ax.plot(xg, init, color=V[1], lw=1.6, label=label_hi)
        (ll,) = ax.plot(xg, init, color=ACCENT, lw=1.6, label=label_lo)
        lines.append((e, lh, ll))
    axs[0].set_ylabel("$u$"); axs[0].legend(frameon=False, fontsize=10, loc="upper left"); fig.tight_layout()
    def upd(i):
        dist = i*per_hi*C_hi*dx*v
        for k, init in enumerate(inits):
            uh, ul = hist[k][i]; e, lh, ll = lines[k]
            e.set_ydata(np.interp((xg - dist + L/2) % L - L/2, xg, init, period=L)); lh.set_ydata(uh); ll.set_ydata(ul)
        fig.suptitle(f"{name}, after a distance $vt = {dist:.1f}$ of a box of length 10", fontsize=13, y=0.99)
    write(fig, upd, scheme.lower(), frames)

two_courant_movie("upwind", "upwind", 1.0, 0.5, r"$C = 1$", r"$C = 0.5$")
two_courant_movie("Lax", "Lax-Friedrichs", 1.0, 0.5, r"$C = 1$", r"$C = 0.5$")
two_courant_movie("LW", "Lax-Wendroff", 1.0, 0.5, r"$C = 1$", r"$C = 0.5$")
