#!/usr/bin/env python3
"""Figures for Lecture 18 (why FTCS blows up, and the schemes that do not: upwind,
Lax-Friedrichs, Lax-Wendroff; amplification factors, phase errors, numerical
diffusion and dispersion). Run from this directory; writes figures/*.png and
prints the numbers quoted on the slides."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "#FAFAF7"; INK = "#182430"; MUTED = "#55636F"
V = ["#440154", "#365C8D", "#1FA187", "#A0DA39"]; ACCENT = "#C2410C"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.2, "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})
def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

# ------------------------------------------- amplification factors xi(theta), theta = k dx
theta = np.linspace(0, np.pi, 400)
def xi(scheme, C, th):
    if scheme == "FTCS":   return 1 - 1j*C*np.sin(th)
    if scheme == "upwind": return 1 - C + C*np.exp(-1j*th)
    if scheme == "Lax":    return np.cos(th) - 1j*C*np.sin(th)
    if scheme == "LW":     return 1 - 1j*C*np.sin(th) - C**2*(1 - np.cos(th))
schemes = [("FTCS", ACCENT), ("upwind", V[1]), ("Lax", V[2]), ("LW", V[0])]
names = {"FTCS": "FTCS", "upwind": "upwind", "Lax": "Lax-Friedrichs", "LW": "Lax-Wendroff"}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
for ax, C in zip(axs, [0.5, 0.9]):
    for s, col in schemes:
        ax.plot(theta, np.abs(xi(s, C, theta)), color=col, label=names[s])
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, np.pi); ax.set_ylim(0, 1.5)
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]); ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_xlabel(r"$k\,\Delta x$"); ax.set_title(r"$C = v\Delta t/\Delta x = %.1f$" % C, fontsize=14)
axs[0].set_ylabel(r"$|\xi|$ per step"); axs[0].legend(frameon=False, fontsize=12, loc="lower left")
fig.tight_layout(); save(fig, "amplification")
for C in [0.5, 0.9]:
    for s, _ in schemes:
        a = np.abs(xi(s, C, theta)); i = np.argmax(a) if s == "FTCS" else np.argmin(a)
        print(f"[xi] C={C}: {names[s]:14s} |xi| {'max' if s=='FTCS' else 'min'} = {a[i]:.4f} at k dx = {theta[i]/np.pi:.2f} pi; at k dx = pi/2: {np.abs(xi(s, C, np.pi/2)):.4f}")
# FTCS growth over one crossing, C = 0.5, 200 cells: 400 steps
for th in [np.pi/2, np.pi/4, 2*np.pi*4/200]:
    print(f"[ftcs] C=0.5 k dx={th:.3f}: |xi|^400 = {np.abs(xi('FTCS', 0.5, th))**400:.3g}")
# amplification at theta for the Gaussian of width 0.5 on 200 cells over L=10: dominant k dx ~ dx/sigma = 0.05/0.5 = 0.1
# upwind: per-step damping at the Gaussian's scale and the peak after 400 steps
# (the peak after a crossing is quoted from make_movies / Lecture 17: 0.707)

# FTCS alone, for the diagnosis slide
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for C, col in [(0.25, V[2]), (0.5, V[1]), (0.9, ACCENT)]:
    ax.plot(theta, np.abs(xi("FTCS", C, theta)), color=col, label=r"$C = %.2g$" % C)
ax.axhline(1, color=MUTED, lw=0.8, ls="--")
ax.set_xlim(0, np.pi); ax.set_ylim(0.9, 1.45)
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]); ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
ax.set_xlabel(r"$k\,\Delta x$"); ax.set_ylabel(r"$|\xi|$ per step"); ax.legend(frameon=False, fontsize=12, loc="upper left")
ax.set_title("FTCS: every wave is amplified", fontsize=14)
fig.tight_layout(); save(fig, "amplification_ftcs")

# ------------------------------------------- phase error: numerical phase speed over v
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
for ax, C in zip(axs, [0.5, 0.9]):
    for s, col in schemes[1:]:
        ph = -np.angle(xi(s, C, theta))              # phase advance per step; exact is C theta
        ratio = np.where(theta > 0, ph/(C*theta), 1.0)
        ax.plot(theta, ratio, color=col, label=names[s])
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, np.pi); ax.set_ylim(0, 1.3)
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]); ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_xlabel(r"$k\,\Delta x$"); ax.set_title(r"$C = %.1f$" % C, fontsize=14)
axs[0].set_ylabel("numerical speed / $v$"); axs[0].legend(frameon=False, fontsize=12, loc="lower left")
fig.tight_layout(); save(fig, "phase")
for C in [0.5, 0.9]:
    for s, _ in schemes[1:]:
        th = np.pi/4; ph = -np.angle(xi(s, C, th))
        print(f"[phase] C={C}: {names[s]:14s} speed ratio at k dx = pi/4: {ph/(C*th):.3f}; at pi/2: {-np.angle(xi(s, C, np.pi/2))/(C*np.pi/2):.3f}")

# ------------------------------------------- the three schemes after one crossing of the box
L, N = 10.0, 200; dx = L/N; xg = -L/2 + dx*np.arange(N); v = 1.0
gauss = lambda x: np.exp(-x**2/(2*0.5**2))
step = np.where(np.abs(xg) < 1, 1.0, 0.0)
def run(scheme, u, C, n):
    for _ in range(n):
        um, up = np.roll(u, 1), np.roll(u, -1)
        if scheme == "upwind": u = (1 - C)*u + C*um
        elif scheme == "Lax":  u = 0.5*(up + um) - 0.5*C*(up - um)
        elif scheme == "LW":   u = u - 0.5*C*(up - um) + 0.5*C**2*(up - 2*u + um)
    return u
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
C = 0.5; n = int(round(L/(v*C*dx)))
for ax, init, name in zip(axs, [gauss(xg), step], ["Gaussian", "step"]):
    ax.plot(xg, init, color=MUTED, lw=1.2, ls="--", label="exact")
    for s, col in schemes[1:]:
        u = run(s, init.copy(), C, n)
        ax.plot(xg, u, color=col, lw=1.6, label=names[s])
        print(f"[crossing] C={C} {name:8s} {names[s]:14s}: max {u.max():.3f}, min {u.min():.3f}, max|u - exact| {np.max(np.abs(u - init)):.3f}")
    ax.set_xlabel("$x$"); ax.set_title(f"{name}, after one crossing at $C = {C}$", fontsize=13)
    ax.set_ylim(-0.3, 1.3)
axs[0].set_ylabel("$u$"); axs[0].legend(frameon=False, fontsize=11)
fig.tight_layout(); save(fig, "crossing")

# ------------------------------------------- stencil diagram: the foot of the characteristic
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, Cn, title in zip(axs, [0.5, 1.0, 1.5],
                         [r"$v\Delta t = 0.5\,\Delta x$", r"$v\Delta t = \Delta x$", r"$v\Delta t = 1.5\,\Delta x$: the foot is outside the stencil"]):
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
    ax.set_title(title, fontsize=13)
fig.tight_layout()
save(fig, "upwind_stencil")

# ------------------------------------------- numerical diffusion coefficients, in units of v dx
for C in [0.25, 0.5, 0.9, 1.0]:
    print(f"[diffusion] C={C}: upwind D = {0.5*(1-C):.3f} v dx; Lax D = {(1-C**2)/(2*C):.3f} v dx")
print("done")
