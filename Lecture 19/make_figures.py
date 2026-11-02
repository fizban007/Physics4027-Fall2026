#!/usr/bin/env python3
"""Figures for Lecture 19 (the diffusion equation: FTCS works but the step is
tiny; BTCS and Crank-Nicolson). Run from this directory; writes figures/*.png
and prints the numbers quoted on the slides."""
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

# ------------------------------------------- the common problem: periodic box, D = 1
L, N, D = 10.0, 200, 1.0
dx = L/N; xg = -L/2 + dx*np.arange(N)
SIG0 = 0.5
gauss = lambda x, s=SIG0: np.exp(-x**2/(2*s**2))
tophat = np.where(np.abs(xg) < 1, 1.0, 0.0)
kgrid = 2*np.pi*np.fft.fftfreq(N, d=dx)
def exact(u0, t):
    """Exact solution of the PDE on the periodic box for the sampled initial data."""
    return np.real(np.fft.ifft(np.fft.fft(u0)*np.exp(-D*kgrid**2*t)))
def gauss_exact(x, t):
    s2 = SIG0**2 + 2*D*t
    return SIG0/np.sqrt(s2)*np.exp(-x**2/(2*s2))

dt_ftcs = dx**2/(2*D)
print(f"[grid] L={L}, N={N}, dx={dx}, D={D}: FTCS limit dt = dx^2/2D = {dt_ftcs:.5f}; diffusion time across the box L^2/D = {L**2/D:.0f}, across a cell dx^2/D = {dx**2/D:.4f}")
print(f"[cost] steps to diffuse across the box at the FTCS limit: 2N^2 = {2*N**2}; for N=1000: {2*1000**2:.0e}; to t=1: {1/dt_ftcs:.0f}")
lam = 4*D/dx**2*np.sin(np.pi*np.arange(1, N//2 + 1)/N)**2
print(f"[stiff] eigenvalues of -D L: largest {lam.max():.0f} (checkerboard, 4D/dx^2), smallest nonzero {lam.min():.3f} ((2 pi/L)^2 = {(2*np.pi/L)**2:.3f}); ratio {lam.max()/lam.min():.0f} ~ N^2/pi^2 = {N**2/np.pi**2:.0f}")

# ------------------------------------------- what diffusion does: a Gaussian and a top hat
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
for t, col in zip([0, 0.25, 1, 4], [INK, V[0], V[1], V[2]]):
    axs[0].plot(xg, exact(gauss(xg), t), color=col, label=f"$t = {t:g}$")
    s = np.sqrt(SIG0**2 + 2*D*t)
    print(f"[spread] Gaussian at t={t}: width sqrt(sigma0^2 + 2Dt) = {s:.2f}, peak {SIG0/s:.3f}")
axs[0].set_title(r"a Gaussian: width $\sqrt{\sigma_0^2 + 2Dt}$", fontsize=14)
for t, col in zip([0, 0.01, 0.1, 1], [INK, V[0], V[1], V[2]]):
    axs[1].plot(xg, exact(tophat, t), color=col, label=f"$t = {t:g}$")
axs[1].set_title("a top hat: the corners go first", fontsize=14)
for ax in axs:
    ax.set_xlim(-5, 5); ax.set_ylim(-0.05, 1.1); ax.set_xlabel("$x$"); ax.legend(frameon=False, fontsize=12)
axs[0].set_ylabel("$u$")
fig.tight_layout(); save(fig, "spreading")

# ------------------------------------------- amplification factors
theta = np.linspace(0, np.pi, 400)
s2 = np.sin(theta/2)**2
def xi(scheme, a, s2):
    if scheme == "exact": return np.exp(-a*(2*np.arcsin(np.sqrt(s2)))**2)   # e^{-D k^2 dt} = e^{-alpha theta^2}
    if scheme == "FTCS":  return 1 - 4*a*s2
    if scheme == "BTCS":  return 1/(1 + 4*a*s2)
    if scheme == "CN":    return (1 - 2*a*s2)/(1 + 2*a*s2)
names = {"exact": "exact", "FTCS": "FTCS", "BTCS": "BTCS (implicit)", "CN": "Crank-Nicolson"}
ticks = ([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])

# FTCS alone, three alpha
fig, ax = plt.subplots(figsize=(7.6, 3.8))
for a, col in [(0.25, V[2]), (0.5, V[1]), (0.6, ACCENT)]:
    ax.plot(theta, xi("FTCS", a, s2), color=col, label=r"FTCS, $\alpha = %.2g$" % a)
    ax.plot(theta, xi("exact", a, s2), color=col, lw=1.0, ls="--")
ax.axhline(1, color=MUTED, lw=0.8, ls=":"); ax.axhline(-1, color=MUTED, lw=0.8, ls=":")
ax.set_xlim(0, np.pi); ax.set_ylim(-1.6, 1.1); ax.set_xticks(*ticks)
ax.set_xlabel(r"$k\,\Delta x$"); ax.set_ylabel(r"$\xi$ per step")
ax.legend(frameon=False, fontsize=11, loc="lower left")
ax.set_title(r"solid: FTCS; dashed: exact $e^{-\alpha (k\Delta x)^2}$", fontsize=13)
fig.tight_layout(); save(fig, "amplification_ftcs")
for a in [0.25, 0.5, 0.6]:
    print(f"[xi] FTCS alpha={a}: xi at k dx = pi is {1 - 4*a:.2f}; exact there {np.exp(-a*np.pi**2):.4f}")
print(f"[xi] FTCS alpha=0.55: xi(pi) = {1-4*0.55:.2f}; |xi|^n reaches 1e16 after n = {16*np.log(10)/np.log(4*0.55-1):.0f} steps")

# all schemes at alpha = 0.5 and 5
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
for ax, a in zip(axs, [0.5, 5.0]):
    ax.plot(theta, xi("exact", a, s2), color=MUTED, lw=1.4, ls="--", label="exact")
    for s, col in [("FTCS", ACCENT), ("BTCS", V[1]), ("CN", V[0])]:
        if s == "FTCS" and a > 0.5: continue
        ax.plot(theta, xi(s, a, s2), color=col, label=names[s])
    ax.axhline(0, color=MUTED, lw=0.6); ax.axhline(-1, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, np.pi); ax.set_ylim(-1.1, 1.1); ax.set_xticks(*ticks)
    ax.set_xlabel(r"$k\,\Delta x$"); ax.set_title(r"$\alpha = D\Delta t/\Delta x^2 = %g$" % a, fontsize=14)
axs[0].set_ylabel(r"$\xi$ per step"); axs[0].legend(frameon=False, fontsize=11, loc="lower left")
fig.tight_layout(); save(fig, "amplification")
for a in [0.5, 5.0, 10.0]:
    for s in ["exact", "BTCS", "CN"]:
        print(f"[xi] alpha={a}: {names[s]:15s} xi at k dx = pi/8: {float(xi(s, a, np.sin(np.pi/16)**2)):.4f}; at pi/2: {float(xi(s, a, 0.5)):.4f}; at pi: {float(xi(s, a, 1.0)):.4f}")

# ------------------------------------------- the three schemes as matrices: one step each
def second_diff(u):
    return np.roll(u, -1) - 2*u + np.roll(u, 1)
def tridiag_periodic_solve(a_diag, a_off, rhs):
    """Solve (a_diag I + a_off L) u = rhs on the periodic grid, L the second difference.
    Circulant, so the FFT does it exactly; the slides say 'tridiagonal solver'."""
    sym = a_diag + a_off*(2*np.cos(kgrid*dx) - 2)
    return np.real(np.fft.ifft(np.fft.fft(rhs)/sym))
def step(scheme, u, a):
    if scheme == "FTCS": return u + a*second_diff(u)
    if scheme == "BTCS": return tridiag_periodic_solve(1.0, -a, u)
    if scheme == "CN":   return tridiag_periodic_solve(1.0, -a/2, u + 0.5*a*second_diff(u))
def run(scheme, u0, a, T):
    dt = a*dx**2/D; n = int(round(T/dt)); u = u0.copy()
    for _ in range(n): u = step(scheme, u, a)
    return u, n

# BTCS and CN against the exact answer at t = 1 for alpha = 10 (40 steps), FTCS at alpha = 0.5 (800 steps)
T = 1.0
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
for ax, u0, name in zip(axs, [gauss(xg), tophat], ["Gaussian", "top hat"]):
    ax.plot(xg, exact(u0, T), color=MUTED, lw=1.4, ls="--", label="exact")
    for scheme, a, col in [("FTCS", 0.5, ACCENT), ("BTCS", 10.0, V[1]), ("CN", 10.0, V[0])]:
        u, n = run(scheme, u0, a, T)
        err = np.max(np.abs(u - exact(u0, T)))
        ax.plot(xg, u, color=col, lw=1.6, label=r"%s, $\alpha = %g$ (%d steps)" % (names[scheme], a, n))
        print(f"[t=1] {name:8s} {names[scheme]:15s} alpha={a:4g}: {n:4d} steps, max error {err:.2e}, peak {u.max():.4f} (exact {exact(u0, T).max():.4f})")
    ax.set_xlim(-5, 5); ax.set_xlabel("$x$"); ax.set_title(f"{name} at $t = 1$", fontsize=13); ax.legend(frameon=False, fontsize=9.5)
axs[0].set_ylabel("$u$")
fig.tight_layout(); save(fig, "implicit_t1")

# ------------------------------------------- BTCS alone at several alpha: stable, and first-order wrong
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.plot(xg, exact(gauss(xg), T), color=MUTED, lw=1.4, ls="--", label="exact")
for a, col in [(0.5, V[2]), (10, V[1]), (100, V[0]), (400, ACCENT)]:
    u, n = run("BTCS", gauss(xg), a, T)
    ax.plot(xg, u, color=col, lw=1.6, label=r"$\alpha = %g$, %d step%s" % (a, n, "" if n == 1 else "s"))
    print(f"[btcs] alpha={a}: {n} steps to t=1, peak {u.max():.4f} vs exact {exact(gauss(xg), T).max():.4f}, max error {np.max(np.abs(u - exact(gauss(xg), T))):.2e}")
ax.set_xlim(-5, 5); ax.set_xlabel("$x$"); ax.set_ylabel("$u$"); ax.legend(frameon=False, fontsize=11)
ax.set_title("BTCS, Gaussian at $t = 1$: never blows up, slowly wrong", fontsize=13)
fig.tight_layout(); save(fig, "btcs_alpha")

# ------------------------------------------- Crank-Nicolson on a top hat at large alpha: the wiggles
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, (a, t) in zip(axs, [(10.0, 0.025), (10.0, 0.25)]):
    ax.plot(xg, exact(tophat, t), color=MUTED, lw=1.4, ls="--", label="exact")
    for scheme, col in [("BTCS", V[1]), ("CN", V[0])]:
        u, n = run(scheme, tophat, a, t)
        ax.plot(xg, u, color=col, lw=1.6, label=f"{names[scheme]} ({n} steps)")
        print(f"[tophat] {names[scheme]:15s} alpha={a} t={t}: min {u.min():.3f}, max {u.max():.3f}, exact max {exact(tophat, t).max():.3f}")
    ax.set_xlim(-3, 3); ax.set_xlabel("$x$"); ax.set_title(r"top hat, $\alpha = %g$, $t = %g$" % (a, t), fontsize=13); ax.legend(frameon=False, fontsize=11)
axs[0].set_ylabel("$u$")
fig.tight_layout(); save(fig, "cn_tophat")

# ------------------------------------------- convergence in dt at fixed dx, Gaussian, T = 1
ref = exact(gauss(xg), T)
alphas = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 400])
fig, ax = plt.subplots(figsize=(7.6, 4.0))
for scheme, col in [("BTCS", V[1]), ("CN", V[0])]:
    errs = []
    for a in alphas:
        u, n = run(scheme, gauss(xg), a, T); errs.append(np.max(np.abs(u - ref)))
    errs = np.array(errs); dts = alphas*dx**2/D
    ax.loglog(dts, errs, "o-", color=col, label=names[scheme], ms=5)
    for a, dt, e in zip(alphas, dts, errs): print(f"[conv] {names[scheme]:15s} alpha={a:5g} dt={dt:.4f}: max error {e:.2e}")
dts = alphas*dx**2/D
ax.loglog(dts, 3*1.65e-4*dts/dts[0], color=MUTED, lw=1, ls=":"); ax.text(dts[2], 3*1.65e-4*dts[2]/dts[0]*1.6, r"$\propto \Delta t$", color=MUTED, fontsize=12)
ax.loglog(dts, 3.68e-3*(dts/dts[7])**2, color=MUTED, lw=1, ls=":"); ax.text(dts[6], 3.68e-3*(dts[6]/dts[7])**2*0.3, r"$\propto \Delta t^2$", color=MUTED, fontsize=12)
u_ftcs, n = run("FTCS", gauss(xg), 0.5, T); e_ftcs = np.max(np.abs(u_ftcs - ref))
ax.loglog([dts[0]], [e_ftcs], "s", color=ACCENT, ms=8, label=r"FTCS at its limit, $\alpha = 1/2$")
print(f"[conv] FTCS alpha=0.5: {n} steps, max error {e_ftcs:.2e}")
ax.set_xlabel(r"$\Delta t$ (fixed $\Delta x = 0.05$)"); ax.set_ylabel("max error at $t = 1$")
ax.legend(frameon=False, fontsize=11, loc="upper left"); ax.set_title("Gaussian, $N = 200$: error against the time step", fontsize=13)
fig.tight_layout(); save(fig, "convergence")

# ------------------------------------------- cell Peclet number: when explicit diffusion costs nothing extra
for Pe in [0.1, 1, 10]:
    print(f"[peclet] v dx / D = {Pe}: dt_diffusion / dt_advection(C=1) = v dx/2D = {Pe/2}")
print("done")
