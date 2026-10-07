#!/usr/bin/env python3
"""Figures for Lecture 12 (boundary value problems).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides.

Part I (shooting): a basketball shot with quadratic drag, the infinite square
well, and the linear Debye-screening problem phi'' = kappa^2 phi, where
shooting fails. Part II (grid): finite-difference errors, the Debye problem on
a grid, the nonlinear version phi'' = kappa^2 sinh(phi) solved with Newton,
and the sparsity of the resulting matrices."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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

def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def secant(F, a, b, tol=1e-10, max_iter=50):
    """Secant iteration; returns the root and the number of evaluations of F."""
    fa, fb = F(a), F(b)
    n_eval = 2
    while abs(b - a) > tol and n_eval < max_iter:
        a, b, fa = b, b - fb * (b - a) / (fb - fa), fb
        fb = F(b)
        n_eval += 1
    return b, n_eval

# ============================================================ Part I
# ------------------------------------------- 1. a basketball shot with drag
# 3-point shot: release 2.0 m high, hoop 3.05 m high, 6.75 m away; speed 10 m/s.
# Drag k = rho C_d A / (2 m) = 1.2 * 0.47 * 0.0456 / (2 * 0.62) = 0.0207 / m.
g, k_drag, v0, D, H = 9.8, 0.0207, 10.0, 6.75, 1.05

def ball(theta, dense=False):
    def f(t, s):
        x, y, vx, vy = s
        v = np.hypot(vx, vy)
        return [vx, vy, -k_drag * v * vx, -g - k_drag * v * vy]
    reach_hoop = lambda t, s: s[0] - D        # stop condition: x = D
    reach_hoop.terminal = True
    return solve_ivp(f, [0, 10], [0, 0, v0 * np.cos(theta), v0 * np.sin(theta)],
                     events=reach_hoop, rtol=1e-10, atol=1e-12, dense_output=dense)

def miss(theta):
    """Height above the hoop when the ball reaches x = D."""
    return ball(theta).y_events[0][0][1] - H

low, n_low = secant(miss, np.radians(30), np.radians(35))
high, n_high = secant(miss, np.radians(60), np.radians(65))
print(f"[ball] low arc {np.degrees(low):.2f} deg ({n_low} IVP solves), "
      f"high arc {np.degrees(high):.2f} deg ({n_high} IVP solves)")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
for i, deg in enumerate(range(25, 71, 5)):
    sol = ball(np.radians(deg), dense=True)
    tt = np.linspace(0, sol.t[-1], 200)
    xy = sol.sol(tt)
    ax.plot(xy[0], xy[1], color=MUTED, lw=1.0, alpha=0.55)
for th, c in ((low, V[1]), (high, V[2])):
    sol = ball(th, dense=True)
    tt = np.linspace(0, sol.t[-1], 300)
    xy = sol.sol(tt)
    ax.plot(xy[0], xy[1], color=c, lw=2.6, label=rf"${np.degrees(th):.1f}^\circ$")
ax.plot([D - 0.23, D + 0.23], [H, H], color=ACCENT, lw=4, solid_capstyle="butt")
ax.text(D, H - 0.55, "hoop", ha="center", fontsize=14, color=ACCENT)
ax.set_xlabel(r"$x$ (m)")
ax.set_ylabel(r"$y$ (m)")
ax.set_xlim(0, 7.3)
ax.set_ylim(-0.3, 4.6)
ax.legend(frameon=False, fontsize=13, loc="upper left")
thetas = np.radians(np.linspace(20, 72, 120))
bx.axhline(0, color=MUTED, lw=1)
bx.plot(np.degrees(thetas), [miss(t) for t in thetas], color=V[0])
bx.plot(np.degrees([low, high]), [0, 0], "o", ms=9, color=ACCENT, zorder=5)
bx.set_xlabel(r"launch angle $\theta$ (degrees)")
bx.set_ylabel(r"$F(\theta) = y(D) - H$ (m)")
bx.set_ylim(-3, 1.4)
fig.tight_layout()
save(fig, "basketball")

# single-panel versions for the slides: the trial shots, and the miss against the angle
fig, ax = plt.subplots(figsize=(8.4, 3.9))
for i, deg in enumerate(range(25, 71, 5)):
    sol = ball(np.radians(deg), dense=True)
    tt = np.linspace(0, sol.t[-1], 200)
    xy = sol.sol(tt)
    ax.plot(xy[0], xy[1], color=MUTED, lw=1.0, alpha=0.55)
for th, c in ((low, V[1]), (high, V[2])):
    sol = ball(th, dense=True)
    tt = np.linspace(0, sol.t[-1], 300)
    xy = sol.sol(tt)
    ax.plot(xy[0], xy[1], color=c, lw=2.8, label=rf"${np.degrees(th):.1f}^\circ$")
ax.plot([D - 0.23, D + 0.23], [H, H], color=ACCENT, lw=4, solid_capstyle="butt")
ax.text(D, H - 0.55, "hoop", ha="center", fontsize=14, color=ACCENT)
ax.set_xlabel(r"$x$ (m)")
ax.set_ylabel(r"$y$ (m)")
ax.set_xlim(0, 7.3)
ax.set_ylim(-0.3, 4.6)
ax.legend(frameon=False, fontsize=14, loc="upper left")
fig.tight_layout()
save(fig, "basketball_shots")

fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.axhline(0, color=MUTED, lw=1)
ax.plot(np.degrees(thetas), [miss(t) for t in thetas], color=V[0])
ax.plot(np.degrees([low, high]), [0, 0], "o", ms=9, color=ACCENT, zorder=5)
ax.set_xlabel(r"launch angle $\theta$ (degrees)")
ax.set_ylabel(r"miss (m)")
ax.set_ylim(-3, 1.4)
fig.tight_layout()
save(fig, "basketball_miss")

# ------------------------------------------- 1b. the idea of shooting, schematic
# y'' = -8, y(0) = 0: every trial is y = s x - 4 x^2; the target y(1) = 1/2 needs s = 4.5.
fig, ax = plt.subplots(figsize=(7.6, 3.9))
xx = np.linspace(0, 1, 200)
for s_try in (2.5, 3.5, 5.5, 6.5):
    ax.plot(xx, s_try * xx - 4 * xx**2, color=MUTED, lw=1.4, ls="--")
    ax.text(1.02, s_try - 4, rf"$s = {s_try}$", fontsize=13, color=MUTED, va="center")
ax.plot(xx, 4.5 * xx - 4 * xx**2, color=V[1], lw=3)
ax.text(1.02, 0.5, r"$s = 4.5$", fontsize=13, color=V[1], va="center")
ax.plot([0], [0], "o", ms=10, color=INK, zorder=5)
ax.plot([1], [0.5], "o", ms=12, color=ACCENT, zorder=5)
ax.annotate(r"known: $y(a) = \alpha$", (0, 0), xytext=(0.06, -1.25), fontsize=14,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1))
ax.annotate(r"missing: $y'(a) = s$", (0.06, 0.27), xytext=(0.0, 2.3), fontsize=14,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1))
ax.annotate(r"target: $y(b) = \beta$", (1, 0.5), xytext=(0.55, -1.25), fontsize=14, color=ACCENT,
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
ax.set_xticks([0, 1])
ax.set_xticklabels([r"$a$", r"$b$"])
ax.set_yticks([])
ax.set_xlim(-0.05, 1.18)
ax.set_ylim(-1.7, 2.7)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
fig.tight_layout()
save(fig, "shooting_idea")

# ------------------------------------------- 2. the infinite square well
# -psi''/2 = E psi on [0, 1] (hbar = m = 1), psi(0) = psi(1) = 0; E_n = n^2 pi^2 / 2.
def psi_end(E, dense=False):
    sol = solve_ivp(lambda x, u: [u[1], -2 * E * u[0]], [0, 1], [0, 1],
                    rtol=1e-11, atol=1e-13, dense_output=dense)
    return sol if dense else sol.y[0, -1]

for n, (a, b) in enumerate(((4.0, 5.5), (18.0, 21.0), (42.0, 46.0)), start=1):
    E, n_ev = secant(psi_end, a, b, tol=1e-12)
    print(f"[well] E_{n} = {E:.10f} (exact {n**2 * np.pi**2 / 2:.10f}), {n_ev} IVP solves")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 3.7))
xs = np.linspace(0, 1, 300)
E1 = np.pi**2 / 2
for E, c, lab in ((3.0, V[1], r"$E = 3$"), (E1, V[0], r"$E = \pi^2/2$"), (7.0, V[2], r"$E = 7$")):
    ax.plot(xs, psi_end(E, dense=True).sol(xs)[0], color=c, label=lab)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\psi(x)$")
ax.legend(frameon=False, fontsize=13, loc="upper center", ncol=3)
ax.set_ylim(-0.45, 0.62)
Es = np.linspace(0, 50, 400)
bx.axhline(0, color=MUTED, lw=1)
bx.plot(Es, [psi_end(E) for E in Es], color=V[0])
En = np.array([1, 4, 9]) * np.pi**2 / 2
bx.plot(En, 0 * En, "o", ms=9, color=ACCENT, zorder=5)
for n, (e, dy) in enumerate(zip(En, (-26, 10, -26)), start=1):
    bx.annotate(rf"$E_{n}$", (e, 0), xytext=(-8, dy), textcoords="offset points", ha="right", fontsize=14)
bx.set_xlabel(r"energy $E$")
bx.set_ylabel(r"$\psi(1)$")
fig.tight_layout()
save(fig, "well")

# single-panel versions for the slides
fig, ax = plt.subplots(figsize=(6.6, 3.9))
for E, c, lab in ((3.0, V[1], r"$E = 3$"), (E1, V[0], r"$E = \pi^2/2$"), (7.0, V[2], r"$E = 7$")):
    ax.plot(xs, psi_end(E, dense=True).sol(xs)[0], color=c, label=lab)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\psi(x;\, E)$")
ax.legend(frameon=False, fontsize=13, loc="upper center", ncol=3)
ax.set_ylim(-0.45, 0.62)
fig.tight_layout()
save(fig, "well_psi")

fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.axhline(0, color=MUTED, lw=1)
ax.plot(Es, [psi_end(E) for E in Es], color=V[0])
ax.plot(En, 0 * En, "o", ms=9, color=ACCENT, zorder=5)
for n, (e, dy) in enumerate(zip(En, (-26, 10, -26)), start=1):
    ax.annotate(rf"$E_{n}$", (e, 0), xytext=(-8, dy), textcoords="offset points", ha="right", fontsize=14)
ax.set_xlabel(r"energy $E$")
ax.set_ylabel(r"$F(E) = \psi(1;\, E)$")
fig.tight_layout()
save(fig, "well_F")

# ------------------------------------------- 3. where shooting fails: Debye screening
# phi'' = kappa^2 phi on [0, 1], phi(0) = phi(1) = 1.  Exact: cosh(kappa (x - 1/2)) / cosh(kappa / 2).
def debye_exact(x, kappa):
    return np.cosh(kappa * (x - 0.5)) / np.cosh(kappa / 2)

def debye_shoot(kappa):
    run = lambda s: solve_ivp(lambda x, u: [u[1], kappa**2 * u[0]], [0, 1], [1, s],
                              method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
    F = lambda s: run(s).y[0, -1] - 1
    s, _ = secant(F, 0.0, -1.0)               # linear in s: one secant step is exact
    return run(s), s

xs = np.linspace(0, 1, 801)
kappas = np.arange(2, 52, 2)
shoot_err = []
for kappa in kappas:
    sol, s = debye_shoot(kappa)
    shoot_err.append(np.max(np.abs(sol.sol(xs)[0] / debye_exact(xs, kappa) - 1)))
    if kappa in (10, 20, 30, 40, 50):
        print(f"[debye shoot] kappa {kappa}: max relative error {shoot_err[-1]:.2e}, "
              f"slope found {s:.15f}, exact {-kappa * np.tanh(kappa / 2):.15f}")
shoot_err = np.array(shoot_err)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 3.7))
kappa = 40
sol, _ = debye_shoot(kappa)
ax.semilogy(xs, debye_exact(xs, kappa), color=MUTED, lw=4, alpha=0.5, label="exact")
ax.semilogy(xs, np.abs(sol.sol(xs)[0]), color=V[1], lw=1.8, label="shooting")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$|\phi(x)|$")
ax.set_title(r"$\kappa = 40$", fontsize=15)
ax.legend(frameon=False, fontsize=13, loc="upper center")
bx.semilogy(kappas, shoot_err, "o-", color=V[1], ms=5)
bx.axhline(1, color=ACCENT, lw=1.2, ls="--")
bx.set_xlabel(r"$\kappa$")
bx.set_ylabel("max relative error")
fig.tight_layout()
save(fig, "debye_shoot")

fig, ax = plt.subplots(figsize=(7.2, 3.9))
kappa = 50
sol, _ = debye_shoot(kappa)
ax.semilogy(xs, debye_exact(xs, kappa), color=MUTED, lw=4, alpha=0.5, label="exact")
ax.semilogy(xs, np.abs(sol.sol(xs)[0]), color=V[1], lw=1.8, label="shooting")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$|\phi(x)|$")
ax.legend(frameon=False, fontsize=14, loc="upper center")
fig.tight_layout()
save(fig, "debye_shoot_single")

# ============================================================ Part II
# ------------------------------------------- 4. finite-difference errors
x0 = 1.0
hs = np.logspace(-1, -13, 121)
fwd = np.abs((np.sin(x0 + hs) - np.sin(x0)) / hs - np.cos(x0))
cen = np.abs((np.sin(x0 + hs) - np.sin(x0 - hs)) / (2 * hs) - np.cos(x0))
print(f"[fd] best forward error {fwd.min():.1e} at h = {hs[fwd.argmin()]:.0e}; "
      f"best central error {cen.min():.1e} at h = {hs[cen.argmin()]:.0e}")
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.loglog(hs, fwd, color=V[1], label="forward")
ax.loglog(hs, cen, color=V[2], label="central")
ax.set_xlabel(r"$h$")
ax.set_ylabel(r"error in $(\sin x)'$ at $x = 1$")
ax.invert_xaxis()
ax.legend(frameon=False, fontsize=13, loc="lower left")
fig.tight_layout()
save(fig, "fd_error")

# ------------------------------------------- 5. the Debye problem on a grid
def laplacian_1d(n_interior, h):
    """(y_{i+1} - 2 y_i + y_{i-1}) / h^2 on the interior points, as a sparse matrix."""
    main = -2.0 * np.ones(n_interior)
    off = np.ones(n_interior - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csc") / h**2

def debye_grid(kappa, N):
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    A = laplacian_1d(N - 1, h) - kappa**2 * sp.identity(N - 1)
    b = np.zeros(N - 1)
    b[0] -= 1.0 / h**2                         # phi_0 = 1 moved to the right-hand side
    b[-1] -= 1.0 / h**2                        # phi_N = 1
    phi = np.empty(N + 1)
    phi[0] = phi[-1] = 1.0
    phi[1:-1] = spla.spsolve(A, b)
    return x, phi

Ns = 2 ** np.arange(5, 15)
fig, ax = plt.subplots(figsize=(6.6, 3.9))
for kappa, c in ((10, V[2]), (30, V[1]), (50, V[0])):
    errs = []
    for N in Ns:
        x, phi = debye_grid(kappa, N)
        errs.append(np.max(np.abs(phi / debye_exact(x, kappa) - 1)))
    errs = np.array(errs)
    slope = np.log(errs[-2] / errs[-1]) / np.log(2)
    print(f"[debye grid] kappa {kappa}: N = {Ns[-1]} max relative error {errs[-1]:.1e}, slope {slope:.2f}; "
          f"N = 1024: {errs[list(Ns).index(1024)]:.1e}")
    ax.loglog(Ns, errs, "o-", color=c, ms=5, label=rf"$\kappa = {kappa}$")
ax.set_xlabel(r"number of grid intervals $N$")
ax.set_ylabel("max relative error")
ax.legend(frameon=False, fontsize=13)
fig.tight_layout()
save(fig, "debye_grid")

# ------------------------------------------- 6. nonlinear: phi'' = kappa^2 sinh(phi), Newton
kappa, phi_b, N = 10.0, 5.0, 200
h = 1.0 / N
x = np.linspace(0, 1, N + 1)
L = laplacian_1d(N - 1, h)
bc = np.zeros(N - 1)
bc[0] = bc[-1] = phi_b / h**2                  # boundary values enter F_1 and F_{N-1}
phi = np.zeros(N - 1)                          # initial guess: zero inside
iterates = [np.concatenate(([phi_b], phi, [phi_b]))]
print("[newton] iteration, max |F|, max |delta|")
for it in range(1, 30):
    F = L @ phi + bc - kappa**2 * np.sinh(phi)
    J = L - sp.diags(kappa**2 * np.cosh(phi), format="csc")
    delta = spla.spsolve(J, -F)
    phi = phi + delta
    iterates.append(np.concatenate(([phi_b], phi, [phi_b])))
    print(f"  {it:2d}  {np.max(np.abs(F)):.2e}  {np.max(np.abs(delta)):.2e}")
    if np.max(np.abs(delta)) < 1e-12:
        break
n_newton = it

fig, ax = plt.subplots(figsize=(6.6, 3.9))
cmap = plt.get_cmap("viridis")
for i, p in enumerate(iterates[:8]):
    ax.plot(x, p, color=cmap(0.9 * i / 7), lw=1.6, label=rf"$k = {i}$" if i in (0, 1, 2, 7) else None)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\phi^{(k)}(x)$")
ax.legend(frameon=False, fontsize=12, loc="upper center", ncol=4)
ax.set_ylim(-0.3, 6.2)
fig.tight_layout()
save(fig, "newton")
print(f"[newton] converged after {n_newton} iterations; phi(1/2) = {phi[N // 2 - 1]:.6e}")

# ------------------------------------------- 7. what the matrices look like
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 4.2))
A1 = laplacian_1d(12, 1.0).toarray()
ax.spy(A1, markersize=7, color=V[1])
ax.set_title(r"1D: 12 unknowns", fontsize=15)
n = 6
I = sp.identity(n)
T = sp.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1])
A2 = (sp.kron(I, T) + sp.kron(T, I)).toarray()
bx.spy(A2, markersize=3.2, color=V[0])
bx.set_title(r"2D: $6 \times 6$ grid, 36 unknowns", fontsize=15)
for a in (ax, bx):
    a.set_xticks([])
    a.set_yticks([])
    for side in a.spines.values():
        side.set_visible(False)
fig.tight_layout()
save(fig, "sparsity")
print(f"[sparsity] 2D 6x6: {np.count_nonzero(A2)} nonzeros of {A2.size}")
