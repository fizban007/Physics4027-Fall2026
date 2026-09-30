#!/usr/bin/env python3
"""Figures for Lecture 11 (Boris pusher, orbit integration, molecular dynamics,
cosmology). Run from this directory; writes figures/*.png and prints the
numbers quoted on the slides.

Needs numpy, scipy, matplotlib; the cosmology figure also needs camb
(pip install camb). The Gauss-Jackson comparison imports GJ8.py from a local
clone of https://github.com/lorcan2440/Gauss-Jackson-Integrator (python_solver/),
located through the GJ8_DIR environment variable; it is not copied into this
repository because the port carries no license of its own. Sections whose
dependency is missing are skipped with a message.

    uv run --with numpy --with scipy --with matplotlib --with camb python make_figures.py
"""
import os
import sys
import time
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

rng = np.random.default_rng(2026)

# ============================================================ Part I: Boris
def boris_push(u, bvec_half):
    """Rotate u about B; bvec_half = (q/m) B dt/2. Exact |u| conservation."""
    t = bvec_half
    s = 2 * t / (1 + t @ t)
    u_prime = u + np.cross(u, t)
    return u + np.cross(u_prime, s)

# ------------------------------------------- 1. uniform B: RK4 vs Boris
W_DT = 0.3                        # omega_c * dt: about 21 steps per gyration
TURNS = 1000
steps_per_turn = 2 * np.pi / W_DT
n_steps = int(TURNS * steps_per_turn)
bz = np.array([0.0, 0.0, 1.0])    # units: omega_c = 1, speed 1, radius 1

def lorentz(state):               # d/dt (x, v) with q/m B = z-hat, E = 0
    return np.concatenate((state[3:], np.cross(state[3:], bz)))

def rk4(state, h):
    k1 = lorentz(state); k2 = lorentz(state + h / 2 * k1)
    k3 = lorentz(state + h / 2 * k2); k4 = lorentz(state + h * k3)
    return state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# gyrocenter at the origin: x = (1, 0), v = (0, -1) circles clockwise? For
# q/m B = +z the motion x' = v, v' = v x z; start at (1, 0) with v = (0, -1).
s_rk = np.array([1.0, 0, 0, 0, -1.0, 0]); x_b = np.array([1.0, 0, 0]); v_b = np.array([0, -1.0, 0])
v_b = boris_push(v_b, -bz * W_DT / 4)   # leapfrog: stagger v back half a step (rotation by -dt/2)
rk_path, b_path, rk_speed, b_speed = [], [], [], []
for i in range(n_steps):
    s_rk = rk4(s_rk, W_DT)
    v_b = boris_push(v_b, bz * W_DT / 2)
    x_b = x_b + v_b * W_DT
    if i > n_steps - 3 * steps_per_turn:
        rk_path.append(s_rk[:2].copy()); b_path.append(x_b[:2].copy())
    if i % 20 == 0:
        rk_speed.append(np.linalg.norm(s_rk[3:])); b_speed.append(np.linalg.norm(v_b))
rk_path, b_path = np.array(rk_path), np.array(b_path)
print(f"uniform B, omega_c dt = {W_DT}, {TURNS} gyrations ({n_steps} steps):")
print(f"  |v| after: RK4 {rk_speed[-1]:.4f}, Boris {b_speed[-1]:.16f}")
print(f"  RK4 energy factor per step 1 - (w dt)^6/72 = {1 - W_DT**6 / 72:.7f}; "
      f"predicted |v|^2 after: {(1 - W_DT**6 / 72) ** n_steps:.4f}, measured {rk_speed[-1]**2:.4f}")
print(f"  Boris rotation per step 2 atan(w dt/2) = {2 * np.arctan(W_DT / 2):.6f} vs w dt = {W_DT}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1, 1.35]})
th = np.linspace(0, 2 * np.pi, 300)
a1.plot(np.cos(th), np.sin(th), color=MUTED, lw=1.0, ls="--", label="exact orbit")
a1.plot(rk_path[:, 0], rk_path[:, 1], color=V[2], lw=1.8, label="RK4")
a1.plot(b_path[:, 0], b_path[:, 1], color=V[0], lw=1.2, label="Boris")
a1.set_aspect("equal"); a1.set_xlim(-1.25, 1.25); a1.set_ylim(-1.25, 1.25)
a1.set_title(f"after {TURNS} gyrations", fontsize=15)
a1.legend(frameon=False, fontsize=12, loc="center")
tt = np.arange(len(rk_speed)) * 20 / steps_per_turn
a2.plot(tt, np.array(rk_speed) ** 2, color=V[2], label="RK4")
a2.plot(tt, np.array(b_speed) ** 2, color=V[0], label="Boris")
a2.set_xlabel("gyrations"); a2.set_ylabel(r"kinetic energy $/\,E_0$")
a2.legend(frameon=False, fontsize=12, loc="lower left")
save(fig, "boris_uniform")

# ------------------------------------------- 2. a proton in the Van Allen belt
RE = 6.371e6; B0 = 3.07e-5                         # Earth radius; equatorial surface field
Q = 1.602176634e-19; M = 1.67262192e-27; C = 2.99792458e8
EK = 10e6 * Q                                       # 10 MeV proton
GAM = 1 + EK / (M * C * C)
VEL = C * np.sqrt(1 - 1 / GAM ** 2)
QM = Q / (GAM * M)                                  # relativistic: gamma is constant in pure B

def dipole(r):
    x, y, z = r
    rr2 = x * x + y * y + z * z
    f = -B0 * RE ** 3 / rr2 ** 2.5                  # Earth's dipole points along -z
    return f * np.array([3 * x * z, 3 * y * z, 2 * z * z - x * x - y * y])

L_SHELL, PITCH = 4.0, np.radians(30.0)
r = np.array([L_SHELL * RE, 0.0, 0.0])
b_eq = np.linalg.norm(dipole(r))
w_c = QM * b_eq
T_GYRO = 2 * np.pi / w_c
u = VEL * np.array([0.0, np.sin(PITCH), np.cos(PITCH)])
DT = T_GYRO / 30
T_RUN = 75.0
n = int(T_RUN / DT)
path = np.empty((n, 3)); mu = np.empty(n)
t0 = time.time()
for i in range(n):
    bh = QM * dipole(r) * DT / 2
    u = boris_push(u, bh)
    r = r + u * DT
    path[i] = r
    bmag = np.linalg.norm(dipole(r)); upar = u @ dipole(r) / bmag
    mu[i] = (u @ u - upar ** 2) / bmag               # proportional to the magnetic moment
wall = time.time() - t0
t_axis = (np.arange(n) + 1) * DT
z = path[:, 2]
ups = t_axis[1:][(z[:-1] < 0) & (z[1:] >= 0)]
phi = np.unwrap(np.arctan2(path[:, 1], path[:, 0]))
drift_rate = (phi[-1] - phi[0]) / (t_axis[-1] - t_axis[0])
lat_max = np.degrees(np.max(np.arcsin(np.abs(z) / np.linalg.norm(path, axis=1))))
print(f"10 MeV proton at L = {L_SHELL}, pitch 30 deg, Boris with dt = T_gyro/30:")
print(f"  gamma = {GAM:.4f}, v/c = {VEL / C:.3f}, B_eq = {b_eq:.2e} T")
print(f"  gyration: period {T_GYRO:.3f} s, radius {VEL * np.sin(PITCH) / w_c / 1e3:.0f} km")
print(f"  bounce: period {np.diff(ups).mean():.2f} s, mirror latitude {lat_max:.1f} deg")
print(f"  drift: period {2 * np.pi / abs(drift_rate) / 60:.2f} min, "
      f"{'westward' if drift_rate < 0 else 'eastward'}")
print(f"  {n} steps in {wall:.1f} s; |u| conserved to {abs(np.linalg.norm(u) - VEL) / VEL:.1e}; "
      f"magnetic moment varies by {np.ptp(mu[int(5 / DT):]) / mu.mean():.1%} (peak to peak)")
bounce_theory = 4 * L_SHELL * RE / VEL * (1.30 - 0.56 * np.sin(PITCH))
print(f"  textbook bounce estimate 4 L R_E / v (1.30 - 0.56 sin alpha) = {bounce_theory:.2f} s")

# RK4 on the same orbit, same step: energy drift
def f_rk(s):
    return np.concatenate((s[3:], QM * np.cross(s[3:], dipole(s[:3]))))
s = np.concatenate(([L_SHELL * RE, 0.0, 0.0], VEL * np.array([0.0, np.sin(PITCH), np.cos(PITCH)])))
for _ in range(n):
    k1 = f_rk(s); k2 = f_rk(s + DT / 2 * k1); k3 = f_rk(s + DT / 2 * k2); k4 = f_rk(s + DT * k3)
    s = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
print(f"  RK4, same step, same 75 s: kinetic energy changed by "
      f"{(np.linalg.norm(s[3:]) ** 2 / VEL ** 2 - 1):.1e} (relative)")

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
fig = plt.figure(figsize=(11.0, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
ax = fig.add_subplot(gs[0], projection="3d")
uu, vv = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
ax.plot_surface(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv),
                color=V[1], alpha=0.35, linewidth=0)
p = path / RE
ax.plot(p[:, 0], p[:, 1], p[:, 2], color=V[0], lw=0.2, alpha=0.8)
ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_zlim(-2.5, 2.5)
ax.set_box_aspect((9, 9, 5), zoom=1.35); ax.view_init(elev=22, azim=-60)
ax.set_axis_off()
ax.set_title("75 s: one drift around the Earth", fontsize=14)
a2 = fig.add_subplot(gs[1])
k = int(6.0 / DT)
rho = np.hypot(path[:k, 0], path[:k, 1]) / RE
a2.plot(rho, path[:k, 2] / RE, color=V[0], lw=0.8)
lat = np.linspace(-np.radians(35), np.radians(35), 200)
a2.plot(L_SHELL * np.cos(lat) ** 3, L_SHELL * np.cos(lat) ** 2 * np.sin(lat), color=MUTED,
        ls="--", lw=1.0, label=r"field line, $L = 4$")
a2.set_xlabel(r"distance from axis ($R_E$)"); a2.set_ylabel(r"$z$ ($R_E$)")
a2.set_title("first 6 s: gyration and bounce", fontsize=14)
a2.legend(frameon=False, fontsize=12, loc="center left")
a2.set_aspect("equal")
save(fig, "van_allen")

# ============================================================ Part II: orbits
GM_E = 398600.4415                                   # km^3/s^2
R_ORB = 7000.0                                       # km, low Earth orbit
W_ORB = np.sqrt(GM_E / R_ORB ** 3)
T_ORB = 2 * np.pi / W_ORB
N_ORB = 100

def circ_exact(t):
    return np.stack((R_ORB * np.cos(W_ORB * t), R_ORB * np.sin(W_ORB * t), 0 * t), axis=-1)

gj8_dir = os.environ.get("GJ8_DIR", os.path.expanduser(
    "~/Academic/Courses/Physics 427 Fall 2025/Lectures/Lecture 11/Gauss-Jackson-Integrator/python_solver"))
try:
    sys.path.insert(0, gj8_dir)
    from GJ8 import gauss_jackson_8
    from scipy.integrate import solve_ivp
    n_eval = [0]
    def acc(t, y, dy):
        n_eval[0] += 1
        return -GM_E * y / np.linalg.norm(y) ** 3
    dt_gj = 60.0
    t_gj, y_gj, _, _ = gauss_jackson_8(acc, np.array([0.0, N_ORB * T_ORB]),
                                       np.array([R_ORB, 0, 0.0]), np.array([0, R_ORB * W_ORB, 0.0]), dt_gj)
    evals_gj = n_eval[0]
    err_gj = np.linalg.norm(y_gj - circ_exact(t_gj), axis=1) * 1e6          # mm

    f1 = lambda t, s: np.hstack((s[3:], -GM_E * s[:3] / np.linalg.norm(s[:3]) ** 3))
    s0 = np.array([R_ORB, 0, 0, 0, R_ORB * W_ORB, 0.0])
    n_rk = evals_gj // 4; h = N_ORB * T_ORB / n_rk
    s = s0.copy(); t_rk = [0.0]; err_rk = [0.0]
    for i in range(n_rk):
        k1 = f1(0, s); k2 = f1(0, s + h / 2 * k1); k3 = f1(0, s + h / 2 * k2); k4 = f1(0, s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t_rk.append((i + 1) * h); err_rk.append(np.linalg.norm(s[:3] - circ_exact((i + 1) * h)) * 1e6)
    print(f"circular orbit at {R_ORB:.0f} km (period {T_ORB / 60:.1f} min), {N_ORB} orbits:")
    print(f"  GJ8, dt = {dt_gj:.0f} s: {evals_gj} evaluations, final error {err_gj[-1]:.2f} mm")
    print(f"  RK4 with the same evaluations (h = {h:.0f} s): final error {err_rk[-1] / 1e6:.0f} km")
    dop = {}
    for tol in (1e-9, 1e-10, 1e-12, 1e-13):
        sol = solve_ivp(f1, (0, N_ORB * T_ORB), s0, method="DOP853", rtol=tol, atol=tol * 1e-3,
                        dense_output=True)
        dop[tol] = sol
        e = np.linalg.norm(sol.y[:3, -1] - circ_exact(N_ORB * T_ORB)) * 1e6
        print(f"  DOP853 (adaptive, 8th order) rtol {tol:g}: {sol.nfev} evaluations, final error {e:.2f} mm")
    sol = dop[1e-10]
    t_d = np.linspace(0, N_ORB * T_ORB, 4000)
    err_d = np.linalg.norm(sol.sol(t_d)[:3].T - circ_exact(t_d), axis=1) * 1e6

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.semilogy(np.array(t_rk) / T_ORB, np.array(err_rk) + 1e-9, color=V[2], label=f"RK4, {evals_gj // 1000}k evaluations")
    ax.semilogy(t_d / T_ORB, err_d + 1e-9, color=V[1], label=f"DOP853, {sol.nfev // 1000}k evaluations")
    ax.semilogy(t_gj / T_ORB, err_gj + 1e-9, color=V[0], label=f"Gauss-Jackson 8, {evals_gj // 1000}k evaluations")
    ax.axhline(1e6, color=MUTED, lw=0.8, ls=":"); ax.text(101, 1e6, "1 km", color=MUTED, fontsize=12, va="center")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":"); ax.text(101, 1.0, "1 mm", color=MUTED, fontsize=12, va="center")
    ax.set_ylim(1e-4, 1e11)
    ax.set_xlabel("time (orbits)"); ax.set_ylabel("position error (mm)")
    ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=1)
    save(fig, "gj8_compare")
except ImportError as exc:
    print(f"skipping the Gauss-Jackson comparison: {exc}")

# ============================================================ Part III: molecular dynamics
# Lennard-Jones in reduced units (sigma = epsilon = m = 1), 2D, periodic box.
R_CUT = 2.5

def lj_forces(x, box):
    d = x[:, None, :] - x[None, :, :]
    d -= box * np.round(d / box)                      # minimum image
    r2 = (d ** 2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    inside = r2 < R_CUT ** 2
    inv6 = np.where(inside, 1.0 / r2 ** 3, 0.0)
    fmag = np.where(inside, 24 * inv6 * (2 * inv6 - 1) / r2, 0.0)   # |F|/r
    forces = (fmag[:, :, None] * d).sum(1)
    shift = 4 * (R_CUT ** -12 - R_CUT ** -6)
    pot = 0.5 * np.where(inside, 4 * inv6 * (inv6 - 1) - shift, 0.0).sum()
    return forces, pot

def verlet(x, v, box, dt, n, record=None):
    f, _ = lj_forces(x, box)
    for i in range(n):
        v = v + dt / 2 * f
        x = (x + dt * v) % box
        f, _ = lj_forces(x, box)
        v = v + dt / 2 * f
        if record is not None:
            record(i, x, v)
    return x, v

def square_lattice(n_side, box):
    g = (np.arange(n_side) + 0.5) * box / n_side
    return np.array([(a, b) for a in g for b in g])

# ------------------------------------------- 5. speeds relax to Maxwell-Boltzmann
N_SIDE = 20; N_MD = N_SIDE ** 2; RHO = 0.30
box = np.sqrt(N_MD / RHO)
x = square_lattice(N_SIDE, box) + rng.normal(0, 0.05, (N_MD, 2))
ang = rng.uniform(0, 2 * np.pi, N_MD)
V0 = 1.5
v = V0 * np.c_[np.cos(ang), np.sin(ang)]
v -= v.mean(0)                                        # no net momentum
DT_MD = 0.005
snap = {}
energies = []
pooled = []
def rec(i, x, v):
    if (i + 1) in (40, 400, 4000):
        snap[i + 1] = np.linalg.norm(v, axis=1)
    if i >= 3000 and i % 100 == 0:                 # pool late snapshots for a smooth histogram
        pooled.append(np.linalg.norm(v, axis=1))
    if i % 50 == 0:
        energies.append(0.5 * (v ** 2).sum() + lj_forces(x, box)[1])
t0 = time.time()
x, v = verlet(x, v, box, DT_MD, 4000, rec)
kT = (v ** 2).sum() / (2 * N_MD)                       # 2D equipartition: <v^2> = 2 kT
print(f"MD: {N_MD} Lennard-Jones atoms in 2D, density {RHO}, velocity Verlet dt = {DT_MD}, "
      f"4000 steps in {time.time() - t0:.0f} s")
print(f"  all speeds start at {V0}; final kT = {kT:.3f}; total energy varies by "
      f"{np.ptp(energies) / abs(np.mean(energies)):.1e} (relative)")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
vs = np.linspace(0, 4.5, 300)
bins = np.linspace(0, 4.5, 31)
ax.axvline(V0, color=V[3], lw=3, label=r"$t = 0$: every speed $= 1.5$")
ax.hist(snap[40], bins=bins, density=True, histtype="step", color=V[2], lw=1.8,
        label=f"$t = {40 * DT_MD:g}$")
ax.hist(np.concatenate(pooled), bins=bins, density=True, color=V[1], alpha=0.55,
        label=r"$t = 15$ to $20$")
ax.plot(vs, vs / kT * np.exp(-vs ** 2 / (2 * kT)), color=V[0], lw=2.2,
        label=r"Maxwell-Boltzmann, $\frac{v}{kT}\,e^{-v^2/2kT}$")
ax.set_xlabel("speed"); ax.set_ylabel("distribution")
ax.legend(frameon=False, fontsize=11, loc="upper right")
save(fig, "md_maxwell")

# ------------------------------------------- 6. running the gas backwards (Loschmidt)
N_SIDE_L = 10; N_L = N_SIDE_L ** 2
box_l = np.sqrt(N_L / RHO)
x0 = square_lattice(N_SIDE_L, box_l) + rng.normal(0, 0.05, (N_L, 2))
ang = rng.uniform(0, 2 * np.pi, N_L)
v0 = V0 * np.c_[np.cos(ang), np.sin(ang)]; v0 -= v0.mean(0)
t_rev = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
miss = []
for T_R in t_rev:
    nstep = int(round(T_R / DT_MD))
    x1, v1 = verlet(x0.copy(), v0.copy(), box_l, DT_MD, nstep)
    x2, v2 = verlet(x1, -v1, box_l, DT_MD, nstep)
    d = x2 - x0; d -= box_l * np.round(d / box_l)
    miss.append(np.sqrt((d ** 2).sum(1).mean()))
print("Loschmidt: run forward for T, reverse all velocities, run T again; rms distance from start:")
print("  " + "  ".join(f"T={a:g}:{b:.0e}" for a, b in zip(t_rev, miss)))
lg = np.log(np.array(miss)); good = (np.array(miss) > 1e-13) & (np.array(miss) < 1e-2)
rate = np.polyfit(np.array(t_rev)[good], lg[good], 1)[0]
print(f"  exponential growth rate {rate:.2f} per unit time: error x e every {1 / rate:.2f}")
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.semilogy(t_rev, miss, "o-", color=V[0], ms=6)
ax.axhline(1.0, color=MUTED, lw=0.8, ls=":"); ax.text(0.8, 1.5, r"atom size $\sigma$", color=MUTED, fontsize=12)
ax.axhline(1e-16 * box_l, color=MUTED, lw=0.8, ls=":")
ax.text(0.8, 2.5e-16 * box_l, "rounding", color=MUTED, fontsize=12)
ax.set_xlabel(r"time $T$ before reversing"); ax.set_ylabel("distance from the start")
save(fig, "md_loschmidt")

# ------------------------------------------- 7. cooling into a crystal
N_SIDE_C = 16; N_C = N_SIDE_C ** 2; RHO_C = 0.90
box_c = np.sqrt(N_C / RHO_C)
x = rng.uniform(0, box_c, (N_C, 2))
# push overlapping random atoms apart with a few tiny, damped steps
for _ in range(400):
    f, _ = lj_forces(x, box_c)
    step = np.clip(1e-4 * f, -0.05, 0.05)
    x = (x + step) % box_c
v = rng.normal(0, np.sqrt(1.0), (N_C, 2)); v -= v.mean(0)
def cool(target, n):
    global x, v
    for _ in range(n // 50):
        x, v = verlet(x, v, box_c, DT_MD, 50)
        kt = (v ** 2).sum() / (2 * N_C)
        v *= np.sqrt(target / kt)                      # rescale to the target temperature
t0 = time.time()
x_hot_snap = None
cool(2.5, 1000); x_hot = x.copy()
for target in np.linspace(2.5, 0.05, 25):
    cool(target, 400)
cool(0.05, 2000)
print(f"crystallization: {N_C} atoms at density {RHO_C}, cooled from kT = 2.5 to 0.05 "
      f"in {time.time() - t0:.0f} s")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.2))
for a, pts, ttl in ((a1, x_hot, r"hot, $kT = 2.5$: fluid"), (a2, x, r"cooled, $kT = 0.05$: crystal")):
    a.scatter(pts[:, 0], pts[:, 1], s=26, color=V[1], edgecolor=V[0], lw=0.5)
    a.set_xlim(0, box_c); a.set_ylim(0, box_c); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=14)
save(fig, "md_crystal")

# ============================================================ Part IV: cosmology
try:
    import camb
    t0 = time.time()
    pars = camb.set_params(H0=67.4, ombh2=0.0224, omch2=0.120, As=2.1e-9, ns=0.965,
                           tau=0.054, lmax=2500)
    res = camb.get_results(pars)
    dl = res.get_cmb_power_spectra(pars, CMB_unit="muK")["total"][:, 0]
    wall = time.time() - t0
    ell = np.arange(len(dl))
    peaks = [i for i in range(150, 1800) if dl[i] > dl[i - 1] and dl[i] > dl[i + 1]]
    der = res.get_derived_params()
    print(f"CAMB {camb.__version__}: TT spectrum to l = 2500 in {wall:.1f} s; peaks at l = {peaks[:4]}")
    print(f"  z_* = {der['zstar']:.0f}, sound horizon r_s = {der['rstar']:.1f} Mpc, "
          f"100 theta_* = {der['thetastar']:.4f}")
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    m = ell >= 2
    ax.plot(ell[m], dl[m], color=V[0], lw=1.8)
    for pk in peaks[:3]:
        ax.annotate(f"$\\ell = {pk}$", (pk, dl[pk]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=12)
    ax.set_xlim(2, 2500); ax.set_ylim(0, 6500)
    ax.set_xlabel(r"multipole $\ell$"); ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$ ($\mu$K$^2$)")
    save(fig, "cmb_tt")
except ImportError as exc:
    print(f"skipping the CMB figure: {exc}")
