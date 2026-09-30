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
# mplot3d depth-sorts whole artists, not pixels, so a single line would be
# drawn entirely in front of the Earth. Split the path by which side of the
# Earth's center it is on, as seen from the camera, and draw far half,
# Earth, near half in that order.
ELEV, AZIM = 22, -60
ax.computed_zorder = False
eye = np.array([np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)),
                np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM)),
                np.sin(np.radians(ELEV))])
p = path / RE
near = p @ eye > 0
for side, z_order, alpha in ((~near, 1, 0.55), (near, 3, 0.9)):
    q = p.copy(); q[~side] = np.nan                   # NaN breaks the line
    ax.plot(q[:, 0], q[:, 1], q[:, 2], color=V[0], lw=0.2, alpha=alpha, zorder=z_order)
uu, vv = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
ax.plot_surface(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv),
                color=V[1], alpha=0.9, linewidth=0, shade=True, zorder=2)
ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_zlim(-2.5, 2.5)
ax.set_box_aspect((9, 9, 5), zoom=1.35); ax.view_init(elev=ELEV, azim=AZIM)
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
from matplotlib.colors import LinearSegmentedColormap
# Kinetic-energy colours for a light background: cold atoms pale and
# recessive, hot atoms saturated and dark, so the energetic ones stand out.
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#b8c4d0", "#f4a261", "#e63946", "#6a040f"])

def hot_on_top(pts, ke, vmax, s_cold, s_hot):
    """Sort atoms so the hottest are drawn last, and size them by energy."""
    order = np.argsort(ke)
    size = s_cold + (s_hot - s_cold) * np.clip(ke[order] / vmax, 0, 1)
    return pts[order], ke[order], size

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

# ------------------------------------------- 5. cooling into a crystal (animation)
from matplotlib.animation import FFMpegWriter
DT_MD = 0.005
N_SIDE_C = 16; N_C = N_SIDE_C ** 2; RHO_C = 0.90
box_c = np.sqrt(N_C / RHO_C)
x = rng.uniform(0, box_c, (N_C, 2))
for _ in range(400):                                   # push overlapping random atoms apart
    f, _ = lj_forces(x, box_c)
    x = (x + np.clip(1e-4 * f, -0.05, 0.05)) % box_c
v = rng.normal(0, 1.0, (N_C, 2)); v -= v.mean(0)

KT_HOT, KT_COLD = 2.5, 0.05
RESCALE = 50                                           # steps between velocity rescalings
FRAME = 20                                             # steps between frames
N_EQUIL, N_COOL, N_HOLD = 1000, 12000, 1500

def temperature(v):
    return (v ** 2).sum() / (2 * len(v))               # 2D equipartition

def rescale(v, target):
    return v * np.sqrt(target / temperature(v))

def psi6(x, box, r_nb=1.5):
    """Global bond-orientational order: |<exp(6 i theta)>| over neighbor bonds.
    About 0 in a disordered fluid, near 1 in a single triangular crystal."""
    d = x[:, None, :] - x[None, :, :]
    d -= box * np.round(d / box)
    r2 = (d ** 2).sum(-1); np.fill_diagonal(r2, np.inf)
    nb = r2 < r_nb ** 2
    theta = np.arctan2(d[..., 1], d[..., 0])
    per_atom = np.where(nb, np.exp(6j * theta), 0).sum(1) / np.maximum(nb.sum(1), 1)
    return abs(per_atom.mean())

# equilibrate hot (not filmed)
for _ in range(N_EQUIL // RESCALE):
    x, v = verlet(x, v, box_c, DT_MD, RESCALE)
    v = rescale(v, KT_HOT)

# film the cooling and the hold, recording every FRAME steps
frames_x, frames_ke, frames_t, frames_kt, targets, frames_psi = [], [], [], [], [], []
t0 = time.time()
total = N_COOL + N_HOLD
f, _ = lj_forces(x, box_c)
for i in range(total):
    v = v + DT_MD / 2 * f
    x = (x + DT_MD * v) % box_c
    f, _ = lj_forces(x, box_c)
    v = v + DT_MD / 2 * f
    target = KT_HOT + (KT_COLD - KT_HOT) * min(i / N_COOL, 1.0)
    if (i + 1) % RESCALE == 0:
        v = rescale(v, target)
    if i % FRAME == 0:
        frames_x.append(x.copy()); frames_ke.append(0.5 * (v ** 2).sum(1))
        frames_t.append(i * DT_MD); frames_kt.append(temperature(v)); targets.append(target)
        frames_psi.append(psi6(x, box_c))
print(f"cooling animation: {N_C} atoms at density {RHO_C}, kT {KT_HOT} -> {KT_COLD} over "
      f"{N_COOL} steps, then {N_HOLD} held; {len(frames_x)} frames; MD took {time.time() - t0:.0f} s")

with plt.rc_context({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm"}):
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_axes([0.01, 0.07, 0.47, 0.90])
    ax.set_xlim(0, box_c); ax.set_ylim(0, box_c); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(MUTED)
    sc = ax.scatter(frames_x[0][:, 0], frames_x[0][:, 1], s=70, c=frames_ke[0],
                    cmap=HEAT, vmin=0, vmax=4.0, edgecolor=INK, lw=0.3)
    at = fig.add_axes([0.62, 0.18, 0.27, 0.62])
    tt = np.array(frames_t)
    at.plot(tt, targets, color=MUTED, lw=1.0, ls="--", label="target")
    kt_line, = at.plot([], [], color=V[0], lw=1.6, label="measured")
    dot, = at.plot([], [], "o", color=V[0], ms=6)
    at.set_xlim(0, tt[-1]); at.set_ylim(0, 2.9)
    at.set_xlabel("time"); at.set_ylabel(r"temperature $kT$")
    at.spines["top"].set_visible(False); at.spines["right"].set_visible(False)
    at.legend(frameon=False, fontsize=11, loc="upper right")
    ap = fig.add_axes([0.62, 0.18, 0.27, 0.62], sharex=at, frameon=False)
    ap.yaxis.tick_right(); ap.yaxis.set_label_position("right")
    ap.set_ylim(0, 1.05); ap.set_ylabel(r"crystal order $\psi_6$", color=V[2])
    ap.tick_params(axis="y", colors=V[2]); ap.tick_params(axis="x", bottom=False, labelbottom=False)
    psi_line, = ap.plot([], [], color=V[2], lw=1.6)
    label = fig.text(0.62, 0.86, "", fontsize=16, color=INK)
    fig.text(0.245, 0.02, "colour: kinetic energy of each atom (dark red = hot)", ha="center", fontsize=11, color=MUTED)

    def draw(k):
        pts, ke, size = hot_on_top(frames_x[k], frames_ke[k], 4.0, 60, 95)
        sc.set_offsets(pts); sc.set_array(ke); sc.set_sizes(size)
        kt_line.set_data(tt[:k + 1], frames_kt[:k + 1]); dot.set_data([tt[k]], [frames_kt[k]])
        psi_line.set_data(tt[:k + 1], frames_psi[:k + 1])
        label.set_text(f"$kT = {frames_kt[k]:.2f}$     $\\psi_6 = {frames_psi[k]:.2f}$")

    t0 = time.time()
    writer = FFMpegWriter(fps=30, codec="libx264", bitrate=2400,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, "figures/md_cooling.mp4", dpi=100):
        for k in range(len(frames_x)):
            draw(k); writer.grab_frame()
    draw(len(frames_x) - 1)
    fig.savefig("figures/md_cooling_poster.png", dpi=100, facecolor=PAPER)
    plt.close(fig)
psi = np.array(frames_psi); kt_arr = np.array(frames_kt)
order_onset = kt_arr[np.argmax(psi > 0.5)]
print(f"  psi6: {psi[:25].mean():.2f} while hot, {psi[-25:].mean():.2f} at the end; "
      f"first exceeds 0.5 at kT = {order_onset:.2f}")
print(f"  wrote figures/md_cooling.mp4 ({os.path.getsize('figures/md_cooling.mp4') / 1e6:.1f} MB, "
      f"{len(frames_x) / 30:.0f} s at 30 fps) in {time.time() - t0:.0f} s")

# ------------------------------------------- 6. a meteorite hits a crystal (animation)
from scipy.spatial import cKDTree
A_LAT = 2 ** (1 / 6)                                   # Lennard-Jones minimum
NX_I, NY_I = 100, 40                                   # slab: atoms per row, rows
LX_I, LY_I = NX_I * A_LAT, 1000.0                      # periodic in x; open in y
slab = np.array([((i + 0.5 * (j % 2)) * A_LAT, 2.0 + j * A_LAT * np.sqrt(3) / 2)
                 for j in range(NY_I) for i in range(NX_I)])
held = slab[:, 1] < 2.0 + 1.5 * A_LAT * np.sqrt(3) / 2 # bottom two rows held in place
surface = slab[:, 1].max()
SHELLS, V_HIT = 2, 16.0                                # 19-atom hexagonal cluster
c0 = np.array([LX_I / 2, surface + 4.0 + SHELLS * A_LAT])
cluster = np.array([c0 + A_LAT * np.array([q + r / 2, r * np.sqrt(3) / 2])
                    for q in range(-SHELLS, SHELLS + 1) for r in range(-SHELLS, SHELLS + 1)
                    if abs(q + r) <= SHELLS])
x = np.vstack([slab, cluster]); held = np.r_[held, np.zeros(len(cluster), bool)]
v = rng.normal(0, np.sqrt(0.02), x.shape); v[held] = 0
v[len(slab):] = [0.0, -V_HIT]

def slab_forces(x):
    """Lennard-Jones forces with a neighbor search (cKDTree), periodic in x."""
    tree = cKDTree(np.c_[x[:, 0] % LX_I, np.clip(x[:, 1], 0, LY_I - 1)], boxsize=[LX_I, LY_I])
    pairs = tree.query_pairs(R_CUT, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = x[i] - x[j]; d[:, 0] -= LX_I * np.round(d[:, 0] / LX_I)
    r2 = (d ** 2).sum(1); inv6 = 1 / r2 ** 3
    fij = (24 * inv6 * (2 * inv6 - 1) / r2)[:, None] * d
    f = np.zeros_like(x); np.add.at(f, i, fij); np.add.at(f, j, -fij)
    pot = (4 * inv6 * (inv6 - 1) - 4 * (R_CUT ** -12 - R_CUT ** -6)).sum()
    f[held] = 0
    return f, pot

DT_I, T_FILM, FRAME_I = 0.001, 12.0, 20
f, pot = slab_forces(x)
e_start = 0.5 * (v ** 2).sum() + pot
film_x, film_ke, film_t = [], [], []
t0 = time.time()
for n in range(int(T_FILM / DT_I) + 1):
    if n % FRAME_I == 0:
        film_x.append(x.copy()); film_ke.append(0.5 * (v ** 2).sum(1)); film_t.append(n * DT_I)
    v = v + DT_I / 2 * f
    x = x + DT_I * v
    f, pot = slab_forces(x)
    v = v + DT_I / 2 * f
e_end = 0.5 * (v ** 2).sum() + pot
print(f"impact: {len(cluster)}-atom cluster at speed {V_HIT} (kinetic energy "
      f"{0.5 * len(cluster) * V_HIT ** 2:.0f}) into a {len(slab)}-atom crystal; "
      f"{int(T_FILM / DT_I)} steps in {time.time() - t0:.0f} s")
print(f"  total energy conserved to {abs(e_end - e_start) / abs(e_start):.1e} (relative); "
      f"{int((x[:, 1] > surface + 3).sum())} atoms thrown above the surface")

with plt.rc_context({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm"}):
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_axes([0.01, 0.07, 0.98, 0.86])
    ax.set_xlim(0, LX_I); ax.set_ylim(0, surface + 22); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    sc = ax.scatter(film_x[0][:, 0] % LX_I, film_x[0][:, 1], s=9, c=film_ke[0],
                    cmap=HEAT, vmin=0, vmax=1.0, lw=0)
    title = fig.text(0.5, 0.95, "", ha="center", fontsize=15, color=INK)
    fig.text(0.5, 0.02, "colour: kinetic energy of each atom (dark red = hot)",
             ha="center", fontsize=11, color=MUTED)

    def draw(k):
        pts, ke, size = hot_on_top(np.c_[film_x[k][:, 0] % LX_I, film_x[k][:, 1]], film_ke[k],
                                   1.0, 7, 20)
        sc.set_offsets(pts); sc.set_array(ke); sc.set_sizes(size)
        title.set_text(f"a {len(cluster)}-atom cluster hits a {len(slab)}-atom crystal     "
                       f"$t = {film_t[k]:.1f}$")

    t0 = time.time()
    writer = FFMpegWriter(fps=30, codec="libx264", bitrate=3000,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, "figures/md_impact.mp4", dpi=100):
        for k in range(len(film_x)):
            draw(k); writer.grab_frame()
    draw(int(3.0 / (DT_I * FRAME_I)))                  # poster: the shock at t = 3
    fig.savefig("figures/md_impact_poster.png", dpi=100, facecolor=PAPER)
    plt.close(fig)
print(f"  wrote figures/md_impact.mp4 ({os.path.getsize('figures/md_impact.mp4') / 1e6:.1f} MB, "
      f"{len(film_x) / 30:.0f} s at 30 fps) in {time.time() - t0:.0f} s")

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
