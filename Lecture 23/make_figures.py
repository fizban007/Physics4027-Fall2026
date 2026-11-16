#!/usr/bin/env python3
"""Figures for Lecture 23 (statistical mechanics and the Metropolis algorithm).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides. The Goodstein page is made separately by goodstein.typ."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

rng = np.random.default_rng(4027)

# ------------------------------------------- a three-state Markov chain
T = np.array([[0.5, 0.3, 0.2],
              [0.2, 0.6, 0.2],
              [0.1, 0.3, 0.6]])
w, v = np.linalg.eig(T.T)
pi = np.real(v[:, np.argmin(abs(w - 1))]); pi /= pi.sum()
print(f"[markov] stationary distribution pi = {np.round(pi, 4)}, second eigenvalue {sorted(abs(w))[-2]:.3f}")
steps = 12
fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
for ax, start, title in [(axes[0], np.array([1.0, 0, 0]), "start in state 1"),
                         (axes[1], np.array([0, 0, 1.0]), "start in state 3")]:
    p = start.copy(); hist = [p.copy()]
    for n in range(steps):
        p = p @ T; hist.append(p.copy())
    hist = np.array(hist)
    for j in range(3):
        ax.plot(range(steps + 1), hist[:, j], "-o", ms=4, color=V[j])
        ax.text(steps + 0.5, pi[j], rf"state {j+1}: $\pi_{j+1} = {pi[j]:.3f}$", color=V[j], va="center", fontsize=16)
    ax.set_xlabel("step $n$"); ax.set_title(title, fontsize=17)
    ax.set_xlim(0, steps + 8); ax.set_ylim(0, 1.02); ax.set_xticks([0, 4, 8, 12])
axes[0].set_ylabel("probability $p_n(j)$")
fig.tight_layout(); save(fig, "markov")

# ------------------------------------------- 2D Ising model, checkerboard Metropolis
L = 100
J = 1.0
ii, jj = np.indices((L, L))
mask = [(ii + jj) % 2 == c for c in (0, 1)]

def energy_per_spin(s):
    return -J * np.sum(s * (np.roll(s, 1, 0) + np.roll(s, 1, 1))) / s.size

def sweep(s, beta):
    """One Metropolis sweep: every spin is offered a flip once, in two
    checkerboard halves so that numpy can do each half at once. Each half
    has the same acceptance rule as the single-spin algorithm."""
    for m in mask:
        nb = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
        dE = 2 * J * s * nb
        flip = m & (rng.random((L, L)) < np.exp(-beta * np.maximum(dE, 0)))
        s[flip] *= -1
    return s

# snapshots
snaps = {}
for kT in (2.0, 2.27, 3.0):
    s = rng.choice([-1, 1], size=(L, L))
    for n in range(3000):
        sweep(s, 1 / kT)
    snaps[kT] = s.copy()
    print(f"[snapshot] kT={kT}: after 3000 sweeps E/N = {energy_per_spin(s):.3f}, M = {s.mean():+.3f}")
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
cmap = ListedColormap([V[1], "#F2E8C9"])
for ax, (kT, s) in zip(axes, snaps.items()):
    ax.imshow(s, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(rf"$k_BT = {kT}\,J$, $M = {s.mean():+.2f}$", fontsize=16)
fig.tight_layout(); save(fig, "snapshots")

# burn-in history at kT = 2 from a cold and a hot start
kT = 2.0; beta = 1 / kT; n_sweeps = 400
hist = {}
for name, s in [("all spins up", np.ones((L, L), dtype=int)),
                ("random spins", rng.choice([-1, 1], size=(L, L)))]:
    E = [energy_per_spin(s)]; M = [s.mean()]
    for n in range(n_sweeps):
        sweep(s, beta); E.append(energy_per_spin(s)); M.append(s.mean())
    hist[name] = (np.array(E), np.array(M))
E_cold = hist["all spins up"][0]; E_hot = hist["random spins"][0]
print(f"[history] kT=2: cold start E/N {E_cold[0]:.3f} -> {E_cold[-100:].mean():.4f}; "
      f"hot start E/N {E_hot[0]:.3f} -> {E_hot[-100:].mean():.4f}; "
      f"Onsager exact u = -1.7459")
first = next(n for n in range(n_sweeps) if abs(E_hot[n] - E_hot[-100:].mean()) < 0.02)
print(f"[history] hot start within 0.02 of its final energy after {first} sweeps; "
      f"|M| at sweep 50: {abs(hist['random spins'][1][50]):.2f}, at 400: {abs(hist['random spins'][1][-1]):.2f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for (name, (E, M)), c in zip(hist.items(), (V[1], ACCENT)):
    axes[0].plot(E, color=c, label=name); axes[1].plot(M, color=c, label=name)
axes[0].axhline(-1.7459, color=MUTED, lw=1, ls=":")
axes[0].text(n_sweeps * 0.55, -1.7459 - 0.03, "exact, $-1.746$", color=MUTED, va="top", ha="left", fontsize=12)
axes[0].set_xlabel("sweeps"); axes[0].set_ylabel("$E/N$ (units of $J$)")
axes[1].set_xlabel("sweeps"); axes[1].set_ylabel("$M$")
axes[0].legend(frameon=False, fontsize=13); axes[0].set_title(r"$k_BT = 2J$, $100\times100$", fontsize=15)
axes[1].set_ylim(-1.05, 1.05)
fig.tight_layout(); save(fig, "history")

# ------------------------------------------- why uniform sampling fails
n_rand = 20000
E_rand = np.array([energy_per_spin(rng.choice([-1, 1], size=(L, L))) for _ in range(n_rand)])
s = snaps[2.0].copy(); E_eq = []
for n in range(2000):
    sweep(s, 1 / 2.0); E_eq.append(energy_per_spin(s))
E_eq = np.array(E_eq)
dE = (E_rand.mean() - E_eq.mean()) * L * L
print(f"[overlap] random states: E/N = {E_rand.mean():+.4f} +- {E_rand.std():.4f}; "
      f"equilibrium at kT=2: {E_eq.mean():.4f} +- {E_eq.std():.4f}; "
      f"a random state weighs exp(-beta dE) = exp(-{dE/2:.0f}) relative to an equilibrium one")
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.hist(E_rand, bins=60, density=True, color=MUTED, alpha=0.8, label="random states (uniform)")
ax.hist(E_eq, bins=60, density=True, color=V[1], alpha=0.9, label=r"equilibrium at $k_BT = 2J$")
ax.set_xlim(-1.85, 0.1); ax.set_xlabel("$E/N$ (units of $J$)"); ax.set_ylabel("density")
ax.set_yticks([])
ax.annotate("", xy=(E_eq.mean() + 0.02, 12), xytext=(E_rand.mean() - 0.04, 12),
            arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=2))
ax.text(-0.85, 14, rf"$\Delta E = {dE/L/L:.2f}\,N J$, \quad relative weight $e^{{-\Delta E/k_BT}} = e^{{-{dE/2:.0f}}}$",
        color=ACCENT, ha="center", fontsize=15)
ax.set_ylim(0, 40)
ax.legend(frameon=False, fontsize=13, loc="upper center")
fig.tight_layout(); save(fig, "overlap")
print("done")
