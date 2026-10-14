#!/usr/bin/env python3
"""Figures for Lecture 14 (matrices with structure: tridiagonal, Cholesky; QR).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides.

The timing figure reads tridiagonal_timing.txt, written by tridiagonal_timing.cpp,
and ../Lecture 13/lu_timing.txt for the dense LU of Lecture 13:
    g++ -std=c++20 -O2 tridiagonal_timing.cpp -o tridiagonal_timing
    ./tridiagonal_timing > tridiagonal_timing.txt
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ------------------------------------------- 1. tridiagonal vs dense
dense = np.loadtxt("../Lecture 13/lu_timing.txt")
tri = np.loadtxt("tridiagonal_timing.txt")
for row in tri:
    print(f"[tridiagonal] n = {int(row[0])}: {row[1]:.3g} s, max error {row[2]:.1e}")
# dense extrapolated with n^3 from its largest measured size
n_big = 1e6
dense_big = dense[-1, 1] * (n_big / dense[-1, 0])**3
print(f"[dense] extrapolated to n = 1e6: {dense_big:.2e} s = {dense_big / 3.15e7:.1f} years")

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.loglog(dense[:, 0], dense[:, 1], "o-", color=V[0], ms=6, label="dense LU (Lecture 13)")
ne = np.logspace(np.log10(dense[-1, 0]), 6, 20)
ax.loglog(ne, dense[-1, 1] * (ne / dense[-1, 0])**3, color=V[0], lw=1.4, ls="--")
ax.loglog(tri[:, 0], tri[:, 1], "o-", color=V[2], ms=6, label="tridiagonal")
ax.plot([n_big], [dense_big], "o", ms=8, mfc=PAPER, mec=V[0], mew=2)
ax.annotate(r"9 years", (n_big, dense_big), xytext=(-12, -4), textcoords="offset points",
            ha="right", va="center", fontsize=14, color=V[0])
t_tri_big = tri[tri[:, 0] == n_big, 1][0]
ax.annotate(rf"{t_tri_big * 1e3:.0f} ms", (n_big, t_tri_big), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=14, color=V[2])
ax.set_xlabel(r"number of unknowns $n$")
ax.set_ylabel("time (s)")
ax.set_ylim(3e-6, 3e9)
ax.legend(frameon=False, fontsize=13, loc="upper left")
fig.tight_layout()
save(fig, "tridiagonal_timing")

# ------------------------------------------- 2. Cholesky: correlated random numbers
Sigma = np.array([[4.0, 2.0], [2.0, 5.0]])
L = np.linalg.cholesky(Sigma)
print("[cholesky] L =", L.tolist())
rng = np.random.default_rng(3)
z = rng.standard_normal((2, 3000))
w = L @ z
print("[cholesky] sample covariance of L z:", np.cov(w).round(2).tolist())
fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.1), sharex=True, sharey=True)
axs[0].plot(z[0], z[1], ".", ms=2.2, color=V[1], alpha=0.6)
axs[0].set_title(r"random pairs", fontsize=14)
axs[1].plot(w[0], w[1], ".", ms=2.2, color=V[2], alpha=0.6)
axs[1].set_title(r"the same pairs, multiplied by $\mathbf{L}$", fontsize=14)
for a in axs:
    a.set_aspect("equal")
    a.set_xlim(-8, 8)
    a.set_ylim(-8, 8)
    a.set_xticks([-6, 0, 6])
    a.set_yticks([-6, 0, 6])
fig.tight_layout()
save(fig, "correlated")

# ------------------------------------------- 3. the QR algorithm
A = np.array([[4.0, 2.0, 1.0], [2.0, 1.0, 0.0], [1.0, 0.0, 3.0]])
lam = np.sort(np.linalg.eigvalsh(A))[::-1]
print("[qr] eigenvalues:", lam.round(5).tolist())
Q, R = np.linalg.qr(A)
print("[qr] Q =\n", Q.round(3), "\n R =\n", R.round(3))
Ak = A.copy()
diags, offs = [np.diag(Ak).copy()], [np.abs(Ak - np.diag(np.diag(Ak))).max()]
for k in range(1, 31):
    Q, R = np.linalg.qr(Ak)
    Ak = R @ Q
    diags.append(np.diag(Ak).copy())
    offs.append(np.abs(Ak - np.diag(np.diag(Ak))).max())
    if k in (1, 2, 5, 10, 20, 30):
        print(f"[qr] k = {k}: diagonal {np.diag(Ak).round(4).tolist()}, max off-diagonal {offs[-1]:.1e}")
Ak10 = A.copy()
for k in range(10):
    Q, R = np.linalg.qr(Ak10)
    Ak10 = R @ Q
print("[qr] A_10 =\n", Ak10.round(4))
diags = np.array(diags)
ks = np.arange(len(offs))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
for i, c in enumerate((V[0], V[1], V[2])):
    axs[0].plot(ks, diags[:, i], "o-", ms=4, color=c, lw=1.8)
    axs[0].axhline(lam[i], color=MUTED, lw=1, ls="--")
axs[0].set_xlabel(r"iteration $k$")
axs[0].set_ylabel(r"diagonal of $\mathbf{A}_k$")
axs[1].semilogy(ks, offs, "o-", ms=4, color=ACCENT, lw=1.8)
axs[1].set_xlabel(r"iteration $k$")
axs[1].set_ylabel("largest off-diagonal entry")
fig.tight_layout()
save(fig, "qr_iteration")
