#!/usr/bin/env python3
"""Figures for Lecture 13 (linear systems: Gaussian elimination, pivoting, LU).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides.

The timing figure reads lu_timing.txt, written by lu_timing.cpp:
    g++ -std=c++20 -O2 lu_timing.cpp -o lu_timing && ./lu_timing > lu_timing.txt
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

# ------------------------------------------- 0. the worked example
A = np.array([[1., 2, 3], [4, 6, 7], [5, 8, 9]])
b = np.array([1., 2, 4])
L = np.array([[1., 0, 0], [4, 1, 0], [5, 1, 1]])
U = np.array([[1., 2, 3], [0, -2, -5], [0, 0, -1]])
assert np.allclose(L @ U, A)
print("[example] x =", np.linalg.solve(A, b), " det =", np.linalg.det(A), "= prod(diag U) =", np.prod(np.diag(U)))

# ------------------------------------------- 1. a small pivot
# [[eps, 1], [1, 1]] x = [1, 2]; exact x1 = 1/(1 - eps), x2 = (1 - 2 eps)/(1 - eps).
def solve_no_pivot(eps):
    m = 1.0 / eps                        # multiplier for row 2
    x2 = (2.0 - m * 1.0) / (1.0 - m * 1.0)
    x1 = (1.0 - x2) / eps
    return x1, x2

def solve_pivot(eps):                    # rows swapped: [[1, 1], [eps, 1]] x = [2, 1]
    m = eps
    x2 = (1.0 - m * 2.0) / (1.0 - m * 1.0)
    x1 = 2.0 - x2
    return x1, x2

epss = np.logspace(-1, -20, 120)
err_np, err_p = [], []
for e in epss:
    exact = 1.0 / (1.0 - e)
    err_np.append(abs(solve_no_pivot(e)[0] / exact - 1))
    err_p.append(abs(solve_pivot(e)[0] / exact - 1))
for e in (1e-4, 1e-8, 1e-12, 1e-17, 1e-20):
    print(f"[pivot] eps {e:.0e}: no pivoting x = {solve_no_pivot(e)}, pivoting x = {solve_pivot(e)}")

fig, ax = plt.subplots(figsize=(6.8, 3.9))
floor = 1e-17
ax.loglog(epss, np.maximum(err_np, floor), color=ACCENT, label="no pivoting")
ax.loglog(epss, np.maximum(err_p, floor), color=V[1], label="partial pivoting")
ax.axhline(1, color=MUTED, lw=1, ls="--")
ax.set_xlabel(r"pivot size $\varepsilon$")
ax.set_ylabel(r"relative error in $x_1$")
ax.set_ylim(5e-18, 1e2)
ax.invert_xaxis()
ax.legend(frameon=False, fontsize=13, loc="upper left")
fig.tight_layout()
save(fig, "small_pivot")

# ------------------------------------------- 2. cost: factor once, solve many
data = np.loadtxt("lu_timing.txt")
n, t_factor, t_solve = data[:, 0], data[:, 1], data[:, 2]
for row in data:
    print(f"[timing] n = {int(row[0])}: factor {row[1]:.3g} s, one solve {row[2]:.3g} s, ratio {row[1]/row[2]:.0f}, residual {row[3]:.1e}")
fig, ax = plt.subplots(figsize=(6.8, 3.9))
ax.loglog(n, t_factor, "o-", color=V[0], ms=6, label="factor: $\\mathbf{A} = \\mathbf{L}\\mathbf{U}$")
ax.loglog(n, t_solve, "o-", color=V[2], ms=6, label="one solve with $\\mathbf{L}, \\mathbf{U}$")
ref = n[2]
ax.loglog(n, t_factor[2] * (n / ref)**3, color=MUTED, lw=1, ls="--")
ax.loglog(n, t_solve[2] * (n / ref)**2, color=MUTED, lw=1, ls="--")
ax.text(n[-2] * 0.9, t_factor[2] * (n[-2] / ref)**3 * 2.2, r"$\propto n^3$", fontsize=14, color=MUTED, ha="right")
ax.text(n[-2] * 0.9, t_solve[2] * (n[-2] / ref)**2 * 2.2, r"$\propto n^2$", fontsize=14, color=MUTED, ha="right")
ax.set_xlabel(r"matrix size $n$")
ax.set_ylabel("time (s)")
ax.legend(frameon=False, fontsize=13, loc="upper left")
fig.tight_layout()
save(fig, "lu_timing")

# ------------------------------------------- 3. why not compute the inverse
import scipy.linalg as sl
rng = np.random.default_rng(1)
n_test = 500
print("[inverse] relative residual |Ax - b| / (|A| |x|) and error in x, LU solve vs x = inv(A) b")
for kappa in (1e2, 1e6, 1e10, 1e14):
    # a random matrix with singular values from 1 down to 1/kappa
    Q1, _ = np.linalg.qr(rng.standard_normal((n_test, n_test)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n_test, n_test)))
    Ak = Q1 @ np.diag(np.logspace(0, -np.log10(kappa), n_test)) @ Q2.T
    x_true = rng.standard_normal(n_test)
    bk = Ak @ x_true
    x_lu = sl.lu_solve(sl.lu_factor(Ak), bk)
    x_inv = np.linalg.inv(Ak) @ bk
    resid = lambda x: np.linalg.norm(Ak @ x - bk) / (np.linalg.norm(Ak, 2) * np.linalg.norm(x))
    err = lambda x: np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    print(f"  kappa {kappa:.0e}: residual LU {resid(x_lu):.1e}, inverse {resid(x_inv):.1e}; "
          f"error LU {err(x_lu):.1e}, inverse {err(x_inv):.1e}")

def second_difference(n):
    return np.diag(-2.0 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)

T = second_difference(1000)
Pt, Lt, Ut = sl.lu(T)
print(f"[inverse] tridiagonal n = 1000: nonzeros in A {np.count_nonzero(T)}, L {np.count_nonzero(Lt)}, "
      f"U {np.count_nonzero(Ut)}, inverse {np.count_nonzero(np.linalg.inv(T))} of {1000 * 1000}")

fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.2))
T12 = second_difference(12)
for a, M, title in ((axs[0], T12, r"$\mathbf{A}$: 34 nonzeros"),
                    (axs[1], np.linalg.inv(T12), r"$\mathbf{A}^{-1}$: all 144 nonzero")):
    shown = np.ma.masked_where(M == 0, np.log10(np.abs(M) + 1e-300))
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(PAPER)
    a.imshow(shown, cmap=cmap, vmin=-1.2, vmax=0.8)
    a.set_title(title, fontsize=15)
    a.set_xticks([])
    a.set_yticks([])
    for side in a.spines.values():
        side.set_color(MUTED)
fig.tight_layout()
save(fig, "inverse_fill")
