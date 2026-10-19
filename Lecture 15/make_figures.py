#!/usr/bin/env python3
"""Figures for Lecture 15 (computing QR: Gram-Schmidt and Householder; the SVD).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides."""
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

def arrow(ax, start, end, color, lw=2.4, ls="-"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=16))

def classical_gram_schmidt(A):
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    for i in range(n):
        u = A[:, i].copy()
        for j in range(i):
            R[j, i] = Q[:, j] @ A[:, i]
            u -= R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(u)
        Q[:, i] = u / R[i, i]
    return Q, R

def modified_gram_schmidt(A):
    m, n = A.shape
    Q, R = A.astype(float).copy(), np.zeros((n, n))
    for i in range(n):
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] /= R[i, i]
        for j in range(i + 1, n):
            R[i, j] = Q[:, i] @ Q[:, j]
            Q[:, j] -= R[i, j] * Q[:, i]
    return Q, R

# ------------------------------------------- 1. Gram-Schmidt in 2D
A2 = np.array([[3.0, -1.0], [4.0, 3.0]])
Q2, R2 = classical_gram_schmidt(A2)
print("[2x2] Gram-Schmidt Q =", Q2.tolist(), " R =", R2.tolist())
a1, a2 = A2[:, 0], A2[:, 1]
e1, e2 = Q2[:, 0], Q2[:, 1]
proj = (e1 @ a2) * e1
print("[2x2] e1 . a2 =", e1 @ a2, " u2 =", (a2 - (e1 @ a2) * e1).tolist(),
      " |u2| =", np.linalg.norm(a2 - (e1 @ a2) * e1), " e1 . e2 =", e1 @ e2)

def draw_gram_schmidt(stage):
    """Stage 0: a1, a2. 1: + e1. 2: + the projection of a2 on e1. 3: + u2.
    4: + e2 (the complete picture). The slide reveals one stage per fragment."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    arrow(ax, (0, 0), a1, V[1])
    arrow(ax, (0, 0), a2, V[2])
    ax.text(a1[0] + 0.1, a1[1], r"$\mathbf{a}_1$", fontsize=17, color=V[1], va="center")
    ax.text(a2[0] - 0.15, a2[1] + 0.2, r"$\mathbf{a}_2$", fontsize=17, color=V[2], ha="center")
    if stage >= 1:
        arrow(ax, (0, 0), e1, V[0], lw=3)
        ax.text(e1[0] + 0.15, e1[1] - 0.05, r"$\mathbf{e}_1$", fontsize=17, color=V[0])
    if stage >= 2:
        arrow(ax, (0, 0), proj, MUTED, lw=1.6, ls="--")
        ax.text(proj[0] + 0.3, proj[1] - 0.2, r"$1.8\,\mathbf{e}_1$", fontsize=15, color=MUTED)
    if stage >= 3:
        arrow(ax, proj, a2, ACCENT)
        ax.text(0.0, 2.55, r"$\mathbf{u}_2$", fontsize=17, color=ACCENT)
    if stage >= 4:
        arrow(ax, (0, 0), e2, V[0], lw=3)
        ax.text(e2[0] - 0.1, e2[1] + 0.15, r"$\mathbf{e}_2$", fontsize=17, color=V[0], ha="right")
    ax.set_xlim(-1.8, 4.6)
    ax.set_ylim(-0.4, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3, 4])
    fig.tight_layout()
    return fig

for stage in range(5):
    save(draw_gram_schmidt(stage), f"gram_schmidt_{stage}")
save(draw_gram_schmidt(4), "gram_schmidt")

# ------------------------------------------- 2. Gram-Schmidt loses orthogonality
eps = 1e-8
Lauchli = np.array([[1, 1, 1], [eps, 0, 0], [0, eps, 0], [0, 0, eps]])
Ql, _ = classical_gram_schmidt(Lauchli)
print("[lauchli] eps = 1e-8, classical Gram-Schmidt Q =\n", Ql.round(4))
print("[lauchli] e2 . e3 =", Ql[:, 1] @ Ql[:, 2], "; 1 + eps^2 == 1:", 1 + eps**2 == 1)
print("[lauchli] Householder max |Q^T Q - I| =", np.abs(np.linalg.qr(Lauchli)[0].T @ np.linalg.qr(Lauchli)[0] - np.eye(3)).max())

rng = np.random.default_rng(0)
m, n = 60, 30
exps = np.arange(0, 16.5, 1.0)
loss = {"cgs": [], "mgs": [], "hh": []}
for k in exps:
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    Vm, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = U @ np.diag(np.logspace(0, -k, n)) @ Vm.T
    for key, f in (("cgs", classical_gram_schmidt), ("mgs", modified_gram_schmidt), ("hh", np.linalg.qr)):
        Q, _ = f(A)
        loss[key].append(np.abs(Q.T @ Q - np.eye(n)).max())
for k, c, mg, h in zip(exps, loss["cgs"], loss["mgs"], loss["hh"]):
    if k in (0, 3, 6, 9, 12, 15):
        print(f"[orthogonality] kappa 1e{int(k)}: classical {c:.1e}, modified {mg:.1e}, Householder {h:.1e}")
fig, ax = plt.subplots(figsize=(6.8, 3.9))
kap = 10.0**exps
ax.loglog(kap, np.maximum(loss["cgs"], 1e-17), "o-", ms=4, color=ACCENT, label="classical Gram-Schmidt")
ax.loglog(kap, np.maximum(loss["mgs"], 1e-17), "o-", ms=4, color=V[2], label="modified Gram-Schmidt")
ax.loglog(kap, np.maximum(loss["hh"], 1e-17), "o-", ms=4, color=V[0], label="Householder")
kappa_lauchli = np.linalg.svd(Lauchli, compute_uv=False)
kappa_lauchli = kappa_lauchli[0] / kappa_lauchli[-1]
print(f"[orthogonality] kappa of the eps example = {kappa_lauchli:.3e} (sqrt(3)/eps = {np.sqrt(3)/eps:.3e})")
ax.axvline(kappa_lauchli, color=INK, lw=1.8, ls="--")
ax.text(kappa_lauchli * 2.2, 1e-13, r"the $\varepsilon$ example", fontsize=13, color=INK, ha="left", va="center")
ax.set_xlabel(r"how close the columns are to dependent, $\kappa$")
ax.set_ylabel(r"$\max|\mathbf{Q}^T\mathbf{Q} - \mathbf{I}|$")
ax.set_ylim(1e-17, 1e2)
ax.legend(frameon=False, fontsize=12, loc="upper left")
fig.tight_layout()
save(fig, "orthogonality")

# ------------------------------------------- 3. a Householder reflection in 2D
a = a1
alpha = -np.sign(a[0]) * np.linalg.norm(a)
u = a - alpha * np.array([1.0, 0.0])
v = u / np.linalg.norm(u)
P = np.eye(2) - 2 * np.outer(v, v)
print("[householder] alpha =", alpha, " u =", u.tolist(), " P =", P.tolist(), " P a =", (P @ a).tolist(),
      " P a2 =", (P @ a2).tolist())
fig, ax = plt.subplots(figsize=(6.4, 4.6))
mirror = np.array([-v[1], v[0]])           # direction of the mirror line, perpendicular to v
ax.plot([-3.2 * mirror[0], 3.2 * mirror[0]], [-3.2 * mirror[1], 3.2 * mirror[1]], color=MUTED, lw=1.4, ls="--")
ax.text(0.75, -0.75, "mirror", color=MUTED, fontsize=14)
arrow(ax, (0, 0), a, V[1])
arrow(ax, (0, 0), P @ a, V[2])
arrow(ax, P @ a, a, ACCENT, lw=1.6)
ax.text(a[0] + 0.15, a[1], r"$\mathbf{a} = (3, 4)$", fontsize=16, color=V[1], va="center")
ax.text((P @ a)[0], -0.55, r"$\mathbf{P}\mathbf{a} = (-5, 0)$", fontsize=16, color=V[2], ha="center")
ax.text(-2.4, 2.25, r"$\mathbf{u} = \mathbf{a} + 5\,\mathbf{e}_1$", fontsize=15, color=ACCENT, ha="right")
ax.set_aspect("equal")
ax.set_xlim(-6.3, 5.6)
ax.set_ylim(-1.0, 4.6)
ax.set_xticks([-5, 0, 3])
ax.set_yticks([0, 4])
fig.tight_layout()
save(fig, "householder")

# sign choice: cancellation in u_1 = a_1 - |a| when a is nearly along e_1
a_close = np.array([1.0, 1e-9])
print("[householder sign] a = (1, 1e-9): a1 - |a| =", a_close[0] - np.linalg.norm(a_close),
      " (exact", -0.5e-18, "); a1 + |a| =", a_close[0] + np.linalg.norm(a_close))

# ------------------------------------------- 4. the SVD: rotate, stretch, rotate
M = np.array([[1.8, 1.2], [0.2, 1.0]])
U, s, Vt = np.linalg.svd(M)
print("[svd 2x2] M =", M.tolist(), " singular values", s.round(4).tolist())
theta = np.linspace(0, 2 * np.pi, 400)
circle = np.vstack([np.cos(theta), np.sin(theta)])
stages = [(circle, r"$\mathbf{x}$: unit circle"),
          (Vt @ circle, r"$\mathbf{V}^T\mathbf{x}$: rotate"),
          (np.diag(s) @ Vt @ circle, r"$\boldsymbol{\Sigma}\mathbf{V}^T\mathbf{x}$: stretch"),
          (U @ np.diag(s) @ Vt @ circle, r"$\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T\mathbf{x} = \mathbf{A}\mathbf{x}$")]
fig, axs = plt.subplots(1, 4, figsize=(13, 3.8))
cmap = plt.get_cmap("viridis")
for ax, (pts, title) in zip(axs, stages):
    ax.scatter(pts[0], pts[1], c=theta, cmap=cmap, s=5)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
# the right singular vectors on the circle, and where they end up
arrow(axs[0], (0, 0), Vt[0], INK, lw=1.8)
arrow(axs[0], (0, 0), Vt[1], INK, lw=1.8)
axs[0].text(*(1.25 * Vt[0]), r"$\mathbf{v}_1$", fontsize=14, ha="center", va="center")
axs[0].text(*(1.3 * Vt[1]), r"$\mathbf{v}_2$", fontsize=14, ha="center", va="center")
arrow(axs[3], (0, 0), s[0] * U[:, 0], INK, lw=1.8)
arrow(axs[3], (0, 0), s[1] * U[:, 1], INK, lw=1.8)
axs[3].text(*(1.2 * s[0] * U[:, 0] + np.array([0.0, 0.25])), r"$\sigma_1\mathbf{u}_1$", fontsize=14, ha="center")
axs[3].text(*(1.6 * s[1] * U[:, 1]), r"$\sigma_2\mathbf{u}_2$", fontsize=14, ha="center", va="center")
fig.tight_layout()
save(fig, "svd_geometry")

# ------------------------------------------- 5. the 4 x 5 example
A45 = np.array([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [0, 4, 0, 0, 0]], dtype=float)
U45, s45, Vt45 = np.linalg.svd(A45)
print("[svd 4x5] singular values", s45.round(6).tolist())
print("[svd 4x5] A (-2,0,0,0,1) =", (A45 @ np.array([-2, 0, 0, 0, 1.0])).tolist(),
      " A (0,0,0,1,0) =", (A45 @ np.array([0, 0, 0, 1, 0.0])).tolist())

# ------------------------------------------- numbers quoted in the step tables
a_h = np.array([3.0, 4.0]); alpha_h = -5.0
u_h = a_h - alpha_h * np.array([1.0, 0.0]); v_h = u_h / np.linalg.norm(u_h)
P_h = np.eye(2) - 2 * np.outer(v_h, v_h)
print("[householder] u =", u_h.tolist(), " v =", v_h.round(6).tolist(), " P =", P_h.round(10).tolist())
print("[householder] P a1 =", (P_h @ a_h).round(10).tolist(), " P a2 =", (P_h @ np.array([-1.0, 3.0])).round(10).tolist(),
      " P^T P - I max =", np.abs(P_h.T @ P_h - np.eye(2)).max())
for i in range(5):
    out = (A45 @ Vt45[i]).round(6).tolist()
    su = (s45[i] * U45[:, i]).round(6).tolist() if i < 4 else "-"
    print(f"[svd 4x5] v{i+1} =", Vt45[i].round(6).tolist(), " A v =", out, " sigma u =", su)
