#!/usr/bin/env python3
"""Figures for Lecture 16 (the SVD: what it is, the closest solution and least
squares, low-rank approximation, PCA). Run from this directory; writes
figures/*.png and prints the numbers quoted on the slides."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def pseudoinverse_solve(A, b, rcond=1e-12):
    """x = V diag(1/sigma_i) U^T b, with 1/sigma_i -> 0 for sigma_i below rcond * sigma_1."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    inv_s = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    return Vt.T @ (inv_s * (U.T @ b)), s

# ------------------------------------------- 1. the SVD: rotate, stretch, rotate (same as Lecture 15)
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

# the 4x5 example of the slides: check every pair A v_i = sigma_i u_i
A45 = np.array([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [0, 4, 0, 0, 0]], dtype=float)
U45, s45, Vt45 = np.linalg.svd(A45)
print("[svd 4x5] singular values", s45.round(6).tolist())
for i in range(5):
    print(f"[svd 4x5] v{i+1} =", Vt45[i].round(6).tolist(), " A v =", (A45 @ Vt45[i]).round(6).tolist())
# the SVD-and-eigenvalues slide: eigenvalues of A^T A are sigma_i^2
print("[svd 4x5] eigenvalues of A^T A:", np.sort(np.linalg.eigvalsh(A45.T @ A45))[::-1].round(6).tolist(),
      " sigma^2:", (s45**2).round(6).tolist())

# ------------------------------------------- 2. least squares: a thrown ball
g_true, v0_true, y0_true = 9.81, 9.0, 1.2
rng = np.random.default_rng(1)
t = np.linspace(0.0, 1.8, 15)
y_exact = y0_true + v0_true * t - 0.5 * g_true * t**2
sigma_meas = 0.15
y = y_exact + sigma_meas * rng.standard_normal(t.size)
A = np.vstack([np.ones_like(t), t, t**2]).T          # columns: 1, t, t^2
a, sv = pseudoinverse_solve(A, y)
a_np = np.linalg.lstsq(A, y, rcond=None)[0]
resid = A @ a - y
print(f"[ball] A is {A.shape[0]} x {A.shape[1]}, singular values {sv.round(3).tolist()}, kappa = {sv[0]/sv[-1]:.1f}")
print(f"[ball] fit a = {a.round(4).tolist()}  (numpy lstsq {a_np.round(4).tolist()})")
print(f"[ball] g from fit = {-2*a[2]:.3f}  v0 = {a[1]:.3f}  y0 = {a[0]:.3f}   true 9.81, 9.0, 1.2")
print(f"[ball] |A a - y| = {np.linalg.norm(resid):.4f}  (sqrt(15) sigma = {np.sqrt(15)*sigma_meas:.3f});"
      f"  A^T (A a - y) = {(A.T @ resid).round(12).tolist()}")
# the slide quotes the first rows of A and the three steps of the solve
for i in [0, 1, 2, 14]:
    print(f"[ball] row {i+1:2d} of A = (1, {t[i]:.3f}, {t[i]**2:.3f})   y_{i+1} = {y[i]:.3f}")
U_b, s_b, Vt_b = np.linalg.svd(A, full_matrices=False)
c = U_b.T @ y
print("[ball] U^T y =", c.round(4).tolist(), " divided by sigma:", (c / s_b).round(4).tolist(),
      " V (c/sigma) =", (Vt_b.T @ (c / s_b)).round(4).tolist())
fig, ax = plt.subplots(figsize=(6.6, 4.0))
tt = np.linspace(0, 1.8, 200)
ax.plot(tt, y0_true + v0_true * tt - 0.5 * g_true * tt**2, color=MUTED, lw=1.2, ls="--", label="true path")
ax.plot(tt, a[0] + a[1] * tt + a[2] * tt**2, color=ACCENT, label=r"least-squares fit, $g = %.2f$" % (-2 * a[2]))
ax.errorbar(t, y, yerr=sigma_meas, fmt="o", color=V[1], ms=5, capsize=2, lw=1.2, label="15 measurements")
ax.set_xlabel("$t$ (s)")
ax.set_ylabel("height (m)")
ax.legend(frameon=False, fontsize=12, loc="lower center")
fig.tight_layout()
save(fig, "lstsq_fit")

# ------------------------------------------- 3. normal equations square the conditioning
x = np.linspace(0, 1, 60)
f = np.cos(3 * x) + 0.2 * np.exp(x)
print("[normal eq] degree | kappa(A) | kappa^2 | residual via SVD | residual via normal equations | coefficient error, normal eq")
for deg in (4, 8, 12, 16):
    Ad = np.vander(x, deg + 1, increasing=True)
    sd = np.linalg.svd(Ad, compute_uv=False)
    kappa = sd[0] / sd[-1]
    a_svd, _ = pseudoinverse_solve(Ad, f)
    try:
        a_ne = np.linalg.solve(Ad.T @ Ad, Ad.T @ f)
        r_ne = np.linalg.norm(Ad @ a_ne - f)
        err = np.linalg.norm(a_ne - a_svd) / np.linalg.norm(a_svd)
    except np.linalg.LinAlgError:
        r_ne, err = np.nan, np.nan
    print(f"[normal eq] {deg:6d} | {kappa:8.1e} | {kappa**2:8.1e} | {np.linalg.norm(Ad @ a_svd - f):8.1e} | {r_ne:8.1e} | {err:8.1e}")

# ------------------------------------------- 4. low-rank approximation of the Project 1 image
img = mpimg.imread("../../Projects/Project 1/image1080.png")
# The disk is drawn in a red-yellow colormap, so the red channel carries the
# brightness; luminance would make the whole disk dark. Crop to the plotting area.
gray = img[12:578, 127:1123, 0]
m_px, n_px = gray.shape
Ui, si, Vti = np.linalg.svd(gray, full_matrices=False)
print(f"[image] {m_px} x {n_px} pixels = {m_px*n_px} numbers; {si.size} singular values; sigma_1 = {si[0]:.1f}")
ranks = [5, 20, 60]
energy = np.cumsum(si**2) / np.sum(si**2)
for k in ranks:
    Ak = (Ui[:, :k] * si[:k]) @ Vti[:k]
    stored = k * (m_px + n_px + 1)
    print(f"[image] rank {k:3d}: stores {stored} numbers = {100*stored/(m_px*n_px):.1f}% of the image;"
          f" max |A - A_k| = {np.linalg.norm(gray - Ak, 2):.2f} = sigma_{k+1} = {si[k]:.2f};"
          f" kept {100*energy[k-1]:.2f}% of sum sigma^2")
fig, axs = plt.subplots(2, 2, figsize=(9, 5.4))
panels = [(gray, "original: %d numbers" % (m_px * n_px))]
for k in ranks:
    Ak = (Ui[:, :k] * si[:k]) @ Vti[:k]
    panels.append((Ak, r"rank %d: %.1f\%% of the numbers" % (k, 100 * k * (m_px + n_px + 1) / (m_px * n_px))))
for ax, (im, title) in zip(axs.ravel(), panels):
    ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
fig.tight_layout()
save(fig, "svd_image")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
idx = np.arange(1, si.size + 1)
a1.semilogy(idx, si, color=V[0], lw=1.8)
for k in ranks:
    a1.axvline(k, color=MUTED, lw=0.8, ls=":")
a1.set_xlabel("$i$"); a1.set_ylabel(r"$\sigma_i$")
a1.set_xlim(0, 300)
a1.set_ylim(0.1, 300)
a1.set_title("singular values of the image", fontsize=14)
a2.plot(idx, 100 * energy, color=ACCENT, lw=1.8)
for k in ranks:
    a2.axvline(k, color=MUTED, lw=0.8, ls=":")
a2.set_xlabel("rank $k$ kept"); a2.set_ylabel(r"$\sum_{i \le k}\sigma_i^2 \,/\, \sum_i \sigma_i^2$ (\%)")
a2.set_xlim(0, 300); a2.set_ylim(90, 100.5)
a2.set_title("how much of the image is kept", fontsize=14)
fig.tight_layout()
save(fig, "svd_spectrum")

# ------------------------------------------- 5. PCA in two dimensions
rng = np.random.default_rng(3)
n_pts = 300
angle = np.radians(30)
R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
X2 = rng.standard_normal((n_pts, 2)) * np.array([3.0, 0.7]) @ R.T + np.array([4.0, 2.0])
mean2 = X2.mean(axis=0)
Xc = X2 - mean2
Up, sp, Vtp = np.linalg.svd(Xc, full_matrices=False)
std_along = sp / np.sqrt(n_pts - 1)
print(f"[pca 2d] mean {mean2.round(3).tolist()}, singular values {sp.round(2).tolist()},"
      f" std along v1, v2 = {std_along.round(3).tolist()} (generated with 3.0 and 0.7)")
print(f"[pca 2d] v1 = {Vtp[0].round(4).tolist()}, angle {np.degrees(np.arctan2(Vtp[0,1], Vtp[0,0])) % 180:.1f} deg (generated 30);"
      f" variance fraction of v1: {sp[0]**2/np.sum(sp**2):.3f}")
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.scatter(X2[:, 0], X2[:, 1], s=9, color=V[1], alpha=0.55)
for i, col in enumerate([ACCENT, V[2]]):
    direction = Vtp[i] * np.sign(Vtp[i][0])
    arrow(ax, mean2, mean2 + 2 * std_along[i] * direction, col, lw=3)
ax.text(*(mean2 + 2 * std_along[0] * Vtp[0] * np.sign(Vtp[0][0]) + np.array([0.3, -0.2])),
        r"$2 s_1 \mathbf{v}_1$", fontsize=15, color=ACCENT)
ax.text(*(mean2 + 2 * std_along[1] * Vtp[1] * np.sign(Vtp[1][0]) + np.array([0.3, 0.1])),
        r"$2 s_2 \mathbf{v}_2$", fontsize=15, color=V[2])
ax.set_aspect("equal")
ax.set_xlabel("quantity 1"); ax.set_ylabel("quantity 2")
fig.tight_layout()
save(fig, "pca_2d")

# ------------------------------------------- 6. PCA on Fisher's irises: 150 flowers, 4 lengths each
# data/iris.csv is the copy shipped with scikit-learn (header: n, p, species names;
# columns: sepal length, sepal width, petal length, petal width in cm, species 0/1/2).
raw = np.loadtxt("data/iris.csv", delimiter=",", skiprows=1)
Xi = raw[:, :4]
species = raw[:, 4].astype(int)
species_names = ["setosa", "versicolor", "virginica"]
measure_names = ["sepal\nlength", "sepal\nwidth", "petal\nlength", "petal\nwidth"]
n_fl = Xi.shape[0]
mean_i = Xi.mean(axis=0)
Xic = Xi - mean_i
Ui, si, Vti = np.linalg.svd(Xic, full_matrices=False)
if Vti[0, 2] < 0:                                   # sign convention: v1 points toward larger petals
    Vti[0] *= -1; Ui[:, 0] *= -1
if Vti[1, 0] > 0:
    Vti[1] *= -1; Ui[:, 1] *= -1
scores = Xic @ Vti.T                                # = U Sigma: each flower's coordinates along v1, v2, ...
var_i = si**2
print(f"[iris] X is {n_fl} x 4; first rows {Xi[0].tolist()}, {Xi[1].tolist()}; column means {mean_i.round(2).tolist()}")
print(f"[iris] singular values {si.round(2).tolist()}; variance fractions {(var_i/var_i.sum()).round(3).tolist()};"
      f" first two together {np.sum(var_i[:2])/var_i.sum():.3f}")
print(f"[iris] v1 = {Vti[0].round(2).tolist()}  v2 = {Vti[1].round(2).tolist()}")
for k in range(3):
    print(f"[iris] {species_names[k]:10s} along v1 from {scores[species==k, 0].min():5.2f} to {scores[species==k, 0].max():5.2f}")
fig, (b1, b2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
for k, col in enumerate([V[0], V[2], ACCENT]):
    b1.scatter(scores[species == k, 0], scores[species == k, 1], s=22, color=col, alpha=0.8, label=species_names[k])
b1.axhline(0, color=MUTED, lw=0.6); b1.axvline(0, color=MUTED, lw=0.6)
b1.set_xlabel(r"coordinate along $\mathbf{v}_1$ (cm)"); b1.set_ylabel(r"along $\mathbf{v}_2$ (cm)")
b1.legend(frameon=False, fontsize=13, loc="upper right")
b1.set_title("each flower by its first two coordinates", fontsize=14)
b2.bar(range(4), Vti[0], color=V[1], width=0.6)
b2.axhline(0, color=MUTED, lw=0.6)
b2.set_xticks(range(4)); b2.set_xticklabels(measure_names, fontsize=12)
b2.set_ylim(-0.3, 1.0)
b2.set_title(r"the components of $\mathbf{v}_1$", fontsize=14)
fig.tight_layout()
save(fig, "pca_iris")
print("done")
