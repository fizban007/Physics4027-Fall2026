#!/usr/bin/env python3
"""Figures for Lecture 21 (random numbers and sampling from distributions).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides."""
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

rng = np.random.default_rng(4027)

# ------------------------------------------- a tiny LCG: full period and a bad choice
def lcg(a, c, m, seed, n):
    out = np.empty(n, dtype=np.int64); x = seed
    for i in range(n):
        x = (a*x + c) % m; out[i] = x
    return out
good = lcg(5, 1, 16, 0, 16); bad = lcg(4, 1, 16, 0, 8)
print("[lcg] m=16, a=5, c=1 from 0:", ", ".join(map(str, good)), "(period 16)")
print("[lcg] m=16, a=4, c=1 from 0:", ", ".join(map(str, bad)), "(stuck)")

# ------------------------------------------- pairs of a small LCG fall on lines
m, a, c = 1024, 37, 1
seq = lcg(a, c, m, 0, m)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
axs[0].scatter(seq[:-1]/m, seq[1:]/m, s=6, color=V[0])
axs[0].set_title(f"$I_{{n+1}} = {a} I_n + {c} \\bmod {m}$: all {m} pairs", fontsize=14)
mt = np.random.default_rng(1).random(m + 1)
axs[1].scatter(mt[:-1], mt[1:], s=6, color=V[1])
axs[1].set_title(f"a modern generator: {m} pairs", fontsize=14)
for ax in axs:
    ax.set_xlabel("$u_n$"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
axs[0].set_ylabel("$u_{n+1}$")
fig.tight_layout(); save(fig, "lcg_pairs")

# ------------------------------------------- RANDU: triples in 15 planes
N = 20000
x = np.empty(N + 2, dtype=np.int64); x[0] = 1
for i in range(1, N + 2):
    x[i] = (65539*x[i-1]) % 2**31
u = x/2**31
planes = np.round(9*u[:-2] - 6*u[1:-1] + u[2:])
print(f"[randu] 9u_n - 6u_(n+1) + u_(n+2) takes {len(np.unique(planes))} integer values, from {planes.min():.0f} to {planes.max():.0f}: {len(np.unique(planes))} planes")
fig = plt.figure(figsize=(11, 4.6))
for k, (elev, azim, title) in enumerate([(30, -60, "the usual view"), (0, np.degrees(np.arctan2(1.5, 1)), "looking along the planes")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.scatter(u[:-2], u[1:-1], u[2:], s=1.2, color=V[0], alpha=0.6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("$u_n$", labelpad=-6); ax.set_ylabel("$u_{n+1}$", labelpad=-6); ax.set_zlabel("$u_{n+2}$", labelpad=-6)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.set_title(title, fontsize=14)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
fig.tight_layout(); save(fig, "randu")

# ------------------------------------------- PDF, CDF and the inverse transform
xs = np.linspace(0, 8, 600)
pdf = xs*np.exp(-xs)                      # Gamma(2, 1): a skewed example
cdf = 1 - (1 + xs)*np.exp(-xs)
u0 = 0.7; x0 = xs[np.searchsorted(cdf, u0)]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
axs[0].plot(xs, pdf, color=V[0]); axs[0].fill_between(xs[xs <= x0], pdf[xs <= x0], color=V[0], alpha=0.25)
axs[0].set_title("$p(x)$: the shaded area is $F(x_0)$", fontsize=14); axs[0].set_ylabel("$p$")
axs[1].plot(xs, cdf, color=V[1])
axs[1].annotate("", xy=(x0, u0), xytext=(0, u0), arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
axs[1].annotate("", xy=(x0, 0), xytext=(x0, u0), arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
axs[1].text(0.15, u0 + 0.04, "$u$", color=ACCENT); axs[1].text(x0 + 0.15, 0.04, "$x = F^{-1}(u)$", color=ACCENT)
axs[1].set_title("$F(x)$: from $u$ to $x$", fontsize=14); axs[1].set_ylabel("$F$"); axs[1].set_ylim(0, 1.02)
for ax in axs: ax.set_xlabel("$x$"); ax.set_xlim(0, 8)
fig.tight_layout(); save(fig, "cdf")

# ------------------------------------------- exponential and power law by inversion
n = 100000
u = rng.random(n)
lam = 1.0
t = -np.log(1 - u)/lam
print(f"[exp] {n} samples of -ln(1-u): mean {t.mean():.4f} (1/lambda = {1/lam}), fraction beyond 5: {np.mean(t > 5):.5f} (e^-5 = {np.exp(-5):.5f})")
alpha, xmin = 2.5, 1.0
xp = xmin*(1 - u)**(1/(1 - alpha))
print(f"[power] alpha={alpha}: largest of {n} samples {xp.max():.0f}, fraction beyond 100: {np.mean(xp > 100):.5f} (100^(1-alpha) = {100**(1-alpha):.5f})")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
bins = np.linspace(0, 10, 51)
axs[0].hist(t, bins=bins, density=True, color=V[1], alpha=0.8, label=f"{n:,} samples")
axs[0].plot(bins, lam*np.exp(-lam*bins), color=ACCENT, label=r"$\lambda e^{-\lambda x}$")
axs[0].set_yscale("log"); axs[0].set_ylim(1e-5, 2); axs[0].set_title(r"exponential: $x = -\ln(1 - u)/\lambda$", fontsize=14)
bins = np.logspace(0, 3, 40)
axs[1].hist(xp, bins=bins, density=True, color=V[1], alpha=0.8, label=f"{n:,} samples")
axs[1].plot(bins, (alpha - 1)/xmin*(bins/xmin)**(-alpha), color=ACCENT, label=r"$p(x) \propto x^{-2.5}$")
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_ylim(1e-8, 3)
axs[1].set_title(r"power law: $x = x_\mathrm{min}(1 - u)^{1/(1 - \alpha)}$", fontsize=14)
for ax in axs: ax.set_xlabel("$x$"); ax.legend(frameon=False, fontsize=12)
axs[0].set_ylabel("$p(x)$")
fig.tight_layout(); save(fig, "inversion")

# ------------------------------------------- Box-Muller
u1, u2 = rng.random(n), rng.random(n)
r = np.sqrt(-2*np.log(u1)); z1, z2 = r*np.cos(2*np.pi*u2), r*np.sin(2*np.pi*u2)
print(f"[boxmuller] mean {z1.mean():.4f}, {z2.mean():.4f}; std {z1.std():.4f}, {z2.std():.4f}; correlation {np.corrcoef(z1, z2)[0,1]:.4f}")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
axs[0].scatter(z1[:4000], z2[:4000], s=3, color=V[0], alpha=0.5)
axs[0].set_xlim(-4, 4); axs[0].set_ylim(-4, 4); axs[0].set_aspect("equal")
axs[0].set_xlabel("$x_1$"); axs[0].set_ylabel("$x_2$"); axs[0].set_title("4000 pairs $(x_1, x_2)$", fontsize=14)
bins = np.linspace(-5, 5, 61)
axs[1].hist(z1, bins=bins, density=True, color=V[1], alpha=0.8, label=f"{n:,} values of $x_1$")
axs[1].plot(bins, np.exp(-bins**2/2)/np.sqrt(2*np.pi), color=ACCENT, label=r"$e^{-x^2/2}/\sqrt{2\pi}$")
axs[1].set_yscale("log"); axs[1].set_ylim(1e-5, 1); axs[1].set_xlabel("$x_1$"); axs[1].legend(frameon=False, fontsize=12)
axs[1].set_title("the marginal distribution", fontsize=14)
fig.tight_layout(); save(fig, "boxmuller")

# ------------------------------------------- rejection sampling
def target(x):                      # unnormalized bumpy density on [0, 4]
    return np.exp(-(x - 1.2)**2/0.3) + 0.6*np.exp(-(x - 2.8)**2/0.5)
xs = np.linspace(0, 4, 400); Z = np.trapezoid(target(xs), xs); M = target(xs).max()/ (1/4)   # envelope: uniform on [0,4] times M
print(f"[reject] uniform envelope on [0,4], height {target(xs).max():.3f}: acceptance = area/box = {Z/(4*target(xs).max()):.3f}")
npts = 400
xr = rng.random(npts)*4; yr = rng.random(npts)*target(xs).max()
acc = yr < target(xr)
fig, ax = plt.subplots(figsize=(7.6, 4.0))
ax.plot(xs, target(xs), color=INK, label="$p(x)$ (up to a constant)")
ax.axhline(target(xs).max(), color=ACCENT, ls="--", label="envelope")
ax.scatter(xr[acc], yr[acc], s=10, color=V[2], label=f"accepted ({acc.sum()})")
ax.scatter(xr[~acc], yr[~acc], s=10, color=MUTED, alpha=0.6, label=f"rejected ({(~acc).sum()})")
ax.set_xlim(0, 4); ax.set_ylim(0, 1.25); ax.set_xlabel("$x$"); ax.legend(frameon=False, fontsize=12, loc="upper right")
fig.tight_layout(); save(fig, "rejection")
print(f"[reject] {npts} proposals, {acc.sum()} accepted ({acc.mean():.2f})")

# ------------------------------------------- directions on a sphere: theta uniform is wrong
npts = 3000
th_wrong = rng.random(npts)*np.pi; ph = rng.random(npts)*2*np.pi
cos_right = 2*rng.random(npts) - 1; th_right = np.arccos(cos_right)
fig = plt.figure(figsize=(11, 4.6))
for k, (th, title) in enumerate([(th_wrong, r"$\theta$ uniform in $[0, \pi]$: wrong"), (th_right, r"$\cos\theta$ uniform in $[-1, 1]$: right")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.scatter(np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th), s=2, color=V[0] if k == 0 else V[2])
    ax.view_init(elev=60, azim=30); ax.set_title(title, fontsize=14)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_box_aspect((1, 1, 1))
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
fig.tight_layout(); save(fig, "sphere")
print(f"[sphere] fraction within 20 degrees of a pole: theta-uniform {np.mean(np.abs(np.cos(th_wrong)) > np.cos(np.radians(20))):.3f}, cos-uniform {np.mean(np.abs(cos_right) > np.cos(np.radians(20))):.3f}, exact {1 - np.cos(np.radians(20)):.3f}")
