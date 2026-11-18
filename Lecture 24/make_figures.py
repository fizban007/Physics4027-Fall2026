#!/usr/bin/env python3
"""Figures for Lecture 24 (Metropolis examples and simulated annealing).
Run from this directory; writes figures/*.png and prints the numbers quoted
in the speaker notes."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = "#FAFAF7"; INK = "#182430"; MUTED = "#55636F"
V = ["#440154", "#365C8D", "#1FA187", "#A0DA39"]; ACCENT = "#C2410C"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "font.size": 19, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.4, "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})
def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

rng = np.random.default_rng(4027)

# ------------------------------------------- two energy levels
def two_level_run(kT, n=100000):
    """Hop down always, hop up with probability exp(-eps/kT); return the
    fraction of steps spent in the excited state. Energies in units of eps."""
    a_up = math.exp(-1.0 / kT); s = 0; n_excited = 0
    u = rng.random(n)
    for i in range(n):
        if s == 1 or u[i] < a_up:
            s = 1 - s
        n_excited += s
    return n_excited / n

t = np.linspace(0.02, 5, 400); pe = 1 / (1 + np.exp(1 / t))
def curve_axes():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(t, pe, color=V[1])
    ax.axhline(0.5, color=MUTED, lw=1.2, ls=":")
    ax.set_xlim(0, 5); ax.set_ylim(0, 0.56); ax.set_yticks([0, 0.25, 0.5]); ax.set_yticklabels(["0", "", r"$\frac{1}{2}$"])
    ax.set_xlabel(r"$k_BT/\varepsilon$"); ax.set_ylabel(r"$P_e$")
    return fig, ax
fig, ax = curve_axes(); fig.tight_layout(); save(fig, "twolevel_curve")
kTs = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
meas = [two_level_run(kT) for kT in kTs]
for kT, m in zip(kTs, meas):
    print(f"[twolevel] kT = {kT} eps: Boltzmann P_e = {1/(1+math.exp(1/kT)):.4f}, run of 1e5 steps gives {m:.4f}")
fig, ax = curve_axes()
ax.lines[0].set_label("Boltzmann distribution")
ax.plot(kTs, meas, "o", ms=10, color=ACCENT, label="runs of the hopping rule", zorder=5, clip_on=False)
ax.legend(frameon=False, fontsize=16, loc="lower right")
fig.tight_layout(); save(fig, "twolevel_check")

# many two-level systems: the probability of a state against that of an energy
NS = 50; kT = 1.0; a_up = math.exp(-1.0 / kT); p_e = a_up / (1 + a_up)
s = np.zeros(NS, dtype=int); n_exc = 0; n_steps = 400000; counts = np.zeros(NS + 1)
sites = rng.integers(0, NS, n_steps); u = rng.random(n_steps)
for i in range(n_steps):
    k = sites[i]
    if s[k] == 1 or u[i] < a_up:        # flipping one system: down always, up with probability a_up
        n_exc += 1 - 2 * s[k]; s[k] = 1 - s[k]
    if i >= 2000: counts[n_exc] += 1
counts /= counts.sum()
ns = np.arange(NS + 1)
binom = np.array([math.comb(NS, n) * p_e**n * (1 - p_e)**(NS - n) for n in ns])
print(f"[degeneracy] {NS} two-level systems at kT = eps: most likely energy n = {int(np.argmax(binom))} eps "
      f"(run: {int(np.argmax(counts))}), mean {NS*p_e:.2f} (run {np.sum(ns*counts):.2f}); "
      f"P(all in ground state) = {binom[0]:.1e}; states with n = 13: {math.comb(NS, 13):.2e}")
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(ns, counts, color=V[1], alpha=0.8, width=0.85, label="energies visited by a run")
ax.plot(ns, binom, "o", ms=5, color=ACCENT, label=r"(number of states) $\times\; e^{-E/k_BT}$")
ax.set_xlim(-1, 32); ax.set_yticks([]); ax.set_xlabel(r"total energy $E/\varepsilon$ of 50 two-level systems"); ax.set_ylabel("$P(E)$")
ax.annotate("lowest energy", xy=(0, 0.002), xytext=(0, 0.045), ha="center", fontsize=15, color=MUTED,
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
ax.legend(frameon=False, fontsize=15, loc="upper right")
fig.tight_layout(); save(fig, "degeneracy")

# ------------------------------------------- particle in a harmonic well
def metropolis_1d(energy, x0, w, kT, n):
    """Metropolis for one coordinate. A move is a Gaussian step of width w."""
    x = x0; e = energy(x); xs = np.empty(n); acc = 0
    steps = rng.normal(0, w, n); u = rng.random(n)
    for i in range(n):
        xp = x + steps[i]; ep = energy(xp)
        if ep <= e or u[i] < math.exp(-(ep - e) / kT):
            x, e = xp, ep; acc += 1
        xs[i] = x
    return xs, acc / n

harm = lambda x: 0.5 * x * x            # units: spring constant 1
gauss = lambda x, kT: np.exp(-x * x / (2 * kT)) / math.sqrt(2 * math.pi * kT)
g = np.linspace(-3.4, 3.4, 400)
KT_COLD, KT_WARM = 0.25, 1.0

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.plot(g, harm(g), color=INK, label="$E(x)$")
ax.fill_between(g, 2.6 * gauss(g, KT_COLD), color=V[1], alpha=0.75, lw=0)
ax.fill_between(g, 2.6 * gauss(g, KT_WARM), color=ACCENT, alpha=0.55, lw=0)
ax.text(0.55, 1.75, "$P(x)$, cold", color=V[1], fontsize=18)
ax.text(1.55, 0.42, "$P(x)$, warm", color=ACCENT, fontsize=18)
ax.text(-2.55, 2.45, "$E(x)$", color=INK, fontsize=18, ha="right")
ax.set_ylim(0, 3.6); ax.set_xlim(-3.4, 3.4); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("$x$")
fig.tight_layout(); save(fig, "harmonic_behavior")

fig, ax = plt.subplots(figsize=(9, 4.2))
gg = np.linspace(-2.6, 2.6, 300); ax.plot(gg, harm(gg), color=INK)
x0 = 1.2
for x1, c in [(0.35, V[2]), (2.05, ACCENT)]:
    ax.annotate("", xy=(x1, harm(x1)), xytext=(x0, harm(x0)),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2.8, shrinkA=9, shrinkB=2, mutation_scale=24))
ax.plot([x0], [harm(x0)], "o", ms=14, color=V[1], zorder=5)
ax.text(-0.15, 1.05, "downhill: always made", color=V[2], ha="center", fontsize=18)
ax.text(1.95, 2.3, r"uphill by $\Delta E$: made with probability $e^{-\Delta E/k_BT}$", color=ACCENT, ha="right", fontsize=18)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("$x$"); ax.set_ylabel("$E(x)$"); ax.set_ylim(-0.15, 3.4)
fig.tight_layout(); save(fig, "harmonic_moves")

xs, acc = metropolis_1d(harm, 12.0, 0.5, 1.0, 1500)
first = int(np.argmax(np.abs(xs) < 1.0))
print(f"[burnin] kT = 1, start x = 12 sigma, w = 0.5 sigma: first within one sigma of the bottom at step {first}")
fig, ax = plt.subplots(figsize=(11, 3.9))
ax.axvspan(0, first, color=ACCENT, alpha=0.13, lw=0)
ax.text(first + 8, 10.3, "burn-in", color=ACCENT, ha="left", fontsize=18)
ax.plot(xs, color=V[1], lw=1.2)
ax.set_xlabel("step"); ax.set_ylabel(r"$x/\sigma$"); ax.set_xlim(0, 500); ax.set_ylim(-4, 13)
fig.tight_layout(); save(fig, "burnin")

fig, ax = plt.subplots(figsize=(9, 4.4))
for kT, c, lab in [(KT_COLD, V[1], "cold"), (KT_WARM, ACCENT, "warm")]:
    sig = math.sqrt(kT)
    xs, acc = metropolis_1d(harm, 0.0, sig, kT, 200000)
    print(f"[harmonic] kT = {kT}: w = sigma = {sig}, acceptance {acc:.2f}, <x^2> = {xs.var():.4f} "
          f"(exact {kT}), <E> = {harm(xs).mean()/kT:.3f} kT (exact 0.5)")
    ax.hist(xs, bins=70, density=True, color=c, alpha=0.55, label=f"positions visited, {lab}")
    ax.plot(g, gauss(g, kT), color=c, lw=2.4)
ax.plot([], [], color=INK, lw=2.4, label="Boltzmann distribution")
ax.set_xlabel("$x$"); ax.set_yticks([]); ax.set_xlim(-3.4, 3.4); ax.set_xticks([0])
ax.legend(frameon=False, fontsize=15, loc="upper right")
fig.tight_layout(); save(fig, "harmonic_check")
for w in (0.05, 2.0, 50.0):
    xs, acc = metropolis_1d(harm, 0.0, w, 1.0, 20000)
    print(f"[width] kT = 1, w = {w} sigma: acceptance {acc:.3f}, <x^2> from 2e4 steps = {xs.var():.2f}")

# ------------------------------------------- particle in a double well
dw = lambda x: (x * x - 1) ** 2 - 0.2 * x
g = np.linspace(-1.7, 1.7, 2001); eg = dw(g)
iL = np.argmin(np.where(g < 0, eg, 9)); iR = np.argmin(np.where(g > 0, eg, 9))
mid = (g > -0.5) & (g < 0.5); iB = np.where(mid)[0][np.argmax(eg[mid])]
print(f"[doublewell] E = (x^2-1)^2 - 0.2x: shallow well x = {g[iL]:.3f}, E = {eg[iL]:.3f}; deep well x = {g[iR]:.3f}, "
      f"E = {eg[iR]:.3f}; barrier x = {g[iB]:.3f}, E = {eg[iB]:.3f}; barrier from the shallow well = {eg[iB]-eg[iL]:.3f}")
fig, ax = plt.subplots(figsize=(8.4, 4.8))
ax.plot(g, eg, color=INK); ax.set_ylim(-0.8, 2.2)
ax.text(g[iL], eg[iL] - 0.14, "shallow well", ha="center", va="top", color=V[1], fontsize=19)
ax.text(g[iR], eg[iR] - 0.14, "deep well", ha="center", va="top", color=V[1], fontsize=19)
ax.text(g[iB], eg[iB] + 0.12, "barrier", ha="center", color=ACCENT, fontsize=19)
ax.plot([g[iL], -0.4], [eg[iL]] * 2, color=ACCENT, lw=1.2, ls=":")
ax.plot([-0.62, g[iB]], [eg[iB]] * 2, color=ACCENT, lw=1.2, ls=":")
ax.annotate("", xy=(-0.5, eg[iB]), xytext=(-0.5, eg[iL]), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=2))
ax.text(-0.44, 0.58, r"$\Delta E_b$", color=ACCENT, fontsize=19, va="center")
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("$x$"); ax.set_ylabel("$E(x)$")
fig.tight_layout(); save(fig, "doublewell_energy")

fig, axes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True)
cases = [(0.4, g[iL], "hot"), (0.05, g[iL], "cold"), (0.05, g[iR], "cold")]
for ax, (kT, x_start, title) in zip(axes, cases):
    xs, acc = metropolis_1d(dw, x_start, 0.3, kT, 200000)
    right = (xs > g[iB]).mean()
    wgt = np.exp(-eg / kT); exact_right = wgt[g > g[iB]].sum() / wgt.sum()
    crossings = int(np.sum(np.diff(np.sign(xs - g[iB])) != 0))
    print(f"[doublewell] kT = {kT}, start x = {x_start:+.2f}: acceptance {acc:.2f}, {crossings} barrier crossings in 2e5 steps, "
          f"fraction of steps in the deep well {right:.3f} (Boltzmann {exact_right:.4f}); "
          f"exp(-barrier/kT) = {math.exp(-(eg[iB]-eg[iL])/kT):.1e}")
    ax.plot(xs[:20000], color=V[1], lw=0.8)
    ax.axhline(g[iB], color=MUTED, lw=1, ls=":")
    ax.set_ylim(-1.8, 1.8); ax.set_yticks([g[iL], g[iR]]); ax.set_yticklabels(["shallow", "deep"])
    ax.text(1.01, 0.5, title, transform=ax.transAxes, rotation=270, va="center", fontsize=18)
xs, acc = metropolis_1d(dw, g[iR], 0.3, 0.4, 200000)
print(f"[doublewell] kT = 0.4, start in the deep well: fraction of steps in the deep well {(xs > g[iB]).mean():.3f}")
axes[2].set_xlabel("step"); axes[2].set_xticks([0, 10000, 20000])
fig.tight_layout(); save(fig, "doublewell_runs")

# annealing in the double well: many runs at once, from random starting positions
def anneal_runs(tau, n_runs=4000, T0=1.0, T_end=0.01, w=0.3, keep=0):
    n = int(tau * math.log(T0 / T_end))
    x = rng.uniform(-1.5, 1.5, n_runs); e = dw(x); track = []
    for i in range(n):
        T = T0 * math.exp(-i / tau)
        xp = x + rng.normal(0, w, n_runs); ep = dw(xp)
        ok = rng.random(n_runs) < np.exp(-np.maximum(ep - e, 0) / T)
        x = np.where(ok, xp, x); e = np.where(ok, ep, e)
        if keep: track.append(x[:keep].copy())
    return x, np.array(track)
taus = [3, 10, 30, 100, 300, 1000, 3000]
fracs = []
for tau in taus:
    x, _ = anneal_runs(tau); fracs.append((x > g[iB]).mean())
    print(f"[anneal] tau = {tau}: {int(tau*math.log(100))} steps, fraction ending in the deep well {fracs[-1]:.3f}")
x, track = anneal_runs(1000, n_runs=200, keep=1)
tr = track[:, 0]; nn = np.arange(len(tr))
last = int(np.max(np.where(np.diff(np.sign(tr - g[iB])) != 0)[0])) if np.any(np.diff(np.sign(tr - g[iB])) != 0) else 0
print(f"[anneal] trace with tau = 1000: last barrier crossing at step {last}, kT there {math.exp(-last/1000):.2f}, ends at x = {tr[-1]:.2f}")
fig, ax = plt.subplots(figsize=(11, 3.9))
ax.plot(nn, tr, color=V[1], lw=0.8)
ax.axhline(g[iB], color=MUTED, lw=1, ls=":")
ax.set_xlabel("step"); ax.set_ylim(-2.4, 2.4); ax.set_xticks([])
ax.set_yticks([g[iL], g[iR]]); ax.set_yticklabels(["shallow", "deep"])
tw = ax.twinx(); tw.plot(nn, np.exp(-nn / 1000), color=ACCENT, lw=2.4)
tw.set_ylabel("temperature", color=ACCENT); tw.set_yticks([]); tw.set_ylim(0, 1.05)
tw.spines["right"].set_visible(True); tw.spines["right"].set_color(ACCENT)
fig.tight_layout(); save(fig, "anneal_trace")
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.semilogx(taus, fracs, "-o", ms=9, color=V[1])
ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])
ax.set_xlabel(r"cooling time $\tau$ (steps)"); ax.set_ylabel("fraction ending in the deep well")
fig.tight_layout(); save(fig, "anneal_fraction")

# ------------------------------------------- the traveling salesman
# one move on six cities: reversing a stretch of the tour removes a crossing
pts = np.array([[0.0, 0.5], [0.9, 1.25], [2.1, 1.3], [3.0, 0.55], [2.2, -0.2], [0.8, -0.25]])
def draw_tour(ax, order, hi, title):
    loop = list(order) + [order[0]]
    for k in range(len(order)):
        p, q = loop[k], loop[k + 1]
        c = ACCENT if (p, q) in hi or (q, p) in hi else V[1]
        ax.plot(pts[[p, q], 0], pts[[p, q], 1], color=c, lw=3 if c == ACCENT else 2.2)
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=24, color=PAPER, mec=INK, mew=1.8, zorder=5)
    for k, (px, py) in enumerate(pts):
        ax.text(px, py, str(k + 1), ha="center", va="center", fontsize=17, zorder=6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=20)
    ax.set_xlim(-0.4, 3.4); ax.set_ylim(-0.6, 1.65)
    for sp in ax.spines.values(): sp.set_visible(False)
before = [0, 1, 4, 3, 2, 5]; after = [0, 1, 2, 3, 4, 5]
tl = lambda o: sum(np.hypot(*(pts[o[k]] - pts[o[(k + 1) % 6]])) for k in range(6))
print(f"[move] six cities: tour 1 2 5 4 3 6 has length {tl(before):.2f}, tour 1 2 3 4 5 6 has length {tl(after):.2f}")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
draw_tour(axes[0], before, {(1, 4), (2, 5)}, r"before: $1\;\,2\;\,\underline{5\;\,4\;\,3}\;\,6$")
draw_tour(axes[1], after, {(1, 2), (4, 5)}, r"after: $1\;\,2\;\,\underline{3\;\,4\;\,5}\;\,6$")
fig.tight_layout(); save(fig, "tsp_move")

NC = 100
crng = np.random.default_rng(2026)
xy = crng.random((NC, 2))
D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
def length(t): return float(D[t, np.roll(t, -1)].sum())
print(f"[tsp] {NC} cities: (N-1)!/2 = 10^{(math.lgamma(NC) - math.log(2)) / math.log(10):.1f} tours")

def anneal_tsp(tour, T0, T_end, n_steps, seed, snap=None):
    """Reverse a random stretch of the tour; Metropolis with the tour length
    as the energy and T falling exponentially from T0 to T_end."""
    r = np.random.default_rng(seed)
    t = tour.copy(); snaps = []; snap_set = set(snap) if snap is not None else set()
    I = r.integers(0, NC, n_steps); Jx = r.integers(0, NC, n_steps); U = r.random(n_steps)
    rate = math.log(T0 / T_end) / n_steps if T0 > 0 else 0
    for n in range(n_steps):
        if n in snap_set: snaps.append(t.copy())
        i, j = sorted((int(I[n]), int(Jx[n])))
        if j - i < 1 or (i == 0 and j == NC - 1): continue
        a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % NC]
        dL = D[a, c] + D[b, d] - D[a, b] - D[c, d]
        T = T0 * math.exp(-rate * n)
        if dL <= 0 or (T > 0 and U[n] < math.exp(-dL / T)):
            t[i:j + 1] = t[i:j + 1][::-1]
    if snap is not None: return t, length(t), snaps
    return t, length(t)

start = crng.permutation(NC)
dls = []
r = np.random.default_rng(5)
for _ in range(4000):
    i, j = sorted(r.integers(0, NC, 2))
    if j - i < 1 or (i == 0 and j == NC - 1): continue
    a, b, c, d = start[i - 1], start[i], start[j], start[(j + 1) % NC]
    dls.append(abs(D[a, c] + D[b, d] - D[a, b] - D[c, d]))
print(f"[tsp] random tour length {length(start):.2f}; typical |dL| of a move on it {np.mean(dls):.2f}")
n_steps = 1_000_000
runs = {"downhill": anneal_tsp(start, 0.0, 0.0, n_steps, 11),
        "fast": anneal_tsp(start, 0.5, 0.002, 20_000, 11),
        "annealed": anneal_tsp(start, 0.5, 0.002, n_steps, 11)}
for k_, (t_, Lf) in runs.items():
    print(f"[tsp] {k_}: final length {Lf:.3f}")
for name, args in [("downhill", (0.0, 0.0, n_steps)), ("annealed", (0.5, 0.002, n_steps))]:
    Ls = [anneal_tsp(start, *args, seed)[1] for seed in range(20, 25)]
    print(f"[tsp] {name}, five seeds: {np.round(Ls, 3)}, mean {np.mean(Ls):.3f}")

# movie: downhill moves only against annealing, same starting tour and random numbers
from matplotlib.animation import FFMpegWriter
frame_steps = sorted(set(list(range(0, 20000, 400)) + list(range(20000, n_steps, 4000))))
_, _, snaps_down = anneal_tsp(start, 0.0, 0.0, n_steps, 11, snap=frame_steps)
_, _, snaps_ann = anneal_tsp(start, 0.5, 0.002, n_steps, 11, snap=frame_steps)
snaps_down.append(runs["downhill"][0]); snaps_ann.append(runs["annealed"][0]); frame_steps.append(n_steps)
print(f"[movie] {len(frame_steps)} frames; dense sampling for the first 20000 steps")
with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 15}):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.08, wspace=0.06)
    lines = []; labels = []
    for ax, title in zip(axes, ("downhill moves only", "annealed")):
        ln, = ax.plot([], [], color=V[1], lw=1.4)
        ax.plot(xy[:, 0], xy[:, 1], "o", ms=5, color=INK)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(title, fontsize=20, pad=10)
        lines.append(ln); labels.append(ax.text(0.5, -0.045, "", transform=ax.transAxes, ha="center", va="top", fontsize=17))
    head = fig.text(0.5, 0.955, "", ha="center", va="center", fontsize=17, color=MUTED)
    def draw(k):
        n = frame_steps[k]
        for ln, lab, tour in zip(lines, labels, (snaps_down[k], snaps_ann[k])):
            loop = np.append(tour, tour[0]); ln.set_data(xy[loop, 0], xy[loop, 1])
            lab.set_text(f"tour length {length(tour):.2f}")
        T = 0.5 * math.exp(-math.log(250) * n / n_steps)
        head.set_text(f"step {n:,}        annealing temperature {T:.3f}")
    writer = FFMpegWriter(fps=24, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "23"])
    with writer.saving(fig, "figures/tsp_anneal.mp4", dpi=100):
        for k in range(len(frame_steps)):
            draw(k); writer.grab_frame()
        for _ in range(60): writer.grab_frame()       # hold the final tours
    fig.savefig("figures/tsp_anneal_poster.png", dpi=100)
    plt.close(fig)
print("done")
