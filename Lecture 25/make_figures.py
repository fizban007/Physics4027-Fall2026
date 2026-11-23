#!/usr/bin/env python3
"""Figures for Lecture 25 (Bayesian inference).
Run from this directory; writes figures/*.png and prints the numbers quoted
in the speaker notes. figures/corner_pulsar.png is not made here."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches

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

def metropolis(nlp, start, widths, n):
    """Metropolis in several parameters. nlp(x) is minus the log of the
    probability to sample, up to a constant. A move adds a Gaussian step to
    every parameter, with its own width for each. Returns the chain and the
    fraction of moves made."""
    x = np.array(start, dtype=float); e = nlp(x)
    chain = np.empty((n, x.size)); made = 0
    steps = rng.normal(0.0, 1.0, (n, x.size)) * np.asarray(widths); u = rng.random(n)
    for i in range(n):
        xp = x + steps[i]; ep = nlp(xp)
        if ep <= e or u[i] < math.exp(-(ep - e)):
            x, e = xp, ep; made += 1
        chain[i] = x
    return chain, made / n

def correlation_steps(x):
    """Integrated autocorrelation time of a series, in steps."""
    x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac /= ac[0]
    total = 1.0
    for k in range(1, n):
        if ac[k] < 0.05: break
        total += 2 * ac[k]
    return total

# ------------------------------------------- the medical test
PREVALENCE, HIT, FALSE_ALARM = 0.001, 0.99, 0.01
POPULATION = 100000
sick = PREVALENCE * POPULATION; healthy = POPULATION - sick
true_pos = HIT * sick; false_pos = FALSE_ALARM * healthy
print(f"[test] of {POPULATION} people {sick:.0f} are sick and {healthy:.0f} healthy; "
      f"positives: {true_pos:.0f} sick + {false_pos:.0f} healthy = {true_pos + false_pos:.0f}; "
      f"P(sick | +) = {true_pos / (true_pos + false_pos):.4f}")
fig, ax = plt.subplots(figsize=(10, 2.3))
ax.barh([0], [true_pos], color=ACCENT, height=0.6)
ax.barh([0], [false_pos], left=[true_pos], color=V[1], height=0.6, alpha=0.85)
ax.text(0, 0.42, f"{true_pos:.0f} are sick", color=ACCENT, ha="left", va="bottom", fontsize=21)
ax.text(true_pos + false_pos / 2, 0, f"{false_pos:.0f} are healthy", color=PAPER, ha="center", va="center", fontsize=21)
ax.set_xlim(0, true_pos + false_pos); ax.set_ylim(-0.4, 1.0); ax.set_yticks([]); ax.set_xticks([])
for side in ax.spines.values(): side.set_visible(False)
ax.set_xlabel(f"the {true_pos + false_pos:.0f} positive tests among {POPULATION:,} people".replace(",", r"\,"))
fig.tight_layout(); save(fig, "test_counts")

# ------------------------------------------- the thrown ball: data and model
# The same 15 measurements as the least-squares example of Lecture 16.
G_TRUE, V0_TRUE, Y0_TRUE, NOISE = 9.81, 9.0, 1.2, 0.15
t = np.linspace(0.0, 1.8, 15)
y = Y0_TRUE + V0_TRUE * t - 0.5 * G_TRUE * t**2 + NOISE * np.random.default_rng(1).standard_normal(t.size)
A = np.vstack([np.ones_like(t), t, -0.5 * t**2]).T           # parameters (y0, v0, g)
best = np.linalg.lstsq(A, y, rcond=None)[0]
cov = NOISE**2 * np.linalg.inv(A.T @ A); err = np.sqrt(np.diag(cov))
chi2 = lambda p: np.sum((A @ p - y)**2) / NOISE**2
ball_nlp = lambda p: 0.5 * chi2(p)
print(f"[ball] least squares: y0 = {best[0]:.3f}, v0 = {best[1]:.3f}, g = {best[2]:.3f}; chi2_min = {chi2(best):.2f} for 15 points")
print(f"[ball] exact errors sigma^2 (A^T A)^-1: y0 {err[0]:.3f}, v0 {err[1]:.3f}, g {err[2]:.3f}; "
      f"correlation of v0 and g = {cov[1, 2] / (err[1] * err[2]):.2f}")

def best_given_g(g):
    """Least-squares y0 and v0 when g is held fixed."""
    p = np.linalg.lstsq(A[:, :2], y - A[:, 2] * g, rcond=None)[0]
    return np.array([p[0], p[1], g])

tt = np.linspace(0, 1.8, 200); curve = lambda p, s=tt: p[0] + p[1] * s - 0.5 * p[2] * s**2
ALTERNATIVES = [(9.3, "--"), (10.5, ":")]          # other values of g, with their line styles
for g_alt, ls in ALTERNATIVES:
    print(f"[ball] parabola with g = {g_alt} and the other two refitted: chi2 = {chi2(best_given_g(g_alt)):.2f}")

def draw_parabolas(ax):
    """The data with the least-squares parabola and the two alternatives."""
    for g_alt, ls in ALTERNATIVES:
        ax.plot(tt, curve(best_given_g(g_alt)), color=MUTED, lw=1.8, ls=ls, label=r"$g = %.1f$" % g_alt)
    ax.plot(tt, curve(best), color=ACCENT, label=r"least squares, $g = %.2f$" % best[2])
    ax.errorbar(t, y, yerr=NOISE, fmt="o", color=V[1], ms=6, capsize=3, lw=1.4, zorder=5)
    ax.set_xlabel("$t$ (s)"); ax.set_ylabel("height (m)")
    h, l = ax.get_legend_handles_labels(); order = [2, 0, 1]
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=16, loc="lower center")

fig, ax = plt.subplots(figsize=(7.6, 5.0))
draw_parabolas(ax)
fig.tight_layout(); save(fig, "ball_data")

# the likelihood of one measurement, drawn schematically on the ball's path
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot(tt, curve(best), color=INK)
ti = 0.5; yi = curve(best, ti); width = 0.55; meas = yi + 0.75 * width
yy = np.linspace(yi - 3 * width, yi + 3 * width, 200)
bell = 0.28 * np.exp(-(yy - yi)**2 / (2 * width**2))
ax.fill_betweenx(yy, ti, ti + bell, color=V[1], alpha=0.35, lw=0); ax.plot(ti + bell, yy, color=V[1], lw=1.8)
ax.plot([ti, ti], [yy[0], yy[-1]], color=MUTED, lw=1.0)
ax.plot([ti], [yi], "o", color=INK, ms=7, zorder=6)
ax.plot([ti], [meas], "o", color=ACCENT, ms=11, zorder=5)
# a bar of height sigma with flat end caps; arrowheads overlap at this length
ax.annotate("", xy=(ti + 0.36, yi + width), xytext=(ti + 0.36, yi),
            arrowprops=dict(arrowstyle="|-|,widthA=0.35,widthB=0.35", color=V[1], lw=1.8, shrinkA=0, shrinkB=0))
ax.text(ti + 0.40, yi + width / 2, r"$\sigma$", color=V[1], va="center", fontsize=20)
ax.text(ti - 0.05, meas, r"measured $y_i$", color=ACCENT, ha="right", va="center", fontsize=18)
ax.annotate(r"predicted $y(t_i;\theta)$", xy=(ti + 0.02, yi - 0.04), xytext=(ti + 0.30, yi - 2.3), color=INK, fontsize=18,
            ha="left", va="center", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
ax.text(1.5, curve(best, 1.5) + 0.25, r"model $y(t;\theta)$", color=INK, ha="left", fontsize=18)
ax.set_xlim(-0.35, 2.3); ax.set_ylim(0.3, 6.6); ax.set_xticks([ti]); ax.set_xticklabels(["$t_i$"]); ax.set_yticks([])
ax.set_xlabel("$t$"); ax.set_ylabel("$y$")
fig.tight_layout(); save(fig, "likelihood_point")

# the range of g shown in the histograms
gg = np.linspace(best[2] - 3.4 * err[2], best[2] + 3.4 * err[2], 300)

# ------------------------------------------- the thrown ball: the chain
BALL_WIDTHS = [0.04, 0.10, 0.10]          # steps in y0 (m), v0 (m/s), g (m/s^2)
BALL_START = [0.0, 5.0, 5.0]; N_BALL = 400000; BURN = 2000
chain, made = metropolis(ball_nlp, BALL_START, BALL_WIDTHS, N_BALL)
e_hist = np.array([ball_nlp(p) for p in chain[:3000]])
settled = int(np.argmax(e_hist < ball_nlp(best) + 3.0))
kept = chain[BURN:]; tau_g = correlation_steps(kept[:, 2])
print(f"[chain] start {BALL_START} has chi2 = {chi2(np.array(BALL_START)):.0f}; chi2/2 first within 3 of the minimum at step {settled}")
print(f"[chain] {N_BALL} steps, {100 * made:.0f}% of moves made, first {BURN} discarded; "
      f"correlation time of g = {tau_g:.0f} steps, so about {len(kept) / tau_g:.0f} independent samples")
for k, name in enumerate(["y0", "v0", "g"]):
    print(f"[chain] {name}: chain {kept[:, k].mean():.3f} +- {kept[:, k].std():.3f}, least squares {best[k]:.3f} +- {err[k]:.3f}")
print(f"[chain] correlation of v0 and g in the chain = {np.corrcoef(kept[:, 1], kept[:, 2])[0, 1]:.2f}")
lo, hi = np.percentile(kept[:, 2], [16, 84])
print(f"[chain] 68% of the g samples lie between {lo:.2f} and {hi:.2f}; fraction with g between 9.6 and 10.2: {np.mean((kept[:, 2] > 9.6) & (kept[:, 2] < 10.2)):.2f}")

# what a list of samples is: a few rows of the chain, and the histogram of the g column
fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.48))
left.set_xlim(0, 1); left.set_ylim(0, 1); left.axis("off")
columns = [0.17, 0.50, 0.83]; headers = [r"$y_0$ (m)", r"$v_0$ (m/s)", r"$g$ (m/s$^2$)"]
left.add_patch(plt.Rectangle((0.68, 0.10), 0.30, 0.90, color=V[1], alpha=0.13, lw=0))
for k, (x, head) in enumerate(zip(columns, headers)):
    left.text(x, 0.93, head, ha="center", va="center", fontsize=19, color=V[1] if k == 2 else INK)
left.plot([0.02, 0.98], [0.855, 0.855], color=MUTED, lw=1.0)
rows = kept[::50000][:6]
for j, row in enumerate(rows):
    for k, x in enumerate(columns):
        left.text(x, 0.76 - 0.102 * j, r"$%.2f$" % row[k], ha="center", va="center", fontsize=19,
                  color=V[1] if k == 2 else INK)
for k, x in enumerate(columns):
    left.text(x, 0.76 - 0.102 * len(rows), r"$\vdots$", ha="center", va="center", fontsize=19, color=V[1] if k == 2 else INK)
left.text(0.34, 0.02, "one row is one sample", ha="center", va="center", fontsize=16, color=MUTED)
few = kept[::200, 2]
right.hist(few, bins=30, color=V[1], alpha=0.8)
right.set_xlim(gg[0], gg[-1]); right.set_xticks([9.0, 9.5, 10.0, 10.5]); right.set_yticks([])
right.set_xlabel(r"$g$ (m/s$^2$)"); right.set_ylabel("number of samples")
fig.add_artist(matplotlib.patches.FancyArrowPatch((0.455, 0.52), (0.535, 0.52), transform=fig.transFigure,
               arrowstyle="-|>", mutation_scale=26, color=V[1], lw=2.2))
print(f"[list] the table shows {len(rows)} rows of the chain; the histogram uses {len(few)} of them")
save(fig, "sample_list")

fig, ax = plt.subplots(figsize=(11, 4.2))
n_show = 6000
ax.axvspan(0, BURN, color=MUTED, alpha=0.15, lw=0)
ax.plot(np.arange(n_show), chain[:n_show, 2], color=V[1], lw=1.2)
ax.text(BURN / 2, 5.4, "burn-in", color=MUTED, ha="center", fontsize=18)
ax.set_xlim(0, n_show); ax.set_xlabel("step"); ax.set_ylabel(r"$g$ (m/s$^2$)")
fig.tight_layout(); save(fig, "ball_chain")

fig, ax = plt.subplots(figsize=(7.2, 5.0))
picks = kept[rng.integers(0, len(kept), 80)]
for p in picks: ax.plot(tt, curve(p), color=V[1], lw=0.8, alpha=0.3)
ax.errorbar(t, y, yerr=NOISE, fmt="o", color=INK, ms=5, capsize=3, lw=1.3, zorder=5)
ax.set_xlabel("$t$ (s)"); ax.set_ylabel("height (m)")
fig.tight_layout(); save(fig, "ball_samples")

def g_histogram(with_exact):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.hist(kept[:, 2], bins=60, density=True, color=V[1], alpha=0.8, label="chain")
    if with_exact:
        ax.plot(gg, np.exp(-(gg - best[2])**2 / (2 * err[2]**2)) / (err[2] * math.sqrt(2 * math.pi)), color=ACCENT,
                label="exact, from least squares")
        ax.legend(frameon=False, fontsize=16, loc="upper left"); ax.set_ylim(0, 1.95)
    ax.set_xlim(gg[0], gg[-1]); ax.set_yticks([]); ax.set_xlabel(r"$g$ (m/s$^2$)"); ax.set_ylabel(r"$P(g\,|\,\mathrm{data})$")
    return fig
fig = g_histogram(False); fig.tight_layout(); save(fig, "ball_hist")
fig = g_histogram(True); fig.tight_layout(); save(fig, "ball_check")

# ------------------------------------------- a decay on a background
A_TRUE, TAU_TRUE, B_TRUE, DECAY_NOISE = 10.0, 2.0, 2.0, 0.4
decay = lambda p, s: p[0] * np.exp(-s / p[1]) + p[2]
def decay_data(t_end, n, seed):
    s = np.linspace(0.0, t_end, n)
    return s, decay([A_TRUE, TAU_TRUE, B_TRUE], s) + DECAY_NOISE * np.random.default_rng(seed).standard_normal(n)
TAU_MAX = 20.0
def decay_nlp(s, d, background=None):
    """chi^2/2 minus the log of the prior. The prior is flat for A > 0,
    0 < tau < TAU_MAX and B >= 0, times an optional Gaussian in B."""
    def nlp(p):
        if p[0] <= 0 or p[1] <= 0 or p[1] > TAU_MAX or p[2] < 0: return math.inf
        e = 0.5 * np.sum((decay(p, s) - d)**2) / DECAY_NOISE**2
        if background is not None: e += 0.5 * ((p[2] - background[0]) / background[1])**2
        return e
    return nlp
DECAY_WIDTHS = [0.25, 0.15, 0.25]; DECAY_START = [8.0, 3.0, 1.0]; N_DECAY = 1000000
def decay_chain(s, d, background=None, widths=DECAY_WIDTHS):
    c, m = metropolis(decay_nlp(s, d, background), DECAY_START, widths, N_DECAY)
    return c[BURN:], m
def summary(x):
    lo, mid, hi = np.percentile(x, [16, 50, 84]); return f"{mid:.2f} +{hi - mid:.2f} -{mid - lo:.2f}"

DECAY_SEED = 19
ts, ds = decay_data(4.0, 20, DECAY_SEED)
dchain, dmade = decay_chain(ts, ds)
names = [r"$A$", r"$\tau$", r"$B$"]
print(f"[decay] 20 points to t = 4, noise {DECAY_NOISE}; {N_DECAY} steps, {100 * dmade:.0f}% of moves made, "
      f"correlation time of tau = {correlation_steps(dchain[::10, 1]) * 10:.0f} steps")
for k, name in enumerate(["A", "tau", "B"]):
    print(f"[decay] {name}: median and 68% interval {summary(dchain[:, k])}; mean {dchain[:, k].mean():.2f}, std {dchain[:, k].std():.2f}")
cc = np.corrcoef(dchain.T)
print(f"[decay] correlations: A-tau {cc[0, 1]:.2f}, A-B {cc[0, 2]:.2f}, tau-B {cc[1, 2]:.2f}")

fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.errorbar(ts, ds, yerr=DECAY_NOISE, fmt="o", color=V[1], ms=6, capsize=3, lw=1.4)
ax.set_xlabel("$t$"); ax.set_ylabel("$y$"); ax.set_ylim(0, 13); ax.set_xlim(-0.2, 4.2)
fig.tight_layout(); save(fig, "decay_data")

def contour_levels(hist, fractions):
    """Heights of a 2D histogram that enclose the given fractions of the samples."""
    flat = np.sort(hist.ravel())[::-1]; cum = np.cumsum(flat) / flat.sum()
    return sorted(flat[np.searchsorted(cum, f)] for f in fractions)
def smooth(hist):
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    for axis in (0, 1):
        hist = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), axis, hist)
    return hist
ranges = [np.percentile(dchain[:, k], [0.1, 99.9]) for k in range(3)]
def pair_panel(ax, i, j, cmap="Blues"):
    """Density of parameter j (horizontal) against parameter i (vertical)."""
    hist, xe, ye = np.histogram2d(dchain[:, j], dchain[:, i], bins=45, range=[ranges[j], ranges[i]])
    hist = smooth(hist).T; xc = 0.5 * (xe[1:] + xe[:-1]); yc = 0.5 * (ye[1:] + ye[:-1])
    ax.pcolormesh(xe, ye, hist, cmap=cmap, vmin=0, vmax=hist.max() * 1.25, rasterized=True)
    ax.contour(xc, yc, hist, levels=contour_levels(hist, [0.95, 0.68]), colors=[V[1], V[0]], linewidths=1.6)

fig, axs = plt.subplots(3, 3, figsize=(8.6, 8.2))
for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        if j > i: ax.axis("off"); continue
        if i == j:
            ax.hist(dchain[:, i], bins=50, range=ranges[i], color=V[1], alpha=0.8); ax.set_yticks([])
            ax.set_title(names[i], fontsize=20, pad=6)
        else:
            pair_panel(ax, i, j); ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
            if j == 0: ax.set_ylabel(names[i], fontsize=21)
            else: ax.set_yticklabels([])
        ax.set_xlim(ranges[j])
        if i == 2: ax.set_xlabel(names[j], fontsize=21)
        else: ax.set_xticklabels([])
        ax.tick_params(labelsize=14); ax.locator_params(nbins=4)
fig.tight_layout(h_pad=0.4, w_pad=0.4); save(fig, "decay_corner")

# two states of the chain from opposite ends of the tau-B band
band = dchain[::500]
order = np.argsort(band[:, 1])
long_state = band[order[int(0.97 * len(band))]]; short_state = band[order[int(0.05 * len(band))]]
e_of = decay_nlp(ts, ds)
for label, p in [("long", long_state), ("short", short_state)]:
    print(f"[pair] {label}-lifetime state: A = {p[0]:.2f}, tau = {p[1]:.2f}, B = {p[2]:.2f}, chi2 = {2 * e_of(p):.1f}")
print(f"[pair] smallest chi2 seen by the chain = {2 * min(e_of(p) for p in dchain[::200]):.1f} for 20 points")
fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axs[0]
pair_panel(ax, 2, 1); ax.set_xlim(ranges[1]); ax.set_ylim(ranges[2])
ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
ax.plot(*long_state[1:], "s", color=ACCENT, ms=11); ax.plot(*short_state[1:], "o", color=V[2], ms=12)
ax.set_xlabel(names[1]); ax.set_ylabel(names[2])
ax = axs[1]; s_long = np.linspace(0, 10, 300)
ax.axvspan(4.0, 10, color=MUTED, alpha=0.10, lw=0)
ax.plot(s_long, decay(long_state, s_long), color=ACCENT, label=r"long $\tau$, low $B$")
ax.plot(s_long, decay(short_state, s_long), color=V[2], label=r"short $\tau$, high $B$")
ax.errorbar(ts, ds, yerr=DECAY_NOISE, fmt="o", color=INK, ms=5, capsize=3, lw=1.3, zorder=5)
ax.text(7.0, 9.3, "no data", color=MUTED, ha="center", fontsize=18)
ax.set_xlim(-0.2, 10); ax.set_ylim(0, 13); ax.set_xlabel("$t$"); ax.set_ylabel("$y$")
ax.legend(frameon=False, fontsize=16, loc="upper right", bbox_to_anchor=(1.0, 0.68))
fig.tight_layout(); save(fig, "decay_pair")

# ------------------------------------------- the prior
BACKGROUND = (2.0, 0.2)            # an earlier measurement of B with no source
informed, m_inf = decay_chain(ts, ds, BACKGROUND)
print(f"[prior] 20 measurements, to t = 4: tau = {summary(dchain[:, 1])} with the flat prior, {summary(informed[:, 1])} with "
      f"B = {BACKGROUND[0]} +- {BACKGROUND[1]} ({100 * m_inf:.0f}% of moves made); B = {summary(dchain[:, 2])} and {summary(informed[:, 2])}")

fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]                        # the plane of tau and B: the band, the measured B, and what is left
pair_panel(ax, 2, 1); ax.set_xlim(ranges[1]); ax.set_ylim(ranges[2])
ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
ax.axhspan(BACKGROUND[0] - BACKGROUND[1], BACKGROUND[0] + BACKGROUND[1], color=ACCENT, alpha=0.16, lw=0)
ax.text(ranges[1][1] - 0.04, BACKGROUND[0] + BACKGROUND[1] + 0.08, r"measured $B$", color=ACCENT, ha="right", va="bottom", fontsize=18)
hist, xe, ye = np.histogram2d(informed[:, 1], informed[:, 2], bins=45, range=[ranges[1], ranges[2]])
hist = smooth(hist).T
ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), hist, levels=contour_levels(hist, [0.95, 0.68]),
           colors=[ACCENT, ACCENT], linewidths=2.0)
ax.set_xlabel(names[1]); ax.set_ylabel(names[2])
ax = axs[1]                        # the posterior of tau with each prior
bins = np.linspace(ranges[1][0], ranges[1][1], 70)
ax.hist(dchain[:, 1], bins=bins, density=True, color=V[1], alpha=0.75, label="flat prior")
ax.hist(informed[:, 1], bins=bins, density=True, histtype="step", color=ACCENT, lw=2.4, label=r"prior from the measured $B$")
ax.set_xlabel(names[1]); ax.set_ylabel(r"$P(\tau\,|\,\mathrm{data})$"); ax.set_yticks([]); ax.set_xlim(ranges[1])
ax.set_ylim(0, 1.42 * ax.get_ylim()[1]); ax.set_xticks([1.5, 2.0, 2.5, 3.0])
ax.legend(frameon=False, fontsize=16, loc="upper right")
fig.tight_layout(); save(fig, "prior_effect")

# for the speaker notes: with more data, out to the background, the same prior changes nothing
t_long, d_long = decay_data(20.0, 80, DECAY_SEED); long_widths = [0.2, 0.07, 0.05]
flat_long = decay_chain(t_long, d_long, widths=long_widths)[0]
informed_long, m_long = decay_chain(t_long, d_long, BACKGROUND, long_widths)
print(f"[prior] 80 measurements, to t = 20: tau = {summary(flat_long[:, 1])} with the flat prior, {summary(informed_long[:, 1])} with "
      f"B = {BACKGROUND[0]} +- {BACKGROUND[1]} ({100 * m_long:.0f}% of moves made); B = {summary(flat_long[:, 2])} and {summary(informed_long[:, 2])}")
