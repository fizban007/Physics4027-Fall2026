#!/usr/bin/env python3
"""Figures for Lecture 22 (Monte Carlo integration and importance sampling).
Run from this directory; writes figures/*.png and prints the numbers quoted
on the slides."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate, special

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

# ------------------------------------------- darts at a quarter circle
N = 2000
x, y = rng.random(N), rng.random(N)
inside = x*x + y*y < 1
est = 4*inside.sum()/N
print(f"[darts] N={N}: {inside.sum()} inside, estimate 4*{inside.sum()}/{N} = {est:.4f} (pi = {np.pi:.4f})")
fig, ax = plt.subplots(figsize=(5.4, 5.4))
th = np.linspace(0, np.pi/2, 200)
ax.plot(np.cos(th), np.sin(th), color=INK, lw=1.8)
ax.scatter(x[inside], y[inside], s=5, color=V[2], label=f"inside: {inside.sum()}")
ax.scatter(x[~inside], y[~inside], s=5, color=MUTED, alpha=0.7, label=f"outside: {(~inside).sum()}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.set_title(rf"$N = {N}$, ${inside.sum()}$ inside: $4 \times {inside.sum()}/{N} = {est:.3f}$", fontsize=15)
fig.tight_layout(); save(fig, "darts")

# ------------------------------------------- the error falls as 1/sqrt(N), and is Gaussian
p = np.pi/4; sig_f = np.sqrt(p*(1 - p))
print(f"[clt] hit-or-miss: p = pi/4 = {p:.4f}, sigma_f = sqrt(p(1-p)) = {sig_f:.4f}, error of pi estimate = 4 sigma_f/sqrt(N) = {4*sig_f:.3f}/sqrt(N)")
Ns = np.logspace(1, 7, 13).astype(int)
repeats = 400
rms = []
for n in Ns:                                   # the number of hits is exactly binomial(N, p)
    k = rng.binomial(n, p, size=repeats)
    rms.append(np.sqrt(np.mean((4*k/n - np.pi)**2)))
rms = np.array(rms)
for n, r in zip(Ns[::2], rms[::2]):
    print(f"[clt]   N = {n:>8d}: rms error {r:.5f}, predicted {4*sig_f/np.sqrt(n):.5f}")
n0 = 1000; k = rng.binomial(n0, p, size=4000); ests = 4*k/n0
sig0 = 4*sig_f/np.sqrt(n0)
print(f"[clt] N = {n0}, 4000 repeats: mean {ests.mean():.4f}, std {ests.std():.4f}, predicted {sig0:.4f}")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].loglog(Ns, rms, "o", color=V[0], label="measured, 400 runs at each $N$")
axs[0].loglog(Ns, 4*sig_f/np.sqrt(Ns), color=ACCENT, label=r"$4\sqrt{p(1-p)/N}$")
axs[0].set_xlabel("$N$"); axs[0].set_ylabel("rms error of the estimate")
axs[0].set_title(r"error $\propto 1/\sqrt{N}$", fontsize=14); axs[0].legend(frameon=False, fontsize=12)
bins = np.linspace(np.pi - 4*sig0, np.pi + 4*sig0, 41)
axs[1].hist(ests, bins=bins, density=True, color=V[1], alpha=0.8, label=f"4000 runs, $N = {n0}$")
axs[1].plot(bins, np.exp(-(bins - np.pi)**2/(2*sig0**2))/(sig0*np.sqrt(2*np.pi)), color=ACCENT, label=rf"Gaussian, $\sigma = {sig0:.3f}$")
axs[1].axvline(np.pi, color=INK, ls=":", lw=1.5)
axs[1].set_xlabel("estimate of $\\pi$"); axs[1].set_title("the estimates are Gaussian", fontsize=14)
axs[1].legend(frameon=False, fontsize=12)
fig.tight_layout(); save(fig, "clt")

# ------------------------------------------- error exponent against dimension
d = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.plot(d, 2/d, "o-", color=V[1], label="trapezoid: $2/d$")
ax.plot(d, 4/d, "s-", color=V[0], label="Simpson: $4/d$")
ax.axhline(0.5, color=ACCENT, label="Monte Carlo: $1/2$")
ax.set_ylim(0, 2.1); ax.set_xlim(0.5, 12.5); ax.set_xticks(d)
ax.set_xlabel("dimension $d$"); ax.set_ylabel("$q$ in error $\\propto N^{-q}$")
ax.legend(frameon=False, fontsize=13)
fig.tight_layout(); save(fig, "exponent")
print("[dim] trapezoid error N^(-2/d) drops below N^(-1/2) at d = 4; Simpson N^(-4/d) at d = 8")

# ------------------------------------------- an integrand with no smoothness at all
def nasty(x):
    return np.sin(1/(x*(2 - x)))**2
# exact value: with t = 1/(x(2-x)), I = int_1^inf sin^2(t) t^-2 (1-1/t)^-1/2 dt
# = 1 - (1/2) int_1^inf g(t) cos 2t dt; t = 1 + v^2 removes the endpoint singularity.
osc = lambda v: 2*(1 + v*v)**(-1.5)*np.cos(2 + 2*v*v)
J = 0.0
edges = np.concatenate([[0], np.sqrt(np.pi*np.arange(1, 40001))])   # one half-period of cos(2v^2) per piece
for a, b in zip(edges[:-1], edges[1:]):
    J += integrate.quad(osc, a, b)[0]
exact_nasty = 1 - 0.5*J
print(f"[nasty] exact value of int_0^2 sin^2(1/(x(2-x))) dx = {exact_nasty:.6f} (tail cut at v = {edges[-1]:.0f}, bound {1/edges[-1]**2:.1e})")
# a sample of f, for sigma_f
xs = rng.random(10**6)*2; fs = nasty(xs)
sig_nasty = fs.std()
print(f"[nasty] <f> = {fs.mean():.4f}, sigma_f = {sig_nasty:.4f}; MC error 2 sigma_f/sqrt(N): N=1e4 {2*sig_nasty/1e2:.4f}, N=1e6 {2*sig_nasty/1e3:.5f}")
def simpson(f, a, b, n):                # composite Simpson with n (even) intervals
    h = (b - a)/n; x = a + h*np.arange(n + 1)
    fx = np.empty(n + 1); fx[1:-1] = f(x[1:-1]); fx[0] = fx[-1] = 0.5   # f oscillates at the ends; take its mean value
    w = np.ones(n + 1); w[1:-1:2] = 4; w[2:-1:2] = 2
    return h/3*np.sum(w*fx)
Nq = 2**np.arange(3, 21)
err_s = np.array([abs(simpson(nasty, 0, 2, n) - exact_nasty) for n in Nq])
Nm = np.logspace(1, 6, 11).astype(int)
err_m = []
for n in Nm:
    e = [abs(2*nasty(rng.random(n)*2).mean() - exact_nasty) for _ in range(40)]
    err_m.append(np.sqrt(np.mean(np.square(e))))
err_m = np.array(err_m)
for n, e in zip(Nq[::3], err_s[::3]):
    print(f"[nasty]   Simpson N = {n:>8d}: error {e:.2e}")
for n, e in zip(Nm[::2], err_m[::2]):
    print(f"[nasty]   MC      N = {n:>8d}: rms error {e:.2e}")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
xp = np.linspace(1e-4, 2 - 1e-4, 40000)
axs[0].plot(xp, nasty(xp), color=V[0], lw=0.6)
axs[0].set_xlim(0, 2); axs[0].set_ylim(0, 1.05); axs[0].set_xlabel("$x$")
axs[0].set_title(r"$f(x) = \sin^2\!\big(1/x(2-x)\big)$", fontsize=14)
axs[1].loglog(Nq, err_s, "s-", color=V[1], label="Simpson, Lecture 5")
axs[1].loglog(Nm, err_m, "o", color=V[0], label="Monte Carlo, rms of 40 runs")
axs[1].loglog(Nm, 2*sig_nasty/np.sqrt(Nm), color=ACCENT, label=r"$2\sigma_f/\sqrt{N}$")
axs[1].set_xlabel("$N$ function evaluations"); axs[1].set_ylabel("error"); axs[1].set_title("error of the two methods", fontsize=14)
axs[1].legend(frameon=False, fontsize=12)
fig.tight_layout(); save(fig, "nasty")

# ------------------------------------------- importance sampling: int_0^1 e^-x / sqrt(x) dx
exact_imp = np.sqrt(np.pi)*special.erf(1)
print(f"[importance] exact int_0^1 e^-x/sqrt(x) dx = sqrt(pi) erf(1) = {exact_imp:.5f}")
n, reps = 10**4, 1000
plain = np.array([(np.exp(-x)/np.sqrt(x)).mean() for x in 1 - rng.random((reps, n))])
imp = np.array([(2*np.exp(-u*u)).mean() for u in rng.random((reps, n))])
print(f"[importance] N = {n}, {reps} runs: plain mean {plain.mean():.4f} std {plain.std():.4f} (largest {plain.max():.3f}); importance mean {imp.mean():.4f} std {imp.std():.4f}")
ui = rng.random(10**6); w = 2*np.exp(-ui*ui)
print(f"[importance] f/p = 2e^-x ranges over [{w.min():.3f}, {w.max():.3f}], sigma(f/p) = {w.std():.4f} -> error {w.std()/np.sqrt(n):.4f} at N = {n}")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
xx = np.linspace(0.002, 1, 500)
axs[0].plot(xx, np.exp(-xx)/np.sqrt(xx), color=V[0], label=r"$f = e^{-x}/\sqrt{x}$")
axs[0].plot(xx, 1/(2*np.sqrt(xx)), color=V[1], ls="--", label=r"$p = 1/2\sqrt{x}$")
axs[0].plot(xx, 2*np.exp(-xx), color=ACCENT, label=r"$f/p = 2e^{-x}$")
axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 8); axs[0].set_xlabel("$x$"); axs[0].legend(frameon=False, fontsize=12)
axs[0].set_title("what gets averaged", fontsize=14)
bins = np.linspace(1.44, 1.56, 61)
axs[1].hist(plain, bins=bins, density=True, color=V[1], alpha=0.75, label=rf"$x$ uniform: $\sigma = {plain.std():.3f}$")
axs[1].hist(imp, bins=bins, density=True, color=V[2], alpha=0.75, label=rf"$x = u^2$: $\sigma = {imp.std():.4f}$")
axs[1].axvline(exact_imp, color=INK, ls=":", lw=1.5)
axs[1].set_xlabel(f"estimate, $N = 10^4$ points"); axs[1].set_title(f"{reps} runs each", fontsize=14)
axs[1].legend(frameon=False, fontsize=12)
fig.tight_layout(); save(fig, "importance")

# ------------------------------------------- a rare event: P(x > 4) for a unit Gaussian
P4 = 0.5*special.erfc(4/np.sqrt(2))
n = 10**6; reps = 200
print(f"[tail] P(x > 4) = {P4:.4e}; with N = 1e6 Gaussian draws the expected number of hits is {n*P4:.1f}")
hits = rng.binomial(n, P4, size=reps)            # the number of x > 4 among N draws is binomial: same statistics, no 1e9 draws
plain_t = hits/n
imp_t = np.empty(reps)
for i in range(reps):
    xi = 4 - np.log(1 - rng.random(n))           # exponential beyond 4, Lecture 21
    imp_t[i] = np.mean(np.exp(-xi*xi/2 + xi - 4))/np.sqrt(2*np.pi)
print(f"[tail] plain: mean {plain_t.mean():.3e}, std {plain_t.std():.2e} ({plain_t.std()/P4:.1%}); importance: mean {imp_t.mean():.4e}, std {imp_t.std():.2e} ({imp_t.std()/P4:.2%})")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
xt = np.linspace(4, 9, 300)
axs[0].semilogy(xt, np.exp(-xt*xt/2)/np.sqrt(2*np.pi), color=V[0], label=r"Gaussian $N(x)$, $x > 4$")
axs[0].semilogy(xt, np.exp(-(xt - 4)), color=V[1], ls="--", label=r"$p(x) = e^{-(x-4)}$")
axs[0].semilogy(xt, np.exp(-xt*xt/2 + xt - 4)/np.sqrt(2*np.pi), color=ACCENT, label=r"$N/p$, the weight")
axs[0].set_xlim(4, 9); axs[0].set_ylim(1e-18, 2); axs[0].set_xlabel("$x$"); axs[0].legend(frameon=False, fontsize=12)
axs[0].set_title("the tail and the proposal", fontsize=14)
bins = np.arange(1.45, 4.85, 0.1)                 # one bin per possible count of hits
axs[1].hist(plain_t*1e5, bins=bins, color=V[1], alpha=0.75, label=rf"count hits: $\sigma = {100*plain_t.std()/P4:.0f}\%$")
axs[1].hist(imp_t*1e5, bins=bins, color=V[2], alpha=0.75, label=rf"importance: $\sigma = {100*imp_t.std()/P4:.2f}\%$")
axs[1].axvline(P4*1e5, color=INK, ls=":", lw=1.5)
axs[1].set_xlabel("estimate of $P(x > 4)$ in units of $10^{-5}$, $N = 10^6$"); axs[1].set_ylabel("runs"); axs[1].set_title(f"{reps} runs each", fontsize=14)
axs[1].legend(frameon=False, fontsize=12, loc="upper left")
fig.tight_layout(); save(fig, "tail")
