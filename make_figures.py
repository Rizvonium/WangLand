"""All figures for the presentation (figures/*.png).

    python make_figures.py            # every figure whose data exist
    python make_figures.py dos potts  # a selection

Conventions: J = k_B = 1, so E is in units of J, T in J/k_B, C and S in k_B.
Simulation = blue markers/lines, exact = dark line. Ordered sizes L use a
one-hue ramp (light = small L, dark = large L).
"""
import math
import os
import sys

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

from wanglandau.thermo import canonical, energy_distribution, equal_height_temperature  # noqa: E402
from wanglandau import exact as X                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- palette ---
INK, INK2 = "#1B2433", "#4A5568"
GRID, AXIS = "#E2E6EB", "#AEB6C1"
BLUE, ORANGE = "#1f6fc5", "#e0681f"
BLUES = ["#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e", "#0a2a57"]
ORANGES = ["#f2a06a", "#e97b35", "#cc5d17", "#9e4510", "#6f300b"]


def style():
    mpl.rcParams.update({
        "font.family": "Inter", "font.size": 20,
        "mathtext.fontset": "custom", "mathtext.rm": "Inter",
        "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
        "mathtext.fallback": "stixsans",
        "axes.labelsize": 22, "axes.titlesize": 22, "axes.titleweight": 600,
        "axes.titlelocation": "left", "axes.titlepad": 14,
        "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 19,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.2, "axes.labelcolor": INK,
        "axes.labelpad": 8, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
        "xtick.major.width": 1.1, "ytick.major.width": 1.1,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.4, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.8,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.transparent": True,
        "axes.facecolor": "none", "figure.facecolor": "none",
    })


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def load(name):
    path = os.path.join(RES, name)
    return np.load(path, allow_pickle=False) if os.path.exists(path) else None


def sub(n, k):
    """~k evenly spaced indices out of n."""
    return np.unique(np.linspace(0, n - 1, k).astype(int))


def panel_label(ax, s):
    ax.set_title(s)


# -------------------------------------------------------- exact (cached) ---
def exact_dos(L):
    path = os.path.join(RES, f"exact_dos_L{L}.npz")
    if not os.path.exists(path):
        E, lng = X.ising_exact_dos(L)
        np.savez_compressed(path, E=E, lng=lng)
    d = np.load(path)
    return d["E"], d["lng"]


def exact_thermo(L, T, tag):
    path = os.path.join(RES, f"exact_thermo_L{L}_{tag}.npz")
    if not os.path.exists(path):
        r = X.ising_kaufman(L, T)
        np.savez_compressed(path, **r)
    return dict(np.load(path))


def exact_cmax(L):
    """Exact specific-heat maximum (golden section on Kaufman's C)."""
    path = os.path.join(RES, "exact_cmax.npz")
    data = dict(np.load(path)) if os.path.exists(path) else {}
    key = f"L{L}"
    if key not in data:
        f = lambda t: X.ising_kaufman(L, [t])["C"][0]
        a, b = 2.15, 2.60
        g = (math.sqrt(5) - 1) / 2
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(40):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = f(d)
        data[key] = np.array([(a + b) / 2, max(fc, fd)])
        np.savez(path, **data)
    return data[key]


# ======================================================================
# 1. demo run
# ======================================================================
def demo_figure(L=16):
    style()
    d = load(f"demo_L{L}.npz")
    if d is None:
        print("  (demo data missing)")
        return
    N = L * L
    E, lng, H, use = d["E"], d["lng"], d["H"], d["use"]
    e = E / N
    fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.4), constrained_layout=True)
    ok = np.isfinite(d["lng_exact"])
    ax[0].plot(e[ok], d["lng_exact"][ok], color=INK, lw=1.8, label="exact")
    k = np.flatnonzero(np.isfinite(lng))[::4]
    ax[0].plot(e[k], lng[k], "o", ms=7, mfc="none", mec=BLUE, mew=1.8, label="Wang–Landau")
    ax[0].set(xlabel="E/N  (J)", ylabel="ln g(E)", xlim=(-2.05, 2.05))
    ax[0].legend(loc="lower center")
    panel_label(ax[0], "(a) density of states")

    h = np.where(use, H, np.nan)
    h = h / np.nanmean(h)
    ax[1].fill_between(e, 0, np.nan_to_num(h), step="mid", color=BLUE, alpha=0.12, lw=0)
    ax[1].step(e, h, where="mid", color=BLUE, lw=1.6)
    ax[1].axhline(0.8, color=ORANGE, lw=2)
    ax[1].text(-1.95, 0.83, r"flatness limit: 80% of $\langle H\rangle$", color=INK, fontsize=18, va="bottom")
    ax[1].set(xlabel="E/N  (J)", ylabel=r"$H(E)\,/\,\langle H\rangle$", ylim=(0, 1.35), xlim=(-2.05, 2.05))
    panel_label(ax[1], "(b) final-stage histogram")

    T = d["T"]
    ax[2].plot(T, d["C_exact"], color=INK, lw=1.8, label="exact")
    k = sub(T.size, 45)
    ax[2].plot(T[k], d["C"][k], "o", ms=7, mfc="none", mec=BLUE, mew=1.8, label="Wang–Landau")
    ax[2].axvline(X.TC_ISING, color=AXIS, lw=1.2)
    ax[2].text(X.TC_ISING + 0.08, 0.15, r"$T_c(\infty)$", color=INK2, fontsize=18)
    ax[2].set(xlabel=r"T  ($J/k_B$)", ylabel=r"C/N  ($k_B$)", ylim=(0, None))
    ax[2].legend(loc="center right")
    panel_label(ax[2], "(c) specific heat")
    save(fig, f"demo_L{L}.png")


# ======================================================================
# 2. density of states vs exact (paper Fig. 1)
# ======================================================================
def fig_dos():
    style()
    sizes = [(32, BLUES[2]), (50, BLUES[4])]
    data = [(L, c, load(f"ising_L{L}.npz")) for L, c in sizes]
    if any(d is None for _, _, d in data):
        print("  (dos data missing)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(16.6, 5.6), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1, 1.08]))
    summary = []
    for L, col, d in data:
        N = L * L
        Ee, le = exact_dos(L)
        e = d["E"] / N
        lng = d["lng"]
        ok = np.isfinite(le) & np.isfinite(lng)
        ax[0].plot(Ee[np.isfinite(le)] / N, le[np.isfinite(le)], color=INK, lw=1.6, zorder=1)
        k = np.flatnonzero(ok)[sub(ok.sum(), 55)]
        ax[0].plot(e[k], lng[k], "o", ms=7.5, mfc="none", mec=col, mew=1.9, zorder=2)
        eps = np.abs(lng - le) / np.abs(le)
        summary.append((L, np.nanmean(eps[ok])))
        ok &= eps > 0                     # E = +-2N fixed by normalisation (eps = 0)
        ax[1].plot(e[ok], eps[ok], color=col, lw=1.6, label=f"L = {L}")
        top = np.nanmax(le)
        ax[0].text(0.0, top + 22, f"{L}×{L}", ha="center", va="bottom", fontsize=19, color=INK)
    ax[0].set(xlabel="E/N  (J)", ylabel="ln g(E)", xlim=(-2.1, 2.1), ylim=(0, 1950))
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="none", mfc="none", mec=BLUES[3], mew=1.9, ms=8,
                                 label="Wang–Landau"),
                          Line2D([], [], color=INK, lw=1.6, label="exact (Beale 1996)")],
                 loc="lower center", bbox_to_anchor=(0.5, -0.01))
    panel_label(ax[0], "(a) ln g(E): simulation vs exact")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="E/N  (J)", ylabel=r"relative error  $\varepsilon(\ln g)$",
              xlim=(-2.1, 2.1), ylim=(1e-7, 1e-1))
    ax[1].legend(loc="upper center", ncol=2)
    txt = "  ·  ".join(f"mean ε, L={L}: {m:.1e}" for L, m in summary)
    ax[1].text(0.5, 0.04, txt, transform=ax[1].transAxes, ha="center", fontsize=17, color=INK2)
    panel_label(ax[1], r"(b) $\varepsilon = |\ln g_{sim} - \ln g_{exact}| \,/\, \ln g_{exact}$")
    save(fig, "dos_vs_exact.png")
    return summary


# ======================================================================
# 3. multiple-range random walk, L = 256
# ======================================================================
def fig_multirange():
    style()
    d = load("ising_L256.npz")
    if d is None:
        print("  (L=256 data missing)")
        return
    N = 256 * 256
    E, lng, Hw, wins = d["E"], d["lng"], d["H_win"], d["windows"]
    e = E / N
    fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.6), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ok = np.isfinite(lng)
    sim = ok & (e <= 0.2 + 1e-12)
    mir = ok & (e > 0.2)
    ax[0].plot(e[sim], lng[sim] / N, color=BLUE, lw=2.6, label="simulated, E/N ≤ 0.2")
    ax[0].plot(e[mir], lng[mir] / N, color=BLUES[0], lw=2.6, label="mirrored, g(E) = g(−E)")
    for k, (lo, hi) in enumerate(wins):                  # window strip
        yy = -0.07 - 0.045 * (k % 2)
        ax[0].plot([e[lo], e[hi]], [yy, yy], color=(BLUES[3] if k % 2 else BLUES[1]), lw=3,
                   solid_capstyle="butt")
    ax[0].text(-1.0, -0.17, f"{len(wins)} windows", ha="center", va="top", fontsize=17, color=INK2)
    ax[0].axhline(math.log(2), color=AXIS, lw=1.1)
    ax[0].text(1.98, math.log(2) + 0.015, "ln 2", ha="right", va="bottom", fontsize=17, color=INK2)
    ax[0].set(xlabel="E/N  (J)", ylabel="ln g(E) / N", xlim=(-2.05, 2.05), ylim=(-0.27, 0.8))
    ax[0].legend(loc="upper right", fontsize=16, bbox_to_anchor=(1.0, 0.93))
    panel_label(ax[0], "(a) joined density of states")

    ec = -math.sqrt(2)
    centers = np.array([(e[lo] + e[hi]) / 2 for lo, hi in wins])
    k0 = int(np.argmin(np.abs(centers - ec)))
    for j, k in enumerate((k0 - 1, k0, k0 + 1)):
        lo, hi = wins[k]
        h = Hw[k, lo:hi + 1]
        h = h / np.nanmean(h)
        ax[1].step(e[lo:hi + 1], h, where="mid", color=(BLUES[1], BLUES[3], ORANGE)[j], lw=1.6,
                   label=f"window {k + 1}")
    ax[1].axhline(0.8, color=INK, lw=1.4)
    ax[1].text(0.03, 0.05, "flatness limit 0.8", transform=ax[1].transAxes, fontsize=17, color=INK)
    ax[1].set(xlabel="E/N  (J)", ylabel=r"$H(E)\,/\,\langle H\rangle$", ylim=(0.6, 1.8))
    ax[1].legend(loc="upper center", ncol=3, fontsize=15, handlelength=1.0, columnspacing=1.0)
    panel_label(ax[1], "(b) neighbouring windows overlap")

    beta = np.full_like(lng, np.nan)
    beta[1:-1] = (lng[2:] - lng[:-2]) / 8.0             # d ln g / dE, dE = 4
    m = sim & np.isfinite(beta) & (e > -1.95) & (e < 0.15)
    ax[2].plot(e[m], beta[m], color=BLUE, lw=1.8)
    ax[2].plot(e[d["joins"]], np.full(len(d["joins"]), -0.07), "|", color=ORANGE, ms=12, mew=1.4)
    ax[2].axhline(1 / X.TC_ISING, color=INK, lw=1.2)
    ax[2].axvline(ec, color=INK, lw=1.2)
    ax[2].text(-0.55, 1 / X.TC_ISING + 0.04, r"$1/T_c$ (Onsager)", fontsize=17, color=INK)
    ax[2].text(ec + 0.04, 0.95, r"$E_c/N=-\sqrt{2}$", fontsize=17, color=INK)
    ax[2].text(-1.9, -0.02, "join points", fontsize=16, color=ORANGE, va="bottom")
    ax[2].set(xlabel="E/N  (J)", ylabel=r"$\beta(E)=d\,\ln g/dE$  (1/J)", xlim=(-2.0, 0.2),
              ylim=(-0.12, 1.1))
    panel_label(ax[2], "(c) microcanonical 1/T")
    save(fig, "multirange_L256.png")


# ======================================================================
# 4. thermodynamics of L = 256 vs exact (paper Figs. 2-3)
# ======================================================================
def T_grid_256():
    return np.unique(np.round(np.concatenate([
        np.linspace(0.1, 8.0, 159), np.linspace(2.15, 2.40, 51)]), 6))


def fig_thermo256():
    style()
    d = load("ising_L256.npz")
    if d is None:
        print("  (L=256 data missing)")
        return
    L, N = 256, 256 * 256
    T = T_grid_256()
    ex = exact_thermo(L, T, "grid")
    th = canonical(d["E"], d["lng"], T, N)
    fig, ax = plt.subplots(2, 3, figsize=(16.6, 5.6), constrained_layout=True,
                           sharex="col", gridspec_kw=dict(height_ratios=[1.75, 1]))
    keys = [("F", "F/N  (J)", "(a) free energy"), ("S", r"S/N  ($k_B$)", "(b) entropy"),
            ("C", r"C/N  ($k_B$)", "(c) specific heat")]
    stats = {}
    for j, (k, lab, ttl) in enumerate(keys):
        a = ax[0, j]
        a.plot(T, ex[k], color=INK, lw=1.8, label="exact")
        idx = sub(T.size, 40) if k != "C" else np.union1d(sub(T.size, 34),
                                                          np.flatnonzero((T > 2.2) & (T < 2.36))[::3])
        a.plot(T[idx], th[k][idx], "o", ms=6.5, mfc="none", mec=BLUE, mew=1.7, label="Wang–Landau")
        a.set(ylabel=lab, xlim=(0, 8.1))
        a.tick_params(labelbottom=False)
        panel_label(a, ttl)
        eps = np.abs(th[k] - ex[k]) / np.abs(ex[k])
        rng = T >= (0.4 if k == "C" else 0.1)
        b = ax[1, j]
        b.semilogy(T[rng], eps[rng], color=BLUE, lw=1.4)
        b.set(xlabel=r"T  ($J/k_B$)", ylabel=fr"$\varepsilon({k})$", xlim=(0, 8.1))
        stats[k] = (eps[rng].mean(), eps[rng].max())
    ax[0, 0].legend(loc="lower left", fontsize=17)
    ax[0, 2].set_ylim(0, None)
    save(fig, "thermo_L256.png")
    return stats


# ======================================================================
# 5. finite-size scaling and order parameter
# ======================================================================
def fig_fss():
    style()
    sizes = [8, 16, 32, 64, 128, 256]
    data = {L: load(f"ising_L{L}.npz") for L in sizes}
    data = {L: v for L, v in data.items() if v is not None}
    if len(data) < 3:
        print("  (fss data missing)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.6), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1.15, 1, 1.1]))
    T = np.linspace(1.6, 3.4, 721)
    peaks = []
    for c, L in zip(BLUES, sizes):
        if L not in data:
            continue
        d = data[L]
        N = L * L
        th = canonical(d["E"], d["lng"], T, N, d["m_abs"])
        ax[0].plot(T, th["C"], color=c, lw=2.2, label=f"L = {L}")
        i = np.argmax(th["C"])
        Tf = np.linspace(T[max(i - 4, 0)], T[min(i + 4, T.size - 1)], 401)
        Cf = canonical(d["E"], d["lng"], Tf, N)["C"]
        tpk, cpk = exact_cmax(L)
        peaks.append((L, Tf[np.argmax(Cf)], Cf.max(), tpk, cpk))
        ax[2].plot(T, th["M"], color=c, lw=2.2, label=f"L = {L}")
    Tx = np.linspace(1.6, 3.4, 181)
    for L in sizes:
        if L in data:
            ex = exact_thermo(L, Tx, "fss")
            ax[0].plot(Tx, ex["C"], color=INK, lw=1.0, ls=(0, (1, 2.2)))
    ax[0].axvline(X.TC_ISING, color=AXIS, lw=1.2)
    ax[0].set(xlabel=r"T  ($J/k_B$)", ylabel=r"C/N  ($k_B$)", xlim=(1.6, 3.4), ylim=(0, None))
    ax[0].legend(loc="upper right", fontsize=17, handlelength=1.2)
    ax[0].text(1.65, 0.05, "dotted: exact", fontsize=16, color=INK2, transform=ax[0].transData)
    panel_label(ax[0], "(a) specific heat vs L")

    p = np.array(peaks)
    lnL = np.log(p[:, 0])
    a0 = 8 * X.KC_ISING**2 / math.pi                         # exact slope 0.4945
    ax[1].plot(p[:, 0], p[:, 4], "s", ms=12, mfc="none", mec=INK, mew=1.8, label="exact (Kaufman)")
    ax[1].plot(p[:, 0], p[:, 2], "o", ms=8, color=BLUE, label="Wang–Landau")
    xx = np.exp(np.linspace(np.log(6), np.log(340), 10))
    b0 = np.mean(p[:, 4] - a0 * lnL)
    ax[1].plot(xx, a0 * np.log(xx) + b0, color=ORANGE, lw=2, zorder=0,
               label=r"slope $8K_c^2/\pi = 0.4945$")
    ax[1].set_xscale("log")
    ax[1].set_xticks(sizes)
    ax[1].set_xticklabels([str(L) for L in sizes])
    ax[1].minorticks_off()
    ax[1].set(xlabel="L  (log scale)", ylabel=r"$C_{max}/N$  ($k_B$)")
    ax[1].legend(loc="upper left", fontsize=17)
    panel_label(ax[1], r"(b) $C_{max}$ grows as ln L")

    ax[2].plot(Tx, X.onsager_magnetisation(Tx), color=INK, lw=1.8, label="Onsager–Yang (L→∞)")
    ax[2].axvline(X.TC_ISING, color=AXIS, lw=1.2)
    ax[2].set(xlabel=r"T  ($J/k_B$)", ylabel=r"$\langle |M| \rangle\,/\,N$", xlim=(1.6, 3.4), ylim=(0, 1.02))
    h, l = ax[2].get_legend_handles_labels()
    ax[2].legend(h[-1:], l[-1:], loc="upper right", fontsize=17)
    ax[2].text(3.38, 0.70, "colours: L as in (a)", ha="right", fontsize=16, color=INK2)
    panel_label(ax[2], "(c) order parameter")
    save(fig, "fss_ising.png")
    np.savetxt(os.path.join(RES, "cmax_table.txt"), p,
               header="L  T_peak(WL)  Cmax/N(WL)  T_peak(exact)  Cmax/N(exact)", fmt="%.6f")
    return p


# ======================================================================
# 6. ten-state Potts model (paper Fig. 4)
# ======================================================================
def fig_potts(sizes=None, q=10):
    style()
    import glob
    if sizes is None:
        sizes = sorted(int(os.path.basename(f).split("_L")[1].split(".")[0])
                       for f in glob.glob(os.path.join(RES, f"potts_q{q}_L*.npz")))
    data = [(L, load(f"potts_q{q}_L{L}.npz")) for L in sizes]
    data = [(L, d) for L, d in data if d is not None]
    if len(data) < 2:
        print("  (potts data missing)")
        return
    tc, (eo, ed) = X.potts_tc(q), X.potts_phase_energies(q)
    fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.6), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    rows, curves = [], []
    for L, d in data:
        N = L * L
        E, lng = d["E"].astype(float), d["lng"]
        Tl = equal_height_temperature(E, lng, tc, e_split=0.5 * (eo + ed) * N)
        P = energy_distribution(E, lng, Tl)
        e = E / N
        lo, hi = e < 0.5 * (eo + ed), e >= 0.5 * (eo + ed)
        i_o = np.flatnonzero(lo)[np.nanargmax(P[lo])]
        i_d = np.flatnonzero(hi)[np.nanargmax(P[hi])]
        Pn = P / P[i_o]
        pmin = np.nanmin(Pn[i_o:i_d + 1])
        rows.append((L, Tl, pmin, e[i_o], e[i_d], d["sweeps"]))
        curves.append((L, e, Pn))
    preferred = [20, 32, 48, 64, 96]                    # curves drawn in (a); fits use all L
    show = [c for c in curves if c[0] in preferred]
    if len(show) < 3:
        show = [curves[int(round(k))] for k in np.linspace(0, len(curves) - 1, min(5, len(curves)))]
    for c, (L, e, Pn) in zip(ORANGES, show):
        ok = np.isfinite(Pn) & (Pn > 0)
        ax[0].semilogy(e[ok], Pn[ok], color=c, lw=2.2, label=f"L = {L}")
    ax[0].axvline(eo, color=INK, lw=1.1)
    ax[0].axvline(ed, color=INK, lw=1.1)
    ax[0].text(eo + 0.012, 2.2, r"$e_{ord}$ (exact)", fontsize=17, color=INK, va="bottom")
    ax[0].text(ed + 0.012, 2.2, r"$e_{dis}$ (exact)", fontsize=17, color=INK, va="bottom")
    ax[0].set(xlabel="E/N  (J)", ylabel=r"$P(E, T_c(L))$  (peaks = 1)", xlim=(-1.84, -0.72),
              ylim=(1e-6, 12))
    ax[0].legend(loc="lower center", ncol=3, fontsize=17, columnspacing=1.2)
    panel_label(ax[0], "(a) energy distribution at the equal-height T")

    r = np.array(rows)
    x = 1.0 / r[:, 0] ** 2
    fit = np.polyfit(x, r[:, 1], 1)
    xx = np.linspace(0, x.max() * 1.08, 20)
    ax[1].plot(xx * 1e4, np.polyval(fit, xx), color=ORANGE, lw=2, zorder=0, label="linear fit")
    ax[1].plot(x * 1e4, r[:, 1], "o", ms=9, color=ORANGES[3], label="Wang–Landau")
    ax[1].plot([0], [tc], "s", ms=11, mfc="none", mec=INK, mew=2, label="exact $T_c$")
    ax[1].set(xlabel=r"$10^4 / L^2$", ylabel=r"$T_c(L)$  ($J/k_B$)")
    ax[1].legend(loc="upper left", fontsize=17)
    ax[1].text(0.97, 0.05, f"extrapolated {fit[1]:.5f}\nexact {tc:.5f}",
               transform=ax[1].transAxes, ha="right", va="bottom", fontsize=17, color=INK2)
    panel_label(ax[1], r"(b) $T_c(L) \to T_c(\infty)$")

    dF = -np.log(r[:, 2])
    fit2, cov2 = np.polyfit(r[:, 0], dF, 1, cov=True)      # all sizes
    err2 = math.sqrt(cov2[0, 0])
    ax[2].plot(r[:, 0], dF, "o", ms=9, color=ORANGES[3])
    xx = np.linspace(0, r[:, 0].max() * 1.05, 20)
    ax[2].plot(xx, np.polyval(fit2, xx), color=ORANGE, lw=2, zorder=0,
               label=f"fit: {fit2[0]:.3f} ± {err2:.3f}")
    ax[2].set(xlabel="L", ylabel=r"$\ln(P_{max}/P_{min})$", ylim=(0, None), xlim=(0, None))
    ax[2].legend(loc="lower right", fontsize=17, title="Wang–Landau, all L", title_fontsize=16)
    ax[2].text(0.04, 0.96, "exact 2σ = 0.0947\n(Borgs–Janke)", transform=ax[2].transAxes,
               ha="left", va="top", fontsize=17, color=INK2)
    panel_label(ax[2], r"(c) barrier grows $\propto$ L")
    save(fig, "potts_q10.png")
    np.savetxt(os.path.join(RES, "potts_table.txt"), r,
               header="L  Tc(L)  Pmin/Pmax  e_ord  e_dis  sweeps", fmt="%.6g")
    with open(os.path.join(RES, "potts_fits.txt"), "w") as fh:
        fh.write(f"Tc(L) = a + b/L^2 (all L): Tc(inf) = {fit[1]:.6f}, exact {tc:.6f}\n")
        fh.write(f"ln(Pmax/Pmin) = c + s*L (all L): s = {fit2[0]:.5f} +- {err2:.5f}, "
                 f"exact 2*sigma_od = 0.0947\n")
    return r, fit, fit2


# ======================================================================
# 7. cost of multiple ranges (L = 32) and convergence (L = 16)
# ======================================================================
def fig_efficiency():
    style()
    s = load("windows_L32_summary.npz")
    c = load("convergence_L16.npz")
    if s is None or c is None:
        print("  (efficiency data missing)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.6), constrained_layout=True)
    rows = s["rows"]
    Ws = np.unique(rows[:, 0])
    L = 32
    Ee, le = exact_dos(L)
    tot, crit, acc = [], [], []
    for W in Ws:
        r = rows[rows[:, 0] == W]
        tot.append(r[:, 2].mean())
        crit.append(r[:, 3].mean())
        errs = []
        for rep in range(int(r[:, 1].max()) + 1):
            d = load(f"windows_L{L}_W{int(W)}_r{rep}.npz")
            ok = np.isfinite(d["lng"]) & np.isfinite(le)
            errs.append(np.mean(np.abs(d["lng"][ok] - le[ok]) / le[ok]))
        acc.append((np.mean(errs), np.min(errs), np.max(errs)))
    tot, crit, acc = np.array(tot), np.array(crit), np.array(acc)
    ax[0].loglog(Ws, tot, "o-", color=BLUES[2], ms=9, label="total (all walkers)")
    ax[0].loglog(Ws, crit, "s-", color=ORANGE, ms=9, label="longest single walker")
    ax[0].loglog(Ws, crit[0] / Ws, color=AXIS, lw=1.4, zorder=0, label=r"ideal  $\propto 1/W$")
    ax[0].set(xlabel="number of energy windows W", ylabel="MC sweeps")
    ax[0].set_xticks(Ws)
    ax[0].set_xticklabels([str(int(w)) for w in Ws])
    ax[0].legend(loc="lower left", fontsize=17)
    panel_label(ax[0], "(a) cost, 32×32 (ln f → 10⁻⁸)")

    ax[1].errorbar(Ws, acc[:, 0], yerr=[acc[:, 0] - acc[:, 1], acc[:, 2] - acc[:, 0]],
                   fmt="o", ms=9, color=BLUES[3], capsize=5, lw=1.6)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks(Ws)
    ax[1].set_xticklabels([str(int(w)) for w in Ws])
    ax[1].set(xlabel="number of energy windows W", ylabel=r"mean $\varepsilon(\ln g)$",
              ylim=(acc[:, 1].min() / 3, acc[:, 2].max() * 3))
    panel_label(ax[1], "(b) accuracy (3 runs each)")

    # convergence: error after every stage, 8 independent runs, L = 16
    snaps, use = c["lng_stages"], c["use"]
    i_lo, i_hi = int(c["i_lo"]), int(c["i_hi"])
    L16 = int(c["L"])
    Ee, le = exact_dos(L16)
    le = le[i_lo:i_hi + 1]
    lnf = c["ln_f"]
    errs = []
    for run in snaps:
        er = []
        for st in run:
            x = np.where(use, st, np.nan)
            x = x - x[0] + math.log(2)
            ok = np.isfinite(x) & np.isfinite(le)
            er.append(np.mean(np.abs(x[ok] - le[ok]) / le[ok]))
        errs.append(er)
    errs = np.array(errs)
    for er in errs:
        ax[2].loglog(lnf, er, color=BLUES[0], lw=1.0, alpha=0.8)
    ax[2].loglog(lnf, errs.mean(0), "o-", color=BLUES[3], ms=6, lw=2.2, label="mean of 8 runs")
    ref = errs.mean(0)[6] * np.sqrt(lnf / lnf[6])
    keep = lnf >= 3e-8
    ax[2].loglog(lnf[keep], ref[keep], color=ORANGE, lw=1.6, label=r"$\propto \sqrt{\ln f}$")
    ax[2].set(xlabel="modification factor  ln f", ylabel=r"mean $\varepsilon(\ln g)$",
              ylim=(1e-4, errs.max() * 2))
    ax[2].invert_xaxis()
    ax[2].legend(loc="upper right", fontsize=17)
    panel_label(ax[2], "(c) error saturates, 16×16")
    save(fig, "efficiency.png")
    np.savetxt(os.path.join(RES, "windows_table.txt"),
               np.column_stack([Ws, tot, crit, acc]),
               header="W  total_sweeps  longest_walker_sweeps  eps_mean  eps_min  eps_max",
               fmt="%.6g")
    np.savetxt(os.path.join(RES, "convergence_table.txt"),
               np.column_stack([lnf, errs.mean(0), errs.std(0)]),
               header="ln_f  mean_eps  std_eps (8 runs, L=16)", fmt="%.6g")
    return Ws, tot, crit, acc, errs


# ======================================================================
# 8. how windows are joined (small run, L = 32, 6 windows)
# ======================================================================
def fig_join():
    style()
    from wanglandau import multirange, Ising, normalise
    path = os.path.join(RES, "join_demo_L32.npz")
    m = Ising(32)
    N = m.N
    if not os.path.exists(path):
        r = multirange("ising", 32, 6, overlap=0.4, e_range=(-2 * N, 0.2 * N),
                       ln_f_final=1e-6, seed=99, n_proc=2)
        pieces = np.full((len(r["parts"]), m.n_levels), np.nan)
        for k, q in enumerate(r["parts"]):
            pieces[k, q["i_lo"]:q["i_hi"] + 1] = np.where(q["use"], q["lng"], np.nan)
        lng = normalise(np.where(r["use"], r["lng"], np.nan), r["use"], m)
        np.savez_compressed(path, pieces=pieces, lng=lng, joins=np.array(r["joins"]),
                            E=m.energies())
    d = np.load(path)
    e = d["E"] / N
    Ee, le = exact_dos(32)
    fig, ax = plt.subplots(2, 1, figsize=(6.24, 5.6), constrained_layout=True, sharex=True)
    cols = [BLUES[1], ORANGE]
    for k, pc in enumerate(d["pieces"]):
        ok = np.isfinite(pc)
        ax[0].plot(e[ok], pc[ok] - np.nanmin(pc), color=cols[k % 2], lw=2.6)
    ax[0].set(ylabel=r"ln $g_k$ − const")
    ax[0].set_title("(a) 6 windows, each up to a constant", fontsize=20)
    ok = np.isfinite(le) & (e <= 0.2)
    ax[1].plot(e[ok], le[ok], color=INK, lw=1.6, label="exact")
    lg = d["lng"]
    k = np.flatnonzero(np.isfinite(lg) & (e <= 0.2))[::14]
    ax[1].plot(e[k], lg[k], "o", ms=6.5, mfc="none", mec=BLUE, mew=1.7, label="joined")
    for j in d["joins"]:
        ax[1].axvline(e[j], color=ORANGE, lw=1.2, zorder=0)
    ax[1].set(xlabel="E/N  (J)", ylabel="ln g(E)", xlim=(-2.05, 0.25))
    ax[1].legend(loc="lower right", fontsize=17)
    ax[1].set_title("(b) joined at E* (orange lines)", fontsize=20)
    save(fig, "join_demo.png")


ALL = dict(demo=demo_figure, dos=fig_dos, multirange=fig_multirange,
           thermo=fig_thermo256, fss=fig_fss, potts=fig_potts, efficiency=fig_efficiency,
           join=fig_join)

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        print("figure:", n)
        ALL[n]()
