"""Production runs for the presentation. Results go to results/*.npz.

    python run_all.py convergence     # L=16, 8 seeds: error vs ln f
    python run_all.py windows         # L=32: cost & accuracy vs number of windows
    python run_all.py ising           # L=8..128 (+50): DOS, thermodynamics, |M|
    python run_all.py potts           # q=10 Potts: first-order double peak
    python run_all.py ising256        # L=256, multiple-range walk (paper Figs 2-3)

Units: J = k_B = 1. Ising runs simulate E/N in [-2, 0.2] and use g(E)=g(-E).
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from wanglandau import Ising, Potts, wang_landau, multirange, normalise, ising_mirror

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
NPROC = mp.cpu_count()
E_TOP = 0.2                      # Ising: simulate E/N <= 0.2, mirror the rest


def pmap(fn, jobs):
    with mp.get_context("fork").Pool(min(NPROC, len(jobs))) as pool:
        return pool.map(fn, jobs, chunksize=1)


def save(name, **arrays):
    path = os.path.join(OUT, name)
    np.savez_compressed(path, **arrays)
    print("  saved", path, flush=True)


def ising_full(model, lng_half, use_half, m_half):
    """Normalise with g(E_min)=2 and mirror to the full energy range."""
    lng = normalise(np.where(use_half, lng_half, np.nan), use_half, model)
    lng, use = ising_mirror(lng, use_half)
    m, _ = ising_mirror(m_half, use_half)
    return lng, use, m


def pad(model, r):
    """Place a single-window result on the full level grid."""
    lng = np.full(model.n_levels, np.nan)
    use = np.zeros(model.n_levels, bool)
    m = np.full(model.n_levels, np.nan)
    sl = slice(r["i_lo"], r["i_hi"] + 1)
    lng[sl], use[sl], m[sl] = r["lng"], r["use"], r["m_abs"]
    return lng, use, m


# --------------------------------------------------------------------------
def _single(args):
    L, seed, lnf, keep = args
    m = Ising(L)
    return wang_landau(m, e_window=(m.E_min, int(E_TOP * m.N)), ln_f_final=lnf,
                       seed=seed, keep_stages=keep)


def task_convergence():
    """Error of ln g after every stage (L = 16, exact DOS known)."""
    L, seeds = 16, range(8)
    res = pmap(_single, [(L, 1000 + s, 1e-9, True) for s in seeds])
    snaps = np.array([r["lng_stages"] for r in res])          # (seed, stage, level)
    save(f"convergence_L{L}.npz", lng_stages=snaps, E=res[0]["E"],
         use=res[0]["use"], ln_f=res[0]["stages"]["ln_f"],
         cum_sweeps=np.array([r["stages"]["cum_sweeps"] for r in res]),
         i_lo=res[0]["i_lo"], i_hi=res[0]["i_hi"], L=L)


def task_windows():
    """Same problem (L=32, E/N in [-2, 0.2]) split into W = 1..16 windows.
    All window jobs share one process pool; for each W we record the total
    effort (sum of sweeps) and the critical path (largest single window)."""
    from wanglandau.sampler import _job, make_windows, join_windows
    L, reps = 32, 3
    m = Ising(L)
    i_last = int((E_TOP * m.N - m.E_min) // m.dE)
    plan, jobs = [], []
    for W in (1, 2, 4, 8, 16):
        wins = make_windows(0, i_last, W, 0.4)
        for rep in range(reps):
            ss = np.random.SeedSequence(500 + 17 * W + rep).spawn(W)
            for k, w in enumerate(wins):
                seed = int(ss[k].generate_state(1)[0] % 2**31)
                plan.append((W, rep, k))
                jobs.append(("ising", L, 2, w, dict(ln_f_final=1e-8, seed=seed)))
    out = pmap(_job, jobs)
    rows = []
    for W in (1, 2, 4, 8, 16):
        for rep in range(reps):
            parts = [o for (w_, r_, k), o in zip(plan, out) if w_ == W and r_ == rep]
            lng, use, joins, mabs = join_windows(parts, m.n_levels)
            lng, use, _ = ising_full(m, lng, use, mabs)
            sw = [p["sweeps"] for p in parts]
            sec = [p["seconds"] for p in parts]
            rows.append((W, rep, sum(sw), max(sw), sum(sec), max(sec)))
            save(f"windows_L{L}_W{W}_r{rep}.npz", E=m.energies(), lng=lng, use=use,
                 windows=np.array([(p["i_lo"], p["i_hi"]) for p in parts]),
                 joins=np.array(joins), per_window=np.array(sw),
                 seconds=np.array(sec))
            print(f"  W={W:2d} rep={rep} total sweeps={sum(sw):,} "
                  f"max/window={max(sw):,} cpu={sum(sec):.1f}s", flush=True)
    save(f"windows_L{L}_summary.npz", rows=np.array(rows, dtype=float),
         cols=np.array(["W", "rep", "sweeps_total", "sweeps_max_window",
                        "cpu_seconds", "max_window_seconds"]))


def task_ising():
    """Single walks for L = 8, 16, 32 and multiple-range walks for 50, 64, 128."""
    singles = pmap(_single, [(8, 81, 1e-8, False), (16, 161, 1e-8, False),
                             (32, 321, 1e-8, False), (32, 322, 1e-8, False)])
    for r, tag in zip(singles, ("L8", "L16", "L32", "L32b")):
        m = Ising(r["L"])
        lng, use, mag = ising_full(m, *pad(m, r))
        save(f"ising_{tag}.npz", E=m.energies(), lng=lng, use=use, m_abs=mag,
             sweeps=r["sweeps"], seconds=r["seconds"], stages=r["stages"],
             H=r["H"], i_lo=r["i_lo"], i_hi=r["i_hi"], L=r["L"])
    for L, W in ((50, 4), (64, 8), (128, 24)):
        r = multirange("ising", L, W, overlap=0.4, e_range=(-2 * L * L, E_TOP * L * L),
                       ln_f_final=1e-8, seed=7000 + L, n_proc=NPROC)
        m = Ising(L)
        lng, use, mag = ising_full(m, r["lng"], r["use"], r["m_abs"])
        save(f"ising_L{L}.npz", E=m.energies(), lng=lng, use=use, m_abs=mag,
             sweeps=r["sweeps"], wall=r["wall"], windows=np.array(r["windows"]),
             joins=np.array(r["joins"]), L=L,
             per_window=np.array([p["sweeps"] for p in r["parts"]]))
        print(f"  L={L}: W={W} sweeps={r['sweeps']:,} wall={r['wall']:.0f}s", flush=True)


def task_ising256(W=128, lnf=1e-8):
    L = 256
    r = multirange("ising", L, W, overlap=0.35, e_range=(-2 * L * L, E_TOP * L * L),
                   ln_f_final=lnf, seed=256256, n_proc=NPROC)
    m = Ising(L)
    lng, use, mag = ising_full(m, r["lng"], r["use"], r["m_abs"])
    parts = r["parts"]
    nl = m.n_levels
    H_win = np.full((len(parts), nl), np.nan)                 # final-stage histograms
    for k, p in enumerate(parts):
        H_win[k, p["i_lo"]:p["i_hi"] + 1] = np.where(p["use"], p["H"], np.nan)
    save(f"ising_L{L}.npz", E=m.energies(), lng=lng, use=use, m_abs=mag,
         sweeps=r["sweeps"], wall=r["wall"], windows=np.array(r["windows"]),
         joins=np.array(r["joins"]), L=L, H_win=H_win.astype(np.float32),
         per_window=np.array([p["sweeps"] for p in parts]),
         seconds_window=np.array([p["seconds"] for p in parts]))
    print(f"  L=256: W={W} sweeps={r['sweeps']:,} wall={r['wall']:.0f}s", flush=True)


def task_potts(sizes=(20, 40, 60, 80), e_top=-0.4, q=10, e_bot=-2.0, width=330,
               lnf=1e-8):
    """q = 10 Potts model, E/N in [e_bot, e_top] (enough for T ~ T_c).
    With e_bot > -2 the ground state is not sampled: ln g is then only known up
    to a constant, which is all that P(E, T) and T_c(L) need."""
    for L in sizes:
        N = L * L
        n_levels = int((e_top - e_bot) * N)
        W = max(1, int(round(n_levels / (width * 0.6))))
        r = multirange("potts", L, W, overlap=0.4,
                       e_range=(int(e_bot * N), int(e_top * N)),
                       q=q, ln_f_final=lnf, seed=1000 + L, n_proc=NPROC)
        m = Potts(L, q)
        lng = np.where(r["use"], r["lng"], np.nan)
        if r["use"][0]:
            lng = normalise(lng, r["use"], m)
        else:
            lng = lng - np.nanmax(lng)
        nl = m.n_levels
        H_win = np.full((len(r["parts"]), nl), np.nan)
        for k, p in enumerate(r["parts"]):
            H_win[k, p["i_lo"]:p["i_hi"] + 1] = np.where(p["use"], p["H"], np.nan)
        save(f"potts_q{q}_L{L}.npz", E=m.energies(), lng=lng, use=r["use"],
             sweeps=r["sweeps"], wall=r["wall"], windows=np.array(r["windows"]),
             joins=np.array(r["joins"]), L=L, q=q, H_win=H_win.astype(np.float32),
             per_window=np.array([p["sweeps"] for p in r["parts"]]))
        print(f"  Potts L={L}: W={W} sweeps={r['sweeps']:,} wall={r['wall']:.0f}s",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--W256", type=int, default=128)
    ap.add_argument("--lnf256", type=float, default=1e-8)
    ap.add_argument("--potts", type=int, nargs="*", default=[20, 40, 60, 80, 100])
    a = ap.parse_args()
    for t in a.tasks:
        t0 = time.time()
        print(f"== {t}", flush=True)
        if t == "ising256":
            task_ising256(a.W256, a.lnf256)
        elif t == "potts":
            task_potts(tuple(a.potts))
        elif t == "potts_fast":
            # narrower energy range and windows around the transition
            small = tuple(L for L in a.potts if L <= 32)
            large = tuple(L for L in a.potts if L > 32)
            if small:
                task_potts(small, e_top=-0.5, e_bot=-2.0, width=250, lnf=1e-7)
            if large:
                task_potts(large, e_top=-0.72, e_bot=-1.84, width=250, lnf=1e-7)
        else:
            globals()[f"task_{t}"]()
        print(f"== {t} done in {time.time() - t0:.0f}s", flush=True)
