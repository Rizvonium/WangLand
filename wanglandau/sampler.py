"""Wang-Landau driver: one energy window, or several windows run in parallel
("multiple-range random walk") and joined afterwards.

Main entry points
-----------------
wang_landau(model, e_window=None, ...)    -> dict   (one random walk)
multirange(model_name, L, n_windows, ...) -> dict   (parallel windows, joined)
"""
import math
import time
import multiprocessing as mp

import numpy as np

from . import kernels as K
from .models import make_model


# --------------------------------------------------------------------------
# one random walk restricted to [e_window[0], e_window[1]]
# --------------------------------------------------------------------------
def wang_landau(model, e_window=None, ln_f0=1.0, ln_f_final=1e-8, flatness=0.8,
                check=10, max_sweeps_stage=10**9, seed=None, measure_below=1e-4,
                keep_stages=False, verbose=False, log=print):
    """Wang-Landau estimate of ln g(E) inside one energy window.

    Parameters (paper defaults)
    ---------------------------
    ln_f0 = 1          initial modification factor  f0 = e
    ln_f_final = 1e-8  stop once f < f_final = exp(1e-8)
    flatness = 0.8     H(E) >= 0.8 <H> for every reachable level
    f -> sqrt(f)       i.e. ln f -> ln f / 2 after every flat histogram
    """
    t0 = time.time()
    E_all = model.energies()
    nl = model.n_levels
    use = model.reachable().copy()
    if e_window is None:
        i_lo, i_hi = 0, nl - 1
    else:
        i_lo = max(0, int(math.ceil((e_window[0] - model.E_min) / model.dE)))
        i_hi = min(nl - 1, int(math.floor((e_window[1] - model.E_min) / model.dE)))
    if seed is None:
        seed = np.random.SeedSequence().entropy % (2**31)
    st = K.new_rng_state(int(seed))          # xoroshiro128+ state

    # start from the ground state and walk greedily into the window
    spins = model.ground_state()
    e_lo, e_hi = E_all[i_lo], E_all[i_hi]
    if model.name == "ising":
        E = K.ising_energy(spins, model.nbr)
        E = K.ising_drive(spins, model.nbr, st, E, e_lo, e_hi, 10**10)
    else:
        E = K.potts_energy(spins, model.nbr)
        E = K.potts_drive(spins, model.nbr, model.q, st, E, e_lo, e_hi, 10**10)
    if not (e_lo <= E <= e_hi):
        raise RuntimeError("could not reach the energy window")
    idx = (E - model.E_min) // model.dE
    M = int(spins.astype(np.int64).sum()) if model.name == "ising" else 0
    state = np.array([idx, M], dtype=np.int64)

    lng = np.zeros(nl)               # ln g(E), start with g(E) = 1
    H = np.zeros(nl)                 # visit histogram of the current stage
    m_sum = np.zeros(nl)
    m_cnt = np.zeros(nl)

    stages, snapshots = [], []
    ln_f, total = ln_f0, 0
    while ln_f > ln_f_final:
        H[:] = 0.0
        measure = model.name == "ising" and ln_f <= measure_below
        ts = time.time()
        sweeps = 0
        while True:
            if model.name == "ising":
                n, flat = K.ising_stage(spins, model.nbr, st, state, lng, H, use,
                                        i_lo, i_hi, ln_f, flatness, check,
                                        max_sweeps_stage, measure, m_sum, m_cnt)
            else:
                n, flat = K.potts_stage(spins, model.nbr, model.q, st, state, lng, H,
                                        use, i_lo, i_hi, ln_f, flatness, check,
                                        max_sweeps_stage)
            sweeps += n
            if flat:
                break
            # never visited after a very long stage -> treat as unreachable
            dead = use[i_lo:i_hi + 1] & (H[i_lo:i_hi + 1] == 0)
            if dead.any():
                use[i_lo:i_hi + 1][dead] = False
                if verbose:
                    log(f"  [warn] {dead.sum()} levels never visited -> excluded")
        total += sweeps
        hw = H[i_lo:i_hi + 1][use[i_lo:i_hi + 1]]
        stages.append((len(stages), ln_f, sweeps, total, hw.min() / hw.mean(),
                       time.time() - ts))
        if keep_stages:
            snapshots.append(lng[i_lo:i_hi + 1].copy())
        if verbose:
            log(f"  stage {len(stages)-1:2d}  ln f = {ln_f:9.3e}  "
                f"sweeps = {sweeps:9d}  min(H)/<H> = {hw.min()/hw.mean():.3f}")
        ln_f /= 2.0                  # f_{i+1} = sqrt(f_i)

    sl = slice(i_lo, i_hi + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_abs = np.where(m_cnt > 0, m_sum / np.maximum(m_cnt, 1), np.nan)
    out = dict(
        model=model.name, L=model.L, N=model.N, q=model.q,
        i_lo=i_lo, i_hi=i_hi, E=E_all[sl], lng=lng[sl], use=use[sl],
        H=H[sl], m_abs=m_abs[sl] / model.N,
        stages=np.array(stages, dtype=[("stage", "i4"), ("ln_f", "f8"),
                                       ("sweeps", "i8"), ("cum_sweeps", "i8"),
                                       ("flatness", "f8"), ("time", "f8")]),
        sweeps=total, seconds=time.time() - t0, seed=int(seed),
        ln_f_final=ln_f_final, flatness_criterion=flatness,
    )
    if keep_stages:
        out["lng_stages"] = np.array(snapshots)
    return out


# --------------------------------------------------------------------------
# multiple-range random walks
# --------------------------------------------------------------------------
def make_windows(i_first, i_last, n_windows, overlap):
    """Split levels [i_first, i_last] into n windows; `overlap` = fraction of
    the window width shared by neighbouring windows."""
    n_levels = i_last - i_first + 1
    if n_windows == 1:
        return [(i_first, i_last)]
    w = int(math.ceil(n_levels / (n_windows - (n_windows - 1) * overlap)))
    step = (n_levels - w) / (n_windows - 1)
    wins = []
    for k in range(n_windows):
        lo = i_first + int(round(k * step))
        wins.append((lo, min(i_last, lo + w - 1)))
    return wins


def _job(args):
    model_name, L, q, (i_lo, i_hi), kw = args
    model = make_model(model_name, L, q)
    E = model.energies()
    return wang_landau(model, e_window=(E[i_lo], E[i_hi]), **kw)


def join_windows(parts, n_levels, trim=0.15):
    """Glue window estimates of ln g(E) together.

    For two neighbouring windows A, B the joining level E* is the point of the
    overlap (edges trimmed) where the slopes d ln g / dE = 1/T(E) of both
    pieces agree best; B is shifted to match A at E*.
    """
    parts = sorted(parts, key=lambda r: r["i_lo"])
    lng = np.full(n_levels, np.nan)
    m_abs = np.full(n_levels, np.nan)
    use = np.zeros(n_levels, dtype=bool)
    p0 = parts[0]
    lng[p0["i_lo"]:p0["i_hi"] + 1] = p0["lng"]
    m_abs[p0["i_lo"]:p0["i_hi"] + 1] = p0["m_abs"]
    use[p0["i_lo"]:p0["i_hi"] + 1] = p0["use"]
    joins = []
    hi_prev, lo_prev = p0["i_hi"], p0["i_lo"]
    for p in parts[1:]:
        a = np.full(n_levels, np.nan)
        b = np.full(n_levels, np.nan)
        a[:] = lng
        b[p["i_lo"]:p["i_hi"] + 1] = np.where(p["use"], p["lng"], np.nan)
        a[~use] = np.nan
        lo, hi = p["i_lo"], hi_prev                # overlap of A and B
        t = max(2, int(trim * (hi - lo + 1)))      # skip the window edges
        cand = np.arange(lo + t, hi - t + 1)
        best, i_star = np.inf, None
        for i in cand:
            sa = a[i + 1] - a[i - 1]               # slope of A ~ 1/T(E)
            sb = b[i + 1] - b[i - 1]               # slope of B
            d = abs(sa - sb)
            if np.isfinite(d) and d < best:
                best, i_star = d, i                # E* = best slope match
        if i_star is None:
            raise RuntimeError("windows do not overlap enough to be joined")
        shift = a[i_star] - b[i_star]              # match ln g at E*
        seg = slice(i_star, p["i_hi"] + 1)
        lng[seg] = p["lng"][i_star - p["i_lo"]:] + shift   # B above E*
        m_abs[seg] = p["m_abs"][i_star - p["i_lo"]:]
        use[seg] = p["use"][i_star - p["i_lo"]:]
        joins.append(i_star)
        hi_prev = p["i_hi"]
    lng[~use] = np.nan
    return lng, use, joins, m_abs


def multirange(model_name, L, n_windows, overlap=0.4, e_range=None, q=10,
               n_proc=None, seed=12345, **wl_kwargs):
    """Wang-Landau with several (overlapping) energy windows in parallel."""
    model = make_model(model_name, L, q)
    E = model.energies()
    if e_range is None:
        i_first, i_last = 0, model.n_levels - 1
    else:
        i_first = int(math.ceil((e_range[0] - model.E_min) / model.dE))
        i_last = int(math.floor((e_range[1] - model.E_min) / model.dE))
    wins = make_windows(i_first, i_last, n_windows, overlap)
    ss = np.random.SeedSequence(seed)
    seeds = [int(s.generate_state(1)[0] % (2**31)) for s in ss.spawn(len(wins))]
    jobs = [(model_name, L, q, w, dict(wl_kwargs, seed=s)) for w, s in zip(wins, seeds)]
    t0 = time.time()
    n_proc = n_proc or min(len(jobs), mp.cpu_count())
    if n_proc > 1:
        with mp.get_context("fork").Pool(n_proc) as pool:
            parts = pool.map(_job, jobs, chunksize=1)
    else:
        parts = [_job(j) for j in jobs]
    wall = time.time() - t0
    lng, use, joins, m_abs = join_windows(parts, model.n_levels)
    return dict(model=model_name, L=L, N=model.N, q=model.q, E=E, lng=lng,
                use=use, m_abs=m_abs, joins=joins, windows=wins, parts=parts,
                wall=wall,
                sweeps=sum(p["sweeps"] for p in parts))


# --------------------------------------------------------------------------
# normalisation and symmetry
# --------------------------------------------------------------------------
def normalise(lng, use, model):
    """Fix the additive constant with the ground-state degeneracy
    (2 for Ising, q for Potts): ln g(E_min) = ln(g0)."""
    i0 = np.flatnonzero(use)[0]
    if i0 != 0:
        raise ValueError("ground state not sampled; cannot normalise")
    return lng - lng[0] + math.log(model.ground_state_degeneracy)


def ising_mirror(arr, use):
    """Ising: g(E) = g(-E) and <|M|>_E = <|M|>_-E for even L.
    Fill E > 0 from E < 0 (index i <-> N - i)."""
    n = arr.size - 1
    out, u = arr.copy(), use.copy()
    for i in range(n // 2 + 1, n + 1):
        out[i] = arr[n - i]
        u[i] = use[n - i]
    return out, u
