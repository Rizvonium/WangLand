"""Numba-compiled inner loops of the Wang-Landau random walk.

One call of a *_stage kernel performs the random walk at a fixed modification
factor ln f until the energy histogram H(E) is flat (or a sweep limit is hit):

    1. propose a single-spin change  E1 -> E2
    2. accept with  p = min(1, g(E1)/g(E2))       (Wang & Landau, eq. 1)
       moves that leave the energy window [i_lo, i_hi] are rejected
    3. update the *current* level:  ln g(E) += ln f ,  H(E) += 1
    4. every `check` sweeps: H is flat if min H >= flatness * <H>

All arrays are indexed by the energy level idx = (E - E_min)/dE.
Random numbers: xoroshiro128+ (state = 2 x uint64), ~1.7x faster than
numpy's Mersenne Twister inside numba.
"""
import numpy as np
from numba import njit, uint64


@njit(inline="always")
def rand(st):
    """xoroshiro128+ generator: uniform double in [0, 1); updates st."""
    s0, s1 = st[0], st[1]
    r = s0 + s1
    s1 ^= s0
    st[0] = ((s0 << uint64(24)) | (s0 >> uint64(40))) ^ s1 ^ (s1 << uint64(16))
    st[1] = (s1 << uint64(37)) | (s1 >> uint64(27))
    return (r >> uint64(11)) * (1.0 / 9007199254740992.0)


def new_rng_state(seed):
    """Seed xoroshiro128+ from an integer through numpy's SeedSequence."""
    st = np.random.SeedSequence(seed).generate_state(2, dtype=np.uint64)
    if not st.any():
        st[0] = 1
    return st


# --------------------------------------------------------------------------
# energies
# --------------------------------------------------------------------------
@njit(cache=True)
def ising_energy(spins, nbr):
    E = 0
    for k in range(spins.size):
        E -= spins[k] * (spins[nbr[k, 0]] + spins[nbr[k, 2]])
    return E


@njit(cache=True)
def potts_energy(spins, nbr):
    E = 0
    for k in range(spins.size):
        if spins[k] == spins[nbr[k, 0]]:
            E -= 1
        if spins[k] == spins[nbr[k, 2]]:
            E -= 1
    return E


@njit(cache=True)
def is_flat(H, use, i_lo, i_hi, flatness):
    """True if min H(E) >= flatness * <H(E)> over all reachable levels."""
    hsum, hmin, n = 0.0, 1e300, 0
    for i in range(i_lo, i_hi + 1):
        if use[i]:
            hsum += H[i]
            n += 1
            hmin = min(hmin, H[i])
    return n > 0 and hmin > 0 and hmin >= flatness * hsum / n


# --------------------------------------------------------------------------
# Ising: one Wang-Landau stage at fixed ln f
# --------------------------------------------------------------------------
@njit(cache=True)
def ising_stage(spins, nbr, st, state, lng, H, use, i_lo, i_hi, ln_f,
                flatness, check, max_sweeps, measure, m_sum, m_cnt):
    """state = [idx, M] (current energy level and magnetisation)."""
    N = spins.size
    idx, M = state[0], state[1]
    sweeps, flat = 0, False
    while sweeps < max_sweeps and not flat:
        for _ in range(check * N):
            k = int(rand(st) * N)                    # random site
            s = np.int64(spins[k])
            h = (np.int64(spins[nbr[k, 0]]) + spins[nbr[k, 1]]
                 + spins[nbr[k, 2]] + spins[nbr[k, 3]])
            j = idx + (s * h) // 2                   # dE = 2sh -> d(idx) = dE/4
            if i_lo <= j <= i_hi:                    # stay inside the window
                d = lng[idx] - lng[j]                # ln[g(E1)/g(E2)]
                if d >= 0.0 or rand(st) < np.exp(d):
                    spins[k] = -s                    # accept the flip
                    idx = j
                    M -= 2 * s
            lng[idx] += ln_f                         # g(E) -> g(E) * f
            H[idx] += 1.0                            # histogram
            if measure:                              # microcanonical <|M|>_E
                m_sum[idx] += abs(M)
                m_cnt[idx] += 1.0
        sweeps += check
        flat = is_flat(H, use, i_lo, i_hi, flatness)
    state[0], state[1] = idx, M
    return sweeps, flat


# --------------------------------------------------------------------------
# q-state Potts: one Wang-Landau stage at fixed ln f
# --------------------------------------------------------------------------
@njit(cache=True)
def potts_stage(spins, nbr, q, st, state, lng, H, use, i_lo, i_hi, ln_f,
                flatness, check, max_sweeps):
    N = spins.size
    idx = state[0]
    sweeps, flat = 0, False
    while sweeps < max_sweeps and not flat:
        for _ in range(check * N):
            k = int(rand(st) * N)
            c = spins[k]
            cn = (c + 1 + int(rand(st) * (q - 1))) % q   # new colour != c
            n_old, n_new = 0, 0
            for m in range(4):
                t = spins[nbr[k, m]]
                if t == c:
                    n_old += 1
                elif t == cn:
                    n_new += 1
            j = idx + n_old - n_new                  # E = -sum delta
            if i_lo <= j <= i_hi:
                d = lng[idx] - lng[j]
                if d >= 0.0 or rand(st) < np.exp(d):
                    spins[k] = cn
                    idx = j
            lng[idx] += ln_f
            H[idx] += 1.0
        sweeps += check
        flat = is_flat(H, use, i_lo, i_hi, flatness)
    state[0] = idx
    return sweeps, flat


# --------------------------------------------------------------------------
# bring a configuration into an energy window [e_lo, e_hi] (greedy walk)
# --------------------------------------------------------------------------
@njit(cache=True)
def _dist(E, e_lo, e_hi):
    if E < e_lo:
        return e_lo - E
    if E > e_hi:
        return E - e_hi
    return 0


@njit(cache=True)
def ising_drive(spins, nbr, st, E, e_lo, e_hi, max_steps):
    N = spins.size
    for _ in range(max_steps):
        if e_lo <= E <= e_hi:
            break
        k = int(rand(st) * N)
        s = np.int64(spins[k])
        h = (np.int64(spins[nbr[k, 0]]) + spins[nbr[k, 1]]
             + spins[nbr[k, 2]] + spins[nbr[k, 3]])
        En = E + 2 * s * h
        if _dist(En, e_lo, e_hi) <= _dist(E, e_lo, e_hi):
            spins[k] = -s
            E = En
    return E


@njit(cache=True)
def potts_drive(spins, nbr, q, st, E, e_lo, e_hi, max_steps):
    N = spins.size
    for _ in range(max_steps):
        if e_lo <= E <= e_hi:
            break
        k = int(rand(st) * N)
        c = spins[k]
        cn = (c + 1 + int(rand(st) * (q - 1))) % q
        n_old, n_new = 0, 0
        for m in range(4):
            t = spins[nbr[k, m]]
            if t == c:
                n_old += 1
            elif t == cn:
                n_new += 1
        En = E + n_old - n_new
        if _dist(En, e_lo, e_hi) <= _dist(E, e_lo, e_hi):
            spins[k] = cn
            E = En
    return E
