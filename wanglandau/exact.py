"""Exact results used to validate the simulation (units J = k_B = 1).

* ising_exact_dos(L)    exact g(E) of the L x L periodic Ising model
                        (Beale, PRL 76, 78 (1996)) from Kaufman's partition
                        function, evaluated in ball arithmetic on N+1 points of
                        the unit circle and inverted with a DFT -> integers.
* ising_kaufman(L, T)   exact finite-lattice F, U, C (Kaufman 1949;
                        Ferdinand & Fisher, Phys. Rev. 185, 832 (1969)).
* Onsager / Yang infinite-lattice results, exact q-state Potts results.
"""
import math

import numpy as np

TC_ISING = 2.0 / math.log(1.0 + math.sqrt(2.0))          # 2.269185...
KC_ISING = 1.0 / TC_ISING


# --------------------------------------------------------------------------
# Kaufman's partition function written as a function of x = exp(-2K)
#   Z = 1/2 (2 sinh 2K)^{N/2} (Z1 + Z2 + Z3 + Z4),   cosh g_l = c - cos(pi l/L)
# Pairing equal g_l turns every factor into a polynomial in c (Chebyshev T_L),
# so Z(x) is a rational function of x and can be evaluated at complex x.
# --------------------------------------------------------------------------
def _kaufman_Q(x, L, cos_odd, cos_even):
    """Q(x) = x^N Z(x) = sum_n g(-2N+4n) x^{2n}  (a polynomial in x^2)."""
    from flint import acb
    N = L * L
    s = (1 - x * x) / (2 * x)               # sinh 2K
    ch = (1 + x * x) / (2 * x)              # cosh 2K
    c = ch * ch / s                          # cosh 2K coth 2K
    Z1 = acb(1)
    Z2 = acb(1)
    for co in cos_odd:                       # l = 1, 3, ..., L-1 (each twice)
        t = (c - co).chebyshev_t(L)          # cosh(L g_l)
        Z1 *= 2 * t + 2                      # [2 cosh(L g/2)]^2
        Z2 *= 2 * t - 2                      # [2 sinh(L g/2)]^2
    P3 = acb(1)
    P4 = acb(1)
    for co in cos_even:                      # l = 2, 4, ..., L-2 (each twice)
        t = (c - co).chebyshev_t(L)
        P3 *= 2 * t + 2
        P4 *= 2 * t - 2
    a, b = 1 - x, x * (1 + x)                # exp(g_0) = a/b = e^{2K} tanh K
    abh = (a * b) ** (L // 2)
    aL, bL = a ** L, b ** L
    cosh0, sinh0 = (aL + bL) / abh, (aL - bL) / abh
    cL = c + 1                               # l = L  (cos pi = -1)
    coshL = 2 * cL.chebyshev_t(L // 2)
    sinhL = 2 * (ch * (1 + s) / s) * cL.chebyshev_u(L // 2 - 1)
    Z = (2 * s) ** (N // 2) * (Z1 + Z2 + cosh0 * coshL * P3 + sinh0 * sinhL * P4) / 2
    return Z * x ** N


def ising_exact_dos(L, extra_bits=256, return_ints=False):
    """Exact density of states of the L x L Ising model (periodic, even L).

    Returns (E, ln g) for all N+1 levels (ln g = -inf where g = 0).
    """
    from flint import acb, arb, ctx
    if L % 2:
        raise ValueError("even L only")
    N = L * L
    M = N + 1
    old = ctx.prec
    ctx.prec = N + 16 * L + extra_bits
    try:
        cos_odd = [arb.cos_pi(arb(2 * r + 1) / L) for r in range(L // 2)]
        cos_even = [arb.cos_pi(arb(2 * r) / L) for r in range(1, L // 2)]
        # x_j^2 = u_j = exp(2 pi i (j + 1/2)/M)  (avoids x = +-1 where s = 0)
        vals = [_kaufman_Q(acb(arb(2 * j + 1) / (2 * M)).exp_pi_i(), L, cos_odd, cos_even)
                for j in range(M)]
        F = acb.dft(vals)                    # F_n = sum_j v_j e^{-2 pi i jn/M}
        g = []
        for n in range(M):
            cn = F[n] * acb(-arb(n) / M).exp_pi_i() / M
            z = cn.real.unique_fmpz()
            if z is None or abs(cn.imag) > 0.25:
                raise ArithmeticError(f"insufficient precision at n={n}")
            g.append(int(z))
    finally:
        ctx.prec = old
    if sum(g) != 2 ** N or g[0] != 2 or g[2] != 2 * N:
        raise ArithmeticError("exact DOS failed its sum rules")
    E = -2 * N + 4 * np.arange(M)
    lng = np.array([math.log(v) if v > 0 else -np.inf for v in g])
    return (E, lng, g) if return_ints else (E, lng)


# --------------------------------------------------------------------------
# exact finite-lattice thermodynamics at real temperature
# --------------------------------------------------------------------------
def _kaufman_lnZ(L, K):
    from mpmath import mpf, cosh, sinh, acosh, log, tanh, cospi, exp
    N = L * L
    K = mpf(K)
    c = cosh(2 * K) ** 2 / sinh(2 * K)
    l1 = l2 = l3 = mpf(0)
    z4 = mpf(1)
    for l in range(2 * L):
        if l == 0:
            g = 2 * K + log(tanh(K))          # signed gamma_0 (Kaufman)
        else:
            g = acosh(c - cospi(mpf(l) / L))
        h = L * g / 2
        if l % 2:
            l1 += log(2 * cosh(h))
            l2 += log(2 * sinh(h))
        else:
            l3 += log(2 * cosh(h))
            z4 *= 2 * sinh(h)
    tot = exp(l1) + exp(l2) + exp(l3) + z4
    return log(mpf(1) / 2) + mpf(N) / 2 * log(2 * sinh(2 * K)) + log(tot)


def ising_kaufman(L, T, dps=40):
    """Exact per-site F, U, C, S of the finite L x L periodic Ising model."""
    from mpmath import mp, mpf, diff
    T = np.atleast_1d(np.asarray(T, float))
    N = L * L
    out = {k: np.empty(T.size) for k in ("F", "U", "C", "S")}
    with mp.workdps(dps):
        f = lambda K: _kaufman_lnZ(L, K)
        for i, t in enumerate(T):
            K = mpf(1) / mpf(t)
            lnZ = f(K)
            d1 = diff(f, K, 1)
            d2 = diff(f, K, 2)
            F = -mpf(t) * lnZ / N
            U = -d1 / N
            out["F"][i] = float(F)
            out["U"][i] = float(U)
            out["C"][i] = float(K * K * d2 / N)
            out["S"][i] = float((U - F) / t)
    out["T"] = T
    return out


def onsager_magnetisation(T):
    """Spontaneous magnetisation of the infinite lattice (Yang 1952)."""
    T = np.asarray(T, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (1.0 - np.sinh(2.0 / T) ** -4) ** 0.125
    return np.where(T < TC_ISING, m, 0.0)


# --------------------------------------------------------------------------
# q-state Potts model (square lattice, E = -sum delta)
# --------------------------------------------------------------------------
def potts_tc(q):
    return 1.0 / math.log(1.0 + math.sqrt(q))


def potts_latent_heat(q, nmax=200):
    """Baxter (1973): latent heat per site at T_c for q > 4."""
    th = math.acosh(math.sqrt(q) / 2.0)
    prod = 1.0
    for n in range(1, nmax):
        prod *= math.tanh(n * th) ** 2
    return 2.0 * (1.0 + 1.0 / math.sqrt(q)) * math.tanh(th / 2.0) * prod


def potts_phase_energies(q):
    """Energies per site of the ordered and disordered phases at T_c."""
    mid = -(1.0 + 1.0 / math.sqrt(q))
    lat = potts_latent_heat(q)
    return mid - lat / 2.0, mid + lat / 2.0
