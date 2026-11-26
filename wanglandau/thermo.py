"""Canonical thermodynamics from a density of states (units J = k_B = 1).

    Z(T) = sum_E g(E) exp(-E/T)                     partition function
    F    = -T ln Z                                  free energy      (eq. 3)
    U    = <E>_T                                    internal energy
    C    = (<E^2>_T - <E>_T^2) / T^2                specific heat    (eq. 4)
    S    = (U - F) / T                              entropy
    |M|  = sum_E <|M|>_E g(E) e^{-E/T} / Z          order parameter

All sums are done with the log-sum-exp trick, because g(E) spans
hundreds of orders of magnitude (ln g ~ N ln 2).
"""
import numpy as np


def canonical(E, lng, T, N, m_abs=None):
    """Return per-site F, U, S, C (and |M|) on the temperature grid T."""
    ok = np.isfinite(lng)
    E = np.asarray(E, float)[ok]
    lng = np.asarray(lng, float)[ok]
    T = np.atleast_1d(np.asarray(T, float))
    a = lng[None, :] - E[None, :] / T[:, None]          # ln[g(E) e^{-E/T}]
    amax = a.max(axis=1, keepdims=True)
    w = np.exp(a - amax)                                 # stable weights
    Z = w.sum(axis=1)
    lnZ = amax[:, 0] + np.log(Z)
    p = w / Z[:, None]                                   # P(E, T)
    U = p @ E
    var = (p * (E[None, :] - U[:, None]) ** 2).sum(axis=1)
    F = -T * lnZ
    out = dict(T=T, lnZ=lnZ, F=F / N, U=U / N, C=var / T**2 / N,
               S=(U - F) / T / N)
    if m_abs is not None:
        m = np.asarray(m_abs, float)[ok]
        good = np.isfinite(m)
        out["M"] = (p[:, good] @ m[good]) / p[:, good].sum(axis=1)
    return out


def energy_distribution(E, lng, T):
    """Canonical energy distribution P(E,T) = g(E) e^{-E/T} / Z."""
    ok = np.isfinite(lng)
    a = np.where(ok, lng - np.asarray(E, float) / T, -np.inf)
    a = a - a[ok].max()
    p = np.exp(a)
    return p / p[ok].sum()


def equal_height_temperature(E, lng, T_guess, e_split, tol=1e-10):
    """Temperature at which the two peaks of P(E,T) (below / above e_split)
    have equal height (Wang & Landau, Fig. 4). Bisection on ln(P1/P2)."""
    E = np.asarray(E, float)
    ok = np.isfinite(lng)
    lo_mask = ok & (E < e_split)
    hi_mask = ok & (E >= e_split)

    def diff(T):
        a = lng - E / T
        return np.nanmax(np.where(lo_mask, a, -np.inf)) - \
            np.nanmax(np.where(hi_mask, a, -np.inf))

    a, b = T_guess * 0.97, T_guess * 1.03
    fa = diff(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = diff(m)
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
        if b - a < tol:
            break
    return 0.5 * (a + b)
