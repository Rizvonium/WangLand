"""Accuracy versus Wang-Landau settings for the 64 x 64 Ising model.

Runs the same multiple-range calculation (E/N in [-2, 0.2]) with four settings
and compares ln g(E), F and C with the exact results (Beale; Kaufman /
Ferdinand-Fisher).  Output: results/accuracy_L64.txt  (about 10 min, 4 cores)

    python accuracy_study.py
"""
import os
import time

import numpy as np

from wanglandau import Ising, multirange, normalise, ising_mirror, canonical
from wanglandau.exact import ising_exact_dos, ising_kaufman

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = [  # (windows W, flatness check interval in sweeps, flatness criterion)
    (8, 10, 0.8),
    (8, 100, 0.8),
    (8, 10, 0.9),
    (2, 10, 0.8),
]


def main(L=64, seed=4242):
    m = Ising(L)
    N = m.N
    _, le = ising_exact_dos(L)
    T = np.linspace(0.4, 8, 153)
    ex = ising_kaufman(L, T)
    rows = []
    for W, check, flat in SETTINGS:
        t0 = time.time()
        r = multirange("ising", L, W, overlap=0.4, e_range=(-2 * N, 0.2 * N),
                       ln_f_final=1e-8, flatness=flat, check=check, seed=seed, n_proc=4)
        lng = normalise(np.where(r["use"], r["lng"], np.nan), r["use"], m)
        lng, _ = ising_mirror(lng, r["use"])
        ok = np.isfinite(lng) & np.isfinite(le)
        th = canonical(m.energies(), lng, T, N)
        eC = np.abs(th["C"] - ex["C"]) / ex["C"]
        eF = np.abs(th["F"] - ex["F"]) / np.abs(ex["F"])
        rows.append((W, check, flat, r["sweeps"], time.time() - t0,
                     np.mean(np.abs(lng[ok] - le[ok]) / le[ok]), eF.max(), eC.mean(), eC.max()))
        print(rows[-1], flush=True)
    np.savetxt(os.path.join(HERE, "results", "accuracy_L64.txt"), np.array(rows),
               header="W  check_sweeps  flatness  sweeps  wall_s  mean_eps_lng  max_eps_F  "
                      "mean_eps_C  max_eps_C   (L=64, 0.4<=T<=8, final ln f 1e-8)",
               fmt="%.6g")


if __name__ == "__main__":
    main()
