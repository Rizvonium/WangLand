"""Quick demonstration run (a few seconds): Wang-Landau for a small Ising lattice.

    python demo.py                 # L = 16, ln f_final = 1e-6
    python demo.py --L 8 --lnf 1e-5

Input  : model, L, ln f0, ln f_final, flatness criterion, seed
Output : results/demo_L{L}.npz   (E, lng, H, stages, thermodynamics)
         results/demo_L{L}_log.txt (this console output, saved as backup)
         figures/demo_L{L}.png    (ln g(E), final histogram, C(T))
"""
import argparse
import io
import os
import sys
import time

import numpy as np

from wanglandau import Ising, wang_landau, normalise, canonical
from wanglandau.exact import ising_exact_dos, TC_ISING

HERE = os.path.dirname(os.path.abspath(__file__))


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            if not st.closed:
                st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            if not st.closed:
                st.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--lnf", type=float, default=1e-6, help="final ln f")
    ap.add_argument("--flat", type=float, default=0.8, help="flatness criterion")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    log = open(os.path.join(HERE, "results", f"demo_L{a.L}_log.txt"), "w")
    out = Tee(sys.stdout, log)

    model = Ising(a.L)
    print("Wang-Landau demo: 2D Ising model, periodic L x L lattice, J = k_B = 1", file=out)
    print(f"  input : L = {a.L} (N = {model.N} spins), {model.n_levels} energy levels,"
          f" seed = {a.seed}", file=out)
    print(f"          ln f0 = 1, ln f_final = {a.lnf:g}, flatness = {a.flat}", file=out)
    t0 = time.time()
    res = wang_landau(model, ln_f_final=a.lnf, flatness=a.flat, seed=a.seed,
                      verbose=True, log=lambda s: print(s, file=out))
    dt = time.time() - t0

    lng = normalise(np.where(res["use"], res["lng"], np.nan), res["use"], model)
    E = res["E"]
    E_ex, lng_ex = ising_exact_dos(a.L)
    ok = np.isfinite(lng) & np.isfinite(lng_ex)
    eps = np.abs(lng[ok] - lng_ex[ok]) / lng_ex[ok]
    T = np.linspace(0.5, 6.0, 221)
    th = canonical(E, lng, T, model.N)
    th_ex = canonical(E_ex, lng_ex, T, model.N)
    i = np.argmax(th["C"])

    lnsum = (np.log(np.nansum(np.exp(lng - np.nanmax(lng)))) + np.nanmax(lng)) / model.N
    print(f"  done  : {res['sweeps']:,} MC sweeps in {dt:.1f} s "
          f"({res['sweeps'] * model.N / dt / 1e6:.0f} M flips/s)", file=out)
    print(f"  output: E[{E.size}] lng[{lng.size}] H[{res['H'].size}] "
          f"stages[{len(res['stages'])}] F,U,S,C(T)[{T.size}]", file=out)
    print(f"  check : ln g(E_min) = {lng[0]:.4f} = ln 2", file=out)
    print(f"          ln(sum g)/N = {lnsum:.5f}  (ln 2 = {np.log(2):.5f})", file=out)
    print(f"  exact : mean eps(ln g) = {eps.mean():.2e}", file=out)
    print(f"          C_max/N = {th['C'][i]:.3f} at T = {T[i]:.3f} "
          f"(exact {th_ex['C'].max():.3f} at {T[np.argmax(th_ex['C'])]:.3f})", file=out)

    np.savez_compressed(os.path.join(HERE, "results", f"demo_L{a.L}.npz"),
                        E=E, lng=lng, H=res["H"], use=res["use"],
                        stages=res["stages"], T=T, F=th["F"], U=th["U"],
                        S=th["S"], C=th["C"], lng_exact=lng_ex,
                        C_exact=th_ex["C"], L=a.L)
    log.close()
    if not a.no_plot:
        from make_figures import demo_figure
        demo_figure(a.L)


if __name__ == "__main__":
    main()
