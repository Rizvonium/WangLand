# Wang–Landau sampling: a random walk in energy space

Computational Physics project: a reproduction of

> F. Wang and D. P. Landau, *Efficient, multiple-range random walk algorithm to
> calculate the density of states*, Phys. Rev. Lett. **86**, 2050 (2001)

The program estimates the density of states g(E) of the 2D Ising model and the
2D 10-state Potts model with Wang–Landau sampling, in a single energy window or
with many overlapping windows run in parallel ("multiple-range random walk"),
and derives F, U, S, C and ⟨|M|⟩ at any temperature. Every result is checked
against exact solutions that are also implemented here.

Units: J = k_B = 1 (energies in J, temperatures in J/k_B, C and S in k_B).

## Layout

| Path | Content |
|---|---|
| `wanglandau/models.py` | Ising and q-state Potts models on an L×L periodic lattice, energy levels |
| `wanglandau/kernels.py` | Numba-compiled Wang–Landau stage (`ising_stage`, `potts_stage`), xoroshiro128+ RNG |
| `wanglandau/sampler.py` | `wang_landau()` (one window, f → √f schedule), `multirange()` (parallel windows), `join_windows()` |
| `wanglandau/thermo.py` | F, U, S, C, ⟨\|M\|⟩ and P(E,T) from ln g(E) (log-sum-exp) |
| `wanglandau/exact.py` | exact g(E) (Beale 1996, via Kaufman's Z and a DFT in ball arithmetic), exact finite-L F, U, S, C (Kaufman / Ferdinand–Fisher), Onsager–Yang, Baxter's Potts results |
| `demo.py` | quick sample run (16×16, ~4 s) with saved console output |
| `run_all.py` | all production runs → `results/*.npz` |
| `make_figures.py` | all figures → `figures/*.png` |
| `accuracy_study.py` | 64×64: accuracy vs flatness, check interval and window count → `results/accuracy_L64.txt` |

## Running

```bash
pip install -r requirements.txt
python demo.py                      # 16x16 demo: results/demo_L16.npz, figures/demo_L16.png
python run_all.py convergence windows ising
python run_all.py ising256          # 256x256, 128 windows (about 25 min on 4 cores)
python run_all.py potts_fast --potts 24 32 40 48 56 64 80 96
python accuracy_study.py            # about 10 min on 4 cores
python make_figures.py              # every figure whose data exist
```

## Method and parameters (as in the paper)

* acceptance `p(E1→E2) = min(1, g(E1)/g(E2))`; after every attempt
  `ln g(E) += ln f`, `H(E) += 1` at the current level
* `ln f0 = 1` (f0 = e), `ln f → ln f / 2` (f → √f) when every `H(E) ≥ 0.8 ⟨H⟩`,
  stop when `ln f < 1e-8` (27 stages); flatness checked every 10 sweeps,
  1 sweep = N spin-flip attempts
* normalisation: `g(E_min) = 2` (Ising) or `q` (Potts); for Ising only
  `E/N ≤ 0.2` is simulated and `g(E) = g(-E)` gives the rest
* multiple ranges: overlapping windows (35–40 %), one walker per window and
  process; a move out of the window is rejected and the current level counted
  again; neighbouring pieces are joined at the energy in the overlap where
  their slopes `d ln g / dE` agree best

## Results (all checked against exact solutions)

| Test | This work | Paper |
|---|---|---|
| ln g(E), 32×32, one walk: mean relative error | 3.4×10⁻⁴ (9.3×10⁵ sweeps) | 3.5×10⁻⁴ (7×10⁵ sweeps) |
| ln g(E), 50×50, 4 windows: mean relative error | 1.1×10⁻⁴ | (figure only) |
| 256×256, multiple ranges | 128 windows, 3.4×10⁶ sweeps, 32 min on 4 cores | 15 windows, 6.1×10⁶ sweeps |
| 256×256 free energy, max ε(F) | 6×10⁻⁵ | < 8×10⁻⁶ |
| 256×256 specific heat, mean ε(C), 0.4 ≤ T ≤ 8 | 2.6 % (1 % at the peak) | 0.39 % |
| 64×64 specific heat, mean / max ε(C) | 1.2 % / 4.4 % (2.0×10⁶ sweeps); 0.67 % / 2.5 % with 90 % flatness (5.8×10⁶) | max < 0.7 % (2×10⁷ sweeps) |
| 2D Ising C_max(L), L = 8…256 | follows the exact (8K_c²/π) ln L growth, within 4 % of exact | – |
| 10-state Potts, T_c(L → ∞) from equal-height peaks | see `results/potts_table.txt` | T_c = 1/ln(1+√10) = 0.701232 |

Other checks: β(E) = d ln g/dE of the joined 256×256 result equals 1/T_c at
E/N = −√2; ⟨|M|⟩(T) approaches the Onsager–Yang curve; the window study
(`results/windows_table.txt`) and the convergence study
(`results/convergence_table.txt`, error saturation below ln f ≈ 10⁻⁶).

All figures are in `figures/`; numbers in `results/*_table.txt`.

## Acknowledgements and self-evaluation

This project was carried out with substantial help from Claude (Anthropic's AI
assistant), which helped write and debug the code, run the simulations, make
the figures and draft the presentation. Software: Python, NumPy, Numba,
Matplotlib, mpmath, python-flint (Arb).

Self-evaluation (out of 100): **[to be filled in by the student]**

## References

1. F. Wang, D. P. Landau, Phys. Rev. Lett. 86, 2050 (2001)
2. P. D. Beale, Phys. Rev. Lett. 76, 78 (1996)
3. B. Kaufman, Phys. Rev. 76, 1232 (1949); A. E. Ferdinand, M. E. Fisher, Phys. Rev. 185, 832 (1969)
4. L. Onsager, Phys. Rev. 65, 117 (1944); C. N. Yang, Phys. Rev. 85, 808 (1952)
5. R. J. Baxter, J. Phys. C 6, L445 (1973); C. Borgs, W. Janke, J. Phys. I France 2, 2011 (1992)
6. B. J. Schulz, K. Binder, M. Müller, D. P. Landau, Phys. Rev. E 67, 067102 (2003)
7. R. E. Belardinelli, V. D. Pereyra, Phys. Rev. E 75, 046701 (2007)
