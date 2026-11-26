"""Wang-Landau (multiple-range random walk) sampling of the density of states.

Reproduction of F. Wang and D. P. Landau, "An efficient, multiple range random
walk algorithm to calculate the density of states", PRL 86, 2050 (2001).
"""
from .models import Ising, Potts, make_model
from .sampler import wang_landau, multirange, normalise, ising_mirror, join_windows
from .thermo import canonical, energy_distribution, equal_height_temperature

__all__ = ["Ising", "Potts", "make_model", "wang_landau", "multirange",
           "normalise", "ising_mirror", "join_windows", "canonical",
           "energy_distribution", "equal_height_temperature"]
