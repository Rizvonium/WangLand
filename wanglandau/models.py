"""Lattice models used with the Wang-Landau sampler.

Both models live on an L x L square lattice with periodic boundary conditions,
nearest-neighbour coupling J = 1 and k_B = 1 (so T is measured in J/k_B).

    Ising :  H = -J * sum_<ij> s_i s_j ,            s_i = +-1
             E = -2N, -2N+4, ..., +2N               (N + 1 levels, step 4)
    Potts :  H = -J * sum_<ij> delta(s_i, s_j),     s_i = 0..q-1
             E = -2N, -2N+1, ..., 0                 (2N + 1 levels, step 1)

Energy levels are stored on an integer grid  idx = (E - E_min) / dE.
"""
import numpy as np


def square_lattice_neighbours(L):
    """nbr[k] = (right, left, down, up) for site k = x + L*y (periodic)."""
    k = np.arange(L * L)
    x, y = k % L, k // L
    nbr = np.empty((L * L, 4), dtype=np.int32)
    nbr[:, 0] = (x + 1) % L + L * y
    nbr[:, 1] = (x - 1) % L + L * y
    nbr[:, 2] = x + L * ((y + 1) % L)
    nbr[:, 3] = x + L * ((y - 1) % L)
    return nbr


class Ising:
    """2D Ising model (ferromagnetic, J = 1)."""

    name = "ising"

    def __init__(self, L):
        if L % 2 or L < 4:
            raise ValueError("use an even L >= 4 (periodic lattice)")
        self.L, self.N = L, L * L
        self.nbr = square_lattice_neighbours(L)
        self.E_min, self.E_max, self.dE = -2 * self.N, 2 * self.N, 4
        self.n_levels = self.N + 1
        self.ground_state_degeneracy = 2          # all up / all down
        self.q = 2

    def energies(self):
        return self.E_min + self.dE * np.arange(self.n_levels)

    def reachable(self):
        """E = -2N+4 and E = 2N-4 cannot occur (one flip costs 8J)."""
        ok = np.ones(self.n_levels, dtype=np.bool_)
        ok[1] = ok[-2] = False
        return ok

    def ground_state(self):
        return np.ones(self.N, dtype=np.int8)


class Potts:
    """2D q-state Potts model (ferromagnetic, J = 1)."""

    name = "potts"

    def __init__(self, L, q=10):
        if L % 2 or L < 4:
            raise ValueError("use an even L >= 4 (periodic lattice)")
        self.L, self.N, self.q = L, L * L, q
        self.nbr = square_lattice_neighbours(L)
        self.E_min, self.E_max, self.dE = -2 * self.N, 0, 1
        self.n_levels = 2 * self.N + 1
        self.ground_state_degeneracy = q          # q uniform states

    def energies(self):
        return self.E_min + self.dE * np.arange(self.n_levels)

    def reachable(self):
        """Breaking bonds costs at least 4 (one spin) or 6, 7 (two spins):
        E - E_min = 1, 2, 3, 5 are impossible."""
        ok = np.ones(self.n_levels, dtype=np.bool_)
        ok[[1, 2, 3, 5]] = False
        return ok

    def ground_state(self):
        return np.zeros(self.N, dtype=np.int8)


def make_model(name, L, q=10):
    name = name.lower()
    if name == "ising":
        return Ising(L)
    if name == "potts":
        return Potts(L, q)
    raise ValueError(f"unknown model {name!r}")
