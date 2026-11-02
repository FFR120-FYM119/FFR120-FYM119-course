"""Implementations of the 2D Ising model.
"""

import numpy as np


def random_spins(size, rng=None):
    """Return a square lattice of independent spins in {-1, +1}."""
    if rng is None:
        rng = np.random.default_rng()
    spins = 2 * rng.integers(0, 2, size=(size, size), dtype=np.int8) - 1
    return spins


def random_spins_rectangular(Lx, Ly, rng=None):
    """Return a rectangular lattice of independent spins in {-1, +1}."""
    if rng is None:
        rng = np.random.default_rng()
    spins = 2 * rng.integers(0, 2, size=(Ly, Lx), dtype=np.int8) - 1
    return spins


def neighbor_sum(spins):
    """Four-neighbor sum using periodic boundaries on both axes."""
    M = (
        np.roll(spins, 1, axis=0) 
        + np.roll(spins, -1, axis=0)
        + np.roll(spins, 1, axis=1) 
        + np.roll(spins, -1, axis=1)
    )
    return M


def magnetization(spins):
    """Return magnetization."""
    m = float(np.mean(spins))
    return m


def energy(spins, J=1.0, H=0.0):
    """Total energy -J sum_<ij> s_i s_j - H sum_i s_i, counting each bond once."""
    right = np.roll(spins, -1, axis=1)
    down = np.roll(spins, -1, axis=0)
    # E = float(-J * np.sum(spins * (right + down)) - H * np.sum(spins))
    E = float(-J * np.mean(spins * (right + down)) - H * np.mean(spins))
    return E


def heat_bath_sweep(spins, T, rng=None, J=1.0, H=0.0, kB=1.0, f=0.1):
    """Update f fraction of random spins.
    """    
    if rng is None:
        rng = np.random.default_rng()
    Ly, Lx = spins.shape
    n = round(f * Lx * Ly)
    indices = rng.choice(Lx * Ly, size=n, replace=False)
    y, x = np.unravel_index(indices, (Ly, Lx))
    # equivalent to:
    # y = indices // Lx
    # x = indices % Lx    
    # because NumPy's default ordering is row-major (C-order)
    M = (
        spins[(y + 1) % Ly, x]
        + spins[(y - 1) % Ly, x]
        + spins[y, (x + 1) % Lx]
        + spins[y, (x - 1) % Lx]
    )
    p_plus = 0.5 * (1.0 + np.tanh((H + J * M) / (kB * T)))
    spins[y, x] = np.where(rng.random(p_plus.shape) < p_plus, 1, -1)


def simulate(spins, T, n_steps=200, 
             rng=None, J=1.0, H=0.0, kB=1.0, f=0.1, n_skip=1):
    """Run and record magnetization"""
    if rng is None:
        rng = np.random.default_rng()
    m = []
    for step in range(n_steps):
        heat_bath_sweep(spins, T, rng, J, H, kB, f)
        if (step + 1) % n_skip == 0:
            m.append(magnetization(spins))
    return np.asarray(m)


""" This below is for the parallel vertical plates.
"""


def magnetization_plates(spins, x1, x2):
    """Return mean spin; exclude fixed-spin plate columns."""
    Ly, Lx = spins.shape
    m = float(np.sum(spins) - np.sum(spins[:, x1]) - np.sum(spins[:, x2]))
    m /= Ly * (Lx - 2)
    return m


def spins_plates(size, x1, x2, s1=1, s2=1, rng=None):
    """Create full-width single-column fixed-spin plates with periodic boundaries."""
    if s1 not in (-1, 1) or s2 not in (-1, 1):
        raise ValueError("Plate spins must be +1 or -1")
    spins = random_spins(size, rng)
    spins[:, x1] = s1
    spins[:, x2] = s2
    return spins


def spins_plates_rectangular(Lx, Ly, x1, x2, s1=1, s2=1, rng=None):
    """Create full-width single-column fixed-spin plates with periodic boundaries."""
    if s1 not in (-1, 1) or s2 not in (-1, 1):
        raise ValueError("Plate spins must be +1 or -1")
    spins = random_spins_rectangular(Lx, Ly, rng)
    spins[:, x1] = s1
    spins[:, x2] = s2
    return spins


def heat_bath_sweep_plates(spins, x1, x2, s1, s2, T, rng=None, J=1.0, H=0.0, kB=1.0, f=0.1):
    """Update f fraction of random spins.
    """    
    if rng is None:
        rng = np.random.default_rng()
    Ly, Lx = spins.shape
    n = round(f * Lx * Ly)
    indices = rng.choice(Lx * Ly, size=n, replace=False)
    y, x = np.unravel_index(indices, (Ly, Lx))
    # equivalent to:
    # y = indices // Lx
    # x = indices % Lx    
    # because NumPy's default ordering is row-major (C-order)
    M = (
        spins[(y + 1) % Ly, x]
        + spins[(y - 1) % Ly, x]
        + spins[y, (x + 1) % Lx]
        + spins[y, (x - 1) % Lx]
    )
    p_plus = 0.5 * (1.0 + np.tanh((H + J * M) / (kB * T)))
    spins[y, x] = np.where(rng.random(p_plus.shape) < p_plus, 1, -1)
    
    # enforce condition
    spins[:, x1] = s1
    spins[:, x2] = s2


def simulate_plates(spins, x1, x2, s1, s2, T, n_steps=200, 
             rng=None, J=1.0, H=0.0, kB=1.0, f=0.1, n_skip=1):
    """Run and record magnetization"""
    if rng is None:
        rng = np.random.default_rng()
    m = []
    for step in range(n_steps):
        heat_bath_sweep_plates(
            spins, x1, x2, s1, s2, T, rng, J, H, kB, f
        )
        if (step + 1) % n_skip == 0:
            m.append(magnetization(spins))
    return np.asarray(m)
