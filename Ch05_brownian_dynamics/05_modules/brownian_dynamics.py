"""Brownian dynamics for FFR120 Chapter 5.

Physical inputs are SI unless explicitly stated otherwise. All random-number
sequences are unseeded by default; pass a Generator only for a controlled test.
The notebook builds each update explicitly before importing these routines.
"""

from __future__ import annotations

import numpy as np

kB = 1.380649e-23  # J/K


def random_walks(
    n_walkers: int, 
    n_steps: int, 
    kind: str = "flip",
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Return positions with shape (n_steps + 1, n_walkers)."""
    if rng is None:
        rng = np.random.default_rng()
    if kind == "flip":
        steps = rng.choice([-1.0, 1.0], size=(n_steps, n_walkers))
    elif kind == "normal":
        steps = rng.standard_normal((n_steps, n_walkers))
    elif kind == "three":
        values = [-1.0, (1 - np.sqrt(3)) / 2, (1 + np.sqrt(3)) / 2]
        steps = rng.choice(values, size=(n_steps, n_walkers))
    else:
        raise ValueError("kind must be 'flip', 'normal', or 'three'")
    positions = np.zeros((n_steps + 1, n_walkers))
    positions[1:] = np.cumsum(steps, axis=0)
    return positions


def free_diffusion(
    n_walkers: int, 
    duration: float, 
    dt: float, 
    diffusion: float,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Exact free-diffusion increments; shape (n_steps + 1, n_walkers)."""
    if rng is None:
        rng = np.random.default_rng()
    n_steps = round(duration / dt)
    increments = (
        np.sqrt(2 * diffusion * dt) 
        * rng.standard_normal((n_steps, n_walkers))
    )
    positions = np.zeros((n_steps + 1, n_walkers))
    positions[1:] = np.cumsum(increments, axis=0)
    return positions


def inertial_langevin(
    noise: np.ndarray, 
    dt: float, 
    mass: float,
    gamma: float, 
    temperature: float,
    velocity_initial: np.ndarray | float = 0.0,
    position_initial: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Implicit friction update equivalent to textbook Eq. (5.6).
    
    noise has shape (n_steps,) or (n_steps, n_walkers). 
    The first axis is time.
    For an equilibrium MSD, sample initial velocities from N(0, kBT/m).
    """
    noise = np.asarray(noise)
    positions = np.empty((noise.shape[0] + 1, *noise.shape[1:]))
    velocities = np.empty_like(positions)
    positions[0] = position_initial
    velocities[0] = velocity_initial
    friction_ratio = gamma * dt / mass
    thermal_kick = np.sqrt(2 * kB * temperature * gamma * dt) / mass
    for step in range(noise.shape[0]):
        velocities[step + 1] = (
            velocities[step] + thermal_kick * noise[step]
        ) / (1 + friction_ratio)
        positions[step + 1] = positions[step] + dt * velocities[step + 1]
    return positions, velocities


def overdamped_free(
    noise: np.ndarray, 
    dt: float, 
    gamma: float,
    temperature: float,
    position_initial: np.ndarray | float = 0.0
) -> np.ndarray:
    """Textbook Eq. (5.7), using the supplied independent normal increments."""
    noise = np.asarray(noise)
    positions = np.empty((noise.shape[0] + 1, *noise.shape[1:]))
    positions[0] = position_initial
    increments = np.sqrt(2 * kB * temperature * dt / gamma) * noise
    positions[1:] = positions[0] + np.cumsum(increments, axis=0)
    return positions


def trapped_overdamped(
    noise: np.ndarray, 
    dt: float, 
    gamma: float,
    temperature: float, 
    stiffness: np.ndarray | float,
    position_initial: np.ndarray | float = 0.0
) -> np.ndarray:
    """Textbook Eq. (5.9). Axes may have different trap stiffnesses.

    Takes an array of N(0,1) noise of shape (steps,) or (steps, dimensions).
    For a scalar stiffness the same restoring coefficient acts on all axes.
    """
    noise = np.asarray(noise)
    positions = np.empty((noise.shape[0] + 1, *noise.shape[1:]))
    positions[0] = position_initial
    drift_factor = 1 - np.asarray(stiffness) * dt / gamma
    kick_factor = np.sqrt(2 * kB * temperature * dt / gamma)
    for step in range(noise.shape[0]):
        positions[step + 1] = (drift_factor * positions[step]
                              + kick_factor * noise[step])
    return positions


def mean_square_displacement(
    positions: np.ndarray,
    lags: np.ndarray) -> np.ndarray:
    """Time-averaged squared displacement at strictly positive integer lags.

    Positions may have shape (time,) or (time, dimensions). For multiple
    independent walkers, calculate one MSD per walker before averaging.
    """
    positions = np.asarray(positions)
    lags = np.asarray(lags, dtype=int)
    values = np.empty(len(lags))
    for i, lag in enumerate(lags):
        if lag < 1 or lag >= len(positions):
            raise ValueError("lags must lie between 1 and n_steps")
        differences = positions[lag:] - positions[:-lag]
        squared = differences**2
        if squared.ndim == 2:
            squared = squared.sum(axis=1)
        values[i] = squared.mean()
    return values


def correlation(
    trajectory: np.ndarray, 
    max_lag: int
) -> np.ndarray:
    """Covariance estimated using a mean-subtracted, finite trajectory.

    The denominator N-lag is used at each lag, so estimates at long lags
    become increasingly noisy. Use max_lag much smaller than N.
    """
    values = np.asarray(trajectory, dtype=float)
    centered = values - values.mean()
    results = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            results[lag] = np.mean(centered**2)
        else:
            results[lag] = np.mean(centered[lag:] * centered[:-lag])
    return results


def stokes_drag(
    radius: float, 
    viscosity: float
) -> float:
    """Stokes friction coefficient of a sphere: gamma = 6*pi*eta*R."""
    return 6 * np.pi * viscosity * radius
