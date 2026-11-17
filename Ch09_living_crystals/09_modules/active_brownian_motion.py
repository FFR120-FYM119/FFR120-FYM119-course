"""Chapter 9, sections 9.1–9.2: two-dimensional active Brownian motion.

Lengths are in micrometres and times in seconds; no random seed is fixed.
"""

import numpy as np


def abp_step(position, angle, dt, speed, diffusion_t, diffusion_r, rng=None):
    """Single time step"""
    if rng is None:
        rng = np.random.default_rng()
    direction = np.array([np.cos(angle), np.sin(angle)])
    position_new = (
        position 
        + speed * dt * direction
        + np.sqrt(2 * diffusion_t * dt) * rng.normal(size=2)
    )
    angle_new = angle + np.sqrt(2 * diffusion_r * dt) * rng.normal()
    return position_new, angle_new


def simulate_abp(steps, dt, speed, diffusion_t, diffusion_r, rng=None):
    """Generate unwrapped positions and unwrapped orientation, including t=0."""
    if rng is None:
        rng = np.random.default_rng()
    positions = np.zeros((steps + 1, 2))
    angles = np.zeros(steps + 1)
    angles[0] = rng.uniform(-np.pi, np.pi)
    for step in range(steps):
        positions[step + 1], angles[step + 1] = abp_step(
            positions[step], 
            angles[step], 
            dt, 
            speed,
            diffusion_t, 
            diffusion_r, 
            rng,
        )
    return positions, angles


def abp_msd_theory(lags, speed, diffusion_t, diffusion_r):
    """Eq. (9.4), evaluated stably also for small DR*t and DR=0."""
    lags = np.asarray(lags, dtype=float)
    if diffusion_r == 0:
        return 4 * diffusion_t * lags + (speed * lags) ** 2
    z = diffusion_r * lags
    # exp(-z) - 1 + z, avoiding cancellation at short times
    bracket = np.expm1(-z) + z
    msd = (
        4 * diffusion_t * lags 
        + (2 * speed ** 2 / diffusion_r ** 2) * bracket
    )
    return msd


def time_averaged_msd(positions, lag_steps):
    """Average squared displacement for each lag along one unwrapped track."""
    msd_t = np.empty(len(lag_steps))
    for index, lag in enumerate(lag_steps):
        if lag < 1 or lag >= len(positions):
            raise ValueError(
                'Lags must lie between 1 and len(positions)-1'
            )
        increments = positions[lag:] - positions[:-lag]
        msd_t[index] = np.mean( np.sum(increments ** 2 , axis=1) )
    return msd_t
