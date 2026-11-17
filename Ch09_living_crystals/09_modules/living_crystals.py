"""Chapter 9, section 9.3: finite-size active particles and phoretic attraction.

Particle positions are in micrometres with periodic square boundaries.
These overdamped position corrections are not hydrodynamic collisions.
"""

import numpy as np


def minimum_image(displacement, box_length):
    """Shortest periodic displacement along each coordinate."""
    return displacement - box_length * np.rint(displacement / box_length)


def initial_positions(number, box_length, radius, rng=None):
    """Rejection-sample nonoverlapping initial positions."""
    if rng is None:
        rng = np.random.default_rng()
    positions = np.empty((number, 2))
    for index in range(number):
        for attempt in range(20000):
            candidate = rng.uniform(0, box_length, size=2)
            offsets = minimum_image(
                positions[:index] - candidate, 
                box_length
            )
            if np.all(np.sum(offsets**2, axis=1) >= (2 * radius)**2):
                positions[index] = candidate
                break
        else:
            raise ValueError('Cannot place particles without overlap; lower density.')
    return positions


def phoretic_drift(positions, radius, strength, cutoff, box_length):
    """Vector sum of attractive Eq. (9.5) pair velocities, with a cutoff.

    Offsets[i, j] points from j to i, so attractive drift is -offset.
    """
    offsets = minimum_image(
        positions[:, None, :] - positions[None, :, :], 
        box_length
    )
    distances = np.linalg.norm(offsets, axis=2)
    interaction = ((distances > 0) & (distances < cutoff))
    safe_distances = np.where(interaction, distances, 1.0)
    coefficients = np.where(
        interaction, 
        -strength * radius**2 / safe_distances**3, 
        0.0
    )
    return np.sum(coefficients[:, :, None] * offsets, axis=1)


def correct_overlaps(positions, radius, box_length, sweeps=35):
    """Midpoint-preserving pair corrections; repeat until contact convergence.

    A vectorized distance search finds candidate overlaps; the actual pair
    corrections are sequential so a particle's latest position is respected.
    This is a geometric projection, not a hydrodynamic collision model.
    """
    positions = positions.copy()
    for sweep in range(sweeps):
        offsets = minimum_image(
            positions[:, None, :] - positions[None, :, :], box_length
        )
        squared_distances = np.sum(offsets**2, axis=2)
        overlap_candidates = np.triu(
            squared_distances < (2 * radius - 1e-8)**2, k=1
        )
        indices_i, indices_j = np.nonzero(overlap_candidates)
        if len(indices_i) == 0:
            break
        for i, j in zip(indices_i, indices_j):
            displacement = minimum_image(positions[i] - positions[j], box_length)
            distance = np.linalg.norm(displacement)
            overlap = 2 * radius - distance
            if overlap <= 0:
                continue
            if distance < 1e-12:
                direction = np.array([1.0, 0.0])
            else:
                direction = displacement / distance
            shift = 0.5 * overlap * direction
            positions[i] += shift
            positions[j] -= shift
        positions %= box_length
    return positions


def step_living(positions, angles, dt, speed, diffusion_t,
                diffusion_r, radius, strength, cutoff, box_length, rng=None):
    """Simultaneous ABP and phoretic drift, then geometric overlap correction."""
    if rng is None:
        rng = np.random.default_rng()
    direction = np.column_stack((np.cos(angles), np.sin(angles)))
    drift = phoretic_drift(positions, radius, strength, cutoff, box_length)
    next_positions = (positions + dt * (speed * direction + drift)
                      + np.sqrt(2 * diffusion_t * dt) * rng.normal(size=positions.shape))
    next_angles = angles + np.sqrt(2 * diffusion_r * dt) * rng.normal(size=len(angles))
    next_positions %= box_length
    next_positions = correct_overlaps(next_positions, radius, box_length)
    return next_positions, next_angles


def cluster_sizes(positions, radius, box_length, contact_factor=1.2):
    """Connected components of the distance graph (contact threshold > 2R)."""
    number = len(positions)
    offsets = minimum_image(positions[:, None] - positions[None, :], box_length)
    close = np.sum(offsets**2, axis=-1) < (contact_factor * 2 * radius)**2
    seen = np.zeros(number, dtype=bool)
    sizes = []
    for start in range(number):
        if seen[start]:
            continue
        seen[start] = True
        frontier = [start]
        size = 0
        while frontier:
            i = frontier.pop()
            size += 1
            for j in np.flatnonzero(close[i] & ~seen):
                seen[j] = True
                frontier.append(int(j))
        sizes.append(size)
    return np.array(sizes, dtype=int)
