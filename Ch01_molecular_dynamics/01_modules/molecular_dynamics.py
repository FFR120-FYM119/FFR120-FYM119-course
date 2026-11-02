"""Chapter 1: transparent CPU molecular-dynamics calculations.

Lennard-Jones calculations use reduced units (m = sigma = epsilon = 1).
"""
import numpy as np


def lennard_jones_potential(distance, epsilon=1.0, sigma=1.0):
    """Calculate Lennard-Jones potential.
    """
    ratio6 = (sigma / np.asarray(distance)) ** 6
    ljp = 4 * epsilon * (ratio6**2 - ratio6)
    return ljp


def lennard_jones_force(positions, epsilon=1.0, sigma=1.0):
    """Calculate Lennard-Jones forces. 
    Full N-by-N pair matrix.
    No cutoff or periodic boundaries.
    positions.shape is (N, s)
    s is the spatial dimention (1, 2, 3, ...)
    """
    displacement = positions[:, None, :] - positions[None, :, :]
    squared = np.sum(displacement ** 2, axis=-1)
    np.fill_diagonal(squared, np.inf)
    inv6 = (sigma ** 2 / squared) ** 3
    coefficient = 24 * epsilon * (2 * inv6 ** 2 - inv6) / squared
    ljf = np.sum(coefficient[..., None] * displacement, axis=1)
    return ljf


def potential_energy(positions, epsilon=1.0, sigma=1.0):
    """Calculate potential energy for Lennard-Jones interaction. 
    """    
    displacement = positions[:, None, :] - positions[None, :, :]
    squared = np.sum(displacement ** 2, axis=-1)
    pairs = np.sqrt(squared[np.triu_indices(len(positions), 1)])
    PE = float(np.sum(lennard_jones_potential(pairs, epsilon, sigma)))
    return PE


def kinetic_energy(velocities, masses=1.0):
    """Calculate kinetic energy of a set of particles. 
    """    
    mass_array = np.asarray(masses)
    KE = 0.5 * np.sum(mass_array * np.sum(velocities**2, axis=1))
    return float(KE)


def reflect_walls(positions, velocities, box_length):
    """Reflect coordinates and normal velocity components.
    """
    for axis in range(positions.shape[1]):
        low = positions[:, axis] < 0
        positions[low, axis] *= -1
        velocities[low, axis] *= -1
        high = positions[:, axis] > box_length
        positions[high, axis] = 2 * box_length - positions[high, axis]
        velocities[high, axis] *= -1
    return positions, velocities


def step_particles(positions, velocities, dt, box_length, masses=1.0):
    """Performs the time evolution over one time step.
    Enforces reflecting walls.
    """
    midpoint = positions + 0.5 * dt * velocities
    forces = lennard_jones_force(midpoint)
    mass_array = np.broadcast_to(masses, (len(positions),))
    acceleration = forces / mass_array[:, None]
    velocities = velocities + dt * acceleration
    positions = midpoint + 0.5 * dt * velocities
    positions, velocities = reflect_walls(positions, velocities, box_length)
    return positions, velocities


def initial_gas(N, box_length, min_separation=1.0, speed=1.0):
    """Initializes positions and velocities.
    Rejection sampling of nonoverlapping initial positions.
    No fixed RNG seed.
    """
    rng = np.random.default_rng()
    positions = []
    for _ in range(N):
        for attempt in range(100000):
            candidate = rng.uniform(1.0, box_length - 1.0, size=2)
            if all(
                np.linalg.norm(candidate - old) >= min_separation 
                for old in positions
            ):
                positions.append(candidate)
                break
        else:
            raise ValueError('Box too crowded for requested separation')
    angles = rng.uniform(0, 2 * np.pi, N)
    velocities = speed * np.column_stack((np.cos(angles), np.sin(angles)))
    return np.array(positions), velocities


def mean_square_displacement(trajectory, lags):
    """Time-averaged MSD for an unwrapped trajectory,
    not a bounded reflected one.
    """
    MSD = np.array(
        [np.mean(np.sum((trajectory[lag:] - trajectory[:-lag])**2, axis=1)) 
         for lag in lags]
    )
    return MSD


""" Below: Brownian disk-specific functions.
    Interaction is only between disk and remaining particles.
    The remaining particles do not mutually interact.
"""

def initial_disk(N, box_length, min_separation=1.0, speed=1.0):
    """Initializes positions and velocities.
    Rejection sampling: no particle may overlap with the disk.
    No fixed RNG seed.
    """
    rng = np.random.default_rng()
    positions = np.array([[1, 1]]) * 0.5 * box_length
    #positions = [[0.5 * box_length, 0.5 * box_length]]
    for _ in range(N - 1):
        for attempt in range(100000):
            candidate = rng.uniform(1.0, box_length - 1.0, size=2)
            if np.linalg.norm(candidate - positions[0, :]) >= min_separation :
                positions = np.vstack((positions, candidate))
                break
        else:
            raise ValueError('Box too crowded for requested separation')
    angles = rng.uniform(0, 2 * np.pi, N)
    velocities = speed * np.column_stack((np.cos(angles), np.sin(angles)))
    velocities[0, :] *= 0
    return np.array(positions), velocities


def lennard_jones_force_disk(positions, epsilon=1.0, sigma=1.0):
    """Calculate Lennard-Jones forces; 
    only between particle 0 and the remaining particles.
    No cutoff or periodic boundaries.
    positions.shape is (N, s)
    s is the spatial dimention (1, 2, 3, ...)
    """
    displacement = positions[:, :] - positions[0, :]
    squared = np.sum(displacement ** 2, axis=1)
    squared[0] = np.inf
    inv6 = (sigma ** 2 / squared) ** 3
    coefficient = 24 * epsilon * (2 * inv6 ** 2 - inv6) / squared
    ljf = coefficient[:, None] * displacement
    ljf[0, :] = -np.sum(ljf, axis=0)
    return ljf


def step_brownian_disk(positions, velocities, dt, box_length, masses):
    """Performs the time evolution over one time step.
    The disk index is 0.
    The remaining masses do not mutually interact.
    Enforces reflecting walls.
    """
    midpoint = positions + 0.5 * dt * velocities
    forces = lennard_jones_force_disk(midpoint)
    mass_array = np.asarray(masses)
    acceleration = forces / mass_array[:, None] 
    velocities = velocities + dt * acceleration
    positions = midpoint + 0.5 * dt * velocities
    positions, velocities = reflect_walls(positions, velocities, box_length)
    return positions, velocities

