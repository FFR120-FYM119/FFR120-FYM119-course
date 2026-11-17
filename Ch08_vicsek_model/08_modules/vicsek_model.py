"""Chapter 8: the Vicsek model in a periodic, two-dimensional square.

All particles have speed ``speed``. Positions are stored in [0, box_length)^2;
angles are in radians. No random-number generator is seeded by default.

The position update uses orientation theta_n, while orientation theta_(n+1)
is calculated synchronously from the old state. This makes the update order
explicit and follows the position rule in textbook Equation (8.4).
"""

import numpy as np
from scipy.spatial import Voronoi, cKDTree


def initial_state(n_particles, box_length, rng=None):
    """Return independent uniformly distributed positions and angles."""
    
    if rng is None:
        rng = np.random.default_rng()
        
    positions = rng.uniform(0.0, box_length, (n_particles, 2))
    angles = rng.uniform(-np.pi, np.pi, n_particles)
    
    return positions, angles


def wrap_positions(positions, box_length):
    """Apply periodic boundaries even when a particle crosses multiple sides."""
    
    return np.mod(positions, box_length)


def minimum_image(displacements, box_length):
    """Return shortest signed displacements on a periodic square."""
    
    shortest_displ = (
        displacements 
        - box_length * np.rint(displacements / box_length)
    )
    
    return shortest_displ


def alignment(angles):
    """Equation (8.5): global orientational order (0 to 1)."""
    
    psi = np.hypot(np.cos(angles).mean(), np.sin(angles).mean())
    
    return psi


def metric_neighbors(positions, box_length, flocking_radius):
    """Neighborhood includes the focal particle (Eq. 8.2)."""
    
    wrapped_positions = wrap_positions(positions, box_length)

    tree = cKDTree(wrapped_positions, boxsize=box_length)
    
    neighbor_indices = tree.query_ball_point(
        wrapped_positions, flocking_radius
    )

    return neighbor_indices


def circular_mean(angles):
    """Mean unit-vector direction; undefined for exactly canceling vectors."""
    
    sine = np.sin(angles).sum()
    cosine = np.cos(angles).sum()
    
    if np.hypot(sine, cosine) < 1e-16:
        return np.nan
    
    return np.arctan2(sine, cosine)


def aligned_angles(neighbors, perceived_angles, noise, dt, rng):
    """All output orientations use the *same* input orientation array."""
    
    next_angles = np.empty_like(perceived_angles)
    
    for j, nearby in enumerate(neighbors):
        direction = circular_mean(perceived_angles[nearby])
        if np.isnan(direction):
            direction = perceived_angles[j]
        next_angles[j] = direction + noise * dt * rng.uniform(-0.5, 0.5)
        
    next_angles = (next_angles + np.pi) % (2 * np.pi) - np.pi
    # equivalent to:
    # next_angles = next_angles % (2 * np.pi)
    # next_angles[next_angles >= np.pi] -= 2 * np.pi
        
    return next_angles


def step_metric(positions, angles, box_length, speed, flocking_radius,
                noise, dt=1.0, rng=None):
    """One synchronous standard Vicsek step with metric interactions."""
    if rng is None:
        rng = np.random.default_rng()
        
    neighbors = metric_neighbors(positions, box_length, flocking_radius)
    
    new_angles = aligned_angles(neighbors, angles, noise, dt, rng)
    
    displacement = dt * speed * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    
    new_positions = wrap_positions(positions + displacement, box_length)
    
    return new_positions, new_angles


def step_delayed(positions, angles, angle_history, delay, box_length,
                 speed, flocking_radius, noise, dt=1.0, rng=None):
    """Delay h >= 0: current neighbors, orientations h steps in the past.

    ``angle_history`` is oldest-to-newest and ends with ``angles``.
    At initial time, the calling notebook can pad prehistory with theta_0.
    This convention treats the sensory neighborhood as sampled *now*.
    """
    if delay < 0:
        raise ValueError('For negative delay use step_predictive().')
    if len(angle_history) <= delay:
        raise ValueError('History must include delay + 1 arrays.')
    if rng is None:
        rng = np.random.default_rng()
        
    neighbors = metric_neighbors(positions, box_length, flocking_radius)
    
    new_angles = aligned_angles(
        neighbors, angle_history[-delay-1], noise, dt, rng)
    
    displacement = dt * speed * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    
    return wrap_positions(positions + displacement, box_length), new_angles



def step_delay(positions, angles, angle_history, delay, box_length,
                 speed, flocking_radius, noise, dt=1.0, rng=None):
    """Delay h >= 0: BOTH neighbors AND orientations h steps in the past.

    ``circmean_history`` is oldest-to-newest and ends with the circular mean 
    calculated with the current neighbours' orientation.
    At initial time, the calling notebook should:
    - start building the history
    - update the position of each particle as if it had no neighbours but itself.
    """
    if delay < 0:
        raise ValueError('For negative delay use step_predictive().')
    if len(angle_history) <= delay:
        raise ValueError('History must include delay + 1 arrays.')
    if rng is None:
        rng = np.random.default_rng()
        
    neighbors = metric_neighbors(positions, box_length, flocking_radius)
    
    new_history = aligned_angles(
        neighbors, angles, noise, dt, rng)
        
    new_angles = angle_history[0]
        
    displacement = dt * speed * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    
    new_positions = wrap_positions(positions + displacement, box_length)
    
    return new_positions, new_angles, new_history



def predict_angles(angle_history, prediction_horizon, window=5):
    """Predict per-particle angles with a least-squares linear trend.

    Unwrap time series before extrapolation; h=0 returns the fitted present
    value, not necessarily the measured present value.
    """
    if len(angle_history) < window or window < 2:
        raise ValueError('At least window >= 2 orientation arrays are needed.')
        
    y = np.unwrap(np.asarray(angle_history[-window:]), axis=0)
    # np.unwrap() is needed because we enforce angles to be
    #             in the interval [-np.pi, np.pi)
    #             this removes jumps 
    
    t = np.arange(window, dtype=float)
    
    center = t.mean()
    slopes = np.sum((
        t[:, None] - center) * (y - y.mean(axis=0)), 
        axis=0) / np.sum((t-center)**2)
    
    prediction = (
        y.mean(axis=0) + slopes * ((window - 1 + prediction_horizon) - center)
    )
    
    return prediction


def step_predictive(positions, angles, angle_history, horizon, box_length,
                    speed, flocking_radius, noise, dt=1.0, window=5, rng=None):
    """Negative-delay interpretation: predict h steps ahead using angle history."""
    if horizon < 0:
        raise ValueError('horizon must be >= 0 (use horizon=abs(delay)).')
    if rng is None:
        rng = np.random.default_rng()
        
    perceived = predict_angles(angle_history, horizon, window=window)
    
    neighbors = metric_neighbors(positions, box_length, flocking_radius)
    
    new_angles = aligned_angles(neighbors, perceived, noise, dt, rng)
    
    displacement = dt * speed * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    
    new_positions = wrap_positions(positions + displacement, box_length)
    
    return new_positions, new_angles


def voronoi_clustering(pos, box_length, flocking_radius):
    """Calculate periodic Voronoi areas and count small cells."""
    
    # Replicate positions in a 3x3 grid (PBC)
    shifts = np.array([
        [i, j]
        for i in [-1, 0, 1]
        for j in [-1, 0, 1]
    ]) * box_length

    points = np.vstack([pos + shift for shift in shifts])

    # Voronoi tessellation
    vor = Voronoi(points)

    # Central copy of the particles
    n_part = len(pos)
   
    central = 4 * n_part  # "4" indicates the central square

    areas = np.zeros(n_part)

    for i in range(n_part):
        region = vor.regions[vor.point_region[central + i]]

        if not region or -1 in region:
            areas[i] = np.inf
            continue

        vertices = vor.vertices[region]

        # Polygon area (shoelace formula)
        x, y = vertices[:, 0], vertices[:, 1]
        areas[i] = 0.5 * abs(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )

    # Count cells smaller than the flocking area
    threshold = np.pi * flocking_radius**2
    c_coeff = np.count_nonzero(areas < threshold) / n_part

    return c_coeff, areas


