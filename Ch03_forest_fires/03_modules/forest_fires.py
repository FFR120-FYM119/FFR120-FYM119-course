"""Forest-fire simulation following Argun, Callegari & Volpe, chapter 3.

A step grows trees independently on empty cells, then attempts one lightning strike
with probability f. One fire consumes an entire four-connected periodic cluster.
No random seed is imposed. Length and time are measured in lattice units and steps.
"""
import numpy as np


def grow(forest, p, rng=None):
    """Return a new array with independent growth at every empty site."""
    rng = np.random.default_rng() if rng is None else rng
    forest |= rng.random(forest.shape) < p
    return forest 


def cluster_sites(forest, start):
    """Iterative flood fill, respecting periodic boundaries (not recursive)."""
    nrows, ncols = forest.shape
    row, col = start
    if not forest[row, col]:
        return []
    visited = np.zeros_like(forest, dtype=bool)
    visited[row, col] = True
    stack = [(row, col)]
    cluster = []
    while stack:
        row, col = stack.pop()
        cluster.append((row, col))
        for r, c in (
            ((row - 1) % nrows, col), ((row + 1) % nrows, col),
            (row, (col - 1) % ncols), (row, (col + 1) % ncols)
        ):
            if forest[r, c] and not visited[r, c]:
                visited[r, c] = True
                stack.append((r, c))
    return cluster


def step(forest, p=0.01, f=0.2, rng=None):
    """Return (new_forest, fire_size, lightning_location, burned_sites).

    fire_size=0 includes no strike and strikes on empty sites.
    """
    rng = np.random.default_rng() if rng is None else rng
    forest = grow(forest, p, rng)
    site = None
    burned = []
    if rng.random() < f:
        site = (int(rng.integers(forest.shape[0])),
                int(rng.integers(forest.shape[1])))
        burned = cluster_sites(forest, site)
        for row, col in burned:
            forest[row, col] = False
    return forest, len(burned), site, burned


def simulate(n=64, p=0.01, f=0.2, steps=5000, burn_in=1000, rng=None,
             compare_random=False):
    """Record positive fire sizes after an equilibration period.

    When compare_random=True, sample a *tree-conditioned* ignition in a
    random configuration containing precisely as many trees as the evolving
    forest immediately before the corresponding fire. This is the matched
    occupied-site comparison in textbook Exercise 3.5.
    """
    rng = np.random.default_rng() if rng is None else rng
    forest = np.zeros((n, n), dtype=bool)
    sizes = [] 
    random_sizes = [] 
    densities = []
    for t in range(burn_in + steps):
        forest = grow(forest, p, rng)
        density = forest.mean()
        if rng.random() < f:
            site = (int(rng.integers(n)), int(rng.integers(n)))
            if forest[site]:
                before_count = int(forest.sum())
                burned = cluster_sites(forest, site)
                for row, col in burned:
                    forest[row, col] = False
                if t >= burn_in:
                    sizes.append(len(burned))
                    if compare_random:
                        random_forest = np.zeros(n * n, dtype=bool)
                        positions = rng.choice(
                            n * n, 
                            size=before_count, 
                            replace=False
                        )
                        random_forest[positions] = True
                        random_forest = random_forest.reshape((n, n))
                        #
                        random_index = rng.choice(positions)
                        random_site = np.unravel_index(
                            random_index, (n, n)
                        )                      
                        random_sizes.append(
                            len(cluster_sites(random_forest, random_site))
                        )
        if t >= burn_in:
            densities.append(density)
        
    simulation = dict(
        fire_sizes=np.array(sizes, dtype=int),
        random_fire_sizes=np.array(random_sizes, dtype=int),
        densities=np.array(densities)
    )
    return simulation


def empirical_ccdf(sizes):
    """Return unique observed event sizes and P(S >= size), including ties."""
    sizes = np.asarray(sizes, dtype=int)
    if sizes.size == 0:
        return np.array([]), np.array([])
                            
    values, counts = np.unique(sizes, return_counts=True)    
    ccdf = np.cumsum(counts[::-1])[::-1] / sizes.size

    return values, ccdf


def exponent_from_ccdf(sizes, lower, upper):
    """Illustrative log-log linear fit; not an unbiased power-law estimator."""
    x, y = empirical_ccdf(sizes)
    selected = (x >= lower) & (x <= upper)
    if selected.sum() < 3:
        raise ValueError('The fitting interval must contain at least 3 distinct fire sizes')
    slope, intercept = np.polyfit(np.log(x[selected]), np.log(y[selected]), 1)
    alpha = 1 - slope
    return alpha, slope, intercept


def synthetic_pareto(alpha, xmin, count, rng=None):
    """power-law sampling"""
    if alpha <= 1 or xmin <= 0:
        raise ValueError('alpha must exceed 1, and xmin must be positive')
    rng = np.random.default_rng() if rng is None else rng
    r_synthetic = xmin * rng.random(count) ** (-1 / (alpha - 1))
    return r_synthetic
