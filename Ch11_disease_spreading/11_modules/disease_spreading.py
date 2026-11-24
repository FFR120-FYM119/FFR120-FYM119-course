"""Chapter 11: spatial stochastic SIR, SIRD, and SIRS models.
"""

import numpy as np

SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2
DECEASED = 3


def initial_agents(n_agents=1000, side=100, infected_fraction=0.01, rng=None):
    """Create agents at independent uniformly random lattice sites."""
    if n_agents < 1 or side < 2 or not 0 <= infected_fraction <= 1:
        raise ValueError(
            "Check population size, lattice side, and infection fraction."
        )
    if rng is None:
        rng = np.random.default_rng()  # Do not seed by default.

    positions = rng.integers(side, size=(n_agents, 2))
    status = np.full(n_agents, SUSCEPTIBLE, dtype=np.int8)
    n_initial = max(
        1, int(round(n_agents * infected_fraction))
    ) if infected_fraction > 0 else 0
    if n_initial:
        selected = rng.choice(n_agents, size=n_initial, replace=False)
        status[selected] = INFECTED
    return positions, status


def step_agents(
    positions, status, side, d, 
    beta, gamma, mu=0.0, alpha=0.0, rng=None
):
    """Advance the model by one time step, 
    modifying both input arrays in place.

    mu: per-step death probability for an existing infected individual.
    gamma: recovery probability *conditional on surviving* the death decision;
           P(recover) = (1-mu)*gamma, P(die) = mu.
    alpha: loss-of-immunity probability for an existing recovered individual.

    Each infected individual independently attempts site-wide infection with
    probability beta. 
    A successful attempt infects ALL susceptible co-occupants.
    """
    if rng is None:
        rng = np.random.default_rng()
    for probability in (d, beta, gamma, mu, alpha):
        if not 0 <= probability <= 1:
            raise ValueError("Probabilities must be between 0 and 1.")

    n_agents = len(status)
    existing_infected = status == INFECTED
    existing_recovered = status == RECOVERED
    susceptible = status == SUSCEPTIBLE

    # Move all living agents, choosing uniformly among the four neighbors.
    moving = (status != DECEASED) & (rng.random(n_agents) < d)
    directions = rng.integers(4, size=np.count_nonzero(moving))
    dx = np.array([1, -1, 0, 0])
    dy = np.array([0, 0, 1, -1])
    positions[moving, 0] = (
        positions[moving, 0] + dx[directions]
    ) % side
    positions[moving, 1] = (
        positions[moving, 1] + dy[directions]
    ) % side

    # One success/failure trial per existing infected agent, not per pair.
    site_index = positions[:, 0] * side + positions[:, 1]
    succeeds = existing_infected & (rng.random(n_agents) < beta)
    infection_sites = np.bincount(
        site_index[succeeds], minlength=side * side) > 0
    newly_infected = susceptible & infection_sites[site_index]

    # Competing death/recovery outcomes, never both for the same agent.
    outcomes = rng.random(n_agents)
    dying = existing_infected & (outcomes < mu)
    recovering = existing_infected & (outcomes >= mu) & (
        outcomes < mu + (1 - mu) * gamma
    )
    losing_immunity = (
        existing_recovered & (rng.random(n_agents) < alpha)
    )

    status[dying] = DECEASED
    status[recovering] = RECOVERED
    status[losing_immunity] = SUSCEPTIBLE
    status[newly_infected] = INFECTED
    return int(np.count_nonzero(newly_infected))


def run_sir(
    n_agents=1000, side=100, infected_fraction=0.01, d=0.8,
    beta=0.6, gamma=0.01, mu=0.0, alpha=0.0,
    lockdown_start=None, lockdown_steps=200, lockdown_d=0.1,
    max_steps=1500, rng=None,
):
    """Run one realization until extinction or a documented step cap.

    Returns time series including t=0 and last states; ended_by_extinction
    tells whether the epidemic actually ended before the step cap.
    """
    if rng is None:
        rng = np.random.default_rng()
    positions, status = initial_agents(
        n_agents, side, infected_fraction, rng
    )
    history = [np.bincount(status, minlength=4)]
    steps = 0

    for step in range(max_steps):
        if not np.any(status == INFECTED):
            break
        movement_probability = d
        if (
            lockdown_start is not None 
            and lockdown_start <= step < lockdown_start + lockdown_steps
        ):
            movement_probability = lockdown_d
        step_agents(
            positions, status, side, movement_probability,
            beta, gamma, mu=mu, alpha=alpha, rng=rng
        )
        history.append(np.bincount(status, minlength=4))
        steps = step + 1

    counts = np.asarray(history, dtype=int)
    extinct = not np.any(status == INFECTED)
    return {
        "time": np.arange(len(counts)),
        "S": counts[:, SUSCEPTIBLE],
        "I": counts[:, INFECTED],
        "R": counts[:, RECOVERED],
        "D": counts[:, DECEASED],
        "positions": positions,
        "status": status,
        "ended_by_extinction": extinct,
        "steps": steps,
        "ever_infected": n_agents - int(counts[-1, SUSCEPTIBLE])
        if alpha == 0 else None,
    }
