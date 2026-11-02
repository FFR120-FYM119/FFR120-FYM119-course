"""Chapter 1: transparent CPU molecular-dynamics calculations.

Lennard-Jones calculations use reduced units (m = sigma = epsilon = 1).
"""
import numpy as np
import molecular_dynamics as md
import pygame

# io and ipywidgets needed to avoid flickering on certain browsers
import io
import ipywidgets as widgets

# needed for animation
from IPython.display import display
from PIL import Image

def animate_gas(
    N=100,
    box_length=15.0,
    v0=1.5,
    m0=2.0,
    nsteps=20000,
    dt=0.001,
    nskip=10,
    window_size=360,
    particle_size=0.5,
):
    """Animate a gas with one heavier tracer particle."""

    scale = window_size / box_length
    particle_radius = round(particle_size * scale)

    color_p = (35, 110, 185)
    color_h = (35, 35, 35)
    color_bg = (250, 250, 250)

    image_widget = widgets.Image(
        format="png", width=window_size, height=window_size
    )
    label = widgets.Label(value="Step: 0")
    display(widgets.VBox([label, image_widget]))

    positions, velocities = md.initial_gas(
        N, box_length, speed=v0
    )

    # Find the particle closest to the center
    target = 0.5 * np.array([box_length, box_length])
    i = np.argmin(np.sum((positions - target)**2, axis=1))

    velocities[i, :] = 0

    masses = np.ones(N)
    masses[i] = m0

    tracer = [positions[i].copy()]

    pygame.init()
    surface = pygame.Surface((window_size, window_size))

    for frame in range(nsteps // nskip):
        for step in range(nskip):
            positions, velocities = md.step_particles(
                positions, velocities, dt, box_length, masses
            )
            tracer.append(positions[i].copy())

        surface.fill(color_bg)

        for x, y in positions:
            pygame.draw.circle(
                surface,
                color_p,
                (int(x * scale), int(y * scale)),
                particle_radius
            )

        pygame.draw.circle(
            surface,
            color_h,
            (
                int(positions[i, 0] * scale),
                int(positions[i, 1] * scale)
            ),
            particle_radius
        )

        label.value = f"Step: {(frame + 1) * nskip}"

        pixels = pygame.surfarray.array3d(surface).swapaxes(0, 1)

        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        image_widget.value = buffer.getvalue()

    pygame.quit()

    return np.array(tracer)

