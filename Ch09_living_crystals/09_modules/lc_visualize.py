"""Embedded pygame animation for Chapter 9.

Draw into a pygame.Surface (never open a native desktop window); encode each
frame into an ipywidgets.Image in the notebook, following Chapter 1 md_visualize.
Requires pygame, ipywidgets, Pillow, and IPython.
"""

import io

import numpy as np
import pygame
import ipywidgets as widgets
from IPython.display import display
from PIL import Image


def animate_abp(steps=120, skip=4, dt=0.01, speed=3.0,
                diffusion_t=0.2, diffusion_r=0.5, window=360):
    """Show self-propulsion, orientation and a growing unwrapped trail."""
    from active_brownian_motion import abp_step

    particle = np.zeros(2)
    angle = np.random.uniform(-np.pi, np.pi)
    rng = np.random.default_rng()
    trail = [particle.copy()]
    surface = pygame.Surface((window, window))
    panel = widgets.Image(format='png', width=window, height=window)
    label = widgets.Label(value='Active Brownian particle')
    display(widgets.VBox([label, panel]))
    scale = 10.0

    for frame in range((steps + skip - 1) // skip):
        for _ in range(min(skip, steps - frame * skip)):
            particle, angle = abp_step(particle, angle, dt, speed,
                                       diffusion_t, diffusion_r, rng)
            trail.append(particle.copy())
        surface.fill((250, 251, 253))
        points = [((window / 2 + p[0] * scale), (window / 2 - p[1] * scale)) for p in trail]
        if len(points) > 1:
            pygame.draw.lines(surface, (116, 154, 181), False, points, 2)
        center = tuple(round(x) for x in points[-1])
        tip = (round(center[0] + 22 * np.cos(angle)),
               round(center[1] - 22 * np.sin(angle)))
        pygame.draw.circle(surface, (24, 101, 174), center, 7)
        pygame.draw.line(surface, (210, 83, 61), center, tip, 3)
        label.value = f'Time: {min((frame + 1) * skip, steps) * dt:.2f} s'
        pixels = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format='PNG')
        panel.value = output.getvalue()
    return np.asarray(trail)




from matplotlib.patches import Circle

def plot_particles(positions, phi, R, v, L, 
                   XFig=3, YFig=3, 
                   show_velocity=False,
                   fig=None, ax=None
):
    """Plot particles with their radii and velocity vectors."""

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(XFig, YFig))
    elif fig is None or ax is None:
        raise ValueError("Provide both fig and ax, or neither.")
    
    ax.plot(positions[:, 0], positions[:, 1], "k.")

    for x, y in positions:
        ax.add_patch(
            Circle(
                (x, y),
                radius=R,
                facecolor="C2",
                edgecolor="C2",
                alpha=0.3
            )
        )
    if show_velocity:
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            v * np.cos(phi),
            v * np.sin(phi),
            color="black",
            angles="xy",
            scale_units="xy",
            scale=1
        )

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Particle positions and velocities")

    return fig, ax

def animate_particles(positions, angles, stepper, frames=40, steps_per_frame=4,
                      box_length=100.0, radius=1.0, window=400, types=None):
    """Animate a supplied chapter-model stepper without implementing physics here.

    stepper(position, angle) returns updated position and angle arrays.
    """
    surface = pygame.Surface((window, window))
    panel = widgets.Image(format='png', width=window, height=window)
    label = widgets.Label(value='Particle configuration')
    display(widgets.VBox([label, panel]))
    scale = window / box_length
    pixel_radius = max(2, round(radius * scale))
    for frame in range(frames):
        for _ in range(steps_per_frame):
            positions, angles = stepper(positions, angles)
        surface.fill((250, 251, 253))
        for i, (x, y) in enumerate(positions):
            color = ((238, 132, 62) if types is not None and types[i]
                     else (38, 119, 182))
            center = (round(x * scale), round(y * scale))
            pygame.draw.circle(surface, color, center, pixel_radius)
            if types is None or types[i]:
                tip = (round(center[0] + pixel_radius * 1.8 * np.cos(angles[i])),
                       round(center[1] - pixel_radius * 1.8 * np.sin(angles[i])))
                pygame.draw.line(surface, (30, 45, 60), center, tip, 1)
        label.value = f'Frame {frame + 1} / {frames}'
        pixels = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format='PNG')
        panel.value = output.getvalue()
    return positions, angles
