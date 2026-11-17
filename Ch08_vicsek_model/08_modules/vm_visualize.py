"""Inline pygame animation for Chapter 8 """

import io

import numpy as np
import pygame
import ipywidgets as widgets
from IPython.display import display
from PIL import Image

import matplotlib.pyplot as plt

import vicsek_model as vm

def plot_particles(pos, vel, box_length, step=np.nan, S=4, fig=None, ax=None):
    """Plot particle positions and velocity vectors."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(S, S))
    else:
        ax.clear()
        if fig is None:
            fig = ax.figure

    ax.quiver(
        pos[:, 0], pos[:, 1],
        vel[:, 0], vel[:, 1],
        angles="xy",
        scale_units="xy",
        scale=0.18
    )
    ax.plot(pos[:, 0], pos[:, 1], ".")

    ax.set_xlim(0, box_length)
    ax.set_ylim(0, box_length)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if not np.isnan(step):
        ax.set_title(f"Step {step}")

    return fig, ax


def animate_vicsek(positions, angles, box_length, *, model='metric',
                  speed=1.0, flocking_radius=2.0, noise=0.1, dt=1.0,
                  k=4, vision_half_angle=np.pi/3, delay=0, prediction_window=5,
                  frames=75, steps_per_frame=5, image_size=430, trail=False):
    """Run and display a live *inline* pygame Vicsek animation.

    The particle model is advanced on the CPU, and pygame draws the frame.
    ``model`` is one of metric, delayed, predictive, topological, or vision.
    This function blocks until its finite number of frames has been drawn;
    rerun the notebook cell with other arguments to explore parameters.

    Returns the final positions and angles. Input arrays are copied.
    """
    positions = np.asarray(positions, dtype=float).copy()
    angles = np.asarray(angles, dtype=float).copy()
    rng = np.random.default_rng()
    history_length = max(delay + 1, prediction_window)
    history = [angles.copy() for _ in range(history_length)]

    picture = widgets.Image(format='png', width=image_size, height=image_size)
    label = widgets.HTML(value='<b>Vicsek simulation</b> — step 0')
    display(widgets.VBox([label, picture]))

    # No display.set_mode(), pygame.display.flip(), or external popup window.
    surface = pygame.Surface((image_size, image_size))
    scale = image_size / box_length

    for frame in range(frames):
        for _ in range(steps_per_frame):
            if model == 'metric':
                positions, angles = vm.step_metric(
                    positions, angles, box_length, speed, flocking_radius,
                    noise, dt=dt, rng=rng)
            elif model == 'delayed':
                positions, angles = vm.step_delayed(
                    positions, angles, history, delay, box_length, speed,
                    flocking_radius, noise, dt=dt, rng=rng)
            elif model == 'predictive':
                positions, angles = vm.step_predictive(
                    positions, angles, history, abs(delay), box_length,
                    speed, flocking_radius, noise, dt=dt,
                    window=prediction_window, rng=rng)
            elif model == 'topological':
                positions, angles = vm.step_topological(
                    positions, angles, box_length, speed, k, noise,
                    dt=dt, rng=rng)
            elif model == 'vision':
                positions, angles = vm.step_vision(
                    positions, angles, box_length, speed, k,
                    vision_half_angle, noise, dt=dt, rng=rng)
            else:
                raise ValueError('Unknown model: ' + model)
            history.append(angles.copy())
            if len(history) > history_length:
                history.pop(0)

        surface.fill((249, 251, 254))
        for (x, y), theta in zip(positions, angles):
            x_pixel = min(image_size - 1, int(x * scale))
            y_pixel = min(image_size - 1, int(y * scale))
            length = max(5, image_size // 55)
            tip_x = x_pixel + int(length * np.cos(theta))
            # Invert screen y so that positive mathematical angles point upward.
            y_screen = image_size - 1 - y_pixel
            tip_y_screen = y_screen - int(length * np.sin(theta))
            pygame.draw.line(surface, (34, 111, 168),
                             (x_pixel, y_screen), (tip_x, tip_y_screen), 2)
            pygame.draw.circle(surface, (34, 111, 168), (x_pixel, y_screen), 3)

        pixels = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format='PNG')
        picture.value = buffer.getvalue()
        label.value = (f'<b>{model} Vicsek</b> — step '
                       f'{(frame + 1) * steps_per_frame}, '
                       f'alignment = {vm.alignment(angles):.3f}')

    return positions, angles
