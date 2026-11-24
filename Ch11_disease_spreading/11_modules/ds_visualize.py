"""Embedded pygame frames for Chapter 11; no desktop window is opened.

This follows the tested-in-Chapter-1 *pattern* of converting a pygame Surface
into PNG bytes and assigning them to an ipywidgets.Image in the notebook.
Actual Colab/browser compatibility must still be checked on target systems.
"""

import io

import numpy as np
import pygame
import ipywidgets as widgets
from PIL import Image
from IPython.display import display

import disease_spreading as ds

COLORS = {
    ds.SUSCEPTIBLE: (51, 126, 187),
    ds.INFECTED: (229, 139, 48),
    ds.RECOVERED: (180, 180, 180),
    ds.DECEASED: (95, 95, 95),
}


def animate_sir(
    n_agents=1000, side=100, infected_fraction=0.01,
    d=0.8, beta=0.6, gamma=0.01, max_steps=360,
    steps_per_frame=4, window_size=430,
    lockdown_start=None, lockdown_steps=200, lockdown_d=0.1,
):
    """pygame animation into a notebook Image widget.
    """
    pixel_size = window_size * 4   # for a better "dpi"
    
    rng = np.random.default_rng()
    positions, status = ds.initial_agents(
        n_agents, 
        side, 
        infected_fraction, 
        rng
    )
    image_widget = widgets.Image(
        format="png", 
        width=window_size, 
        height=window_size
    )
    label = widgets.Label(value="Step: 0")
    display(widgets.VBox([label, image_widget]))

    pygame.init()
    surface = pygame.Surface((pixel_size, pixel_size))
    scale = pixel_size / side
    radius = int(np.ceil(pixel_size / side * 0.45))
    history = [np.bincount(status, minlength=4)]
    step = 0

    # Draw at t=0 too, before any update.
    for frame in range(max_steps // steps_per_frame + 1):
        if frame:
            for _ in range(steps_per_frame):
                if step >= max_steps or not np.any(status == ds.INFECTED):
                    break
                active_d = d
                if lockdown_start is not None and (
                    lockdown_start <= step < lockdown_start + lockdown_steps
                ):
                    active_d = lockdown_d
                    
                ds.step_agents(
                    positions, status, side, 
                    active_d, beta, gamma, 
                    rng=rng
                )
                history.append(np.bincount(status, minlength=4))
                step += 1

        surface.fill((251, 251, 252))
        # 
        for i in range(n_agents):
            if status[i] == ds.DECEASED:
                continue
            x, y = positions[i]
            pixel_x = int((x + 0.5) * scale + (i % 3 - 1) * 1.2)
            pixel_y = int((y + 0.5) * scale + ((i // 3) % 3 - 1) * 1.2)
            pygame.draw.circle(surface, COLORS[int(status[i])],
                               (pixel_x, pixel_y), radius)

        pixels = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        image_widget.value = buffer.getvalue()
        counts = history[-1]
        label.value = (
            f"Step {step} | S={counts[0]}, I={counts[1]}, "
            f"R={counts[2]}" + (
                " | lockdown" if (
                    lockdown_start is not None and ( 
                        lockdown_start <= step < lockdown_start + lockdown_steps
                    )
                ) else ""
            )
        )
        if not np.any(status == ds.INFECTED) or step >= max_steps:
            break

    pygame.quit()
    
    return np.asarray(history)
