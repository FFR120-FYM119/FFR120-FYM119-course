"""pygame display for Ising lattices"""

import os
import sys
import io
import time
import numpy as np


def animate_ising_model(spins, T, n_steps, rng=None,
                        J=1.0, H=0.0, kB=1.0, f=0.1,
                        n_skip=10, window_size=384):
    """Animate the 2D Ising model in a Jupyter notebook."""

    import io
    import pygame
    import ipywidgets as widgets
    from PIL import Image
    from IPython.display import display
    import ising_model as im

    if rng is None:
        rng = np.random.default_rng()

    Ly, Lx = spins.shape
    scale = min(window_size // Lx, window_size // Ly)
    color_bg = (255, 255, 255)
    color_p = (0, 0, 0)

    image_widget = widgets.Image(
        format="png",
        width=Lx * scale,
        height=Ly * scale
    )
    label = widgets.Label(value="Step: 0")
    display(widgets.VBox([label, image_widget]))

    pygame.init()
    surface = pygame.Surface((Lx * scale, Ly * scale))

    for frame in range(n_steps // n_skip):
        for _ in range(n_skip):
            im.heat_bath_sweep(
                spins, T, rng=rng, J=J, H=H, kB=kB, f=f
            )

        surface.fill(color_bg)

        for y in range(Ly):
            for x in range(Lx):
                if spins[y, x] == 1:
                    pygame.draw.rect(
                        surface,
                        color_p,
                        (x * scale, y * scale, scale, scale)
                    )

        pixels = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        image_widget.value = buffer.getvalue()

        label.value = f"Step: {(frame + 1) * n_skip}"

    pygame.quit()

