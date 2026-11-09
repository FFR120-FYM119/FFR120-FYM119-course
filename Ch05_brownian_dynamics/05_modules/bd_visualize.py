"""Notebook-embedded pygame visualization of a live 2D Brownian trajectory.

Based on the off-screen pygame.Surface -> PNG -> ipywidgets.Image route used
in FFR120 Chapter 1. No desktop window is created.
"""

from __future__ import annotations

import io
from collections import deque

import ipywidgets as widgets
import numpy as np
import pygame
from IPython.display import display
from PIL import Image

import brownian_dynamics as bd


def animate_optical_trap(gamma: float, temperature: float,
                         stiffness_x: float = 1e-6,
                         stiffness_y: float = 0.25e-6,
                         dt: float = 2e-4, steps_per_frame: int = 10,
                         frames: int = 240, window_size: int = 360,
                         view_radius_um: float = 0.35) -> widgets.VBox:
    """Display a pausable, notebook-embedded animation with a stiffness slider.

    Play/Pause is supplied by the ipywidgets.Play control. Reset starts
    a fresh stochastic trajectory from the origin. The model is advanced
    *only* when the Play widget delivers the next frame value.
    """
    if dt * max(stiffness_x, stiffness_y) / gamma >= 1:
        raise ValueError("Choose dt much smaller than gamma / stiffness")

    image = widgets.Image(format="png", width=window_size,
                          height=window_size)
    play = widgets.Play(value=0, min=0, max=frames, step=1,
                        interval=90, description="Play / pause")
    reset = widgets.Button(description="Reset trajectory", icon="refresh")
    y_slider = widgets.FloatSlider(
        value=stiffness_y * 1e6, min=0.1, max=2.0, step=0.05,
        description="ky (pN/µm)", continuous_update=False,
        readout_format=".2f", style={"description_width": "initial"},
        layout=widgets.Layout(width="320px"),
    )
    status = widgets.Label(value="Frame 0. Press play to start.")
    surface = pygame.Surface((window_size, window_size))
    rng = np.random.default_rng()
    trace = deque(maxlen=160)
    position = np.zeros(2)
    origin = window_size // 2
    pixels_per_um = window_size / (2 * view_radius_um)
    therm_noise = np.sqrt(2 * bd.kB * temperature * dt / gamma)

    def draw_frame() -> None:
        surface.fill((249, 251, 253))
        pygame.draw.line(surface, (225, 229, 235), (origin, 0),
                         (origin, window_size), 1)
        pygame.draw.line(surface, (225, 229, 235), (0, origin),
                         (window_size, origin), 1)
        # Ellipse of one theoretical standard deviation in each direction.
        sigma_x = np.sqrt(bd.kB * temperature / stiffness_x) * 1e6
        sigma_y = np.sqrt(bd.kB * temperature / (y_slider.value * 1e-6)) * 1e6
        rect = pygame.Rect(0, 0, max(2, int(2 * sigma_x * pixels_per_um)),
                           max(2, int(2 * sigma_y * pixels_per_um)))
        rect.center = (origin, origin)
        pygame.draw.ellipse(surface, (150, 192, 218), rect, 2)
        if len(trace) > 1:
            pygame.draw.lines(surface, (67, 143, 185), False, list(trace), 2)
        pygame.draw.circle(surface, (219, 96, 57),
                           (int(origin + position[0] * 1e6 * pixels_per_um),
                            int(origin - position[1] * 1e6 * pixels_per_um)), 6)
        rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, format="PNG")
        image.value = output.getvalue()

    def next_frame(change: dict) -> None:
        if change["new"] <= change["old"]:
            return
        stiffness = np.array([stiffness_x, y_slider.value * 1e-6])
        for _ in range(steps_per_frame):
            position[:] = (position - stiffness * position * dt / gamma
                           + therm_noise * rng.standard_normal(2))
        trace.append((int(origin + position[0] * 1e6 * pixels_per_um),
                      int(origin - position[1] * 1e6 * pixels_per_um)))
        status.value = (f"Frame {change['new']} / {frames} | "
                        f"t = {change['new'] * steps_per_frame * dt:.2f} s")
        draw_frame()

    def reset_animation(_button: widgets.Button) -> None:
        play.value = 0
        position[:] = 0.0
        trace.clear()
        status.value = "Frame 0. Press play to start."
        draw_frame()

    play.observe(next_frame, names="value")
    reset.on_click(reset_animation)
    y_slider.observe(lambda _: draw_frame(), names="value")
    draw_frame()
    controls = widgets.HBox([play, reset])
    box = widgets.VBox([controls, y_slider, status, image])
    display(box)
    return box
