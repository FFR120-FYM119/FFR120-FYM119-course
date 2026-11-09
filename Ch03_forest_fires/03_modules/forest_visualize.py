"""Inline pygame rendering."""
import io
import numpy as np
import pygame
from PIL import Image

def png_frame(forest, width=384, burned=(), lightning=None):
    """Render off-screen; return PNG bytes for an ipywidgets.Image.value."""
    
    forest_col = (62, 156, 92)
    burned_col = (255, 128, 128)
    lightning_col = (250, 214, 70)
    empty_col = (255, 255, 255)
    
    rows, cols = forest.shape    
    scale = max(1, width // max(rows, cols))
    surface = pygame.Surface((cols * scale, rows * scale))
    surface.fill(empty_col)
    for row, col in np.argwhere(forest):
        pygame.draw.rect(
            surface, 
            forest_col,
            (int(col * scale), int(row * scale), 
             scale, scale)
        )
    for row, col in burned:
        pygame.draw.rect(
            surface, 
            burned_col,
            (col * scale, row * scale, scale, scale)
        )
    if lightning is not None:
        row, col = lightning
        pygame.draw.rect(
            surface, 
            lightning_col,
            (col * scale, row * scale, scale, scale), 
            max(1, scale // 4)
        )
    pixels = pygame.surfarray.array3d(surface).swapaxes(0, 1)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    return buffer.getvalue()
