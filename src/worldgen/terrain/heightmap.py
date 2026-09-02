import numpy as np

from .noise import generate_fractal_noise


def generate_heightmap(
    width: int,
    height: int,
    seed: int,
    wavelength: float,
    octaves: int,
    redistribution: float = 1.0,
) -> np.ndarray:
    noise = generate_fractal_noise(
        width=width,
        height=height,
        seed=seed,
        wavelength=wavelength,
        octaves=octaves,
        persistence=0.25,
        lacunarity=1.5,
    )

    elevation = noise**redistribution

    return elevation
