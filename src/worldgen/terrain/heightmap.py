import numpy as np
import taichi as ti

from .noise import generate_fractal_noise


def generate_land_mask(heightmap: np.ndarray, sea_level: float) -> np.ndarray:
    return heightmap > sea_level


@ti.kernel
def generate_continental_mask_kernel(
    width: ti.int32, height: ti.int32, output: ti.types.ndarray(dtype=ti.f32, ndim=2)
):
    for y, x in output:
        x_norm = 2.0 * x / (width - 1) - 1.0
        y_norm = 2.0 * y / (height - 1) - 1.0

        distance = ti.sqrt(x_norm * x_norm + y_norm * y_norm)

        continental_mask = ti.math.clamp(1.0 - distance, 0.0, 1.0)

        output[y, x] = continental_mask


def generate_continental_mask(width: int, height: int) -> np.ndarray:
    output = np.zeros(shape=(height, width), dtype=np.float32)

    generate_continental_mask_kernel(width=width, height=height, output=output)

    return output


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
