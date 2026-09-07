import numpy as np
import taichi as ti

from .noise import generate_fractal_noise, generate_value_noise, smoothstep


def generate_land_mask(heightmap: np.ndarray, sea_level: float) -> np.ndarray:
    return heightmap > sea_level


@ti.kernel
def generate_continental_mask_kernel(
    width: ti.int32,
    height: ti.int32,
    output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    falloff_power: ti.f32,
):
    for y, x in output:
        x_norm = 2.0 * x / (width - 1) - 1.0
        y_norm = 2.0 * y / (height - 1) - 1.0

        distance = ti.sqrt(x_norm * x_norm + y_norm * y_norm)

        # continental_mask = ti.math.clamp(1.0 - distance, 0.0, 1.0)

        continental_mask = ti.math.clamp(1.0 - distance**falloff_power, 0.0, 1.0)

        output[y, x] = continental_mask


def generate_continental_mask(
    width: int, height: int, falloff_power: float = 4.0
) -> np.ndarray:
    output = np.zeros(shape=(height, width), dtype=np.float32)

    generate_continental_mask_kernel(
        width=width, height=height, output=output, falloff_power=falloff_power
    )

    return output


@ti.kernel
def regional_weight_kernel(
    noise: ti.types.ndarray(dtype=ti.f32, ndim=2),
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    lower_thresh: ti.f32,
    upper_thresh: ti.f32,
):
    denom = upper_thresh - lower_thresh
    for y, x in weights:
        n = noise[y, x]

        if n <= lower_thresh:
            weights[y, x] = 0
        elif n >= upper_thresh:
            weights[y, x] = 1
        else:
            t = (n - lower_thresh) / denom
            w = smoothstep(t)
            weights[y, x] = w


def generate_region_mask(
    width: int,
    height: int,
    seed: int,
    wavelength: float,
    lower_thresh: float = 0.3,
    upper_thresh: float = 0.7,
) -> np.ndarray:
    noise = generate_value_noise(
        width=width, height=height, seed=seed, wavelength=wavelength
    )

    weights = np.zeros(shape=(height, width), dtype=np.float32)

    if lower_thresh < upper_thresh:
        regional_weight_kernel(
            noise=noise,
            weights=weights,
            lower_thresh=lower_thresh,
            upper_thresh=upper_thresh,
        )

    return weights


def generate_heightmap(
    width: int,
    height: int,
    seed: int,
    wavelength: float,
    octaves: int,
    redistribution: float = 1.0,
    persistence: float = 0.25,
    lacunarity: float = 1.5,
) -> np.ndarray:
    noise = generate_fractal_noise(
        width=width,
        height=height,
        seed=seed,
        wavelength=wavelength,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
    )

    elevation = noise**redistribution

    return elevation
