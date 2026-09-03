import numpy as np
import taichi as ti


def generate_white_noise(width: int, height: int, seed: int) -> np.ndarray:
    """
    Generate a 2D array of seeded white noise.

    Each output value is generated independently in the range [0, 1).
    Using the same seed produces the same noise.

    Args:
        width: Width of the output array in samples.
        height: Height of the output array in samples.
        seed: Seed used to initialize the random number generator.

    Returns:
        A 2D NumPy array of shape (height, width) containing white noise.
    """
    rng = np.random.default_rng(seed)
    return rng.random(size=(height, width), dtype=np.float32)


@ti.func
def lerp(v0, v1, t):
    """
    Linearly interpolate between two values.

    Args:
        v0: Starting value.
        v1: Ending value.
        t: Interpolation factor, typically in the range [0, 1].

    Returns:
        The interpolated value between v0 and v1.
    """
    return v0 + t * (v1 - v0)


@ti.func
def smoothstep(t):
    """
    Apply cubic Hermite smoothing to an interpolation factor.

    Smooths the transition from 0 to 1 so that the rate of change is
    zero at both endpoints.

    Args:
        t: Interpolation factor, typically in the range [0, 1].

    Returns:
        The smoothed interpolation factor.
    """
    return 3.0 * t * t - 2.0 * t * t * t


@ti.kernel
def generate_value_noise_kernel(
    grid: ti.types.ndarray(dtype=ti.f32, ndim=2),
    output: ti.types.ndarray(dtype=ti.f32, ndim=2),
    wavelength: ti.f32,
):
    for Y, X in output:
        sample_x = X / wavelength
        sample_y = Y / wavelength

        cell_x = ti.cast(ti.floor(sample_x), ti.int32)
        cell_y = ti.cast(ti.floor(sample_y), ti.int32)

        top_left = grid[cell_y, cell_x]
        top_right = grid[cell_y, cell_x + 1]
        bottom_left = grid[cell_y + 1, cell_x]
        bottom_right = grid[cell_y + 1, cell_x + 1]

        tx = sample_x - cell_x
        ty = sample_y - cell_y

        smooth_tx = smoothstep(t=tx)
        smooth_ty = smoothstep(t=ty)

        top = lerp(
            v0=top_left,
            v1=top_right,
            t=smooth_tx,
        )

        bottom = lerp(
            v0=bottom_left,
            v1=bottom_right,
            t=smooth_tx,
        )

        value = lerp(
            v0=top,
            v1=bottom,
            t=smooth_ty,
        )

        output[Y, X] = value


def generate_value_noise(
    width: int,
    height: int,
    seed: int,
    wavelength: float,
) -> np.ndarray:
    """
    Generate 2D value noise using a seeded random lattice.

    Random values are assigned to points on a coarse lattice. Each output
    sample is calculated by smoothly interpolating between the four lattice
    points surrounding it.

    Larger wavelengths produce broader, smoother features, while smaller
    wavelengths produce finer features.

    Args:
        width: Width of the output array in samples.
        height: Height of the output array in samples.
        seed: Seed used to generate the random lattice values.
        wavelength: Distance, in output samples, between lattice points.

    Returns:
        A 2D NumPy array of shape (height, width) containing value noise
        approximately in the range [0, 1].
    """
    grid_width = int(np.ceil(width / wavelength)) + 1
    grid_height = int(np.ceil(height / wavelength)) + 1

    grid = np.random.default_rng(seed=seed).random(
        size=(grid_height, grid_width),
        dtype=np.float32,
    )

    output = np.zeros(
        shape=(height, width),
        dtype=np.float32,
    )

    generate_value_noise_kernel(grid=grid, output=output, wavelength=wavelength)
    return output


def generate_fractal_noise(
    width: int,
    height: int,
    seed: int,
    wavelength: float,
    octaves: int,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """
    Generate 2D fractal noise by combining multiple octaves of value noise.

    Each successive octave uses a shorter wavelength and a lower amplitude,
    adding progressively finer detail to the result. The combined output is
    normalized by the total amplitude.

    Args:
        width: Width of the output array in samples.
        height: Height of the output array in samples.
        seed: Base seed used to generate deterministic octave noise.
        wavelength: Wavelength of the first and coarsest octave.
        octaves: Maximum number of noise octaves to combine.
        persistence: Factor by which amplitude decreases each octave.
            Lower values reduce the influence of fine-scale detail.
        lacunarity: Factor by which frequency increases each octave.
            Higher values cause successive octaves to introduce
            smaller-scale features more quickly.

    Returns:
        A 2D NumPy array of shape (height, width) containing normalized
        fractal value noise approximately in the range [0, 1].
    """
    output = np.zeros(
        shape=(height, width),
        dtype=np.float32,
    )

    total_amplitude = 0.0

    for i in range(octaves):
        current_wavelength = wavelength / (lacunarity**i)

        if current_wavelength < 1:
            break

        amplitude = persistence**i
        total_amplitude += amplitude

        noise = generate_value_noise(
            width=width,
            height=height,
            seed=seed + i,
            wavelength=current_wavelength,
        )

        output += noise * amplitude

    output /= total_amplitude

    return output
