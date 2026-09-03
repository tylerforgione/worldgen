import numpy as np
import taichi as ti


def generate_white_noise(width: int, height: int, seed: int) -> np.ndarray:
    """
    Generate a deterministic 2D field of white noise.

    Each output sample is generated independently from a uniform distribution
    in the range [0, 1). NumPy is used for seeded random-number generation so
    that the same seed produces the same noise field.

    Args:
        width: Width of the output array in samples.
        height: Height of the output array in samples.
        seed: Seed used to initialize the random number generator.

    Returns:
        A NumPy array of shape (height, width) and dtype float32 containing
        white-noise samples in the range [0, 1).
    """
    rng = np.random.default_rng(seed)
    return rng.random(size=(height, width), dtype=np.float32)


@ti.func
def lerp(v0, v1, t):
    """
    Linearly interpolate between two values inside a Taichi kernel.

    Args:
        v0: Starting value.
        v1: Ending value.
        t: Interpolation factor, typically in the range [0, 1].

    Returns:
        The value interpolated between v0 and v1 by factor t.
    """
    return v0 + t * (v1 - v0)


@ti.func
def smoothstep(t):
    """
    Apply cubic Hermite smoothing inside a Taichi kernel.

    Transforms an interpolation factor so that its first derivative is zero
    at both endpoints, producing smoother transitions between lattice cells.

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
    """
    Compute a value-noise field in parallel using Taichi.

    Each output sample is computed independently by locating its surrounding
    lattice points and smoothly interpolating their values. Taichi parallelizes
    the iteration over the 2D output array, allowing the computation to run on
    the configured CPU or GPU backend.

    The output array is modified in place.

    Args:
        grid: 2D float32 array containing random values at lattice points.
        output: 2D float32 array to fill with interpolated noise values.
        wavelength: Distance, in output samples, between adjacent lattice
            points.
    """
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
    Generate deterministic 2D value noise.

    Creates a coarse lattice of seeded random values using NumPy, then delegates
    interpolation of the output samples to a parallel Taichi kernel.

    Each output sample is determined by the four surrounding lattice values.
    Cubic smoothstep interpolation is used to produce continuous, smoothly
    varying noise across lattice-cell boundaries.

    Larger wavelengths produce broader, smoother features, while smaller
    wavelengths produce finer features.

    Args:
        width: Width of the output array in samples.
        height: Height of the output array in samples.
        seed: Seed used to generate the random lattice.
        wavelength: Distance, in output samples, between adjacent lattice
            points.

    Returns:
        A NumPy array of shape (height, width) and dtype float32 containing
        value noise approximately in the range [0, 1].
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
    Generate deterministic 2D fractal value noise from multiple octaves.

    Combines several value-noise fields at progressively smaller wavelengths.
    Each octave adds finer spatial detail while its contribution is scaled by
    an amplitude determined by persistence.

    Individual value-noise fields are generated using the parallel Taichi
    implementation. Octaves are currently generated and combined sequentially
    in Python.

    Each octave uses a seed derived from the base seed so that the complete
    field remains deterministic while each octave uses a different random
    lattice.

    Octaves whose wavelength would fall below one output sample are skipped.
    The final field is normalized by the sum of the amplitudes of all generated
    octaves.

    Args:
        width: Width of the output array in samples.
        height: Height of the output array in samples.
        seed: Base seed used to derive the seed for each octave.
        wavelength: Wavelength of the first and coarsest octave.
        octaves: Maximum number of octaves to generate.
        persistence: Factor controlling amplitude decay between successive
            octaves. Higher values give fine-scale detail more influence.
        lacunarity: Factor controlling how quickly wavelength decreases between
            successive octaves. Higher values introduce smaller-scale features
            more rapidly.

    Returns:
        A NumPy array of shape (height, width) and dtype float32 containing
        normalized fractal value noise approximately in the range [0, 1].
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
