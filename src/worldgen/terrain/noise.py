import numpy as np
import matplotlib.pyplot as plt

def generate_white_noise(width: int, height: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.random(size=(height, width), dtype=np.float32)

# imprecise lerp
def lerp(v0: float, v1: float, t: float):
    return v0 + t * (v1 - v0)

# cubic Hermite
def smoothstep(t: float):
    return (3 * t**2) - (2 * t**3)

def generate_value_noise(width: int, height: int, seed: int, wavelength: float):
    # find grid height and width
    grid_width = int(np.ceil(width / wavelength)) + 1
    grid_height = int(np.ceil(height / wavelength)) + 1

    # create a coarse grid with random seeded values
    grid = np.random.default_rng(seed=seed).random(size=(grid_height, grid_width), dtype=np.float32)

    # create an output array to store values
    output = np.zeros(shape=(height, width), dtype=np.float32)

    # given the cell (X, Y), we can find the lattic points (corners)
    for Y in range(height):
        for X in range(width):
            sample_x = X / wavelength
            sample_y = Y / wavelength

            cell_x = int(np.floor(sample_x))
            cell_y = int(np.floor(sample_y))

            top_left = grid[cell_y, cell_x]
            top_right = grid[cell_y, cell_x + 1]
            bottom_left = grid[cell_y + 1, cell_x]
            bottom_right = grid[cell_y + 1, cell_x + 1]

            tx = sample_x - cell_x
            ty = sample_y - cell_y
            
            # smooth
            smooth_tx = smoothstep(t=tx)
            smooth_ty = smoothstep(t=ty)

            # interpolate
            top = lerp(v0=top_left, v1=top_right, t=smooth_tx)
            bottom = lerp(v0=bottom_left, v1=bottom_right, t=smooth_tx)

            value = lerp(v0=top, v1=bottom, t=smooth_ty)

            # store interpolated value
            output[Y][X] = value

    return output

def generate_fractal_noise(width: int, height: int, seed: int, wavelength: float, 
                           octaves: int, persistence: float = 0.5, lacunarity: float = 2.0):
    output = np.zeros(shape=(height, width), dtype=np.float32)
    total_amplitude = 0.0
    current_seed = seed

    for i in range(octaves):
        current_wavelength = wavelength / (lacunarity ** i)

        if current_wavelength < 1:
            break
        
        amplitude = persistence ** 1
        total_amplitude += amplitude

        current_seed += 1

        noise = generate_value_noise(width=width, height=height, seed=current_seed, wavelength=current_wavelength)
        output += noise * amplitude

    output /= total_amplitude
    return output

noise = generate_fractal_noise(1024, 1024, 56, 64, 8)
plt.imshow(X=noise, cmap='gray')
plt.show()