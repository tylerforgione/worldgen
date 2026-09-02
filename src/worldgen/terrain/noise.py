import numpy as np
import matplotlib.pyplot as plt

def generate_white_noise(width, height, seed):
    rng = np.random.default_rng(seed)
    return rng.random(size=(height, width), dtype=np.float32)

# imprecise lerp
def lerp(v0, v1, t):
    return v0 + t * (v1 - v0)

def smoothstep(t):
    return (3 * t**2) - (2 * t**3)

def generate_value_noise(width, height, seed, wavelength):
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
            cell_x = X // wavelength
            cell_y = Y // wavelength

            top_left = grid[cell_y][cell_x]
            top_right = grid[cell_y][cell_x + 1]
            bottom_left = grid[cell_y + 1][cell_x]
            bottom_right = grid[cell_y + 1][cell_x + 1]

            tx = (X % wavelength) / wavelength
            ty = (Y % wavelength) / wavelength
            
            # smooth
            smooth_tx = smoothstep(t=tx)
            smooth_ty = smoothstep(t=ty)

            # interpolate
            top = lerp(v0=top_left, v1=top_right, t=smooth_tx)
            bottom = lerp(v0=bottom_left, v1=bottom_right, t=smooth_tx)

            value = lerp(v0=top, v1=bottom, t=smooth_ty)

            # store interpolated value
            output[Y][X] = value

    plt.imshow(X=output, cmap='gray')
    plt.show()

generate_value_noise(512, 512, 12, 32)