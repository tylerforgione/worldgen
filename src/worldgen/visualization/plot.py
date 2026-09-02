import matplotlib.pyplot as plt
import numpy as np


def plot_heightmap_3d(heightmap: np.ndarray, vertical_scale: float = 100.0) -> None:
    x = np.arange(heightmap.shape[1])
    y = np.arange(heightmap.shape[0])

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, heightmap * vertical_scale, cmap="terrain")

    plt.show()
