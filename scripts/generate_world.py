from worldgen.compute import initialize_compute
from worldgen.terrain.heightmap import generate_heightmap
from worldgen.visualization.plot import plot_heightmap_3d

initialize_compute()

heightmap = generate_heightmap(
    width=1024,
    height=1024,
    seed=8974,
    wavelength=256,
    octaves=8,
    redistribution=4.0,
)

plot_heightmap_3d(heightmap)
