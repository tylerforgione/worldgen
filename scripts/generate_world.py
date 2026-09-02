from worldgen.terrain.heightmap import generate_heightmap
from worldgen.visualization.plot import plot_heightmap_3d

heightmap = generate_heightmap(
    width=512,
    height=512,
    seed=47,
    wavelength=32,
    octaves=8,
    redistribution=2.0,
)

plot_heightmap_3d(heightmap)
