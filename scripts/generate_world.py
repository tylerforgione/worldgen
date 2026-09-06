from worldgen.compute import initialize_compute
from worldgen.terrain.heightmap import (
    generate_continental_mask,
    generate_heightmap,
    generate_land_mask,
)
from worldgen.visualization.plot import plot_heightmap_3d

initialize_compute()

heightmap = generate_heightmap(
    width=2048,
    height=2048,
    seed=3982,
    wavelength=256,
    octaves=6,
    redistribution=3.0,
)

continental_mask = generate_continental_mask(
    width=heightmap.shape[1], height=heightmap.shape[0]
)

plot_heightmap_3d(heightmap=heightmap)

shaped_heightmap = heightmap * continental_mask

plot_heightmap_3d(heightmap=shaped_heightmap)
