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
    octaves=4,
    redistribution=4.0,
)

land_mask = generate_land_mask(heightmap=heightmap, sea_level=0.3)

continental_mask = generate_continental_mask(
    width=heightmap.shape[1], height=heightmap.shape[0]
)

plot_heightmap_3d(heightmap=continental_mask)
plot_heightmap_3d(heightmap=land_mask)
plot_heightmap_3d(heightmap=heightmap)
