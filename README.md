# Worldgen

Launch the desktop generator from the project root using the existing environment:

```powershell
.\.worldgen\Scripts\python.exe scripts\generate_world.py
```

You can also run `python -m worldgen.gui` in an environment where this project is installed.
The GUI uses Python's Tkinter and the existing NumPy, Taichi, and Matplotlib dependencies.

Edit the settings and click **Generate**. Tabs show shaped terrain, terrain before
continental shaping, and land above sea level. Drag the 3D previews to rotate them.
**Reset settings** restores the original script's defaults. Changes apply on the
next Generate click and last for the current session.

Persistence controls how strongly later octaves contribute; lacunarity controls
how quickly their wavelengths shrink. Redistribution raises noise to the given power.
Continental falloff uses `clamp(1 - distance ** power, 0, 1)`; larger powers broaden
the interior. Disable the continental mask to preview unmasked terrain.
Sea level changes land classification; vertical scale changes the 3D display.

Generation runs in a background worker. The first run includes Taichi initialization
and kernel compilation. Previews sample up to 128 points per axis; generation uses
the full requested resolution. Try 512 x 512 for quick experiments. Dimensions are
limited to 8192 per axis; large grids can still use substantial memory. Closing
during generation lets the current job finish before the Python process exits.
