"""Small desktop controls for the existing terrain generator."""

import math
import random
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from worldgen.compute import initialize_compute
from worldgen.terrain.heightmap import (
    generate_continental_mask,
    generate_heightmap,
    generate_land_mask,
    generate_region_mask,
    generate_plains,
    generate_mountains,
)

# name, label, default, type, minimum, maximum
FIELDS = (
    ("width", "Width (samples)", 4096, int, 2, 8192),
    ("height", "Height (samples)", 4096, int, 2, 8192),
    ("seed", "Seed", 1, int, 0, 2**32 - 1),
    ("wavelength", "Wavelength (samples)", 1024, float, 1, 1_000_000),
    ("octaves", "Octaves", 8, int, 1, 32),
    ("persistence", "Persistence", 0.25, float, 0, 1),
    ("lacunarity", "Lacunarity", 1.5, float, 1, 10),
    ("redistribution", "Redistribution power", 3.0, float, 0.01, 100),
    ("falloff_power", "Continental falloff power", 4.0, float, 0.01, 100),
    ("sea_level", "Sea level", 0.1, float, 0, 1),
    ("vertical_scale", "3D vertical scale", 100.0, float, 0.01, 1_000_000),
    ("region_seed", "Region seed", 3982, int, 0, 2**32 - 1),
    ("region_wavelength", "Region wavelength (samples)", 1024, float, 1, 1_000_000),
    ("lower_thresh", "Region lower threshold", 0.3, float, 0, 1),
    ("upper_thresh", "Region upper threshold", 0.7, float, 0, 1),
)


FIELDS += tuple(
    (f"{prefix}_{name}", label, default, kind, minimum, maximum)
    for prefix, variation in (("plains", 0.05), ("mountains", 0.6))
    for name, label, default, kind, minimum, maximum in (
        ("width", "Width (samples)", 4096, int, 2, 8192),
        ("height", "Height (samples)", 4096, int, 2, 8192),
        ("seed", "Seed", 1 if prefix == "plains" else 2, int, 0, 2**32 - 1),
        ("wavelength", "Wavelength (samples)", 1024, float, 1, 1_000_000),
        ("base_elevation", "Base elevation", 0.2, float, 0, 1),
        ("elevation_variation", "Elevation variation", variation, float, 0, 1),
    )
)

SECTIONS = {
    "Main noise": ("width", "height", "seed", "wavelength", "octaves",
                   "persistence", "lacunarity", "redistribution"),
    "Continental mask": ("falloff_power",),
    "Land mask": ("sea_level",),
    "Region mask": ("region_seed", "region_wavelength", "lower_thresh", "upper_thresh"),
    "Plains": tuple(name for name, *_ in FIELDS if name.startswith("plains_")),
    "Mountains": tuple(name for name, *_ in FIELDS if name.startswith("mountains_")),
    "Display": ("vertical_scale",),
}
VIEW_TITLES = ("Terrain", "Before shaping", "Land mask", "Region mask", "Plains", "Mountains")


def random_settings(rng=None):
    """Choose bounded experiment settings with matching square dimensions."""
    rng = rng or random.Random()
    values = {name: default for name, _, default, *_ in FIELDS}
    size = rng.choice((128, 256, 512))
    for prefix in ("", "plains_", "mountains_"):
        values[prefix + "width"] = values[prefix + "height"] = size
        values[prefix + "seed"] = rng.randrange(2**32)
        values[prefix + "wavelength"] = size / rng.choice((2, 4, 8))
    values.update(
        octaves=rng.randint(3, 6), persistence=round(rng.uniform(0.2, 0.65), 3),
        lacunarity=round(rng.uniform(1.3, 2.0), 3),
        redistribution=round(rng.uniform(1, 4), 2),
        falloff_power=round(rng.uniform(2, 8), 2),
        sea_level=round(rng.uniform(0.05, 0.3), 3),
        vertical_scale=rng.choice((25, 50, 100, 150)),
        region_seed=rng.randrange(2**32), region_wavelength=size / rng.choice((2, 4)),
        lower_thresh=round(rng.uniform(0.15, 0.45), 3),
        upper_thresh=round(rng.uniform(0.55, 0.85), 3),
        plains_base_elevation=round(rng.uniform(0.1, 0.3), 3),
        plains_elevation_variation=round(rng.uniform(0.02, 0.1), 3),
        mountains_base_elevation=round(rng.uniform(0.1, 0.3), 3),
        mountains_elevation_variation=round(rng.uniform(0.35, 0.7), 3),
    )
    return parse_settings(values)

def parse_settings(values):
    """Reject invalid values before allocating arrays or running kernels."""
    settings = {}
    for name, label, _, number_type, minimum, maximum in FIELDS:
        try:
            value = number_type(values[name])
        except (ValueError, TypeError, OverflowError):
            kind = "whole number" if number_type is int else "number"
            raise ValueError(f"{label} must be a {kind}.") from None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
        settings[name] = value
    if settings["lower_thresh"] >= settings["upper_thresh"]:
        raise ValueError(
            "Region lower threshold must be less than the upper threshold."
        )
    return settings


class WorldGenerator:
    def __init__(self, root):
        self.root = root
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.compute_ready = False
        self.future = None
        self.poll_id = None
        self.variables = {}
        self.continental = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Adjust settings, then click Generate.")
        root.title("World generator")
        root.geometry("980x650")
        root.minsize(820, 600)
        root.protocol("WM_DELETE_WINDOW", self.close)

        controls = ttk.Frame(root, padding=16)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(
            controls, text="World generation", font=("Segoe UI", 14, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        self.section = tk.StringVar(value="Main noise")
        selector = ttk.Combobox(controls, textvariable=self.section,
                               values=tuple(SECTIONS), state="readonly", width=28)
        selector.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        section_host = ttk.Frame(controls)
        section_host.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.section_frames = {}
        field_specs = {field[0]: field for field in FIELDS}
        for title, names in SECTIONS.items():
            frame = ttk.LabelFrame(section_host, text=title, padding=10)
            frame.grid(row=0, column=0, sticky="nsew")
            self.section_frames[title] = frame
            for row, name in enumerate(names):
                _, label, default, *_ = field_specs[name]
                variable = tk.StringVar(value=str(default))
                self.variables[name] = variable
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
                ttk.Entry(frame, textvariable=variable, width=12).grid(
                    row=row, column=1, padx=(12, 0), pady=5)
            if title in ("Continental mask", "Land mask", "Region mask"):
                ttk.Label(frame, text="Uses main noise width and height.").grid(
                    row=len(names), column=0, columnspan=2, sticky="w", pady=8)
            if title in ("Plains", "Mountains"):
                ttk.Label(frame, text="Independent terrain preview.").grid(
                    row=len(names), column=0, columnspan=2, sticky="w", pady=8)
        ttk.Checkbutton(
            self.section_frames["Continental mask"],
            text="Apply to main terrain", variable=self.continental
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=10)
        selector.bind("<<ComboboxSelected>>", lambda event: self.section_frames[self.section.get()].tkraise())
        self.section_frames["Main noise"].tkraise()
        row = 3
        self.generate_button = ttk.Button(
            controls, text="Generate", command=self.generate
        )
        self.generate_button.grid(row=row + 1, column=0, sticky="ew", pady=6)
        ttk.Button(controls, text="Reset settings", command=self.reset).grid(
            row=row + 1, column=1, sticky="ew", padx=(12, 0)
        )
        self.random_button = ttk.Button(controls, text="Random parameters", command=self.randomize)
        self.random_button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.grid(row=row + 2, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(controls, textvariable=self.status, wraplength=280).grid(
            row=row + 3, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            controls,
            text="Previews sample up to 128 points per axis.\n"
            "Generation uses the full resolution.\nTry 512 x 512 for quick experiments.",
            wraplength=280,
            foreground="#666666",
        ).grid(row=row + 4, column=0, columnspan=2, sticky="w", pady=14)

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 12), pady=12)
        self.views = []
        for title in VIEW_TITLES:
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=title)
            figure = Figure(figsize=(6, 5), layout="constrained")
            canvas = FigureCanvasTkAgg(figure, master=frame)
            toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.views.append((figure, canvas))

        root.update_idletasks()
        minimum_height = max(650, controls.winfo_reqheight() + 24)
        minimum_width = max(820, controls.winfo_reqwidth() + 500)
        root.minsize(minimum_width, minimum_height)
        root.geometry(f"{max(980, minimum_width)}x{minimum_height}")

    def reset(self):
        for name, _, default, _, _, _ in FIELDS:
            self.variables[name].set(str(default))
        self.continental.set(True)

    def randomize(self):
        for name, value in random_settings().items():
            self.variables[name].set(str(value))
        self.continental.set(random.choice((True, False)))
        if self.future is None:
            self.status.set("Random parameters ready. Click Generate to preview.")

    def generate(self):
        if self.future is not None:
            return
        try:
            settings = parse_settings(
                {name: var.get() for name, var in self.variables.items()}
            )
        except ValueError as error:
            messagebox.showerror("Check settings", str(error), parent=self.root)
            return
        settings["continental"] = self.continental.get()
        self.generate_button.state(["disabled"])
        self.status.set("Generating... First run includes kernel compilation.")
        self.progress.start(12)
        self.future = self.worker.submit(self.build_preview, settings)
        self.poll_id = self.root.after(100, self.poll)

    def build_preview(self, settings):
        # Taichi stays on one worker; Tk and Matplotlib stay on the UI thread.
        start = perf_counter()
        if not self.compute_ready:
            initialize_compute()
            self.compute_ready = True
        heightmap = generate_heightmap(
            **{
                name: settings[name]
                for name in (
                    "width",
                    "height",
                    "seed",
                    "wavelength",
                    "octaves",
                    "redistribution",
                    "persistence",
                    "lacunarity",
                )
            }
        )
        xs = np.linspace(
            0, settings["width"] - 1, min(128, settings["width"]), dtype=int
        )
        ys = np.linspace(
            0, settings["height"] - 1, min(128, settings["height"]), dtype=int
        )
        before = heightmap[np.ix_(ys, xs)]
        if settings["continental"]:
            heightmap *= generate_continental_mask(
                settings["width"], settings["height"], settings["falloff_power"]
            )
        land = generate_land_mask(heightmap, settings["sea_level"])
        regions = generate_region_mask(
            width=settings["width"],
            height=settings["height"],
            seed=settings["region_seed"],
            wavelength=settings["region_wavelength"],
            lower_thresh=settings["lower_thresh"],
            upper_thresh=settings["upper_thresh"],
        )
        terrain_previews = []
        for prefix, generator in (("plains", generate_plains), ("mountains", generate_mountains)):
            data = generator(**{
                name: settings[f"{prefix}_{name}"]
                for name in ("width", "height", "seed", "wavelength", "base_elevation", "elevation_variation")
            })
            px = np.linspace(0, data.shape[1] - 1, min(128, data.shape[1]), dtype=int)
            py = np.linspace(0, data.shape[0] - 1, min(128, data.shape[0]), dtype=int)
            terrain_previews.append((px, py, data[np.ix_(py, px)]))
            del data
        return (
            settings,
            xs,
            ys,
            before,
            heightmap[np.ix_(ys, xs)],
            land[np.ix_(ys, xs)],
            float(land.mean()),
            perf_counter() - start,
            regions[np.ix_(ys, xs)],
            terrain_previews,
        )

    def poll(self):
        self.poll_id = None
        future = self.future
        if future is None:
            return
        if not future.done():
            self.poll_id = self.root.after(100, self.poll)
            return
        try:
            # The worker records generation errors on its completed future.
            error = future.exception()
            if error is not None:
                self.status.set("Generation failed. Adjust settings and try again.")
                messagebox.showerror("Generation failed", str(error), parent=self.root)
            else:
                self.show_preview(future.result())
        finally:
            self.future = None
            self.progress.stop()
            self.generate_button.state(["!disabled"])

    def show_preview(self, result):
        settings, xs, ys, before, terrain, land, land_fraction, seconds, regions, extra = (
            result
        )
        x, y = np.meshgrid(xs, ys)
        for (figure, canvas), data, title in zip(
            self.views,
            (terrain, before, land, regions, extra[0][2], extra[1][2]),
            VIEW_TITLES,
        ):
            figure.clear()
            if title in ("Land mask", "Region mask"):
                axes = figure.add_subplot(111)
                preview = axes.imshow(
                    data,
                    origin="lower",
                    cmap="gray" if title == "Region mask" else "Blues_r",
                    vmin=0,
                    vmax=1,
                    extent=(0, settings["width"] - 1, 0, settings["height"] - 1),
                    interpolation="nearest",
                )
                if title == "Region mask":
                    axes.set_title(
                        "Plains (dark) / mountains (light)\nRegion weights — preview only"
                    )
                    figure.colorbar(preview, ax=axes, label="Mountain weight")
                else:
                    axes.set_title("Land (light) / water (dark)")
            else:
                if title in ("Plains", "Mountains"):
                    px, py, _ = extra[0 if title == "Plains" else 1]
                    x, y = np.meshgrid(px, py)
                axes = figure.add_subplot(111, projection="3d")
                axes.plot_surface(
                    x,
                    y,
                    data * settings["vertical_scale"],
                    cmap="terrain",
                    rcount=data.shape[0],
                    ccount=data.shape[1],
                    vmin=0,
                    vmax=settings["vertical_scale"],
                )
                upper = max(1.0, float(data.max()))
                axes.set_zlim(0, upper * settings["vertical_scale"])
                axes.set_zlabel("Scaled elevation")
                axes.set_title(title)
            axes.set_xlabel("X (samples)")
            axes.set_ylabel("Y (samples)")
            canvas.draw_idle()
        self.status.set(
            f"Generated {settings['width']} x {settings['height']} in {seconds:.2f}s.\n"
            f"Seed {settings['seed']} | Land {land_fraction:.1%}"
        )

    def close(self):
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
        self.worker.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    root = tk.Tk()
    WorldGenerator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
