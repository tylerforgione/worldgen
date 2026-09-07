"""Small desktop controls for the existing terrain generator."""

import math
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from worldgen.compute import initialize_compute
from worldgen.terrain.heightmap import (
    generate_continental_mask,
    generate_heightmap,
    generate_land_mask,
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
)


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
        for row, (name, label, default, _, _, _) in enumerate(FIELDS, start=1):
            variable = tk.StringVar(value=str(default))
            self.variables[name] = variable
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(controls, textvariable=variable, width=12).grid(
                row=row, column=1, padx=(12, 0), pady=5
            )
        row = len(FIELDS) + 1
        ttk.Checkbutton(
            controls, text="Apply continental mask", variable=self.continental
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=10)
        self.generate_button = ttk.Button(
            controls, text="Generate", command=self.generate
        )
        self.generate_button.grid(row=row + 1, column=0, sticky="ew", pady=6)
        ttk.Button(controls, text="Reset settings", command=self.reset).grid(
            row=row + 1, column=1, sticky="ew", padx=(12, 0)
        )
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
        for title in ("Terrain", "Before shaping", "Land mask"):
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
        return (
            settings,
            xs,
            ys,
            before,
            heightmap[np.ix_(ys, xs)],
            land[np.ix_(ys, xs)],
            float(land.mean()),
            perf_counter() - start,
        )

    def poll(self):
        self.poll_id = None
        if not self.future.done():
            self.poll_id = self.root.after(100, self.poll)
            return
        try:
            self.show_preview(self.future.result())
        except Exception as error:
            self.status.set("Generation failed. Adjust settings and try again.")
            messagebox.showerror("Generation failed", str(error), parent=self.root)
        finally:
            self.future = None
            self.progress.stop()
            self.generate_button.state(["!disabled"])

    def show_preview(self, result):
        settings, xs, ys, before, terrain, land, land_fraction, seconds = result
        x, y = np.meshgrid(xs, ys)
        for (figure, canvas), data, title in zip(
            self.views,
            (terrain, before, land),
            ("Terrain", "Before shaping", "Land mask"),
        ):
            figure.clear()
            if title == "Land mask":
                axes = figure.add_subplot(111)
                axes.imshow(
                    data,
                    origin="lower",
                    cmap="Blues_r",
                    vmin=0,
                    vmax=1,
                    extent=(0, settings["width"] - 1, 0, settings["height"] - 1),
                    interpolation="nearest",
                )
                axes.set_title("Land (light) / water (dark)")
            else:
                axes = figure.add_subplot(111, projection="3d")
                axes.plot_surface(
                    x,
                    y,
                    data * settings["vertical_scale"],
                    cmap="terrain",
                    rcount=len(ys),
                    ccount=len(xs),
                    vmin=0,
                    vmax=settings["vertical_scale"],
                )
                axes.set_zlim(0, settings["vertical_scale"])
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
