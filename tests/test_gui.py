"""Run with python -m unittest discover -s tests."""

import time
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np

from worldgen.gui import FIELDS, WorldGenerator, parse_settings
from worldgen.terrain.noise import generate_fractal_noise


class SettingsTests(unittest.TestCase):
    def test_invalid_settings(self):
        defaults = {name: str(default) for name, _, default, *_ in FIELDS}
        for name, value in (
            ("width", "1"), ("height", "3.5"), ("seed", "-1"),
            ("octaves", "0"), ("wavelength", "0.5"),
            ("persistence", "nan"), ("lacunarity", "0"),
            ("redistribution", "inf"), ("sea_level", "2"),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_settings({**defaults, name: value})


class GuiTests(unittest.TestCase):
    def test_generation_controls_and_recovery(self):
        root = tk.Tk()
        root.withdraw()
        app = WorldGenerator(root)
        errors = []
        root.report_callback_exception = lambda *args: errors.append(args)
        try:
            app.poll()  # Polling while idle should be harmless.
            self.assertIsNone(app.poll_id)
            for name, value in {"width": 48, "height": 32, "wavelength": 16,
                                "octaves": 4}.items():
                app.variables[name].set(str(value))
            with patch("worldgen.gui.messagebox.showerror") as show_error:
                app.generate()
                first_future = app.future
                self.assertIn("disabled", app.generate_button.state())
                self.wait_for_preview(root, app)
                result = first_future.result()
                self.assertEqual(result[4].shape, (32, 48))
                self.assertTrue(np.isfinite(result[4]).all())
                self.assertTrue((result[4] <= result[3]).all())
                np.testing.assert_array_equal(result[4][0], 0)
                np.testing.assert_array_equal(result[5], result[4] > 0.1)
                expected = app.worker.submit(
                    generate_fractal_noise, 48, 32, 1, 16, 4, 0.25, 1.5
                ).result() ** 3.0
                np.testing.assert_allclose(result[3], expected)
                self.assertEqual(len(app.views[0][0].axes), 1)
                self.assertFalse(errors)
                show_error.assert_not_called()

                app.continental.set(False)
                app.variables["persistence"].set("0.8")
                app.variables["lacunarity"].set("2.0")
                app.generate()
                second_future = app.future
                self.wait_for_preview(root, app)
                second = second_future.result()
                np.testing.assert_array_equal(second[3], second[4])
                self.assertFalse(np.allclose(result[3], second[3]))
                expected = app.worker.submit(
                    generate_fractal_noise, 48, 32, 1, 16, 4, 0.8, 2.0
                ).result() ** 3.0
                np.testing.assert_allclose(second[3], expected)

                with patch.object(app, "build_preview", side_effect=RuntimeError("test error")):
                    app.generate()
                    self.wait_for_preview(root, app)
                show_error.assert_called_once()
                self.assertNotIn("disabled", app.generate_button.state())
                self.assertFalse(errors)
        finally:
            app.close()

    def wait_for_preview(self, root, app):
        deadline = time.monotonic() + 45
        while app.future is not None and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        self.assertIsNone(app.future, "Generation did not finish within 45 seconds")
        for _, canvas in app.views:
            canvas.draw()


if __name__ == "__main__":
    unittest.main()
