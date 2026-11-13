import csv
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

import torch


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
import kc_string
from kc_string_forward import ForwardString


class KirchhoffCarrierStringTest(unittest.TestCase):
    def test_rank_one_update_matches_direct_matrix_solve(self):
        right_hand_side = torch.tensor([0.2, -0.8, 0.3], dtype=torch.float64)
        for direction in ([0.0, 0.0, 0.0], [1.5, -0.3, 0.7]):
            with self.subTest(direction=direction):
                q = torch.tensor(direction, dtype=torch.float64)
                effective_mass = 0.4
                matrix = effective_mass * torch.eye(3, dtype=torch.float64) + torch.outer(q, q)
                expected = torch.linalg.solve(matrix, right_hand_side)
                actual = kc_string.solve_rank_one(right_hand_side, q, 1 / effective_mass)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)

    def test_second_difference_matches_simply_supported_mode(self):
        interval_count = 8
        spacing = 1 / interval_count
        points = torch.arange(1, interval_count, dtype=torch.float64)
        mode = torch.sin(math.pi * points / interval_count)
        eigenvalue = -4 * math.sin(math.pi / (2 * interval_count))**2 / spacing**2
        actual = kc_string.second_spatial_derivative(mode, spacing)
        torch.testing.assert_close(actual, eigenvalue * mode, rtol=1e-12, atol=1e-12)

    def test_output_matches_reference_matlab_script(self):
        octave = shutil.which("octave-cli") or shutil.which("octave")
        self.assertIsNotNone(
            octave,
            "GNU Octave is required for this test. Install the project environment "
            "with `pixi install`, then run `pixi run python test/kc_string.py`.",
        )

        # Keep the upstream spatial/time resolution, but only simulate a short prefix.
        duration_for_test = 0.0001
        oversampling_factor_for_test = 512

        python_output, python_sample_rate = kc_string.simulate_string(
            duration_seconds=duration_for_test,
            oversampling_factor=oversampling_factor_for_test,
        )

        reference_script = (
            project_root
            / "nLinStringsConv-SAV"
            / "KCstring"
            / "KCStringSAV_noSplit.m"
        )
        reference_text = reference_script.read_text()

        replacements = {
            "SAV = true;": "SAV = false;",
            "dampOn = false;": "dampOn = true;",
            "OSfac = 512;": f"OSfac = {oversampling_factor_for_test};",
            "durSec = 0.06;": f"durSec = {duration_for_test};",
        }
        for original, replacement in replacements.items():
            self.assertIn(original, reference_text)
            reference_text = reference_text.replace(original, replacement, 1)

        # Keep the upstream simulation intact, but skip its plots and write SR + Out.
        plot_section = "%%%%% Play & Plot"
        self.assertIn(plot_section, reference_text)
        reference_text = reference_text.split(plot_section, 1)[0]

        with tempfile.TemporaryDirectory() as temp_directory:
            temp_path = Path(temp_directory)
            output_csv = temp_path / "reference_output.csv"
            matlab_path = str(output_csv).replace("'", "''")
            reference_text += (
                "\ndlmwrite(\n"
                f"    '{matlab_path}', [SR, Out(:)'], 'precision', '%.17g'\n"
                ");\n"
            )
            script_path = temp_path / "reference_run.m"
            script_path.write_text(reference_text)

            completed = subprocess.run(
                [octave, "--no-gui", "--quiet", str(script_path)],
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"Octave failed:\n{completed.stdout}\n{completed.stderr}",
            )

            with output_csv.open(newline="") as csv_file:
                values = [float(value) for value in next(csv.reader(csv_file))]

        reference_sample_rate = int(values[0])
        reference_output = torch.tensor(values[1:], dtype=torch.float64)

        self.assertEqual(python_sample_rate, reference_sample_rate)
        torch.testing.assert_close(
            python_output,
            reference_output,
            rtol=1e-9,
            atol=1e-12,
        )
        # Use the original dt rather than its floored, reported sample rate.
        intervals, _, time_step, _ = kc_string.simulation_grid(oversampling_factor_for_test)
        forward = ForwardString(intervals=intervals, sample_rate=1 / time_step)
        native_output = forward(torch.ones(8, dtype=torch.float64),
                                (python_output.numel() - 2) * time_step)
        torch.testing.assert_close(native_output, reference_output, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
