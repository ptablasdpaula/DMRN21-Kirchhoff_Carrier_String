import csv
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


class KirchhoffCarrierStringTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
