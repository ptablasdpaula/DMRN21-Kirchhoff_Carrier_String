"""Regression checks for figure generation; synthetic data stay in temp dirs."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt


MODULE = Path(__file__).resolve().parents[1] / "benchmarks" / "figure_comparison_plot.py"
spec = importlib.util.spec_from_file_location("figure_comparison_plot", MODULE)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class ComparisonFigureTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"method": "fdtd", "batch": 1, "status": "ok", "median_seconds": 20.0, "peak_gpu_bytes": 2_000_000_000},
            {"method": "fdtd", "batch": 2, "status": "oom", "median_seconds": None, "peak_gpu_bytes": None},
            {"method": "fdtd", "batch": 4, "status": "ok", "median_seconds": 22.0, "peak_gpu_bytes": 3_000_000_000},
            {"method": "cached3", "batch": 1, "status": "ok", "median_seconds": 1.5, "peak_gpu_bytes": 1_000_000_000},
            {"method": "cached1", "batch": 1, "status": "ok", "median_seconds": 1.6, "peak_gpu_bytes": 900_000_000},
            {"method": "diaz", "batch": 1, "status": "timeout", "median_seconds": None, "peak_gpu_bytes": None},
        ]

    def test_failed_cases_create_gaps_and_cached1_is_not_plotted(self):
        fig, (time_axis, memory_axis) = plot.create_figure(plot.normalize_rows({"rows": self.rows}))
        try:
            self.assertEqual([line.get_label() for line in time_axis.lines], ["fdtd", "cached3", "diaz"])
            self.assertEqual([line.get_label() for line in memory_axis.lines], ["fdtd", "cached3", "diaz"])
            self.assertTrue(math.isnan(time_axis.lines[0].get_ydata()[1]))
            self.assertTrue(math.isnan(memory_axis.lines[0].get_ydata()[1]))
            self.assertEqual(memory_axis.lines[0].get_ydata()[0], 2.0)
            self.assertEqual(time_axis.get_yscale(), "log")
            self.assertEqual(memory_axis.get_yscale(), "log")
        finally:
            plt.close(fig)

    def test_output_and_failure_table(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, prefix = directory / "input.json", directory / "output" / "comparison"
            data.write_text(json.dumps({"rows": self.rows}), encoding="utf-8")
            plot.main([str(data), "--output-prefix", str(prefix)])
            for ext in (".pdf", ".png", ".csv", ".md"):
                self.assertGreater(prefix.with_suffix(ext).stat().st_size, 0)
            markdown = prefix.with_suffix(".md").read_text(encoding="utf-8")
            self.assertIn("OOM", markdown)
            self.assertIn("TIMEOUT", markdown)
            self.assertIn("1.600 s / 0.900 GB", markdown)
            with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 32)
            missing = next(row for row in rows if row["batch"] == "128" and row["method"] == "diaz")
            self.assertEqual(missing["status"], "not_measured")
            self.assertEqual(missing["peak_gpu_bytes"], "")

    def test_rejects_duplicates_and_invalid_successes(self):
        with self.assertRaises(ValueError):
            plot.normalize_rows(self.rows + [self.rows[0]])
        bad = dict(self.rows[0], median_seconds=float("nan"))
        with self.assertRaises(ValueError):
            plot.normalize_rows([bad])

    def test_rejects_changed_sources_but_accepts_running_partials(self):
        with self.assertRaisesRegex(ValueError, "sources changed"):
            plot.normalize_rows({"metadata": {"status": "invalid_source_changed"}, "rows": self.rows})
        result = plot.normalize_rows({"metadata": {"status": "running"}, "rows": self.rows})
        self.assertEqual(len(result), len(self.rows))

    def test_empty_and_failure_only_partials_render(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, rows in (("empty", []), ("failed", [self.rows[1]])):
                source, prefix = directory / f"{name}.json", directory / name
                source.write_text(json.dumps({"metadata": {"status": "running"}, "rows": rows}), encoding="utf-8")
                plot.main([str(source), "--output-prefix", str(prefix)])
                self.assertGreater(prefix.with_suffix(".png").stat().st_size, 0)
                self.assertGreater(prefix.with_suffix(".pdf").stat().st_size, 0)

    def test_gpu_label_uses_actual_row_or_metadata(self):
        for device in ("NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB HBM3"):
            row = dict(self.rows[3], gpu_name=device)
            index = plot.normalize_rows([row])
            fig, (time_axis, _) = plot.create_figure(index)
            try:
                self.assertIn(device, time_axis.get_title(loc="right"))
                self.assertIn("10 gradients", time_axis.get_title(loc="right"))
            finally:
                plt.close(fig)
            self.assertEqual(plot.gpu_label({}, {"gpu_name": device}), device)

    def test_missing_and_conflicting_gpu_labels_are_honest(self):
        self.assertEqual(plot.gpu_label({}), "GPU not recorded")
        index = plot.normalize_rows([
            dict(self.rows[0], gpu_name="NVIDIA A100-PCIE-40GB"),
            dict(self.rows[3], gpu_name="NVIDIA H100 80GB HBM3"),
        ])
        self.assertEqual(plot.gpu_label(index), "Multiple GPU models recorded")


if __name__ == "__main__":
    unittest.main()
