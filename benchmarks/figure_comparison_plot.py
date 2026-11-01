#!/usr/bin/env python3
"""Render the batch comparison from *measured* case records.

The input is either a JSON array or an object with a ``rows`` array. Each row
contains ``method``, ``batch``, ``status``, ``median_seconds`` and
``peak_gpu_bytes``. Failed and missing cases are reported in the table and are
never plotted. Memory is shown in decimal GB (10^9 bytes).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MarkerPath
import numpy as np


BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
METHODS = ("fdtd", "cached3", "diaz", "cached1")
LABELS = {
    "fdtd": "Autodiff FDTD (direct PyTorch)",
    "cached3": "Cached reconstruction (3 scalars)",
    "diaz": "Díaz modal synthesis (JAX)",
    "cached1": "Cached reconstruction (1 scalar)",
}
PLOTTED = ("fdtd", "cached3", "diaz")
STYLES = {"fdtd": ":", "cached3": "-", "diaz": "--"}
BLUE, RED = "#1263a6", "#b52e35"


def sine_marker():
    """A line-shaped marker that remains a sine wave in both colours."""
    x = np.linspace(-1.0, 1.0, 31)
    vertices = np.column_stack((x, 0.55 * np.sin(np.pi * x)))
    codes = [MarkerPath.MOVETO] + [MarkerPath.LINETO] * (len(x) - 1)
    return MarkerPath(vertices, codes)


def normalize_rows(payload):
    """Check records and index them without converting failures to zeroes."""
    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("status") == "invalid_source_changed":
            raise ValueError("Benchmark sources changed during the run; its results cannot be published")
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Input must be a list or an object containing a rows list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each row must be an object")
        method, batch, status = row.get("method"), row.get("batch"), row.get("status")
        if method not in METHODS or not isinstance(batch, int) or isinstance(batch, bool) or batch not in BATCHES:
            raise ValueError(f"Unsupported method or batch in row: {row!r}")
        if not isinstance(status, str) or not status:
            raise ValueError(f"Missing status in row: {row!r}")
        key = (method, batch)
        if key in result:
            raise ValueError(f"Duplicate measurement for {method}, batch {batch}")
        if status == "ok":
            for field in ("median_seconds", "peak_gpu_bytes"):
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Successful {method}, batch {batch} needs positive finite {field}")
        result[key] = row
    return result


def gpu_label(index, metadata=None):
    """Identify the measured GPU without inferring it from the job or filename.

    Successful rows emitted by the workers contain the actual GPU name. An
    optional metadata name can identify it before any successful row exists.
    If names disagree, avoid attributing the figure to one particular model.
    """
    names = []
    if isinstance(metadata, dict):
        for key in ("gpu_name", "gpu_model", "gpu"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
                break
    for row in index.values():
        value = row.get("gpu_name")
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
    distinct = {name.casefold(): name for name in names}
    if not distinct:
        return "GPU not recorded"
    if len(distinct) > 1:
        return "Multiple GPU models recorded"
    return next(iter(distinct.values()))


def create_figure(index, metadata=None):
    """Create a dual-axis figure; a missing/failed case creates a line gap."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax_time = plt.subplots(figsize=(8.2, 5.3), layout="constrained")
    ax_mem = ax_time.twinx()
    markers = {"fdtd": "o", "cached3": "*", "diaz": sine_marker()}
    sizes = {"fdtd": 6.2, "cached3": 10.0, "diaz": 12.5}
    for method in PLOTTED:
        times, memory = [], []
        for batch in BATCHES:
            row = index.get((method, batch), {})
            if row.get("status") == "ok":
                times.append(row["median_seconds"])
                memory.append(row["peak_gpu_bytes"] / 1e9)
            else:
                times.append(float("nan"))
                memory.append(float("nan"))
        for axis, values, colour in ((ax_time, times, BLUE), (ax_mem, memory, RED)):
            # Differently patterned red strokes keep blue visible if the two
            # axes happen to place curves directly on top of one another.
            pattern = STYLES[method] if axis is ax_time else (0, (4, 3))
            axis.plot(BATCHES, values, color=colour, linestyle=pattern,
                      linewidth=1.7, marker=markers[method], markersize=sizes[method],
                      markerfacecolor="none" if method != "cached3" else colour,
                      markeredgecolor=colour, markeredgewidth=1.5, label=method,
                      zorder=3)

    ax_time.set_xscale("log", base=2)
    ax_time.set_xticks(BATCHES, [str(batch) for batch in BATCHES])
    ax_time.set_xlim(0.82, 157)
    ax_time.set_yscale("log")
    ax_mem.set_yscale("log")
    # A running partial may contain only failures or no completed cases yet.
    # Give both logarithmic axes positive bounds until real points arrive.
    if not any(index.get((method, batch), {}).get("status") == "ok"
               for method in PLOTTED for batch in BATCHES):
        ax_time.set_ylim(0.1, 10)
        ax_mem.set_ylim(0.1, 10)
    ax_time.set_xlabel("Batch size (strings)")
    ax_time.set_ylabel("Computation time (s)", color=BLUE)
    ax_mem.set_ylabel("Sampled peak GPU memory (GB)", color=RED)
    ax_time.tick_params(axis="y", colors=BLUE, which="both")
    ax_mem.tick_params(axis="y", colors=RED, which="both")
    ax_time.spines["left"].set_color(BLUE)
    ax_mem.spines["right"].set_color(RED)
    ax_time.spines["top"].set_visible(False)
    ax_mem.spines["top"].set_visible(False)
    ax_time.grid(which="major", axis="y", color="#dce3e9", linewidth=0.7)
    ax_time.set_axisbelow(True)
    ax_time.set_title("Kirchhoff–Carrier: time and GPU memory", loc="left", fontweight="bold", pad=12)
    duration = (metadata or {}).get("duration") or (metadata or {}).get("duration_seconds") or 4
    oversampling = (metadata or {}).get("oversampling") or 3
    ax_time.set_title(f"{gpu_label(index, metadata)}\n{duration:g} s · 44.1 kHz · {oversampling}× · 10 gradients", loc="right", fontsize=8, color="#52606d", pad=12)
    handles = [Line2D([], [], color="#303944", linestyle=STYLES[m], marker=markers[m],
                      markersize=sizes[m], markerfacecolor="none" if m != "cached3" else "#303944",
                      markeredgewidth=1.4, label=LABELS[m]) for m in PLOTTED]
    ax_time.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.96,
                   fontsize=8.5, handlelength=2.2)
    return fig, (ax_time, ax_mem)


def _display(row):
    if row is None:
        return "—"
    if row["status"] != "ok":
        return row["status"].upper()
    return f'{row["median_seconds"]:.3f} s / {row["peak_gpu_bytes"] / 1e9:.3f} GB'


def write_tables(index, prefix):
    """Emit long-form CSV and a compact table including all requested batches."""
    csv_path, md_path = prefix.with_suffix(".csv"), prefix.with_suffix(".md")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("batch", "method", "status", "median_seconds", "peak_gpu_bytes", "peak_gpu_gb"))
        writer.writeheader()
        for batch in BATCHES:
            for method in METHODS:
                row = index.get((method, batch))
                ok = row is not None and row["status"] == "ok"
                writer.writerow({"batch": batch, "method": method, "status": row["status"] if row else "not_measured",
                                 "median_seconds": row["median_seconds"] if ok else "",
                                 "peak_gpu_bytes": row["peak_gpu_bytes"] if ok else "",
                                 "peak_gpu_gb": row["peak_gpu_bytes"] / 1e9 if ok else ""})
    lines = ["# Batch comparison", "", "Ten gradients (eight physical parameters, pluck and pickup positions). Cells show median seconds / peak GPU GB; — means not measured.", "",
             "| Batch | Autodiff FDTD | Cached: 3 scalars | Díaz JAX modal | Cached: 1 scalar |",
             "|---:|---:|---:|---:|---:|"]
    for batch in BATCHES:
        lines.append("| " + str(batch) + " | " + " | ".join(_display(index.get((m, batch))) for m in METHODS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON list or object containing rows")
    parser.add_argument("--output-prefix", required=True, type=Path, help="Destination path without extension")
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    index = normalize_rows(payload)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_tables(index, args.output_prefix)
    fig, _ = create_figure(index, metadata)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output_prefix.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
