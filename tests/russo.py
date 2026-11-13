"""Run Russo et al.'s ``KCStringSAV_noSplit.m`` in Octave with our parameters and grid.

Only inputs are changed in the script: ``SAV = false``, damping on, the physical
parameters, the grid and step count, the excitation, the pluck and pickup
positions and, for order 4, the five-point second difference. The reference's
initialisation and update are left as they are. The script's plotting section is
dropped and its raw pickup output is written in binary, so no precision is lost.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "nLinStringsConv-SAV/KCstring/KCStringSAV_noSplit.m"
PARAMETERS = ("rho", "T0", "radius", "E", "L", "amplitude")
# Five-point second difference closed by odd reflection; the script's Dxxxx = Dxx.'*Dxx is then its square.
FIVE_POINT = (
    "e = ones(nPoints,1); Dxx = spdiags([-e 16*e -30*e 16*e -e], -2:2, nPoints, nPoints); "
    "Dxx(1,1) = -29; Dxx(end,end) = -29; Dxx = Dxx/(12*h^2);"
)
# Readout at a pickup between nodes: the sine interpolant of the interior nodes at x (fraction of L).
INTERPOLATED_PICKUP = "nn = (1:nPoints)'; wOut = (2/N) * (sin(pi*nn*(1:nPoints)/N)' * sin(pi*nn*{x:.17g}));"


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r} in the reference script.")
    return text.replace(old, new, 1)


def _script(physical, intervals, rate, steps, excitation, order):
    text = SCRIPT.read_text()
    text = _replace_once(text, "SAV = true;", "SAV = false;")
    text = _replace_once(text, "dampOn = false;", "dampOn = true;")
    if excitation == "raised_cosine":
        text = _replace_once(text, "initType = 1;", "initType = 2;")
    if order == 4:
        text = _replace_once(text, "Dxx = Dxp*Dxm;", FIVE_POINT)
    assignments = "\n".join(f"{name} = {value:.17g};" for name, value in zip(PARAMETERS, physical[:6], strict=True))
    text = _replace_once(text, "Area = pi*radius^2;", assignments + "\nArea = pi*radius^2;")
    text = _replace_once(
        text, "sigmaL = sigma0;", f"sigma0 = {physical[6]:.17g}; sigma1 = {physical[7]:.17g};\nsigmaL = sigma0;"
    )
    # Node j sits at x = j/N, so a fraction x of L is node x N (the raised-cosine loop accepts non-integers).
    text = _replace_once(text, "excitPos = outPos;", f"excitPos = {physical[8] * intervals:.17g};")
    pickup_node = physical[9] * intervals
    if abs(pickup_node - round(pickup_node)) < 1e-9:
        text = _replace_once(text, "outPos = floor((nPoints + 1)/2);", f"outPos = {round(pickup_node)};")
    else:
        text = _replace_once(
            text,
            "outPos = floor((nPoints + 1)/2);",
            "outPos = floor((nPoints + 1)/2);\n" + INTERPOLATED_PICKUP.format(x=physical[9]),
        )
        text = _replace_once(text, "Out(1) = uPrev(outPos);", "Out(1) = wOut.'*uPrev;")
        text = _replace_once(text, "Out(2) = u(outPos);", "Out(2) = wOut.'*u;")
        text = _replace_once(text, "Out(n+2)= uNext(outPos);", "Out(n+2)= wOut.'*uNext;")
    # Replace the script's own grid choice with ours.
    before, rest = text.split("% grid spacing\n", 1)
    _, after = rest.split("nPoints = N - 1;", 1)
    grid = f"N = {intervals}; h = L/N; k = 1/{rate}; SR = {rate}; timeSamples = {steps};\n\n"
    text = before + "% grid spacing\n" + grid + "nPoints = N - 1;" + after
    text = text.split("%%%%% Play & Plot", 1)[0]
    return text + "\nfid = fopen('out.bin', 'wb', 'ieee-le'); fwrite(fid, Out(:), 'double'); fclose(fid);\n"


def run(octave, physical, intervals, rate, steps, excitation, order):
    """Raw pickup output (steps + 2,) of the reference script for ten physical values."""
    with tempfile.TemporaryDirectory(prefix="kc-russo-") as folder:
        folder = Path(folder)
        (folder / "run.m").write_text(_script(list(physical), intervals, rate, steps, excitation, order))
        subprocess.run([octave, "--no-gui", "--quiet", "run.m"], cwd=folder, check=True, capture_output=True)
        return torch.from_numpy(np.fromfile(folder / "out.bin", dtype="<f8").copy())
