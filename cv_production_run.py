# %%

#!/usr/bin/env python3
"""
Run Cyclic Voltammetry (CV) on EmStat4X (macOS) using PalmSens MethodSCRIPT examples.

This script:
- Uses clean CV parameters (mV, mV/s)
- Generates a minimal (robust) MethodSCRIPT (no '#' comments, no extra blank lines)
- Runs N scans by repeating the single-scan script from Python (most reliable)
- Parses packed results into SI units (V, A) regardless of whether parser returns strings-with-units or floats
- Writes a PSTrace-like CSV (UTF-16LE + BOM) with grouped columns per scan
- Plots all scans overlayed

Prereqs (in your conda env):
  pip install pyserial numpy matplotlib

Run:
  python cv_test.py
"""

import datetime
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr

import palmsens.instrument
import palmsens.mscript
import palmsens.serialport

LOG = logging.getLogger(__name__)
# %%
# =========================
# User configuration
# =========================
DEVICE_PORT = "/dev/cu.usbmodem0147324338351"   # ✅ your EmStat4X USB port
BAUD_RATE =  921600                               # ✅ EmStat4 / EmStat4X
OUTPUT_PATH = "output"
FIGURES_PATH = "figures"

NAME = " 4TPA try 4"  # for labeling files and plots; will be stripped


@dataclass
class CVParams:
    # CV parameters
    E_begin_mV: float = -500
    Vertex1_mV: float = -500
    Vertex2_mV: float = 850
    Step_mV: float = 10
    ScanRate_mV_s: float = 100 # 50

    # Repeat N scans (Python repeats the one-scan MethodSCRIPT)
    n_scans: int = 3

    # Current range / autoranging (as in PalmSens example)
    range_max: str = "100u"
    autorange_min: str = "1n"
    autorange_max: str = "100u"

    # Short CA loop before CV to let autoranging settle
    pre_autorange_step_mV: float = 100
    pre_autorange_n: int = 50 # 50



# %%
# =========================
# Helpers
# =========================
def set_thesis_mpl_style(dpi: int = 150) -> None:
    mpl.rcParams.update({
        "savefig.dpi": dpi,
        "figure.dpi": dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # If LaTeX is not installed on the machine running the notebook,
        # set this to False.
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{siunitx}",

        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "legend.frameon": False,

        "figure.constrained_layout.use": True,
    })

set_thesis_mpl_style(dpi=150)

def generate_colors_from_theme(number_of_colors, theme=None, colors_to_pick=None):
    """Return a list of colors from a matplotlib colormap."""
    if theme is None:
        theme = 'viridis'
    if colors_to_pick is None:
        colors_to_pick = list(range(number_of_colors))

    # cmap = mpl.cm.get_cmap(theme, number_of_colors if number_of_colors > 1 else 2)
    cmap = mpl.colormaps.get_cmap(theme).resampled(number_of_colors if number_of_colors > 1 else 2)
    denom = max(number_of_colors - 1, 1)
    return [cmap(i / denom) for i in colors_to_pick]

def _to_si(x) -> float:
    """
    Accept either:
      - strings like "10007779 nV" or "2066770 aA"
      - numeric floats / numpy scalars (already SI)
    Return float in SI (V or A).
    """
    # Already numeric?
    if isinstance(x, (int, float)):
        return float(x)

    # numpy scalar?
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return float(x)
    except Exception:
        pass

    # Otherwise assume "value unit"
    s = str(x).strip()
    parts = s.split()
    if len(parts) != 2:
        raise ValueError(f"Cannot parse value '{x}'")

    val_str, unit = parts
    val = float(val_str)

    if unit.endswith(("V", "A")):
        pref = unit[:-1]
        scale = {
            "a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9,
            "u": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3,
        }.get(pref)
        if scale is None:
            raise ValueError(f"Unknown prefix in '{unit}' from token '{x}'")
        return val * scale

    raise ValueError(f"Unknown unit token '{unit}' in '{x}'")

def split_into_scans_by_restart(
    E: List[float],
    I: List[float],
    beginV: float,
    stepV: float,
    n_scans: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Split a concatenated multi-scan CV into N scans by detecting when E resets to E_begin.
    Works well because each meas_loop_cv starts at E_begin again.
    """
    if n_scans <= 1:
        return [E], [I]

    E = np.asarray(E, float)
    I = np.asarray(I, float)

    # tolerance: quarter of the step, but not too tiny
    tol = max(abs(stepV) * 0.25, 1e-6)

    # indices where E is (approximately) E_begin
    is_begin = np.abs(E - beginV) < tol

    # scan boundaries: a "begin" that appears after we've moved away from begin
    # and after some minimum points to avoid the very first begin
    boundaries = [0]
    min_gap = 20  # prevent false triggers very close together

    last_boundary = 0
    for idx in range(1, len(E)):
        if idx - last_boundary < min_gap:
            continue
        # transition into begin region
        if is_begin[idx] and not is_begin[idx - 1]:
            boundaries.append(idx)
            last_boundary = idx
            if len(boundaries) == n_scans:
                break

    # If we found fewer boundaries than expected, fall back to equal splitting
    if len(boundaries) < n_scans:
        # crude but safe fallback: split into n_scans chunks
        cuts = np.linspace(0, len(E), n_scans + 1, dtype=int).tolist()
        scans_E = [E[cuts[i]:cuts[i+1]].tolist() for i in range(n_scans)]
        scans_I = [I[cuts[i]:cuts[i+1]].tolist() for i in range(n_scans)]
        return scans_E, scans_I

    # add end
    boundaries.append(len(E))

    scans_E = []
    scans_I = []
    for i in range(n_scans):
        s0, s1 = boundaries[i], boundaries[i + 1]
        scans_E.append(E[s0:s1].tolist())
        scans_I.append(I[s0:s1].tolist())

    return scans_E, scans_I

# def build_single_scan_cv_mscript_minimal(p: CVParams) -> str:
#     """
#     Minimal, robust MethodSCRIPT:
#     - NO comment lines starting with '#'
#     - NO internal blank lines (only final newline)
#     This avoids '#!0003' when comment/blank lines are treated as online commands.
#     """
#     lines = [
#         "e",
#         "var c",
#         "var p",
#         "set_pgstat_mode 2",
#         "set_max_bandwidth 40",
#         f"set_range ba {p.range_max}",
#         f"set_autoranging ba {p.autorange_min} {p.autorange_max}",
#         "set_e 0m",
#         "cell_on",
#         f"meas_loop_ca p c 0m {int(p.pre_autorange_step_mV)}m {int(p.pre_autorange_n)}",
#         "endloop",
#         f"meas_loop_cv p c {p.E_begin_mV:.0f}m {p.Vertex1_mV:.0f}m {p.Vertex2_mV:.0f}m {p.Step_mV:.0f}m {p.ScanRate_mV_s:.0f}m",
#         "\tpck_start",
#         "\tpck_add p",
#         "\tpck_add c",
#         "\tpck_end",
#         "endloop",
#         "on_finished:",
#         "cell_off",
#         "",  # ensures a trailing newline
#     ]
#     return "\n".join(lines) + "\n"


def build_multi_scan_cv_mscript_minimal(p: CVParams) -> str:
    """
    One MethodSCRIPT run with N consecutive CV scans.
    Minimal = no '#' comments, no internal blank lines.
    """
    lines = [
        "e",
        "var c",
        "var p",
        "set_pgstat_mode 2",
        "set_max_bandwidth 40",
        f"set_range ba {p.range_max}",
        f"set_autoranging ba {p.autorange_min} {p.autorange_max}",
        "set_e 0m",
        "cell_on",
        f"meas_loop_ca p c 0m {int(p.pre_autorange_step_mV)}m {int(p.pre_autorange_n)}",
        "endloop",
    ]

    # N back-to-back CV scans
    for _ in range(int(p.n_scans)):
        lines += [
            f"meas_loop_cv p c {p.E_begin_mV:.0f}m {p.Vertex1_mV:.0f}m {p.Vertex2_mV:.0f}m {p.Step_mV:.0f}m {p.ScanRate_mV_s:.0f}m",
            "\tpck_start",
            "\tpck_add p",
            "\tpck_add c",
            "\tpck_end",
            "endloop",
        ]

    lines += [
        "on_finished:",
        "cell_off",
        "",  # trailing newline
    ]
    return "\n".join(lines) + "\n"

# def parse_cv_result_lines(result_lines: List[str]) -> Tuple[List[float], List[float]]:
#     """
#     Parse packed CV result into SI arrays (E in V, I in A).
#     Fail fast on instrument errors like '#!0003' or 'v!0003'.
#     """
#     for line in result_lines:
#         if "!" in line:
#             raise RuntimeError(f"Instrument reported error: {line.strip()}")

#     curves = palmsens.mscript.parse_result_lines(result_lines)
#     E_vals = palmsens.mscript.get_values_by_column(curves, 0)
#     I_vals = palmsens.mscript.get_values_by_column(curves, 1)

#     E = [_to_si(v) for v in E_vals]
#     I = [_to_si(i) for i in I_vals]

#     if not E or not I:
#         raise RuntimeError("No datapoints parsed (no packed data received).")

#     return E, I


def parse_cv_scans_from_result_lines(result_lines: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Return scans_E, scans_I where each scan corresponds to one packed curve returned by the parser.
    This is robust for multi-scan scripts because each meas_loop_cv + pck_end becomes one curve.
    """
    # Fail fast on instrument errors
    for line in result_lines:
        if "!" in line:
            raise RuntimeError(f"Instrument reported error: {line.strip()}")

    curves = palmsens.mscript.parse_result_lines(result_lines)
    if not curves:
        raise RuntimeError("No curves parsed (no packed data received).")

    scans_E: List[List[float]] = []
    scans_I: List[List[float]] = []

    for curve in curves:
        E_scan = []
        I_scan = []
        for package in curve:
            # package is usually [E, I] (either strings-with-units or numeric)
            if len(package) < 2:
                continue
            E_scan.append(_to_si(package[0]))
            I_scan.append(_to_si(package[1]))

        if E_scan and I_scan:
            scans_E.append(E_scan)
            scans_I.append(I_scan)

    if not scans_E:
        raise RuntimeError("Parsed curves contained no datapoints.")

    return scans_E, scans_I

def write_pstrace_like_csv(
    out_path: str,
    scans_E: List[List[float]],
    scans_I: List[List[float]],
    dt_file: datetime.datetime,
    dt_measure: datetime.datetime,
) -> None:
    """
    PSTrace-ish CSV:
    - UTF-16LE with BOM
    - Date/time header lines
    - Column groups per scan: [E1, I1, E2, I2, ...]
    - Units row
    """
    n = len(scans_E)
    if n == 0:
        raise ValueError("No scans provided for CSV export.")

    maxlen = max(len(x) for x in scans_E)

    header = []
    for i in range(n):
        header += [f"Cyclic Voltammetry: CV i vs E Scan {i+1}", ""]

    header2 = []
    for _ in range(n):
        header2 += ["Date and time measurement:", dt_measure.strftime("%Y-%m-%d %H:%M:%S")]

    units = []
    for _ in range(n):
        units += ["V", "A"]

    rows = []
    for k in range(maxlen):
        r = []
        for i in range(n):
            if k < len(scans_E[i]):
                r += [f"{scans_E[i][k]:.6f}", f"{scans_I[i][k]:.12g}"]
            else:
                r += ["", ""]
        rows.append(",".join(r))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    content_lines = [
        f"Date and time:,{dt_file.strftime('%Y-%m-%d %H:%M:%S')}",
        "Notes:",
        "," * (2 * n - 1),
        ",".join(header),
        ",".join(header2) + ",",
        ",".join(units),
        *rows,
    ]

    with open(out_path, "wb") as f:
        f.write(b"\xff\xfe")  # UTF-16LE BOM
        f.write(("\r\n".join(content_lines) + "\r\n").encode("utf-16le"))


def main():
    logging.basicConfig(level=logging.DEBUG, format='[%(module)s] %(message)s', stream=sys.stdout)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ts = datetime.datetime.now()

    params = CVParams(
    )

    # Generate the one-scan MethodSCRIPT
    mscript_text = build_multi_scan_cv_mscript_minimal(params)
    mscript_path = os.path.join(OUTPUT_PATH, "cv_generated_multi_scan.mscr")
    with open(mscript_path, "wt", encoding="ascii", newline="\n") as f:
        f.write(mscript_text)

    # scans_E: List[List[float]] = []
    # scans_I: List[List[float]] = []

    # # Connect once, run N scans
    # with palmsens.serialport.Serial(DEVICE_PORT, BAUD_RATE, 1) as comm:
    #     device = palmsens.instrument.Instrument(comm)
    #     LOG.info("Connected to %s.", device.get_device_type())

    #     for scan_idx in range(params.n_scans):
    #         LOG.info("Starting scan %d/%d ...", scan_idx + 1, params.n_scans)

    #         device.send_script(mscript_path)
    #         result_lines = device.readlines_until_end()

    #         # Save raw per-scan output (debug + provenance)
    #         raw_path = os.path.join(
    #             OUTPUT_PATH, ts.strftime(f"cv_raw_%Y%m%d-%H%M%S_scan{scan_idx+1}.txt")
    #         )
    #         with open(raw_path, "wt", encoding="ascii") as f:
    #             f.writelines(result_lines)

    #         E, I = parse_cv_result_lines(result_lines)
    #         scans_E.append(E)
    #         scans_I.append(I)

    #         LOG.info("Scan %d: %d points", scan_idx + 1, len(E))

    # Generate the multi-scan MethodSCRIPT
    mscript_text = build_multi_scan_cv_mscript_minimal(params)
    mscript_path = os.path.join(OUTPUT_PATH, "cv_generated_multi_scan.mscr")
    with open(mscript_path, "wt", encoding="ascii", newline="\n") as f:
        f.write(mscript_text)

    # Connect once, run ONCE (multi-scan)
    with palmsens.serialport.Serial(DEVICE_PORT, BAUD_RATE, 1) as comm:
        device = palmsens.instrument.Instrument(comm)
        LOG.info("Connected to %s.", device.get_device_type())

        LOG.info("Starting multi-scan CV (%d scans) ...", params.n_scans)
        device.send_script(mscript_path)
        result_lines = device.readlines_until_end()

    # Save raw output
    raw_path = os.path.join(OUTPUT_PATH, ts.strftime("cv_raw_%Y%m%d-%H%M%S.txt"))
    with open(raw_path, "wt", encoding="ascii") as f:
        f.writelines(result_lines)

    # Parse to one long E/I trace
    # E_long, I_long = parse_cv_result_lines(result_lines)

    # # Split into scans
    # beginV = params.E_begin_mV * 1e-3
    # stepV = params.Step_mV * 1e-3
    # scans_E, scans_I = split_into_scans_by_restart(E_long, I_long, beginV, stepV, params.n_scans)

    scans_E, scans_I = parse_cv_scans_from_result_lines(result_lines)

    LOG.info("Parsed %d scan(s) from result.", len(scans_E))

    # Write PSTrace-like CSV with grouped columns per scan
    csv_path = os.path.join(OUTPUT_PATH, ts.strftime(f"cv_%Y%m%d-%H%M%S_{NAME.strip()}.csv"))
    write_pstrace_like_csv(csv_path, scans_E, scans_I, dt_file=ts, dt_measure=ts)
    LOG.info("Wrote %s", csv_path)

    # Plot: overlay scans
    plt.figure(figsize=(3.5, 3.0))
    colors = generate_colors_from_theme(len(scans_E), theme="cmr.emerald")
    for i in range(len(scans_E)):
        c = colors[i]
        plt.plot(np.array(scans_E[i]), 1e6 * np.array(scans_I[i]), label=f"Scan {i+1}", color=c, linewidth=1.5)
    plt.xlabel("E (V)")
    plt.ylabel(r"I ($\mu$A)")
    plt.title(f"Cyclic Voltammetry of {NAME}")
    plt.grid(True)
    plt.minorticks_on()
    
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, ts.strftime(f"cv_%Y%m%d-%H%M%S_{NAME.strip()}.pdf")), dpi=400)
    plt.show()
    
    # plt.savefig(os.path.join(FIGURES_PATH, ts.strftime(f"cv_%Y%m%d-%H%M%S_{NAME.strip()}.pdf")), dpi=400)


if __name__ == "__main__":
    main()
# %%
