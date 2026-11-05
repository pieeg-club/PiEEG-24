#!/usr/bin/env python3
"""
PiEEG-24 — record all 24 channels for 1 minute and save to Excel.

    python3 record.py                    # electrodes (CHnSET 0x00) -> normal_YYYYMMDD_HHMMSS.xlsx
    python3 record.py --mode short       # inputs shorted (0x01)
    python3 record.py --mode test        # internal test signal (0x05)
    python3 record.py --seconds 30 --out my.xlsx
    python3 record.py --mock             # test without hardware

Values are raw µV straight from the ADS1299 (no filter), so electrode
channels include their DC offset — filter afterwards if needed.

Sheet "data":    columns  ch 1 ... ch 24   (µV, one row per sample, 250 rows/s)
Sheet "summary": mean, rms noise, min, max, peak-to-peak for every channel

Needs pieeg24.py in the same folder and:  pip install openpyxl
"""

import argparse
import sys
import time

import numpy as np

from pieeg24 import PiEEG24, NUM_CHANNELS

try:
    from openpyxl import Workbook
except ImportError:
    sys.exit("openpyxl is missing -> run:  pip install openpyxl   "
             "(or: sudo apt install python3-openpyxl)")

FS = 250


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="normal", choices=["normal", "short", "test"])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    a.out = a.out or time.strftime(f"{a.mode}_%Y%m%d_%H%M%S.xlsx")

    n = int(a.seconds * FS)
    data = np.empty((n, NUM_CHANNELS))

    with PiEEG24(sample_rate=FS, mode=a.mode, mock=a.mock) as dev:   # CHnSET 0x00 / 0x01 / 0x05
        print("Chip IDs:", ", ".join(f"0x{c:02X}" for c in dev.chip_ids), "(expect 0x3E)")
        dev.read(FS // 2)                       # let it settle 0.5 s
        dev.bad_frames = [0, 0, 0]
        print(f"Recording {a.seconds:g} s, all 24 channels, mode = {a.mode} ...")
        t0 = time.perf_counter()
        for i in range(n):
            data[i] = dev.read_sample()
            if (i + 1) % (FS * 10) == 0:
                print(f"  {(i + 1) // FS:3d} s")
        rate = n / (time.perf_counter() - t0)
        bad = list(dev.bad_frames)

    print(f"Done: {n} samples, {rate:.1f} SPS, bad frames per chip {bad}")
    print(f"Writing {a.out} ...")

    names = [f"ch {i + 1}" for i in range(NUM_CHANNELS)]
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("data")
    ws.append(names)
    for row in np.round(data, 3).tolist():
        ws.append(row)

    s = wb.create_sheet("summary")
    s.append(["channel", "mean µV", "rms µV", "min µV", "max µV", "p-p µV"])
    centered = data - data.mean(axis=0)
    for ch in range(NUM_CHANNELS):
        col = data[:, ch]
        s.append([names[ch], round(col.mean(), 3), round(centered[:, ch].std(), 3),
                  round(col.min(), 3), round(col.max(), 3), round(np.ptp(col), 3)])
    s.append([])
    s.append(["samples", n]); s.append(["sample rate SPS", round(rate, 1)])
    s.append(["bad frames U5/U6/U7", str(bad)])
    s.append(["mode", a.mode]); s.append(["recorded", time.strftime("%Y-%m-%d %H:%M:%S")])
    wb.save(a.out)

    print("\nrms (µV) per channel:")
    rms = centered.std(axis=0)
    for c in range(3):
        print("  " + "  ".join(f"ch{ch + 1:<2} {rms[ch]:5.2f}" for ch in range(c * 8, c * 8 + 8)))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
