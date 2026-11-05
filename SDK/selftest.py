#!/usr/bin/env python3
"""
PiEEG-24 hardware self-test — no electrodes needed.

1. SPI/CS check: reads the ID register of all three ADS1299 (expect 0x3E).
2. DRDY check: every chip must deliver frames at the configured rate.
3. Shorted inputs: noise floor of each of the 24 channels.
4. Internal test signal: the ADS1299 1 Hz square wave must appear, with the
   same shape and amplitude, on every channel.

Writes a PNG report (default: pieeg24_selftest.png) and prints PASS/FAIL.

    python3 selftest.py            # on the Pi
    python3 selftest.py --mock     # dry run without hardware
"""

import argparse
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pieeg24 import PiEEG24, NUM_CHANNELS, CHIPS, ChipTimeout

NOISE_MAX_UV = 10.0   # rms limit with inputs shorted (pieeg-server scale, gain 1)
TEST_TOL = 0.25       # each channel's test-signal p2p within ±25 % of the median
CORR_MIN = 0.95       # correlation with the median test waveform


def collect(dev, mode, seconds):
    dev.set_mode(mode)
    dev.read(int(dev.sample_rate * 0.5))          # settle 0.5 s after reconfig
    dev.bad_frames = [0, 0, 0]
    t0 = time.perf_counter()
    x = dev.read(int(dev.sample_rate * seconds))
    return x, len(x) / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="pieeg24_selftest.png")
    ap.add_argument("--seconds", type=float, default=4.0)
    a = ap.parse_args()

    ok = True
    fs = 250
    try:
        dev = PiEEG24(sample_rate=fs, mock=a.mock).open()
    except ChipTimeout as e:
        print("FAIL:", e); return 1

    with dev:
        print("== 1. Chip ID (SPI + chip select)")
        for i, cid in enumerate(dev.chip_ids):
            good = cid is not None and (cid & 0x1F) == 0x1E
            ok &= good
            cs = CHIPS[i][2]
            print(f"   chip {i + 1} {CHIPS[i][0]:<13} CS=GPIO{cs if cs is not None else '8 (CE0)':<8}"
                  f" ID=0x{cid:02X}  {'OK' if good else 'FAIL (expect 0x3E; 0x00/0xFF = no SPI response)'}")

        ids_ok = [(c or 0) & 0x1F == 0x1E for c in dev.chip_ids]
        if ids_ok[0] and not (ids_ok[1] or ids_ok[2]):
            print("   HINT: chip 1 answers but 2 and 3 don't -> they have no clock. They get it\n"
                  "         from U5's CLK pin (shared CLK net). Check U6/U7 CLKSEL resistors\n"
                  "         (pull-down = external clock) and the CLK trace/vias from U5 pin 37.")
        elif ids_ok[0] and not all(ids_ok):
            bad = [i + 1 for i, k in enumerate(ids_ok) if not k]
            print(f"   HINT: only chip(s) {bad} fail -> check that chip's CS line, CLKSEL,\n"
                  f"         DVDD/AVDD and soldering of its TQFP pins.")

        print("== 2. Data-ready / sample rate")
        try:
            short, rate = collect(dev, "short", a.seconds)
        except ChipTimeout as e:
            print("   FAIL:", e); return 1
        good = abs(rate - fs) / fs < 0.05
        ok &= good
        print(f"   {rate:.1f} SPS measured (expected {fs})  {'OK' if good else 'FAIL'}")
        n = len(short)
        for i, bf in enumerate(dev.bad_frames):
            good = bf <= 0.01 * n
            ok &= good
            print(f"   chip {i + 1} {CHIPS[i][0]:<13} valid status frames: {n - min(bf, n)}/{n}  "
                  f"{'OK' if good else 'FAIL (status byte not 0xC0 -> chip not converting / not answering)'}")

        print("== 3. Shorted-input noise (µV rms)")
        noise = short.std(axis=0)
        stuck = np.ptp(short, axis=0) == 0
        noise_ok = (noise < NOISE_MAX_UV) & ~stuck
        ok &= bool(noise_ok.all())
        for c in range(3):
            row = "  ".join(f"E{ch + 1:<2}{noise[ch]:5.2f}{'' if noise_ok[ch] else '!'}"
                            for ch in range(c * 8, c * 8 + 8))
            print("  ", row)

        print("== 4. Internal test signal (µV p-p)")
        test, _ = collect(dev, "test", a.seconds)
        tc = test - test.mean(axis=0)
        p2p = np.percentile(tc, 99, axis=0) - np.percentile(tc, 1, axis=0)
        ref = np.median(tc, axis=1)
        corr = np.array([np.corrcoef(tc[:, ch], ref)[0, 1] if tc[:, ch].std() > 0 else 0
                         for ch in range(NUM_CHANNELS)])
        med = np.median(p2p)
        test_ok = (np.abs(p2p - med) <= TEST_TOL * med) & (corr > CORR_MIN) & (med > 100)
        ok &= bool(test_ok.all())
        for c in range(3):
            row = "  ".join(f"E{ch + 1:<2}{p2p[ch]:6.0f}{'' if test_ok[ch] else '!'}"
                            for ch in range(c * 8, c * 8 + 8))
            print("  ", row)

        dev.set_mode("normal")

    # ---- report figure
    fig, axes = plt.subplots(6, 4, figsize=(16, 13), sharex=True)
    tt = np.arange(len(tc)) / fs
    colors = ["#2a78d6", "#d6512a", "#1f9e6e"]
    for ch, axx in enumerate(axes.flatten()):
        axx.plot(tt, tc[:, ch], color=colors[ch // 8], lw=1)
        status = "OK" if (test_ok[ch] and noise_ok[ch]) else "FAIL"
        axx.set_title(f"E{ch + 1}  {p2p[ch]:.0f} µVpp  noise {noise[ch]:.2f} µV  {status}",
                      fontsize=9, color="black" if status == "OK" else "red")
        axx.tick_params(labelsize=7)
        axx.grid(alpha=0.3)
    for axx in axes[-1]:
        axx.set_xlabel("s", fontsize=8)
    fig.suptitle(f"PiEEG-24 self-test — internal test signal on all 24 channels — "
                 f"chip IDs {', '.join(f'0x{c:02X}' for c in dev.chip_ids)} — "
                 f"{'PASS' if ok else 'FAIL'}", fontsize=13,
                 color="green" if ok else "red")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"\nReport saved to {a.out}")
    print("RESULT:", "PASS — all 24 channels working" if ok else "FAIL — see channels marked '!'")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
