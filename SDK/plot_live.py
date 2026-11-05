#!/usr/bin/env python3
"""
Live 24-channel plot for PiEEG-24.

    python3 plot_live.py                 # electrode input, 1-40 Hz band-pass + 50 Hz notch
    python3 plot_live.py --mode test     # internal 1 Hz square wave on all 24 channels
    python3 plot_live.py --mode short    # inputs shorted (noise floor)
    python3 plot_live.py --raw           # no filtering (just DC removed)
    python3 plot_live.py --notch 60      # 60 Hz mains (Americas)
    python3 plot_live.py --save rec.csv  # also record everything to CSV
    python3 plot_live.py --mock          # no hardware: synthetic data

Keys in the window:  n = normal,  t = test signal,  s = shorted,
                     +/- = zoom µV scale,  a = autoscale each channel.
"""

import argparse
import csv
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pieeg24 import Acquisition, NUM_CHANNELS, ChipTimeout

try:
    from scipy import signal as sps
except ImportError:
    sps = None

GROUP_COLORS = ["#2a78d6", "#d6512a", "#1f9e6e"]  # chip 1 / 2 / 3


class Reader:
    """Pulls samples from the acquisition PROCESS, filters, keeps a ring buffer."""

    def __init__(self, acq, seconds, fs, lo, hi, notch, raw, save):
        self.acq, self.fs = acq, fs
        self.buf = np.zeros((int(seconds * fs), NUM_CHANNELS))
        self.raw_last = np.zeros(NUM_CHANNELS)
        self.count = 0
        self.error = None
        self.rate = 0.0
        self._hist = deque(maxlen=50)
        self.sos = None
        if not raw and sps is not None:
            parts = [sps.butter(2, [lo, hi], btype="band", fs=fs, output="sos")]
            if notch:
                b, a = sps.iirnotch(notch, 30, fs=fs)
                parts.append(sps.tf2sos(b, a))
            self.sos = np.vstack(parts)
        self.zi = None
        self.writer = None
        self.t0 = time.time()
        if save:
            self._f = open(save, "w", newline="")
            self.writer = csv.writer(self._f)
            self.writer.writerow(["time"] + [f"E{i + 1}" for i in range(NUM_CHANNELS)])

    def poll(self):
        x = self.acq.get()
        self.error = self.acq.error
        if self.acq.reset_flag:                 # registers changed -> restart filter
            self.acq.reset_flag = False
            self.zi = None
        if not len(x):
            return
        if self.writer:
            for k, s in enumerate(x):
                t = self.t0 + (self.count + k) / self.fs
                self.writer.writerow([f"{t:.4f}"] + [f"{v:.2f}" for v in s])
        y = x
        if self.sos is not None:
            if self.zi is None:
                self.zi = sps.sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
            y, self.zi = sps.sosfilt(self.sos, x, axis=0, zi=self.zi)
        k = min(len(y), len(self.buf))
        self.buf = np.roll(self.buf, -k, axis=0)
        self.buf[-k:] = y[-k:]
        self.raw_last = x[-1]
        self.count += len(x)
        self._hist.append((time.perf_counter(), self.count))
        if len(self._hist) > 5:
            (t0, c0), (t1, c1) = self._hist[0], self._hist[-1]
            self.rate = (c1 - c0) / (t1 - t0)

    def snapshot(self):
        return self.buf, self.raw_last, self.count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="normal", choices=["normal", "test", "short"])
    ap.add_argument("--rate", type=int, default=250, choices=[250, 500, 1000])
    ap.add_argument("--seconds", type=float, default=5.0, help="window length")
    ap.add_argument("--lo", type=float, default=1.0)
    ap.add_argument("--hi", type=float, default=40.0)
    ap.add_argument("--notch", type=float, default=50.0, help="0 to disable")
    ap.add_argument("--raw", action="store_true", help="no filtering")
    ap.add_argument("--scale", type=float, default=0, help="µV per channel spacing (0 = auto)")
    ap.add_argument("--save", help="CSV file to record to")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    raw = a.raw or a.mode == "test"   # don't band-pass the square wave
    if not raw and sps is None:
        print("scipy not installed -> showing unfiltered data (pip install scipy)")
    acq = Acquisition(sample_rate=a.rate, mode=a.mode, mock=a.mock).start()
    print("Chip IDs:", ", ".join(f"0x{c:02X}" for c in acq.chip_ids), "(ADS1299 = 0x3E)")
    state_mode = {"mode": a.mode}
    rd = Reader(acq, a.seconds, a.rate, a.lo, a.hi, a.notch, raw, a.save)

    state = {"scale": a.scale, "auto": a.scale == 0}
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.05)
    t = np.arange(len(rd.buf)) / a.rate - a.seconds
    lines, labels = [], []
    for ch in range(NUM_CHANNELS):
        (ln,) = ax.plot(t, np.zeros_like(t), lw=0.8, color=GROUP_COLORS[ch // 8])
        lines.append(ln)
    ax.set_yticks(-np.arange(NUM_CHANNELS))
    ax.set_yticklabels([f"E{ch + 1}" for ch in range(NUM_CHANNELS)])
    for ytl in ax.get_yticklabels():
        ytl.set_color(GROUP_COLORS[(int(ytl.get_text()[1:]) - 1) // 8])
    for ch in range(NUM_CHANNELS):
        labels.append(ax.text(-a.seconds * 1.0, -ch, "", fontsize=7, ha="right", va="center",
                              transform=ax.transData, color="#666", clip_on=False))
        labels[-1].set_position((-a.seconds - a.seconds * 0.075, -ch))
    for b in (7.5, 15.5):
        ax.axhline(-b, color="#bbb", lw=0.6, ls="--")
    ax.set_ylim(-NUM_CHANNELS + 0.3, 0.7)
    ax.set_xlim(-a.seconds, 0)
    ax.set_xlabel("time (s)")
    ax.grid(axis="x", alpha=0.3)

    def on_key(ev):
        if ev.key in ("n", "t", "s"):
            mode = {"n": "normal", "t": "test", "s": "short"}[ev.key]
            acq.set_mode(mode)
            state_mode["mode"] = mode
        elif ev.key in ("+", "="):
            state["auto"] = False
            state["scale"] = max(state["scale"] or 100, 1) / 2
        elif ev.key == "-":
            state["auto"] = False
            state["scale"] = (state["scale"] or 100) * 2
        elif ev.key == "a":
            state["auto"] = True
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        rd.poll()
        data, last, count = rd.snapshot()
        filled = min(count, len(data))
        d = data[-filled:] if filled else data
        d = d - d.mean(axis=0) if filled else d
        p2p = np.ptp(d, axis=0) if filled > 1 else np.ones(NUM_CHANNELS)
        if state["auto"]:
            spacing = np.where(p2p > 0, p2p, 1.0) * 1.1
        else:
            spacing = np.full(NUM_CHANNELS, state["scale"] or 100.0)
        for ch in range(NUM_CHANNELS):
            y = np.full(len(data), np.nan)
            if filled:
                y[-filled:] = d[:, ch] / spacing[ch] - ch
            lines[ch].set_ydata(y)
            labels[ch].set_text(f"{p2p[ch]:,.0f} µVpp\nDC {last[ch]:,.0f}")
        scale_txt = "auto/channel" if state["auto"] else f"{state['scale']:.0f} µV/div"
        title = (f"PiEEG-24  mode={state_mode['mode']}  {rd.rate:5.1f} SPS  samples={count}  "
                 f"scale={scale_txt}  bad frames={acq.bad_frames}")
        if rd.error:
            title = f"ERROR: {rd.error}"
            ax.title.set_color("red")
        ax.set_title(title, fontsize=10)
        return lines

    _anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    try:
        plt.show()
    finally:
        acq.stop()


if __name__ == "__main__":
    try:
        main()
    except ChipTimeout as e:
        print("ERROR:", e)
