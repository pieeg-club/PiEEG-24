#!/usr/bin/env python3
"""
PiEEG-24 — look at ONE channel in a big plot.

The chosen channel is set to electrode input (CHnSET = 0x00); all other 23
channels are shorted (CHnSET = 0x01), so only your electrode is "live".

    python3 one_channel.py            # uses CHANNEL below
    python3 one_channel.py 10         # channel 10
    python3 one_channel.py 6 --raw    # no filter
    python3 one_channel.py --mock     # test without hardware

Keys in the window:
    Up / Right   next channel        Down / Left   previous channel
    + / -        zoom in / out       a             autoscale
    f            filter on / off

Needs pieeg24.py in the same folder.
"""

# ====================== EASY SETTINGS ======================
CHANNEL = 10        # 1 ... 24   (E1-E8 = U5, E9-E16 = U6, E17-E24 = U7)
GAIN    = 1         # 1, 2, 4, 6, 8, 12 or 24
NOTCH   = 50        # mains filter: 50 (Europe/Asia) or 60 (Americas), 0 = off
BAND    = (1, 40)   # band-pass in Hz
SECONDS = 5         # width of the window
# ===========================================================

import argparse
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pieeg24 import Acquisition

try:
    from scipy import signal as sps
except ImportError:
    sps = None

GAIN_BITS = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4, 12: 5, 24: 6}
SHORT = 0x01
FS = 250


def chset_for(channel, gain):
    """Selected channel -> electrode input, every other channel -> shorted (0x01)."""
    live = (GAIN_BITS[gain] << 4) | 0x00
    return [live if ch == channel else SHORT for ch in range(1, 25)]


class Reader:
    """Pulls samples from the acquisition PROCESS (no GIL fights -> no spikes)."""

    def __init__(self, acq, channel):
        self.acq = acq
        self.channel = channel
        self.buf = np.zeros(SECONDS * FS)
        self.raw = np.zeros(SECONDS * FS)
        self.count = 0
        self.filter_on = True
        self.error = None
        self.sos = None
        if sps is not None:
            parts = [sps.butter(2, BAND, btype="band", fs=FS, output="sos")]
            if NOTCH:
                b, a = sps.iirnotch(NOTCH, 30, fs=FS)
                parts.append(sps.tf2sos(b, a))
            self.sos = np.vstack(parts)
        self.zi = None

    def set_channel(self, ch):
        self.channel = ch
        self.acq.set_channels(chset_for(ch, GAIN))

    def poll(self):
        x = self.acq.get()
        self.error = self.acq.error
        if self.acq.reset_flag:                 # channel switched -> clear
            self.acq.reset_flag = False
            self.zi = None
            self.buf[:] = 0; self.raw[:] = 0; self.count = 0
        if not len(x):
            return
        x = x[:, self.channel - 1] / GAIN
        y = x
        if self.sos is not None and self.filter_on:
            if self.zi is None:
                self.zi = sps.sosfilt_zi(self.sos) * x[0]
            y, self.zi = sps.sosfilt(self.sos, x, zi=self.zi)
        k = min(len(x), len(self.buf))
        self.buf = np.roll(self.buf, -k); self.buf[-k:] = y[-k:]
        self.raw = np.roll(self.raw, -k); self.raw[-k:] = x[-k:]
        self.count += len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", nargs="?", type=int, default=CHANNEL)
    ap.add_argument("--raw", action="store_true", help="start with filter off")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if not 1 <= a.channel <= 24:
        sys.exit("channel must be 1..24")
    if GAIN not in GAIN_BITS:
        sys.exit(f"GAIN must be one of {list(GAIN_BITS)}")

    acq = Acquisition(sample_rate=FS, mock=a.mock).start()
    print("Chip IDs:", ", ".join(f"0x{c:02X}" for c in acq.chip_ids))
    rd = Reader(acq, a.channel)
    rd.set_channel(a.channel)
    rd.filter_on = not a.raw and sps is not None
    state = {"auto": True, "ylim": 100.0}

    fig, (ax, axf) = plt.subplots(2, 1, figsize=(13, 8),
                                  gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.35, left=0.07, right=0.98, top=0.92, bottom=0.08)
    t = np.arange(SECONDS * FS) / FS - SECONDS
    (line,) = ax.plot(t, np.zeros_like(t), lw=1, color="#2a78d6")
    ax.set_xlim(-SECONDS, 0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("µV")
    ax.grid(alpha=0.3)
    (fline,) = axf.plot([], [], lw=1, color="#d6512a")
    axf.set_xlim(0, 60)
    axf.set_xlabel("frequency (Hz)")
    axf.set_ylabel("µV")
    axf.grid(alpha=0.3)
    info = fig.text(0.07, 0.95, "", fontsize=12, weight="bold")

    def on_key(ev):
        ch = rd.channel
        if ev.key in ("up", "right"):
            rd.set_channel(ch % 24 + 1)
        elif ev.key in ("down", "left"):
            rd.set_channel((ch - 2) % 24 + 1)
        elif ev.key in ("+", "="):
            state["auto"] = False; state["ylim"] /= 2
        elif ev.key == "-":
            state["auto"] = False; state["ylim"] *= 2
        elif ev.key == "a":
            state["auto"] = True
        elif ev.key == "f" and sps is not None:
            rd.filter_on = not rd.filter_on; rd.zi = None
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        rd.poll()
        y, raw, n = rd.buf.copy(), rd.raw.copy(), rd.count
        n = min(n, len(y))
        shown = np.full_like(y, np.nan)
        if n > 1:
            seg = y[-n:] - (0 if rd.filter_on else y[-n:].mean())
            shown[-n:] = seg
            if state["auto"]:
                lim = max(np.percentile(np.abs(seg[min(n - 1, FS // 2):]), 99.5) * 1.2, 5)
            else:
                lim = state["ylim"]
            ax.set_ylim(-lim, lim)
            state["ylim"] = lim
            # spectrum of the last 2 s (or what we have)
            m = min(n, 2 * FS)
            s = seg[-m:] - seg[-m:].mean()
            spec = np.abs(np.fft.rfft(s * np.hanning(m))) * 2 / (np.hanning(m).sum())
            fr = np.fft.rfftfreq(m, 1 / FS)
            fline.set_data(fr, spec)
            axf.set_ylim(0, max(spec[fr > 0.5].max() * 1.1, 1) if m > FS // 2 else 1)
            rms, p2p = seg[-m:].std(), np.ptp(seg[-m:])
        else:
            rms = p2p = 0
        line.set_ydata(shown)
        ch = rd.channel
        chip = ["U5", "U6", "U7"][(ch - 1) // 8]
        txt = (f"CH {ch}  ({chip}, input {(ch - 1) % 8 + 1})   gain {GAIN}   "
               f"{'filter %g-%g Hz' % BAND + (', notch %d' % NOTCH if NOTCH else '') if rd.filter_on else 'RAW'}   "
               f"rms {rms:.1f} µV   p-p {p2p:.1f} µV   DC {raw[-1]:,.0f} µV   "
               f"bad frames {acq.bad_frames}   [↑/↓ change channel]")
        if rd.error:
            txt = f"ERROR: {rd.error}"
        info.set_text(txt)
        info.set_color("red" if rd.error else "black")
        return line, fline

    _anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    try:
        plt.show()
    finally:
        acq.stop()


if __name__ == "__main__":
    main()
