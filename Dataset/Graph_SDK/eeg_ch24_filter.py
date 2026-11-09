"""
Clean and filter one EEG channel (ch 24) to show blinks and left/right eye movements.

Steps:
  1. Load the data (one value per line, first line can be a header like "ch 24")
  2. Remove anomalies (e.g. 482372.253) and fill them by interpolation
  3. 50 Hz notch filter (power-line noise)
  4. Band-pass filter tuned for eye activity (blinks + left/right movements)
  5. Plot: cleaned raw signal vs. filtered signal

Usage:
  pip install numpy scipy matplotlib pandas openpyxl
  python eeg_ch24_filter.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ---------------- SETTINGS (change these) ----------------
FILE = "ter.xlsx"  # .xlsx (sheet "data") or .txt
FS = 248.9             # sampling rate in Hz (from the "summary" sheet)
LOW_CUT = 0.1          # Hz. Keep low so left/right eye movements stay visible
HIGH_CUT = 15.0        # Hz. Blinks and eye movements are mostly below ~15 Hz
NOTCH_FREQ = 50.0      # Hz. Power-line noise (use 60 in the Americas)
# ----------------------------------------------------------


def load(path):
    if path.lower().endswith(".xlsx"):
        import pandas as pd
        return pd.read_excel(path, sheet_name="data")["ch 24"].to_numpy(float)
    vals = []
    with open(path) as f:
        for line in f:
            try:
                vals.append(float(line.strip()))
            except ValueError:
                pass  # skip header / empty lines
    return np.array(vals)


def remove_anomalies(x, k=10.0):
    """Mark points far from the median (robust MAD test) and interpolate them."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    bad = np.abs(x - med) > k * 1.4826 * mad
    idx = np.arange(len(x))
    y = x.copy()
    y[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return y, np.where(bad)[0]


def notch(x, fs, f0, q=30.0):
    if f0 >= fs / 2:
        print(f"Skipping {f0} Hz notch: sampling rate {fs} Hz is too low.")
        return x
    b, a = iirnotch(f0, q, fs)
    return filtfilt(b, a, x)


def bandpass(x, fs, lo, hi, order=4):
    nyq = fs / 2
    hi = min(hi, 0.95 * nyq)
    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


raw = load(FILE)
clean, bad_idx = remove_anomalies(raw)
print(f"{len(raw)} samples, {len(raw)/FS:.1f} s at {FS} Hz")
for i in bad_idx:
    print(f"Removed anomaly at row {i + 1} (sample {i}): {raw[i]}")

filt = notch(clean - np.mean(clean), FS, NOTCH_FREQ)
filt = bandpass(filt, FS, LOW_CUT, HIGH_CUT)

t = np.arange(len(raw)) / FS

fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
ax[0].plot(t, clean, color="#555", lw=0.7)
ax[0].set_title("Ch 24 – after anomaly removal (raw, unfiltered)")
ax[0].set_ylabel("Amplitude (µV)")
ax[0].grid(alpha=0.3)

ax[1].plot(t, filt, color="#1f5fbf", lw=0.9)
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_title(f"Ch 24 – 50 Hz notch + band-pass {LOW_CUT}–{HIGH_CUT} Hz "
                "(blinks = sharp peaks, left/right = step-like shifts)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amplitude (µV)")
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ch24_filtered.png", dpi=150)
plt.show()
