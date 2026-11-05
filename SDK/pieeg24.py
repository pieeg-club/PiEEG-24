"""
pieeg24 — Python SDK for the PiEEG-24 shield (3 x ADS1299, 24 channels).

Wiring traced from the PiEEG-24 V2.0 Gerber files (matches the silkscreen):

    chip  ref  electrodes  CS                        DRDY
    1     U5   E1  - E8    pin 24 / GPIO8  (CE0)     pin 37 / GPIO26
    2     U6   E9  - E16   pin 35 / GPIO19 (manual)  pin 33 / GPIO13
    3     U7   E17 - E24   pin 40 / GPIO21 (manual)  pin 29 / GPIO5

    shared: MOSI pin 19 (GPIO10), MISO pin 21 (GPIO9), SCLK pin 23 (GPIO11),
            CLK net shared by all three chips, START tied to GND
            (conversions are started with the START command).

CLOCK: U5's CLKSEL is pulled up (internal oscillator); U6/U7 have CLKSEL
pull-down footprints, i.e. they take their clock from the shared CLK net.
Unlike the PiEEG-16, chip 1 must therefore OUTPUT its clock (CONFIG1.CLK_EN=1,
0xB6 @ 250 SPS) before chips 2/3 can respond at all — without it they read
back ID 0x00 and never assert DRDY. The driver does this first, then waits for
chips 2/3 to power-on-reset. All 24 channels then share one clock (in sync).

Register setup, SPI mode and scaling otherwise follow pieeg-server
(pieeg_server/hardware.py). Chip 3: CS on GPIO21, DRDY on GPIO5.

Requirements on the Pi:  SPI enabled (raspi-config -> Interface -> SPI),
`pip install spidev numpy`.  SPI1 must NOT be enabled (it would claim GPIO19/21).
Only Linux GPIO chardev ioctls are used — no gpiod / RPi.GPIO needed.

Quick use:

    from pieeg24 import PiEEG24
    with PiEEG24() as dev:
        print(dev.chip_ids)          # three ADS1299 IDs, expect 0x3E each
        data = dev.read(250)         # (250, 24) array in microvolts
"""

from __future__ import annotations

import atexit
import glob
import math
import os
import struct
import time

import numpy as np

try:
    import fcntl
except ImportError:  # not Linux: only mock mode works
    fcntl = None

try:
    import spidev
except ImportError:
    spidev = None

__all__ = ["PiEEG24", "Acquisition", "ChipTimeout", "NUM_CHANNELS", "CHIPS"]

NUM_CHANNELS = 24

# (name, spidev (bus, dev), CS gpio or None for hardware CE0, DRDY gpio)
CHIPS = (
    ("U5 (E1-E8)",   (0, 0), None, 26),
    ("U6 (E9-E16)",  (0, 1), 19,   13),
    ("U7 (E17-E24)", (0, 1), 21,    5),
)
# Chips 2 and 3 are opened on spidev0.1 so the kernel toggles CE1 (GPIO7,
# header pin 26), which is not connected on the PiEEG-24 — CE0 (chip 1) stays
# high while we talk to them, and the real CS is driven by hand.

# ---- ADS1299 commands / registers --------------------------------------
CMD_WAKEUP, CMD_STOP, CMD_RESET = 0x02, 0x0A, 0x06
CMD_START, CMD_RDATAC, CMD_SDATAC = 0x08, 0x10, 0x11
REG_ID, REG_CONFIG1, REG_CONFIG2, REG_CONFIG3 = 0x00, 0x01, 0x02, 0x03
CH_REGS = tuple(range(0x05, 0x0D))  # CH1SET..CH8SET

# CONFIG1 data-rate bits (DAISY_EN=0 → multi-readback, CLK_EN=0)
_DR_BITS = {250: 0b110, 500: 0b101, 1000: 0b100, 2000: 0b011}

# CHnSET values (gain 1, like pieeg-server)
MODES = {
    "normal": 0x00,  # electrode input
    "short":  0x01,  # inputs shorted -> noise floor
    "test":   0x05,  # internal ~1 Hz square-wave test signal (CONFIG2 = 0xD4)
}

# Scaling identical to pieeg-server so numbers match its dashboard/recordings:
#   uV = 4.5e6 * code / (2^24 - 1)
# (The ADS1299 datasheet LSB is 2*VREF/gain/2^24, i.e. 2x this value.
#  Pass scale="datasheet" to get datasheet-correct microvolts.)
_SCALES = {"pieeg": 4.5e6 / 16777215, "datasheet": 4.5e6 / 2**23}

FRAME_BYTES = 27  # 3 status + 8 ch * 3

# ---- Linux GPIO chardev v1 ioctls (same as pieeg-server) ----------------
_GPIO_GET_CHIPINFO = 0x8044B401
_GPIO_GET_LINEHANDLE = 0xC16CB403
_GPIOHANDLE_GET_VALUES = 0xC040B408
_GPIOHANDLE_SET_VALUES = 0xC040B409
_REQ_INPUT, _REQ_OUTPUT = 1, 2


class ChipTimeout(RuntimeError):
    """A chip's DRDY never went low — chip not running or DRDY not wired."""


def _find_gpiochip() -> str:
    """Pick the 40-pin-header GPIO controller (Pi 3/4: bcm2835/2711, Pi 5: rp1)."""
    best = None
    for path in sorted(glob.glob("/dev/gpiochip*")):
        try:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            buf = bytearray(68)
            fcntl.ioctl(fd, _GPIO_GET_CHIPINFO, buf)
            os.close(fd)
        except OSError:
            continue
        label = buf[32:64].split(b"\0")[0].decode(errors="replace")
        if label.startswith("pinctrl-rp1") or label.startswith("pinctrl-bcm"):
            return path
        best = best or path
    if best is None:
        raise RuntimeError("No /dev/gpiochip* found — is this a Raspberry Pi?")
    return best


def _pi_model() -> str:
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return f.read().rstrip(b"\0").decode(errors="replace")
    except OSError:
        return ""


class _Line:
    def __init__(self, chip_fd: int, offset: int, output: bool, default: int = 1,
                 label: bytes = b"pieeg24"):
        buf = bytearray(364)
        struct.pack_into("I", buf, 0, offset)
        struct.pack_into("I", buf, 256, _REQ_OUTPUT if output else _REQ_INPUT)
        buf[260] = default & 1
        buf[324:324 + len(label[:31])] = label[:31]
        struct.pack_into("I", buf, 356, 1)
        try:
            fcntl.ioctl(chip_fd, _GPIO_GET_LINEHANDLE, buf)
        except OSError as e:
            raise RuntimeError(
                f"Cannot claim GPIO{offset} ({e}). Something else holds it: stop it with\n"
                f"  sudo systemctl stop pieeg-server; sudo pkill -f pieeg-server; pkill -f plot_live; pkill -f selftest\n"
                f"In Thonny press Stop (Ctrl+F2) before running again. "
                f"Also check SPI1 / audio overlays are off.") from e
        self.fd = struct.unpack_from("i", buf, 360)[0]
        self._buf = bytearray(64)

    def get(self) -> int:
        fcntl.ioctl(self.fd, _GPIOHANDLE_GET_VALUES, self._buf)
        return self._buf[0]

    def set(self, v: int):
        self._buf[0] = v & 1
        fcntl.ioctl(self.fd, _GPIOHANDLE_SET_VALUES, self._buf)

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class PiEEG24:
    """Driver for the PiEEG-24 shield.

    Parameters
    ----------
    sample_rate : 250, 500, 1000 or 2000 SPS
    mode        : "normal" | "short" | "test"  (see MODES)
    spi_hz      : SPI clock; default 4 MHz on Pi 4, 2 MHz on Pi 5 (as pieeg-server)
    gpiochip    : override GPIO chardev path (auto-detected by default)
    scale       : "pieeg" (same numbers as pieeg-server) or "datasheet"
    mock        : True -> synthetic data, no hardware (for testing the plots)
    timeout     : seconds to wait for a chip's DRDY before raising ChipTimeout
    drdy        : "first" (default) — like the original PiEEG-16 script: wait
                  only for chip 1's DRDY (GPIO26), then read all three chips.
                  Valid because all chips run on chip 1's clock.
                  "each" — wait for every chip's own DRDY (GPIO26/13/5).
    """

    def __init__(self, sample_rate: int = 250, mode: str = "normal",
                 spi_hz: int | None = None, gpiochip: str | None = None,
                 scale: str = "pieeg", mock: bool = False, timeout: float = 0.1,
                 drdy: str = "first"):
        if drdy not in ("first", "each"):
            raise ValueError('drdy must be "first" or "each"')
        self.drdy = drdy
        if sample_rate not in _DR_BITS:
            raise ValueError(f"sample_rate must be one of {sorted(_DR_BITS)}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        self.sample_rate = sample_rate
        self.mode = mode
        self.mock = mock
        self.timeout = timeout
        self._scale = _SCALES[scale]
        self._spi_hz = spi_hz
        self._gpiochip = gpiochip
        self._spi = {}
        self._cs = {}
        self._drdy = {}
        self._chip_fd = -1
        self.chip_ids = [None, None, None]
        self.bad_frames = [0, 0, 0]  # frames whose status byte was not 0xC_
        self._opened = False
        self._mock_t = 0
        self._mock_next = 0.0

    # ------------------------------------------------------------ lifecycle
    def open(self):
        if self._opened:
            return self
        if self.mock:
            self.chip_ids = [0x3E, 0x3E, 0x3E]
            self._mock_next = time.perf_counter()
            self._opened = True
            return self
        if spidev is None or fcntl is None:
            raise RuntimeError("spidev not installed (pip install spidev) or not on Linux. "
                               "Use PiEEG24(mock=True) to test without hardware.")
        if self._spi_hz is None:
            self._spi_hz = 2_000_000 if "Raspberry Pi 5" in _pi_model() else 4_000_000
        path = self._gpiochip or _find_gpiochip()
        self._chip_fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        try:
            for i, (_, busdev, cs, drdy) in enumerate(CHIPS):
                if busdev not in self._spi:
                    s = spidev.SpiDev()
                    s.open(*busdev)
                    s.max_speed_hz = self._spi_hz
                    s.mode = 0b01
                    s.bits_per_word = 8
                    s.lsbfirst = False
                    self._spi[busdev] = s
                if cs is not None:
                    self._cs[i] = _Line(self._chip_fd, cs, True, 1, b"pieeg24_cs%d" % (i + 1))
                if i == 0 or self.drdy == "each":
                    self._drdy[i] = _Line(self._chip_fd, drdy, False, 0, b"pieeg24_drdy%d" % (i + 1))
            # Chip 1 first: it drives the shared CLK net for chips 2 and 3.
            self._init_chip(0)
            # ADS1299 power-on reset needs 2^18 clock cycles (~128 ms) once a
            # clock is present; chips 2/3 have had none until now.
            time.sleep(0.2)
            for i in (1, 2):
                self._init_chip(i)
            self._start_all()
        except Exception:
            self.close()
            raise
        self._opened = True
        atexit.register(self.close)   # free GPIO/SPI even if the script crashes
        return self

    def close(self):
        if not self.mock:
            for i in range(3):
                if i in self._cs or i == 0:
                    try:
                        self._cmd(i, CMD_SDATAC)
                        self._cmd(i, CMD_STOP)
                    except Exception:
                        pass
        for s in self._spi.values():
            s.close()
        for ln in list(self._cs.values()) + list(self._drdy.values()):
            ln.close()
        self._spi.clear(); self._cs.clear(); self._drdy.clear()
        if self._chip_fd >= 0:
            os.close(self._chip_fd)
            self._chip_fd = -1
        self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ low level
    def _xfer(self, chip: int, data: list[int]) -> list[int]:
        spi = self._spi[CHIPS[chip][1]]
        cs = self._cs.get(chip)
        if cs:
            cs.set(0)
        try:
            return spi.xfer2(list(data))
        finally:
            if cs:
                cs.set(1)

    def _read_frame(self, chip: int) -> list[int]:
        spi = self._spi[CHIPS[chip][1]]
        cs = self._cs.get(chip)
        if cs:
            cs.set(0)
        try:
            return spi.readbytes(FRAME_BYTES)
        finally:
            if cs:
                cs.set(1)

    def _cmd(self, chip: int, c: int):
        self._xfer(chip, [c])

    def _wreg(self, chip: int, reg: int, val: int):
        self._xfer(chip, [0x40 | reg, 0x00, val & 0xFF])

    def _rreg(self, chip: int, reg: int) -> int:
        return self._xfer(chip, [0x20 | reg, 0x00, 0x00])[2]

    def _init_chip(self, i: int):
        # CONFIG1: 0x90 | DR (0x96 @ 250 SPS, as pieeg-server). Chip 1 also
        # sets CLK_EN (bit 5) so its oscillator drives the shared CLK net.
        cfg1 = 0x90 | _DR_BITS[self.sample_rate] | (0x20 if i == 0 else 0)
        self._cmd(i, CMD_WAKEUP)
        self._cmd(i, CMD_STOP)
        self._cmd(i, CMD_RESET)        # (chip 1: RESET clears CLK_EN …)
        time.sleep(0.002)
        self._cmd(i, CMD_SDATAC)
        time.sleep(0.001)
        if i == 0:
            self._wreg(i, REG_CONFIG1, cfg1)   # … so turn the clock out back on at once
        self.chip_ids[i] = self._rreg(i, REG_ID)
        regs = [(0x14, 0x80), (REG_CONFIG1, cfg1), (REG_CONFIG2, 0xD4),
                (REG_CONFIG3, 0xFF), (0x04, 0x00), (0x0D, 0x00), (0x0E, 0x00),
                (0x0F, 0x00), (0x10, 0x00), (0x11, 0x00), (0x15, 0x20), (0x17, 0x00)]
        regs += [(r, MODES[self.mode]) for r in CH_REGS]
        for reg, val in regs:
            self._wreg(i, reg, val)

    def _start_all(self):
        # RDATAC + START back-to-back, chip 3 -> 2 -> 1. Chip 1 starts LAST, so
        # with the shared clock its DRDY always falls after chips 2/3 already
        # have their sample: reading all three on DRDY1 gets the same sample
        # instant from every chip, with a constant sub-millisecond offset.
        for i in (2, 1, 0):
            self._cmd(i, CMD_RDATAC)
        for i in (2, 1, 0):
            self._cmd(i, CMD_START)

    # ------------------------------------------------------------ public API
    def set_mode(self, mode: str):
        """Switch all 24 channels to 'normal', 'short' or 'test' while running."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        self.mode = mode
        self.set_channels([MODES[mode]] * NUM_CHANNELS)

    def set_channels(self, chset: list[int]):
        """Write CHnSET for all 24 channels, e.g. [0x01]*24 with one 0x00."""
        if len(chset) != NUM_CHANNELS:
            raise ValueError("need 24 CHnSET values")
        self.chset = list(chset)
        if self.mock:
            return
        for i in range(3):
            self._cmd(i, CMD_STOP)
            self._cmd(i, CMD_SDATAC)
            for k, r in enumerate(CH_REGS):
                self._wreg(i, r, chset[i * 8 + k] & 0xFF)
        self._start_all()

    def read_registers(self, chip: int) -> dict[int, int]:
        """Dump registers 0x00-0x17 of one chip (0, 1 or 2). Pauses streaming briefly."""
        if self.mock:
            return {}
        self._cmd(chip, CMD_SDATAC)
        regs = {r: self._rreg(chip, r) for r in range(0x18)}
        self._cmd(chip, CMD_RDATAC)
        return regs

    def _wait_ready(self, chip: int):
        # In RDATAC mode DRDY goes low when a new frame is ready and goes back
        # high once we clock the frame out, so "wait until low" yields exactly
        # one new frame per conversion even though the three chips are polled
        # one after another.
        ln = self._drdy[chip]
        deadline = time.perf_counter() + self.timeout
        while ln.get():
            if time.perf_counter() > deadline:
                name, _, cs, drdy = CHIPS[chip]
                raise ChipTimeout(
                    f"Chip {chip + 1} {name}: DRDY (GPIO{drdy}) never went low. "
                    f"Check CS (GPIO{cs if cs is not None else '8/CE0'}), power and DRDY wiring.")

    def _decode(self, chip: int, raw: list[int]) -> np.ndarray:
        if (raw[0] & 0xF0) != 0xC0:
            self.bad_frames[chip] += 1
        b = np.frombuffer(bytes(raw[3:27]), dtype=np.uint8).reshape(8, 3).astype(np.int32)
        v = (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
        v = np.where(v & 0x800000, v - (1 << 24), v)
        return v * self._scale

    def read_sample(self) -> np.ndarray:
        """Block until one new 24-channel sample is available. Returns (24,) µV."""
        if not self._opened:
            self.open()
        if self.mock:
            return self._mock_sample()
        out = np.empty(NUM_CHANNELS)
        if self.drdy == "first":
            self._wait_ready(0)                       # DRDY of chip 1 only
            frames = [self._read_frame(i) for i in range(3)]   # then all chips at once
            for i, f in enumerate(frames):
                out[i * 8:(i + 1) * 8] = self._decode(i, f)
            return out
        for i in range(3):
            self._wait_ready(i)
            out[i * 8:(i + 1) * 8] = self._decode(i, self._read_frame(i))
        return out

    def read(self, n: int) -> np.ndarray:
        """Read n consecutive samples. Returns (n, 24) array in µV."""
        return np.stack([self.read_sample() for _ in range(n)])

    def stream(self):
        """Infinite generator of (timestamp, (24,) µV) tuples."""
        while True:
            s = self.read_sample()
            yield time.time(), s

    # ------------------------------------------------------------ mock
    def _mock_sample(self) -> np.ndarray:
        self._mock_next += 1.0 / self.sample_rate
        d = self._mock_next - time.perf_counter()
        if d > 0:
            time.sleep(d)
        t = self._mock_t / self.sample_rate
        self._mock_t += 1
        rng = np.random.default_rng(self._mock_t)
        ch = np.arange(NUM_CHANNELS)
        if self.mode == "short":
            return rng.normal(0, 1.0, NUM_CHANNELS)
        if self.mode == "test":
            sq = 1875.0 * (1 if (t % 1.0) < 0.5 else -1)
            return sq + rng.normal(0, 1.0, NUM_CHANNELS)
        alpha = 15 * np.sin(2 * math.pi * 10 * t + ch * 0.3)
        mains = 8 * np.sin(2 * math.pi * 50 * t)
        drift = 2000 + 300 * ch + 40 * np.sin(2 * math.pi * 0.1 * t + ch)
        return drift + alpha + mains + rng.normal(0, 4, NUM_CHANNELS)


# ======================================================================
# Acquisition in a separate process — use this for anything with a GUI.
#
# A reader *thread* next to matplotlib gets blocked by the GIL for 5-50 ms
# while the plot redraws. One sample is only 4 ms long, so late reads clock
# out a frame while the chip is already writing the next one -> corrupted
# single samples (spikes), "bad frames" and ~230 SPS instead of 250.
# A separate process has its own GIL and its own CPU core, so it reads
# exactly like record.py does and just hands finished samples over.
# ======================================================================
import multiprocessing as _mp
import queue as _queue


def _acq_main(kw, data_q, cmd_q, info_q):
    try:
        try:
            os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})   # own core
        except (AttributeError, OSError):
            pass
        dev = PiEEG24(**kw).open()
        info_q.put(("ids", dev.chip_ids))
        batch = []
        while True:
            try:
                cmd, arg = cmd_q.get_nowait()
                if cmd == "stop":
                    break
                if cmd == "mode":
                    dev.set_mode(arg)
                elif cmd == "chset":
                    dev.set_channels(arg)
                data_q.put(("reset", None))
            except _queue.Empty:
                pass
            batch.append(dev.read_sample())
            if len(batch) >= 5:                     # hand over every 20 ms
                data_q.put((np.array(batch), list(dev.bad_frames)))
                batch = []
        dev.close()
    except Exception as e:
        info_q.put(("error", f"{type(e).__name__}: {e}"))


class Acquisition:
    """PiEEG24 running in its own process.

        acq = Acquisition(sample_rate=250, mode="normal").start()
        x = acq.get()            # (n, 24) µV — all samples since the last call
        acq.set_mode("short")    # or acq.set_channels([...24 CHnSET values...])
        acq.stop()

    Same keyword arguments as PiEEG24.
    """

    def __init__(self, **kw):
        self.kw = kw
        self.chip_ids = None
        self.bad_frames = [0, 0, 0]
        self.error = None
        self.count = 0
        self.reset_flag = False
        self._p = None

    def start(self, timeout: float = 10.0):
        ctx = _mp.get_context("fork")
        self._data, self._cmd, self._info = ctx.Queue(), ctx.Queue(), ctx.Queue()
        self._p = ctx.Process(target=_acq_main, daemon=True,
                              args=(self.kw, self._data, self._cmd, self._info))
        self._p.start()
        kind, val = self._info.get(timeout=timeout)
        if kind == "error":
            self._p.join(1)
            raise RuntimeError(val)
        self.chip_ids = val
        atexit.register(self.stop)
        return self

    def get(self) -> np.ndarray:
        """All new samples (n, 24); empty (0, 24) if none. Non-blocking."""
        try:
            kind, val = self._info.get_nowait()
            if kind == "error":
                self.error = val
        except _queue.Empty:
            pass
        out = []
        while True:
            try:
                arr, bad = self._data.get_nowait()
            except _queue.Empty:
                break
            if isinstance(arr, str):          # "reset" after a register change
                out = []
                self.reset_flag = True
                continue
            out.append(arr)
            self.bad_frames = bad
        if not out:
            return np.empty((0, NUM_CHANNELS))
        x = np.vstack(out)
        self.count += len(x)
        return x

    def set_mode(self, mode: str):
        self._cmd.put(("mode", mode))

    def set_channels(self, chset):
        self._cmd.put(("chset", list(chset)))

    def stop(self):
        if self._p is not None and self._p.is_alive():
            self._cmd.put(("stop", None))
            self._p.join(2)
            if self._p.is_alive():
                self._p.terminate()
        self._p = None
