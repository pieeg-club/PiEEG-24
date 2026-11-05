#!/usr/bin/env python3
"""
Free the PiEEG GPIO + SPI pins: stops pieeg-server and kills any other
process that still holds /dev/gpiochip* or /dev/spidev* (old scripts,
another IDLE/Thonny window ...).

    python3 reset_gpio.py

or put this at the top of any PiEEG script:

    import reset_gpio; reset_gpio.free()
"""

import glob
import os
import signal
import subprocess
import time


def _holders():
    me = {os.getpid(), os.getppid()}
    found = {}
    for p in glob.glob("/proc/[0-9]*"):
        pid = int(p.split("/")[-1])
        if pid in me:
            continue
        try:
            for fd in os.listdir(p + "/fd"):
                t = os.readlink(f"{p}/fd/{fd}")
                if t.startswith("/dev/gpiochip") or t.startswith("/dev/spidev"):
                    cmd = open(p + "/cmdline", "rb").read().replace(b"\0", b" ").decode().strip()
                    found[pid] = cmd
                    break
        except OSError:
            continue
    return found


def free(verbose=True):
    subprocess.run(["sudo", "-n", "systemctl", "stop", "pieeg-server"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for pid, cmd in _holders().items():
        if verbose:
            print(f"killing {pid}: {cmd[:80]}")
        try:
            os.kill(pid, signal.SIGKILL)
        except PermissionError:   # owned by root -> need sudo
            subprocess.run(["sudo", "-n", "kill", "-9", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except ProcessLookupError:
            pass
    time.sleep(0.3)
    left = _holders()
    if verbose:
        print("GPIO + SPI are free" if not left else f"still busy: {left}")
    return not left


if __name__ == "__main__":
    free()
