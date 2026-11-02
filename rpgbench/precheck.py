"""Best-effort pre-run check: battery and heavy background apps.

Nothing here ever raises, reads input or stops the benchmark.
"""
import importlib
import sys
from dataclasses import dataclass, field

# Lowercase Windows process names that tend to steal CPU/GPU time.
HEAVY_APPS = (
    "obs64.exe",
    "obs32.exe",
    "blender.exe",
    "resolve.exe",
    "afterfx.exe",
    "adobe premiere pro.exe",
    "handbrake.exe",
    "furmark.exe",
    "cinebench.exe",
    "prime95.exe",
    "xmrig.exe",
    "steam.exe",
    "epicgameslauncher.exe",
    "unrealeditor.exe",
    "3dsmax.exe",
)


@dataclass
class PrecheckResult:
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    on_battery: object = None
    heavy_apps: list = field(default_factory=list)


def _windows_battery():
    """Return True on battery, False when plugged in, None when unknown."""
    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = PowerStatus()
    ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    if not ok:
        return None
    if status.ACLineStatus == 0:
        return True
    if status.ACLineStatus == 1:
        return False
    return None


def _battery_from_psutil(psutil_mod):
    battery = psutil_mod.sensors_battery()
    if battery is None:
        return None
    plugged = battery.power_plugged
    if plugged is None:
        return None
    return not plugged


def run_precheck(import_fn=None, win_battery_fn=None, platform_name=None):
    """Return a PrecheckResult. Safe to call anywhere; never raises."""
    result = PrecheckResult()
    try:
        _run(result, import_fn, win_battery_fn, platform_name)
    except Exception as exc:
        result.notes.append("pre-run check limited: %s" % type(exc).__name__)
    return result


def _run(result, import_fn, win_battery_fn, platform_name):
    import_fn = import_fn or importlib.import_module
    platform_name = platform_name or sys.platform
    psutil_mod = None
    try:
        psutil_mod = import_fn("psutil")
    except Exception:
        result.notes.append("psutil not installed - heavy-app check skipped")

    on_battery = None
    if psutil_mod is not None:
        try:
            on_battery = _battery_from_psutil(psutil_mod)
        except Exception:
            on_battery = None
    if on_battery is None and platform_name.startswith("win"):
        try:
            fn = win_battery_fn or _windows_battery
            on_battery = fn()
        except Exception:
            on_battery = None
    if on_battery is None:
        result.notes.append("battery status unavailable - battery check skipped")
    result.on_battery = on_battery
    if on_battery:
        result.warnings.append(
            "Running on battery: plug in for a fair score.")

    if psutil_mod is None:
        return
    found = []
    try:
        for proc in psutil_mod.process_iter(["name"]):
            try:
                name = proc.info.get("name")
            except Exception:
                continue
            if not name:
                continue
            if str(name).lower() in HEAVY_APPS and name not in found:
                found.append(str(name))
    except Exception:
        result.notes.append("process list unavailable - heavy-app check skipped")
    result.heavy_apps = found
    for name in found:
        result.warnings.append(
            "Heavy app running: %s - close it for a fair score." % name)
