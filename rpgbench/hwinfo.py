"""Minimal, privacy-safe hardware description (no hostname, user or paths)."""
import importlib
import os
import platform
import sys

GENERIC_CPU = "Unknown CPU"


def _cpu_model():
    name = None
    try:
        if sys.platform.startswith("win"):
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            try:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            finally:
                winreg.CloseKey(key)
    except Exception:
        name = None
    if not name:
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8",
                      errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1]
                        break
        except Exception:
            name = None
    if not name:
        try:
            name = platform.processor()
        except Exception:
            name = None
    name = " ".join(str(name).split()) if name else ""
    return name or GENERIC_CPU


def _ram_from_ctypes():
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys)


def total_ram_bytes(import_fn=None):
    """Total physical RAM in bytes, or None when it cannot be determined."""
    import_fn = import_fn or importlib.import_module
    try:
        total = int(import_fn("psutil").virtual_memory().total)
        if total > 0:
            return total
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            total = _ram_from_ctypes()
            if total and total > 0:
                return total
        except Exception:
            pass
    try:
        total = int(os.sysconf("SC_PAGE_SIZE")) * int(
            os.sysconf("SC_PHYS_PAGES"))
        if total > 0:
            return total
    except Exception:
        pass
    return None


def _physical_cores(import_fn):
    try:
        count = import_fn("psutil").cpu_count(logical=False)
        return int(count) if count else None
    except Exception:
        return None


def _os_string():
    try:
        return "%s %s (%s)" % (platform.system(), platform.release(),
                               platform.version())
    except Exception:
        return "Unknown OS"


def collect(torch_mod, backend_label, gpu_name, import_fn=None):
    """Return the hardware dictionary stored in the results document."""
    import_fn = import_fn or importlib.import_module
    ram = total_ram_bytes(import_fn)
    threads = None
    try:
        threads = int(torch_mod.get_num_threads())
    except Exception:
        threads = None
    torch_version = getattr(torch_mod, "__version__", None)
    return {
        "cpu_model": _cpu_model(),
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_physical": _physical_cores(import_fn),
        "ram_gb": round(ram / (1024 ** 3), 1) if ram else None,
        "gpu_name": gpu_name,
        "torch_version": str(torch_version) if torch_version else "unknown",
        "backend": backend_label,
        "python_version": platform.python_version(),
        "os": _os_string(),
        "torch_threads": threads,
    }
