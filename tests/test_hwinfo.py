import getpass
import os
import platform
from types import SimpleNamespace

from rpgbench import hwinfo

KEYS = {"cpu_model", "cpu_cores_logical", "cpu_cores_physical", "ram_gb",
        "gpu_name", "torch_version", "backend", "python_version", "os",
        "torch_threads"}


def no_psutil(name):
    raise ImportError(name)


def test_exact_key_set_with_fake_torch():
    fake = SimpleNamespace(__version__="9.9", get_num_threads=lambda: 6)
    info = hwinfo.collect(fake, "CPU", None, no_psutil)
    assert set(info) == KEYS
    assert info["torch_version"] == "9.9"
    assert info["torch_threads"] == 6
    assert info["backend"] == "CPU"
    assert info["gpu_name"] is None
    assert info["cpu_model"]


def test_works_without_torch_and_psutil():
    info = hwinfo.collect(None, "CPU", None, no_psutil)
    assert set(info) == KEYS
    assert info["torch_threads"] is None
    assert info["cpu_cores_physical"] is None


def test_ram_gb_number_or_none():
    info = hwinfo.collect(None, "CPU", None, no_psutil)
    ram = info["ram_gb"]
    assert ram is None or (isinstance(ram, float) and ram > 0)


def test_psutil_values_used():
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=16 * 1024 ** 3),
        cpu_count=lambda logical=True: 4)
    info = hwinfo.collect(None, "CPU", None, lambda n: fake_psutil)
    assert info["ram_gb"] == 16.0
    assert info["cpu_cores_physical"] == 4
    assert hwinfo.total_ram_bytes(lambda n: fake_psutil) == 16 * 1024 ** 3


def test_no_hostname_or_username():
    info = hwinfo.collect(None, "CPU", None, no_psutil)
    text = " ".join(str(v) for v in info.values()).lower()
    tokens = [platform.node(), os.environ.get("COMPUTERNAME", ""),
              os.environ.get("USERNAME", "")]
    try:
        tokens.append(getpass.getuser())
    except Exception:
        pass
    for token in tokens:
        if len(token) >= 3:
            assert token.lower() not in text


def test_never_raises_on_broken_torch():
    class Broken:
        def get_num_threads(self):
            raise RuntimeError("x")

    info = hwinfo.collect(Broken(), "CPU", None, no_psutil)
    assert info["torch_threads"] is None
