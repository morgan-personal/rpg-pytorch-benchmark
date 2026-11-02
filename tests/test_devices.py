from types import SimpleNamespace

import pytest

from conftest import make_fake_torch
from rpgbench import devices

GIB = 1024 ** 3


def no_dml(name):
    raise ImportError(name)


def yes_dml(name):
    return SimpleNamespace(device=lambda: "dml-device",
                           device_name=lambda i: "Fake Radeon")


def test_rocm():
    t = make_fake_torch(cuda_available=True, hip="6.1")
    assert devices.detect_backend(t, no_dml).label == "AMD ROCm"


def test_cuda():
    t = make_fake_torch(cuda_available=True, hip=None)
    info = devices.detect_backend(t, no_dml)
    assert info.label == "NVIDIA CUDA"
    assert info.kind == "cuda"


def test_mps():
    t = make_fake_torch(mps_available=True)
    assert devices.detect_backend(t, no_dml).label == "Apple MPS"


def test_directml():
    t = make_fake_torch()
    assert devices.detect_backend(t, yes_dml).label == "DirectML"


def test_cpu():
    t = make_fake_torch()
    info = devices.detect_backend(t, no_dml)
    assert (info.label, info.kind) == ("CPU", "cpu")


def test_directml_import_raising_other_exception():
    def boom(name):
        raise RuntimeError("broken dll")

    assert devices.detect_backend(make_fake_torch(), boom).label == "CPU"


def test_cuda_check_raising_moves_on():
    t = make_fake_torch(cuda_raises=True, mps_available=True)
    assert devices.detect_backend(t, no_dml).label == "Apple MPS"
    t = make_fake_torch(cuda_raises=True)
    assert devices.detect_backend(t, yes_dml).label == "DirectML"
    assert devices.detect_backend(t, no_dml).label == "CPU"


def test_missing_attributes_never_raise():
    assert devices.detect_backend(SimpleNamespace(), no_dml).label == "CPU"


def test_cpu_alloc_cap():
    assert devices.cpu_alloc_cap(None) == 1 * GIB
    assert devices.cpu_alloc_cap(0) == 1 * GIB
    assert devices.cpu_alloc_cap(8 * GIB) == 2 * GIB
    assert devices.cpu_alloc_cap(64 * GIB) == 4 * GIB


def test_gpu_alloc_cap():
    assert devices.gpu_alloc_cap(None) == 2 * GIB
    assert devices.gpu_alloc_cap(4 * GIB) == 2 * GIB
    assert devices.gpu_alloc_cap(24 * GIB) == 4 * GIB


def test_cpu_context():
    t = make_fake_torch()
    ctx = devices.cpu_context(t, 8 * GIB)
    assert ctx.label == "CPU" and ctx.kind == "cpu"
    assert ctx.supports_fp16 is False
    assert ctx.alloc_cap_bytes == 2 * GIB
    assert ctx.sync(None) is None


def test_cuda_context_sync_and_cap():
    t = make_fake_torch(cuda_available=True, hip="6.1", free_bytes=6 * GIB)
    info = devices.detect_backend(t, no_dml)
    ctx = devices.make_context(t, info, no_dml)
    assert ctx.label == "AMD ROCm"
    assert ctx.alloc_cap_bytes == 3 * GIB
    assert ctx.gpu_name == "Fake GPU"
    assert ctx.supports_fp16
    ctx.sync(None)
    assert t.calls["cuda_sync"] == 1


def test_cuda_context_without_mem_info():
    t = make_fake_torch(cuda_available=True)
    ctx = devices.make_context(t, devices.detect_backend(t, no_dml), no_dml)
    assert ctx.alloc_cap_bytes == 2 * GIB


def test_mps_context_sync():
    t = make_fake_torch(mps_available=True)
    ctx = devices.make_context(t, devices.detect_backend(t, no_dml), no_dml)
    ctx.sync(None)
    assert t.calls["mps_sync"] == 1


def test_directml_context_readback_sync():
    reads = []

    class FakeTensor:
        def reshape(self, *shape):
            return self

        def __getitem__(self, idx):
            return SimpleNamespace(item=lambda: reads.append(idx) or 1.0)

    t = make_fake_torch()
    ctx = devices.make_context(t, devices.BackendInfo("DirectML", "directml"),
                               yes_dml)
    assert ctx.device == "dml-device"
    assert ctx.gpu_name == "Fake Radeon"
    assert ctx.supports_fp16 is False
    ctx.sync(FakeTensor())
    assert reads == [0]
    ctx.sync(None)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        devices.make_context(make_fake_torch(),
                             devices.BackendInfo("X", "weird"), no_dml)
