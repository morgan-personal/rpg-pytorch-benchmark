"""Backend detection and device contexts. torch is always passed in."""
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rpgbench import config

LABEL_ROCM = "AMD ROCm"
LABEL_CUDA = "NVIDIA CUDA"
LABEL_MPS = "Apple MPS"
LABEL_DML = "DirectML"
LABEL_CPU = "CPU"

KIND_CPU = "cpu"
KIND_CUDA = "cuda"
KIND_MPS = "mps"
KIND_DML = "directml"


@dataclass
class BackendInfo:
    label: str
    kind: str


@dataclass
class DeviceContext:
    label: str
    kind: str
    device: Any
    sync: Callable
    supports_fp16: bool
    alloc_cap_bytes: int
    gpu_name: Optional[str] = None


def detect_backend(torch_mod, import_fn=None):
    """Pick the best available backend. Never raises."""
    import_fn = import_fn or importlib.import_module
    try:
        if torch_mod.cuda.is_available():
            hip = None
            try:
                hip = getattr(torch_mod.version, "hip", None)
            except Exception:
                hip = None
            if hip:
                return BackendInfo(LABEL_ROCM, KIND_CUDA)
            return BackendInfo(LABEL_CUDA, KIND_CUDA)
    except Exception:
        pass
    try:
        if torch_mod.backends.mps.is_available():
            return BackendInfo(LABEL_MPS, KIND_MPS)
    except Exception:
        pass
    try:
        import_fn("torch_directml")
        return BackendInfo(LABEL_DML, KIND_DML)
    except Exception:
        pass
    return BackendInfo(LABEL_CPU, KIND_CPU)


def cpu_alloc_cap(ram_bytes):
    """Largest single allocation allowed on the CPU."""
    if not ram_bytes or ram_bytes <= 0:
        return config.CPU_ALLOC_UNKNOWN_BYTES
    return int(min(ram_bytes * config.CPU_ALLOC_RAM_FRACTION,
                   config.CPU_ALLOC_MAX_BYTES))


def gpu_alloc_cap(free_bytes):
    """Largest single allocation allowed on a GPU (None when not queryable)."""
    if not free_bytes or free_bytes <= 0:
        return config.GPU_ALLOC_UNKNOWN_BYTES
    return int(min(free_bytes * config.GPU_ALLOC_FREE_FRACTION,
                   config.GPU_ALLOC_MAX_BYTES))


def _noop_sync(tensor=None):
    return None


def cpu_context(torch_mod, ram_bytes):
    """Context for the always-benchmarked CPU device."""
    return DeviceContext(
        label=LABEL_CPU,
        kind=KIND_CPU,
        device=torch_mod.device("cpu"),
        sync=_noop_sync,
        supports_fp16=False,
        alloc_cap_bytes=cpu_alloc_cap(ram_bytes),
        gpu_name=None,
    )


def _directml_sync(tensor=None):
    # DirectML has no synchronize(); reading one element back from the result
    # waits for the in-order queue to finish the work that produced it.
    if tensor is not None:
        tensor.reshape(-1)[0].item()


def make_context(torch_mod, info, import_fn=None):
    """Build the context for a detected GPU backend. May raise."""
    import_fn = import_fn or importlib.import_module
    if info.kind == KIND_CUDA:
        name = None
        try:
            name = str(torch_mod.cuda.get_device_name(0))
        except Exception:
            name = None
        free = None
        try:
            free = torch_mod.cuda.mem_get_info()[0]
        except Exception:
            free = None

        def cuda_sync(tensor=None):
            torch_mod.cuda.synchronize()

        return DeviceContext(info.label, KIND_CUDA, torch_mod.device("cuda"),
                             cuda_sync, True, gpu_alloc_cap(free), name)
    if info.kind == KIND_MPS:
        def mps_sync(tensor=None):
            torch_mod.mps.synchronize()

        return DeviceContext(info.label, KIND_MPS, torch_mod.device("mps"),
                             mps_sync, True, gpu_alloc_cap(None),
                             "Apple GPU")
    if info.kind == KIND_DML:
        dml = import_fn("torch_directml")
        name = None
        try:
            name = str(dml.device_name(0))
        except Exception:
            name = None
        return DeviceContext(info.label, KIND_DML, dml.device(),
                             _directml_sync, False, gpu_alloc_cap(None), name)
    raise ValueError("unsupported backend: %s" % info.label)
