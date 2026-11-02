"""Shared pytest setup: project root on sys.path and a fake torch factory."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_fake_torch(cuda_available=False, hip=None, mps_available=False,
                    free_bytes=None, cuda_raises=False):
    """Build a SimpleNamespace that mimics the parts of torch used by devices."""
    calls = {"cuda_sync": 0, "mps_sync": 0}

    def cuda_is_available():
        if cuda_raises:
            raise RuntimeError("driver exploded")
        return cuda_available

    def cuda_sync(*args, **kwargs):
        calls["cuda_sync"] += 1

    def mps_sync(*args, **kwargs):
        calls["mps_sync"] += 1

    cuda = SimpleNamespace(
        is_available=cuda_is_available,
        synchronize=cuda_sync,
        get_device_name=lambda idx=0: "Fake GPU",
    )
    if free_bytes is not None:
        cuda.mem_get_info = lambda idx=0: (free_bytes, free_bytes * 2)

    def device(name):
        return "device:" + str(name)

    return SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(hip=hip),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)),
        mps=SimpleNamespace(synchronize=mps_sync),
        device=device,
        __version__="0.0-fake",
        get_num_threads=lambda: 4,
        calls=calls,
    )
