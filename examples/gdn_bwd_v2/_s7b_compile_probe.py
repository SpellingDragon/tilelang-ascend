# S7b compile-only gate probe (process file; docker cp to container /tmp).
# Env: docker exec cann910b-l bash -c 'source /usr/local/Ascend/ascend-toolkit/set_env.sh;
#      PYTHONPATH=/tlasc python3 /tmp/s7b_compile_probe.py'
# (bash is mandatory: sh=dash silently skips set_env.sh and fakes bisheng misses)
import sys

sys.path.insert(0, "/tmp")
import torch as _torch

_orig_device = _torch.device


def _device(s=None, *a, **k):
    if isinstance(s, str) and s.startswith("npu"):
        return _orig_device("cpu")
    return _orig_device(s, *a, **k)


_torch.device = _device
if not hasattr(_torch, "npu"):

    class _NpuShim:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def current_stream():
            class _S:
                npu_stream = 0

            return _S()

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def set_device(*a, **k):
            pass

        @staticmethod
        def get_device_name():
            return "s7b-shim"

    _torch.npu = _NpuShim

import gdn_bwd_v2 as m

C, BV, BK = 64, 64, 64
CONFIGS = [
    ("L0", 1, 2, 256, 64, 64),
    ("TSC", 1, 16, 256, 128, 128),  # scaled true-shape (H16 DK128 DV128)
]

failed = False
for name, B, H, L, DK, DV in CONFIGS:
    for dt in ("float16", "float32"):
        try:
            k0 = m.gdn_bwd_v2_cumsum(B, H, L, C)
            s0 = k0.get_kernel_source()
            print(f"S7B-COMPILE-{name}-{dt}-cumsum OK src={len(s0)}")
            k1 = m.gdn_bwd_v2_solve_a(B, H, L, DK, C, BK, dtype=dt, accum_dtype="float")
            s1 = k1.get_kernel_source()
            print(f"S7B-COMPILE-{name}-{dt}-solve_a OK src={len(s1)}")
            k2 = m.gdn_bwd_v2_scan1(B, H, L, DK, DV, C, BV, BK, dtype=dt, accum_dtype="float")
            s2 = k2.get_kernel_source()
            print(f"S7B-COMPILE-{name}-{dt}-scan1 OK src={len(s2)}")
            k3 = m.gdn_bwd_v2_scan2a(B, H, L, DK, DV, C, BV, BK, dtype=dt, accum_dtype="float")
            s3 = k3.get_kernel_source()
            print(f"S7B-COMPILE-{name}-{dt}-scan2a OK src={len(s3)}")
            k4 = m.gdn_bwd_v2_scan2b(B, H, L, DK, DV, C, BV, BK, dtype=dt, accum_dtype="float")
            s4 = k4.get_kernel_source()
            print(f"S7B-COMPILE-{name}-{dt}-scan2b OK src={len(s4)}")
        except Exception as e:  # noqa: BLE001 - report and keep the matrix rolling
            failed = True
            print(f"S7B-COMPILE-{name}-{dt} FAIL {type(e).__name__}: {e}")

print("S7B-ALL-OK" if not failed else "S7B-FAIL")
sys.exit(1 if failed else 0)
