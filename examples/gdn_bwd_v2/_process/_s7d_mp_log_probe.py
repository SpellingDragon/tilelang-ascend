# S7d diagnosis probe -- error B: capture the REAL LiveInterval table straight
# from /tlasc/src/transform/ascend_memory_planning.cc DLOG output (enable with
# TVM_LOG_DEBUG=1). One scan-kernel compile per invocation; the Ascend memory
# planner prints per buffer `Buffer <name>: [start, end], size=N` and the
# LinearScanAllocator trace, then fails exactly like the compile gate does.
# (process file; docker cp to container /tmp. sandbox forbids host /tmp writes.)
# Usage (bash mandatory):
#   docker cp _s7d_mp_log_probe.py cann910b-l:/tmp/s7d_mp_log_probe.py
#   docker cp gdn_bwd_v2.py cann910b-l:/tmp/gdn_bwd_v2.py
#   docker exec cann910b-l bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh;
#     PYTHONPATH=/tlasc TVM_LOG_DEBUG=1 python3 /tmp/s7d_mp_log_probe.py <name>
#     > /tmp/s7d_mp_<name>.log 2>&1'
# <name> in {L0-fp16, TSC-fp16, TSC-fp32} -- scan kernel shapes only.
import os
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
            return "s7d-shim"

    _torch.npu = _NpuShim

import gdn_bwd_v2 as m

NAME = sys.argv[1] if len(sys.argv) > 1 else "TSC-fp16"
CASES = {
    "L0-fp16": (1, 2, 256, 64, 64, "float16"),
    "TSC-fp16": (1, 16, 256, 128, 128, "float16"),
    "TSC-fp32": (1, 16, 256, 128, 128, "float32"),
}
B, H, L, DK, DV, dt = CASES[NAME]
C, BV, BK = 64, 64, 64

print(f"S7D-MP-{NAME} START", flush=True)
try:
    k = m.gdn_bwd_v2_scan(B, H, L, DK, DV, C, BV, BK, dtype=dt, accum_dtype="float")
    src = k.get_kernel_source()
    print(f"S7D-MP-{NAME} OK src={len(src)}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"S7D-MP-{NAME} FAIL {type(e).__name__}: {e}", flush=True)
    sys.exit(1)
sys.exit(0)
