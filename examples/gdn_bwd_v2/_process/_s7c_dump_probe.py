# S7c diagnosis probe: dump IR right before CombineCV to identify workspace_10.
# (process file; docker cp to container /tmp. sandbox forbids host /tmp writes,
#  kept here per _s7b_compile_probe.py precedent)
# Usage (bash mandatory):
#   docker cp _s7c_dump_probe.py cann910b-l:/tmp/s7c_dump_probe.py
#   docker cp gdn_bwd_v2.py cann910b-l:/tmp/gdn_bwd_v2.py
#   docker exec cann910b-l bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh;
#     export PYTHONPATH=/tlasc:$PYTHONPATH; cd /tmp; python3 /tmp/s7c_dump_probe.py'
# Outputs: /tmp/ir_pre_combinecv.txt in container + S7C-COMBINECV OK/FAIL line.
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
            return "s7c-shim"

    _torch.npu = _NpuShim

import gdn_bwd_v2 as m
import tvm
import tilelang
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_platform

impl = m.gdn_bwd_v2_scan.__jit_impl__
prim = impl.func(1, 2, 256, 64, 64, 64, 64, 64, dtype="float16", accum_dtype="float")
platform = determine_platform("auto")
target = tvm.target.Target({"kind": "llvm", "model": "auto"})
mod = tvm.IRModule({prim.attrs["global_symbol"]: prim})
for gvar, f in mod.functions_items():
    mod[gvar] = f.with_attr("npu_platform", platform)

# S7c knob: S7C_DROP_TAIL=1 drops TL_ASCEND_TAIL_MASK from pass_configs
# (diagnosis only; kernel file untouched).
pcfg = dict(impl.pass_configs)
import os

if os.environ.get("S7C_DROP_TAIL") == "1":
    pcfg.pop(tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK, None)
print("S7C-PCFG", sorted(str(k) for k in pcfg))

with tilelang.transform.PassContext(config=pcfg):
    mod = LowerAndLegalize(mod, target)
    mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.CrossCorePipeline()(mod)
    ir = mod.script()
    with open("/tmp/ir_pre_combinecv.txt", "w") as fh:
        fh.write(ir)
    print("S7C-DUMP-OK platform=" + str(platform))
    try:
        tilelang.transform.CombineCV()(mod)
        print("S7C-COMBINECV OK")
    except Exception as e:  # noqa: BLE001
        print(f"S7C-COMBINECV FAIL {type(e).__name__}: {e}")
