# S7d diagnosis probe -- error A: dump scan's IR right before codegen (post
# LowerAndLegalize + OptimizeForTarget) and show the surviving scalar `tir.exp`
# with its enclosing-loop context; solve_a dumped as the green control.
# (process file; docker cp to container /tmp. sandbox forbids host /tmp writes.)
# Usage (bash mandatory):
#   docker cp _s7d_ir_probe.py cann910b-l:/tmp/s7d_ir_probe.py
#   docker cp gdn_bwd_v2.py cann910b-l:/tmp/gdn_bwd_v2.py
#   docker exec cann910b-l bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh;
#     PYTHONPATH=/tlasc python3 /tmp/s7d_ir_probe.py'
# Outputs: /tmp/s7d_ir_scan_l0.txt + /tmp/s7d_ir_solve_l0.txt + S7D-IR-* lines.
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
import tvm
import tilelang
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget
from tilelang.utils.target import determine_platform


def run_one(name, prim, path):
    platform = determine_platform("auto")
    target = tvm.target.Target({"kind": "llvm", "model": "auto"})
    mod = tvm.IRModule({prim.attrs["global_symbol"]: prim})
    for gvar, f in mod.functions_items():
        mod[gvar] = f.with_attr("npu_platform", platform)
    with tilelang.transform.PassContext(config=impl_pass_configs):
        mod = LowerAndLegalize(mod, target)
        mod = OptimizeForTarget(mod, target, platform)
    ir = mod.script()
    with open(path, "w") as fh:
        fh.write(ir)
    lines = ir.splitlines()
    hits = [i for i, ln in enumerate(lines) if "tir.exp" in ln]
    print(f"S7D-IR-{name} dump={path} tir.exp_hits={len(hits)} "
          f"ascend_exp={ir.count('ascend_exp')}")
    for h in hits[:3]:
        print(f"S7D-IR-{name} ---- context around line {h + 1} ----")
        for j in range(max(0, h - 12), min(len(lines), h + 6)):
            mark = ">>" if j == h else "  "
            print(f"S7D-IR-{name} {mark} {lines[j]}")
    print(f"S7D-IR-{name} DONE")


impl = m.gdn_bwd_v2_scan.__jit_impl__
impl_pass_configs = dict(impl.pass_configs)
prim = impl.func(1, 2, 256, 64, 64, 64, 64, 64, dtype="float16", accum_dtype="float")
try:
    run_one("SCAN-L0", prim, "/tmp/s7d_ir_scan_l0.txt")
except Exception as e:  # noqa: BLE001
    print(f"S7D-IR-SCAN-L0 ERROR {type(e).__name__}: {e}")

impl2 = m.gdn_bwd_v2_solve_a.__jit_impl__
impl_pass_configs = dict(impl2.pass_configs)
prim2 = impl2.func(1, 2, 256, 64, 64, 64, dtype="float16", accum_dtype="float")
try:
    run_one("SOLVE-L0", prim2, "/tmp/s7d_ir_solve_l0.txt")
except Exception as e:  # noqa: BLE001
    print(f"S7D-IR-SOLVE-L0 ERROR {type(e).__name__}: {e}")

print("S7D-IR-DONE")
