# S7d diagnosis probe -- error A: which exp(1D) form survives AscendLowerParallelToVector
# as scalar `tir.exp` (codegen "Unresolved call Op(tir.exp)") vs gets lowered to ascend_exp.
# (process file; docker cp to container /tmp. sandbox forbids host /tmp writes.)
# Usage (bash mandatory):
#   docker cp _s7d_exp_probe.py cann910b-l:/tmp/s7d_exp_probe.py
#   docker exec cann910b-l bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh;
#     PYTHONPATH=/tlasc python3 /tmp/s7d_exp_probe.py'
# Prints per-variant verdict: S7D-EXP-<variant> LOWERED-OK / LEFT-SCALAR.
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

import tilelang
import tvm
from tilelang import language as T
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_platform

CFG5 = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}
CFG_SYNC = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}


def build_and_check(variant, pcfg):
    if variant == "oop":

        @tilelang.jit(out_idx=[-1], pass_configs=pcfg)
        def ker(B, H, L, C):
            @T.prim_func
            def main(G: T.Tensor([B, H, L], "float"), S: T.Tensor([B, H, L], "float")):  # type: ignore
                with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
                    g_ub = T.alloc_ub([C], "float")
                    e_ub = T.alloc_ub([C], "float")
                    T.copy(G[cid, 0, 0], g_ub)
                    for i in T.Parallel(C):
                        e_ub[i] = T.exp(g_ub[i])
                    T.copy(e_ub, S[cid, 0, 0])

            return main

    elif variant == "inplace":

        @tilelang.jit(out_idx=[-1], pass_configs=pcfg)
        def ker(B, H, L, C):
            @T.prim_func
            def main(G: T.Tensor([B, H, L], "float"), S: T.Tensor([B, H, L], "float")):  # type: ignore
                with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
                    g_ub = T.alloc_ub([C], "float")
                    T.copy(G[cid, 0, 0], g_ub)
                    for i in T.Parallel(C):
                        g_ub[i] = T.exp(g_ub[i])
                    T.copy(g_ub, S[cid, 0, 0])

            return main

    elif variant == "compound":

        @tilelang.jit(out_idx=[-1], pass_configs=pcfg)
        def ker(B, H, L, C):
            @T.prim_func
            def main(G: T.Tensor([B, H, L], "float"), S: T.Tensor([B, H, L], "float")):  # type: ignore
                with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
                    g_ub = T.alloc_ub([C], "float")
                    T.copy(G[cid, 0, 0], g_ub)
                    for i in T.Parallel(C):
                        g_ub[i] = T.exp(0.0 - g_ub[i])
                    T.copy(g_ub, S[cid, 0, 0])

            return main

    elif variant == "split":  # fwd chunk_h :91-96 verbatim shape: neg loop then plain exp
        @tilelang.jit(out_idx=[-1], pass_configs=pcfg)
        def ker(B, H, L, C):
            @T.prim_func
            def main(G: T.Tensor([B, H, L], "float"), S: T.Tensor([B, H, L], "float")):  # type: ignore
                with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
                    g_ub = T.alloc_ub([C], "float")
                    zero_ub = T.alloc_ub([C], "float")
                    T.copy(G[cid, 0, 0], g_ub)
                    T.tile.fill(zero_ub, 0.0)
                    for i in T.Parallel(C):
                        g_ub[i] = zero_ub[i] - g_ub[i]
                    for i in T.Parallel(C):
                        g_ub[i] = T.exp(g_ub[i])
                    T.copy(g_ub, S[cid, 0, 0])

            return main

    elif variant == "tileexp":  # v1 already-verified form: T.tile.exp on [C] buffer
        @tilelang.jit(out_idx=[-1], pass_configs=pcfg)
        def ker(B, H, L, C):
            @T.prim_func
            def main(G: T.Tensor([B, H, L], "float"), S: T.Tensor([B, H, L], "float")):  # type: ignore
                with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
                    g_ub = T.alloc_ub([C], "float")
                    T.copy(G[cid, 0, 0], g_ub)
                    T.tile.exp(g_ub, g_ub)
                    T.copy(g_ub, S[cid, 0, 0])

            return main

    elif variant == "oop2d":  # 2D control -- solve_a green shape
        @tilelang.jit(out_idx=[-1], pass_configs=pcfg)
        def ker(B, H, L, C):
            @T.prim_func
            def main(G: T.Tensor([B, H, L], "float"), S: T.Tensor([B * H, C, C], "float")):  # type: ignore
                with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
                    g_ub = T.alloc_ub([C, C], "float")
                    T.tile.fill(g_ub, 0.0)
                    for (i, j) in T.Parallel(C, C):
                        g_ub[i, j] = T.exp(g_ub[i, j])
                    T.copy(g_ub, S[cid, 0, 0])

            return main

    impl = ker.__jit_impl__
    prim = impl.func(1, 2, 256, 64)
    platform = determine_platform("auto")
    target = tvm.target.Target({"kind": "llvm", "model": "auto"})
    mod = tvm.IRModule({prim.attrs["global_symbol"]: prim})
    for gvar, f in mod.functions_items():
        mod[gvar] = f.with_attr("npu_platform", platform)
    with tilelang.transform.PassContext(config=pcfg):
        mod = LowerAndLegalize(mod, target)
    ir = mod.script()
    n_scalar = ir.count("tir.exp")
    n_ascend = ir.count("ascend_exp")
    verdict = "LOWERED-OK" if n_scalar == 0 and n_ascend > 0 else (
        "LEFT-SCALAR" if n_scalar > 0 else "NO-EXP-FOUND")
    print(f"S7D-EXP-{variant}-{'cfg5' if len(pcfg) > 1 else 'sync'} "
          f"{verdict} tir.exp={n_scalar} ascend_exp={n_ascend}")


failed = False
for variant in ("oop", "inplace", "compound", "split", "tileexp", "oop2d"):
    for tag, pcfg in (("cfg5", CFG5), ("sync", CFG_SYNC)):
        try:
            build_and_check(variant, dict(pcfg))
        except Exception as e:  # noqa: BLE001
            failed = True
            print(f"S7D-EXP-{variant}-{tag} ERROR {type(e).__name__}: {e}")

print("S7D-EXP-DONE" if not failed else "S7D-EXP-PROBE-ERROR")
sys.exit(1 if failed else 0)
