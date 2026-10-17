# w11 exp-shape micro-probe (process file; container parse check, then card).
#
# Verdicts so far: M1 (E chain) PASS; M2 (+glast chain) FAIL; P1 (64-elem
# norm-mask exp) PASS 1.6487; P2 (narrow copy + 8-lane mask=255 exp) FAIL
# 507015 -- a 9-line kernel reproduces the fault. P2-P6 verdicts:
#   P3 (narrow copy only)       FAIL  <- the ONE faulting shape
#   P4 (fill8 + exp8)           PASS  <- 8-lane exp/fill clean
#   P6 (fill8 only)             PASS
#   P5 (fill + narrow + exp64)  FAIL  <- narrow copy still inside
# Root cause: 1-elem UB->UB DataCopy (P3 FAIL / P4,P6 PASS). Production fix:
# scalar `glast_ub[0] = g_ub[...]` in scan1/scan2a/scan2b. mode=7 re-verifies
# the fixed glast chain (scalar read + 8-lane exp + 8-elem GM copy).
# One process per mode:
#   python3 _w11_v_exp_probe.py <mode>
import sys

sys.path.insert(0, "/tmp")
import torch  # noqa: E402

try:
    import torch_npu  # noqa: E402,F401
except ImportError:  # device-less container: shim npu so the factory can compile
    _orig_device = torch.device

    def _device(s=None, *a, **k):
        if isinstance(s, str) and s.startswith("npu"):
            return _orig_device("cpu")
        return _orig_device(s, *a, **k)

    torch.device = _device

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

    torch.npu = _NpuShim

import tilelang  # noqa: E402
from tilelang import language as T  # noqa: E402

from gdn_bwd_v2 import pass_configs  # noqa: E402


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def exp_probe(C, mode, accum_dtype="float"):
    @T.prim_func
    def main(
        G: T.Tensor([C], accum_dtype),  # type: ignore
        O: T.Tensor([C], accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            g_ub = T.alloc_ub([C], accum_dtype)
            e_ub = T.alloc_ub([C], accum_dtype)
            glast_ub = T.alloc_ub([8], accum_dtype)
            glastc_ub = T.alloc_ub([C], accum_dtype)
            T.tile.fill(g_ub, 0.5)
            if mode == 1:
                for i in T.Parallel(C):
                    e_ub[i] = T.exp(g_ub[i])
                T.copy(e_ub, O[0])
            else:
                if mode == 2:
                    T.copy(g_ub[C - 1:C], glast_ub[0:1])
                    T.tile.exp(glast_ub, glast_ub)
                else:
                    if mode == 3:
                        T.copy(g_ub[C - 1:C], glast_ub[0:1])
                    else:
                        if mode == 4:
                            T.tile.fill(glast_ub, 0.5)
                            T.tile.exp(glast_ub, glast_ub)
                        else:
                            if mode == 5:
                                T.tile.fill(glastc_ub, 0.0)
                                T.copy(g_ub[C - 1:C], glastc_ub[0:1])
                                T.tile.exp(glastc_ub, glastc_ub)
                            else:
                                if mode == 7:
                                    T.tile.fill(glast_ub, 0.5)
                                    glast_ub[0] = g_ub[C - 1]
                                    T.tile.exp(glast_ub, glast_ub)
                                else:
                                    T.tile.fill(glast_ub, 0.5)
                if mode == 5:
                    T.copy(glastc_ub, O[0])
                else:
                    T.copy(glast_ub, O[0])

    return main


if __name__ == "__main__":
    mode = int(sys.argv[1])
    C = 64
    g = torch.full((C,), 0.5, dtype=torch.float32).npu()
    try:
        out = exp_probe(C, mode)(g)
        torch.npu.synchronize()
        print(f"EXP-P{mode}-PASS {float(out[0]) if mode == 2 else float(out[1]):.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"EXP-P{mode}-FAIL {type(e).__name__}: {str(e)[:150]}")
    print("EXPPROBE-DONE")
