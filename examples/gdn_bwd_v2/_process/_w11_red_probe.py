# w11 reduce-shape micro-probe — scan2a E 段两个 reduce 形态单测
#   mode=8: T.reduce_sum(x_ub[C,BV], redh_ub[BV], dim=1)          (L703 形态, 输出 64 elem)
#   mode=9: rowb_ub[1,BV] = red1_ub[BV]; reduce_sum(rowb_ub, scal_ub[0:1], dim=1)
#           (L712-714 形态, 输出 1 elem -- 与 P3 同族的"奇长度输出"嫌疑)
# 一进程一 mode；数值可算（0.5 常数矩阵 → 行和 = BV*0.5）。
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
def red_probe(C, BV, mode, accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor([C, BV], accum_dtype),  # type: ignore
        O: T.Tensor([C], accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            x_ub = T.alloc_ub([C, BV], accum_dtype)
            redh_ub = T.alloc_ub([BV], accum_dtype)
            red1_ub = T.alloc_ub([BV], accum_dtype)
            rowb_ub = T.alloc_ub([1, BV], accum_dtype)
            scal_ub = T.alloc_ub([8], accum_dtype)
            T.copy(A[0, 0], x_ub)
            if mode == 8:
                T.reduce_sum(x_ub, redh_ub, dim=1)
                T.copy(redh_ub, O[0])
            else:
                for j in T.Parallel(BV):
                    red1_ub[j] = x_ub[0, j]
                for j in T.Parallel(BV):
                    rowb_ub[0, j] = red1_ub[j]
                T.reduce_sum(rowb_ub, scal_ub[0:1], dim=1)
                T.copy(scal_ub, O[0])

    return main


if __name__ == "__main__":
    mode = int(sys.argv[1])
    C, BV = 64, 64
    a = torch.full((C, BV), 0.5, dtype=torch.float32).npu()
    try:
        out = red_probe(C, BV, mode)(a)
        torch.npu.synchronize()
        print(f"RED-P{mode}-PASS out0={float(out[0]):.4f} (expect 32.0)")
    except Exception as e:  # noqa: BLE001
        print(f"RED-P{mode}-FAIL {type(e).__name__}: {str(e)[:150]}")
    print("REDPROBE-DONE")
