# w11 P10/P12 micro-probe — 64x64 VEC shapes from scan2a Gamma segment.
#   mode=10: exp(64x64)            <- scan2a L651 (scan1 only does 64-elem exp)
#   mode=12: mul(64x64) three-loop <- scan2a L656-660 (raw .* Gam .* M)
#   mode=13: ub->ub full copy 4096 <- scan2a L655 T.copy(da_ub, pk2_ub)
# One process per mode: python3 _w11_v2d_probe.py <mode>
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
def v2d_probe(C, mode, accum_dtype="float"):
    @T.prim_func
    def main(
        G: T.Tensor([C], accum_dtype),  # type: ignore
        O: T.Tensor([C, C], accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            x_ub = T.alloc_ub([C, C], accum_dtype)
            aux_ub = T.alloc_ub([C, C], accum_dtype)
            zer_ub = T.alloc_ub([C, C], accum_dtype)
            g_ub = T.alloc_ub([C], accum_dtype)
            T.tile.fill(g_ub, 0.5)
            T.tile.fill(x_ub, 0.5)
            T.tile.fill(zer_ub, 0.0)
            if mode == 10:
                for (i, j) in T.Parallel(C, C):
                    aux_ub[i, j] = T.exp(x_ub[i, j])
                T.copy(aux_ub, O[0, 0])
            else:
                if mode == 12:
                    for (i, j) in T.Parallel(C, C):
                        aux_ub[i, j] = g_ub[i] - g_ub[j]
                    for (i, j) in T.Parallel(C, C):
                        aux_ub[i, j] = zer_ub[i, j] - aux_ub[i, j]
                    for (i, j) in T.Parallel(C, C):
                        aux_ub[i, j] = T.exp(aux_ub[i, j])
                    for (i, j) in T.Parallel(C, C):
                        x_ub[i, j] = x_ub[i, j] * aux_ub[i, j]
                else:  # mode 13: full-shape UB -> UB
                    T.copy(x_ub, aux_ub)
                T.copy(aux_ub, O[0, 0])

    return main


if __name__ == "__main__":
    mode = int(sys.argv[1])
    C = 64
    g = torch.full((C,), 0.5, dtype=torch.float32).npu()
    try:
        out = v2d_probe(C, mode)(g)
        torch.npu.synchronize()
        # mode10: exp(0.5)=1.6487 ; mode12: x*exp(0.5)=0.8244 ; mode13: 0.5
        print(f"V2D-P{mode}-PASS out[0,0]={float(out[0, 0]):.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"V2D-P{mode}-FAIL {type(e).__name__}: {str(e)[:150]}")
    print("V2DPROBE-DONE")
