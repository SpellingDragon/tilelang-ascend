# w11P20 gemm micro — 64^3 GEMM 形态总表
#   mode=0: fp32 plain      C = A @ B
#   mode=1: fp32 transpose_B C = A @ B^T   (B given as [N,K])
#   mode=2: fp32 transpose_A C = A^T @ B   (A given as [K,M])
#   mode=3: fp16 transpose_A (control)
#   mode=4: fp16 plain (control)
#   mode=5: fp32 plain via workaround: B -> UB -> transpose -> L1, gemm tB
#   mode=6: fp16 plain via the same workaround (control)
# Driver compares against fp64 numpy golden: nan/inf counts + max abs err.
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    import torch_npu  # noqa: E402,F401
except ImportError:  # container parse: shim so the factory can compile
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


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def gemm64_probe(mode, dtype="float32"):
    M, N, K = 64, 64, 64
    m_a, n_a = (K, M) if mode == 2 else (M, K)   # transpose_A: A is [K,M]
    k_b, n_b = (N, K) if mode == 1 else (K, N)   # transpose_B: B is [N,K]

    @T.prim_func
    def main(
        A: T.Tensor([m_a, n_a], dtype),  # type: ignore
        B: T.Tensor([k_b, n_b], dtype),  # type: ignore
        C: T.Tensor([M, N], "float"),  # type: ignore
    ):
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            a_l1 = T.alloc_L1([m_a, n_a], dtype)
            b_l1 = T.alloc_L1([k_b, n_b], dtype)
            o_l0 = T.alloc_L0C([M, N], "float")
            T.copy(A[0, 0], a_l1)
            if mode >= 5:
                # workaround path: B [K,N] -> UB -> transpose -> L1 [N,K] -> tB gemm
                b_ub = T.alloc_ub([K, N], dtype)
                bt_ub = T.alloc_ub([N, K], dtype)
                io_ub = T.alloc_ub([K, N], dtype)
                bt_l1 = T.alloc_L1([N, K], dtype)
                T.copy(B[0, 0], io_ub)
                T.copy(io_ub, b_ub)
                T.tile.transpose(bt_ub, b_ub)
                T.copy(bt_ub, bt_l1)
                T.gemm_v0(a_l1, bt_l1, o_l0, transpose_B=True, init=True)
            else:
                T.copy(B[0, 0], b_l1)
                if mode == 0 or mode == 4:
                    T.gemm_v0(a_l1, b_l1, o_l0, init=True)
                else:
                    if mode == 1:
                        T.gemm_v0(a_l1, b_l1, o_l0, transpose_B=True, init=True)
                    else:
                        T.gemm_v0(a_l1, b_l1, o_l0, transpose_A=True, init=True)
            T.copy(o_l0, C[0, 0])

    return main


if __name__ == "__main__":
    mode = int(sys.argv[1])
    dtype = "float16" if mode in (3, 4, 6) else "float32"
    torch.manual_seed(3)
    M = N = K = 64
    m_a, n_a = (K, M) if mode == 2 else (M, K)
    k_b, n_b = (N, K) if mode == 1 else (K, N)
    a = torch.randn(m_a, n_a)
    b = torch.randn(k_b, n_b)
    if dtype == "float16":
        a, b = a.half(), b.half()
    an, bn = a.double().numpy(), b.double().numpy()
    if mode == 1:
        gold = an @ bn.T
    elif mode == 2:
        gold = an.T @ bn
    else:
        gold = an @ bn
    try:
        out = gemm64_probe(mode, dtype=dtype)(a.npu(), b.npu()).float().cpu().numpy()
        nan = int(np.isnan(out).sum())
        inf = int(np.isinf(out).sum())
        err = float(np.nanmax(np.abs(out - gold))) if nan == 0 else float("nan")
        print(f"GP{mode}-PASS({dtype}) nan={nan} inf={inf} max_abs_err={err:.3e} "
              f"gold_max={float(np.abs(gold).max()):.3e}")
        # bad-position distribution (fractal-boundary vs random pattern)
        bad = np.argwhere(~np.isfinite(out))
        if bad.shape[0]:
            rows = np.unique(bad[:, 0])
            cols = np.unique(bad[:, 1])
            print(f"GP{mode}-BAD rows={rows[:16].tolist()}{'...' if rows.size > 16 else ''} "
                  f"nrows={rows.size} cols={cols[:16].tolist()}{'...' if cols.size > 16 else ''} "
                  f"ncols={cols.size}")
        finite = np.isfinite(out)
        if int(finite.sum()):
            rel = np.abs(out[finite] - gold[finite]) / (np.abs(gold[finite]) + 1e-12)
            print(f"GP{mode}-ERR finite={int(finite.sum())} max_rel={float(rel.max()):.3e} "
                  f"med_rel={float(np.median(rel)):.3e} "
                  f"out_max={float(np.abs(out[finite]).max()):.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"GP{mode}-FAIL({dtype}) {type(e).__name__}: {str(e)[:150]}")
    print("GEMMPROBE-DONE")
