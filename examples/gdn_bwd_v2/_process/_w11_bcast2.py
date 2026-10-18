# _w11_bcast2.py — TA broadcast probes, standalone (no scan2a context).
#   v0: T.tile.broadcast([1,64] -> [64,64])  (axis-0 / column) with row-slice fill
#   v1: same but the [1,64] source is filled via a plain 1D copy (shape-1 alloc)
#   v2: row broadcast control: dst[i,j] = src1d[i] via T.Parallel (known-good path)
# A correct v0 makes dst[i,j] = src[j] = j; the D15 degradation makes it 0 everywhere.
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402

_GOLDEN_ONLY = "--golden-only" in sys.argv

if not _GOLDEN_ONLY:
    import tilelang  # noqa: E402
    from tilelang import language as T  # noqa: E402

    def pass_configs():
        from gdn_bwd_v2 import pass_configs as _pc

        return _pc

    @tilelang.jit(out_idx=[1], pass_configs=pass_configs())
    def bcast_col_rowfill():
        @T.prim_func
        def main(B: T.Tensor([64], "float"), C: T.Tensor([64, 64], "float")):  # type: ignore
            with T.Kernel(1, threads=1, is_npu=True) as (cid):
                b_ub = T.alloc_ub([1, 64], "float")
                c_ub = T.alloc_ub([64, 64], "float")
                T.copy(B[0], b_ub[0, :])
                T.tile.broadcast(c_ub, b_ub)
                T.copy(c_ub, C[0, 0])

        return main

    @tilelang.jit(out_idx=[1], pass_configs=pass_configs())
    def bcast_col_1dcopy():
        @T.prim_func
        def main(B: T.Tensor([64], "float"), C: T.Tensor([64, 64], "float")):  # type: ignore
            with T.Kernel(1, threads=1, is_npu=True) as (cid):
                b1 = T.alloc_ub([64], "float")
                b_ub = T.alloc_ub([1, 64], "float")
                c_ub = T.alloc_ub([64, 64], "float")
                T.copy(B[0], b1)
                T.copy(b1, b_ub)
                T.tile.broadcast(c_ub, b_ub)
                T.copy(c_ub, C[0, 0])

        return main

    @tilelang.jit(out_idx=[1], pass_configs=pass_configs())
    def row_ctrl():
        @T.prim_func
        def main(B: T.Tensor([64], "float"), C: T.Tensor([64, 64], "float")):  # type: ignore
            with T.Kernel(1, threads=1, is_npu=True) as (cid):
                b_ub = T.alloc_ub([64], "float")
                c_ub = T.alloc_ub([64, 64], "float")
                T.copy(B[0], b_ub)
                for (i, j) in T.Parallel(64, 64):
                    c_ub[i, j] = b_ub[i] + 5.0      # row broadcast (known-good)
                T.copy(c_ub, C[0, 0])

        return main


def main():
    b = torch.arange(64, dtype=torch.float32)
    gold_col = np.tile(b.numpy(), (64, 1))            # C[i,j] = j
    gold_row = np.tile(b.numpy()[:, None], (1, 64)) + 5.0  # C[i,j] = i + 5
    if _GOLDEN_ONLY:
        print(f"GOLD col[0,:4]={gold_col[0,:4].tolist()} row[0,:4]={gold_row[0,:4].tolist()}")
        print("BCAST2-GOLDEN-ONLY-DONE")
        return

    for name, fac, gold in (("v0_rowfill", bcast_col_rowfill, gold_col),
                            ("v1_1dcopy", bcast_col_1dcopy, gold_col),
                            ("v2_row_ctrl", row_ctrl, gold_row)):
        try:
            out = fac()(b.npu()).float().cpu().numpy().reshape(64, 64)
            deg = bool(np.allclose(out, out[:, :1]) and not np.allclose(out[:, :1], gold[:, :1]))
            print(f"{name}: max_abs_err={float(np.abs(out-gold).max()):.3e} "
                  f"degraded={deg} out[0,:4]={np.round(out[0,:4],2).tolist()} "
                  f"out[63,:4]={np.round(out[63,:4],2).tolist()}")
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAIL {type(e).__name__}: {str(e)[:120]}")
    print("BCAST2-DONE")


if __name__ == "__main__":
    main()
