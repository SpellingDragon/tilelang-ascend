# _w11_bcast_probe.py — 5-line canary: does 1D->2D COLUMN broadcast work on card?
#   c[i,j] = b[j] + 5  (b is 1D, j is the SECOND dim) -- exactly the pattern the
#   official docs promise (api-compute.md:231 / t_parallel.md 3.5) and the one
#   that silently degraded to j=0 before D15 was found.
#   b = [0..63] so a degradation is unmistakable: C[i,j] becomes 5 instead of j+5.
#   This also tells whether the ta_pkg in use contains the #1219 fallback
#   ("lower un-vectorizable T.Parallel to serial").
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
    def bcast_probe():
        @T.prim_func
        def main(B: T.Tensor([64], "float"), C: T.Tensor([64, 64], "float")):  # type: ignore
            with T.Kernel(1, threads=1, is_npu=True) as (cid):
                b_ub = T.alloc_ub([64], "float")
                c_ub = T.alloc_ub([64, 64], "float")
                T.copy(B[0], b_ub)
                for (i, j) in T.Parallel(64, 64):
                    c_ub[i, j] = b_ub[j] + 5.0
                T.copy(c_ub, C[0, 0])

        return main


def main():
    b = torch.arange(64, dtype=torch.float32)
    gold = np.tile(b.numpy(), (64, 1)) + 5.0
    print(f"GOLD C[0,0..3]={np.round(gold[0, :4], 2).tolist()} C[63,60..63]={np.round(gold[63, 60:], 2).tolist()}")
    if _GOLDEN_ONLY:
        print("BCAST-GOLDEN-ONLY-DONE")
        return
    out = bcast_probe()(b.npu()).float().cpu().numpy().reshape(64, 64)
    print(f"CARD C[0,0..3]={np.round(out[0, :4], 2).tolist()} C[63,60..63]={np.round(out[63, 60:], 2).tolist()}")
    r = float(np.abs(out - gold).max())
    # degradation signature: all columns equal b[0]+5
    deg = bool(np.allclose(out, out[:, :1]) and not np.allclose(out[:, :1], gold[:, :1]))
    print(f"BCAST max_abs_err={r:.3e} degraded_to_j0={deg}")
    print("VERDICT column-broadcast-BROKEN (j collapsed to 0)" if deg else
          ("VERDICT column-broadcast-OK" if r < 1e-5 else "VERDICT column-broadcast-PARTIAL"))
    print("BCAST-DONE")


if __name__ == "__main__":
    main()
