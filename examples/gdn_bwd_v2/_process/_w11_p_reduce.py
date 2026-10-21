# _w11_p_reduce.py — 第 4 个验证点：ones-GEMM 做 rowsum（dbeta/dg 全靠它）
#
#   rows = X @ ones(DV, 16)   的第一列 == rowsum(X)
# 纯 cube（L1 + L0C），单次 GEMM。v2 的 dbeta/dg 全部依赖这个形态，但它从未被
# 单独验证过；若这里不过，dbeta/dg 的误差就不必再往别处找。
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

torch.manual_seed(7)

_pass = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_rowsum(B, H, C, DV, dtype="float16", accum="float"):
    @T.prim_func
    def main(X: T.Tensor([B, H, C, DV], dtype), Ones: T.Tensor([DV, 16], dtype),  # type: ignore
             R: T.Tensor([B, H, C, 16], accum)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            x_l1 = T.alloc_L1([C, DV], dtype)
            ones_l1 = T.alloc_L1([DV, 16], dtype)
            o_l0 = T.alloc_L0C([C, 16], accum)
            T.copy(X[b, h, 0, 0], x_l1)
            T.copy(Ones[0, 0], ones_l1)
            T.gemm_v0(x_l1, ones_l1, o_l0, init=True)
            T.copy(o_l0, R[b, h, 0, 0])

    return main


def main():
    B, H, C, DV = 1, 2, 64, 64
    x = torch.randn(B, H, C, DV, dtype=torch.float16)
    ones = torch.ones(DV, 16, dtype=torch.float16)
    r = k_rowsum(B, H, C, DV)(x.npu(), ones.npu())
    torch.npu.synchronize()
    got = r.float().cpu().numpy()          # [B,H,C,16]
    ref = x.double().numpy().sum(-1)       # [B,H,C]
    d1 = np.abs(got[..., 0] - ref)
    d2 = np.abs(got[..., 15] - ref)
    rel1 = float(d1.max() / (np.abs(ref).max() + 1e-30))
    rel15 = float(d2.max() / (np.abs(ref).max() + 1e-30))
    spread = float(np.abs(got[..., 0] - got[..., 15]).max())
    print(f"ROWS rel_col0={rel1:.3e} rel_col15={rel15:.3e} col_spread={spread:.3e} "
          f"goldmax={float(np.abs(ref).max()):.3e}")
    print("ROWS verdict=%s" % ("PASS" if max(rel1, rel15) < 5e-3 else "FAIL"))
    print("ROWS-DONE")


if __name__ == "__main__":
    main()
