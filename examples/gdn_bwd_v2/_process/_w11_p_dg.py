# _w11_p_dg.py — 第 12 个验证点：dg 的 suffix-sum（B14 形态，纯 vector）
#
#   dg[i] = carry + sum_{r >= i} dgam[r]
# 与 cumsum 同族（纯 vec 的前缀和），carry 是后一个 chunk 的总和（反向扫描的递推态）。
# 判读：DGSS PASS ⇒ dg 的组装基元成立；跨 chunk 只需按 chunk 反序把 carry 传下去。
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
def k_suffix(B, H, C, dtype="float16", accum="float"):
    @T.prim_func
    def main(DG: T.Tensor([B, H, C], "float"), Carry: T.Tensor([B, H], "float"),  # type: ignore
             O: T.Tensor([B, H, C], "float")):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            s = T.alloc_ub([C], accum)
            c0 = T.alloc_ub([1], accum)
            dg = T.alloc_ub([C], accum)
            T.copy(DG[b, h, 0], dg)
            T.copy(Carry[b, h], c0)
            for i in T.serial(C):
                s[C - 1 - i] = dg[C - 1 - i]
            # suffix sum: walk from the tail, accumulating into a scalar
            for i in T.serial(C):
                c0[0] = c0[0] + s[C - 1 - i]
                s[C - 1 - i] = c0[0]
            T.copy(s, O[b, h, 0])

    return main


def main():
    B, H, C = 2, 3, 64
    dg = torch.randn(B, H, C, dtype=torch.float32)
    carry = torch.randn(B, H, dtype=torch.float32)
    o = k_suffix(B, H, C)(dg.npu(), carry.npu())
    torch.npu.synchronize()
    got = o.float().cpu().numpy()
    d = dg.double().numpy()
    c = carry.double().numpy()
    ref = np.zeros_like(d)
    for i in range(C):
        ref[:, :, i] = c + d[:, :, i:].sum(-1)
    rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
    print(f"DGSS rel={rel:.3e} goldmax={float(np.abs(ref).max()):.3e} "
          f"finite={int(np.isfinite(got).sum())}/{got.size}")
    print("DGSS verdict=%s" % ("PASS" if rel < 5e-3 else "FAIL"))
    print("DGSS-DONE")


if __name__ == "__main__":
    main()
