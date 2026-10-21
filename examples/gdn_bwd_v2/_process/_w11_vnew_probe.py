# _w11_vnew_probe.py — D22 的最小判定：专用 A lane 在"单核单 chunk"下能否接线
#
# 背景：v2 里 Vnew = U0 + W·H_in^T 的 W 项贡献 ≈ 0（notes [71]），三种修法全失败
# （新 lane 不接线 / GM staging 崩 bisheng / 复用读破坏 cv-sync 配对）。
# 但 v2 的 liveness 极复杂；本探针在最小形态下只问一件事：
#   "W 用一条独立的 2-D L1（只作 A），H 用另一条（只作 B），GEMM 能不能算出 W@H^T？"
#
# 判读：
#   VNP PASS  ⇒ 新 lane 在简单形态可用 ⇒ v3 的路可行（v2 的失败是 liveness 冲突）
#   VNP FAIL  ⇒ 与 lane 无关，是 GEMM 形态本身（则 v3 也要换写法）
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

torch.manual_seed(7)
C = DV = 64
BK = 64

_pass = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def vnew_kernel(C, DV, BK, dtype="float16", accum="float"):
    @T.prim_func
    def main(
        W: T.Tensor([C, BK], dtype),     # type: ignore  (-W 的一半)
        H: T.Tensor([DV, BK], dtype),    # type: ignore  (H_in 的一半)
        Out: T.Tensor([C, DV], accum),   # type: ignore
    ):
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            w_l1 = T.alloc_L1([C, BK], dtype)    # A operand only
            h_l1 = T.alloc_L1([DV, BK], dtype)   # B operand only
            o_l0 = T.alloc_L0C([C, DV], accum)
            T.copy(W[0, 0], w_l1)
            T.copy(H[0, 0], h_l1)
            T.gemm_v0(w_l1, h_l1, o_l0, transpose_B=True, init=True)
            T.copy(o_l0, Out[0, 0])
    return main


def main():
    w = torch.randn(C, BK, dtype=torch.float16)
    h = torch.randn(DV, BK, dtype=torch.float16)
    out = vnew_kernel(C, DV, BK)(w.npu(), h.npu())
    torch.npu.synchronize()
    got = out.float().cpu().numpy()
    ref = w.double().numpy() @ h.double().numpy().T
    d = np.abs(got - ref)
    rel = float(d.max() / (np.abs(ref).max() + 1e-30))
    print(f"VNP rel={rel:.3e} nonfinite={int((~np.isfinite(got)).sum())} "
          f"absmax={float(np.abs(got).max()):.3e} goldmax={float(np.abs(ref).max()):.3e}")
    print("VNP verdict=%s" % ("PASS" if rel < 5e-3 else "FAIL"))
    print("VNP-DONE")


if __name__ == "__main__":
    main()
