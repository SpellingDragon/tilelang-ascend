# _w11_vec_probe.py — A 路第一块砖：纯 vector kernel（GM→UB→运算→GM）
#
# 背景（notes [83]）：本 fork 的 cv-sync 对 cube/vec 混合面约束极窄，任何新增
# 混合段都可能触发 auto-workspace 配对失衡。而所有"绿"的组件都是单侧的
# （cumsum 纯 vec、fwd 六件纯 cube）。本探针验证纯 vec 侧：
#   O = beta .* V   （只有 UB 运算，无 L1 / 无 GEMM）
# 判读：VECP PASS ⇒ 拆 cube/vec 的方案可行，vector 段可以独立成 kernel。
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
def beta_v(B, H, C, DV, dtype="float16", accum="float"):
    @T.prim_func
    def main(
        V: T.Tensor([B, H, C, DV], dtype),     # type: ignore
        Beta: T.Tensor([B, H, C], "float"),    # type: ignore
        O: T.Tensor([B, H, C, DV], dtype),     # type: ignore
    ):
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            v16 = T.alloc_ub([C, DV], dtype)
            v32 = T.alloc_ub([C, DV], accum)
            be = T.alloc_ub([C], accum)
            T.copy(V[b, h, 0, 0], v16)
            T.copy(Beta[b, h, 0], be)
            T.copy(v16, v32)
            for i, j in T.Parallel(C, DV):
                v32[i, j] = v32[i, j] * be[i]
            T.copy(v32, v16)
            T.copy(v16, O[b, h, 0, 0])

    return main


def main():
    B, H, C, DV = 1, 2, 64, 64
    v = torch.randn(B, H, C, DV, dtype=torch.float16)
    beta = torch.rand(B, H, C, dtype=torch.float32)
    out = beta_v(B, H, C, DV)(v.npu(), beta.npu())
    torch.npu.synchronize()
    got = out.float().cpu().numpy()
    ref = v.double().numpy() * beta.double().numpy()[..., None]
    d = np.abs(got - ref)
    rel = float(d.max() / (np.abs(ref).max() + 1e-30))
    print(f"VECP rel={rel:.3e} nonfinite={int((~np.isfinite(got)).sum())} "
          f"absmax={float(np.abs(got).max()):.3e} goldmax={float(np.abs(ref).max()):.3e}")
    print("VECP verdict=%s" % ("PASS" if rel < 5e-3 else "FAIL"))
    print("VECP-DONE")


if __name__ == "__main__":
    main()
