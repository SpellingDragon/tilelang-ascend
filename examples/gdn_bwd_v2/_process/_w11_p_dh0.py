# _w11_p_dh0.py — 第 5 个验证点：dh0 = e_do^T Q - dVnew^T W（D17 重审）
#
# 两项都是 transpose_A 的 GEMM，且 **W 只作 B**（与 v2 里 w_full 兼作 A 的冲突不同）。
# 只对 chunk 0 判：golden 的 dh0 是"对初始状态的梯度"，等于 chunk 0 的 Delta'。
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


@tilelang.jit(out_idx=[4], pass_configs=_pass)
def k_dh(B, H, C, DV, DK, dtype="float16", accum="float"):
    """cube: DH = EDO^T Q - DVN^T W  (two transposed-A gemms into one L0C)."""

    @T.prim_func
    def main(EDO: T.Tensor([B, H, C, DV], dtype), Q: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             DVN: T.Tensor([B, H, C, DV], dtype), W: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             DH: T.Tensor([B, H, DV, DK], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            edo_l1 = T.alloc_L1([C, DV], dtype)   # A (transposed)
            q_l1 = T.alloc_L1([C, DK], dtype)     # B
            dvn_l1 = T.alloc_L1([C, DV], dtype)   # A (transposed)
            w_l1 = T.alloc_L1([C, DK], dtype)     # B
            o_l0 = T.alloc_L0C([DV, DK], accum)
            T.copy(EDO[b, h, 0, 0], edo_l1)
            T.copy(Q[b, h, 0, 0], q_l1)
            T.copy(DVN[b, h, 0, 0], dvn_l1)
            T.copy(W[b, h, 0, 0], w_l1)
            T.gemm_v0(edo_l1, q_l1, o_l0, transpose_A=True, init=True)
            T.gemm_v0(dvn_l1, w_l1, o_l0, transpose_A=True, init=False)
            T.copy(o_l0, DH[b, h, 0, 0])

    return main


def main():
    from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd, _golden_item

    B, H, T_len, DK, DV, C = 1, 1, 256, 64, 64, 64
    (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T_len, DK, DV, "float16", "zero", False)
    Qn, Kn, Vn = q.double().numpy()[0, 0], k.double().numpy()[0, 0], v.double().numpy()[0, 0]
    Bn, Gn, Hn, DOn = (beta.double().numpy()[0, 0], g.double().numpy()[0, 0],
                       h0.double().numpy()[0, 0], do.double().numpy()[0, 0])
    _, cache = _np_chunk_fwd(T_len, DK, DV, Qn, Kn, Vn, Bn, Gn, Hn, C)
    dh0_g = np.asarray(_golden_item(gold, "dh0", 0), np.float64).reshape(DV, DK)

    def f16(x):
        return torch.from_numpy(np.asarray(x, np.float64).astype(np.float16))[None, None].to("npu")

    ca = cache[0]
    sl = slice(0, C)
    e_do = (np.asarray(ca["e_vec"], np.float64)[:, None] * DOn[sl]).astype(np.float16)
    dvn = np.asarray(ca["vnew"], np.float64).astype(np.float16)
    wm = np.asarray(ca["w_mat"], np.float64).astype(np.float16)
    out = k_dh(B, H, C, DV, DK)(f16(e_do), f16(Qn[sl]), f16(dvn), f16(wm))
    got = out.float().cpu().numpy()[0, 0].astype(np.float64)
    ref_math = (e_do.astype(np.float64).T @ Qn[sl]) - (dvn.astype(np.float64).T @ wm.astype(np.float64))
    rel_k = float(np.abs(got - ref_math).max() / (np.abs(ref_math).max() + 1e-30))
    rel_g = float(np.abs(got - dh0_g).max() / (np.abs(dh0_g).max() + 1e-30))
    rel_alt = float(np.abs(got - (e_do.astype(np.float64).T @ Qn[sl]
                                 - np.asarray(ca["vnew"], np.float64).T @ wm.astype(np.float64))).max()
                    / (np.abs(dh0_g).max() + 1e-30))
    print(f"DH0 rel_vs_math={rel_k:.3e} rel_vs_golden={rel_g:.3e} goldmax={float(np.abs(dh0_g).max()):.3e}")
    print("DH0 verdict=%s" % ("PASS" if rel_vs_math_ok(rel_k) else "FAIL"))
    print("DH0-DONE")


def rel_vs_math_ok(x):
    return x < 5e-3


if __name__ == "__main__":
    main()
