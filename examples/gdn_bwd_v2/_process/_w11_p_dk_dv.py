# _w11_p_dk_dv.py — 一次验两个梯度形态：dk = dqkt^T Q，dv = beta .* (A^T dVnew)
#
# 两者都是"transpose_A 的单次 GEMM + L0C 直出 GM"，各自独立 launch（单侧）。
# dk 用 golden 的 dqkt；dv 用 golden 的 dVnew 与 A（beta 的乘法留在 host，
# 因此本探针只判 GEMM 与转置形态）。
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
def k_dk(B, H, C, DK, dtype="float16", accum="float"):
    """cube: dk = dqkt^T Q"""

    @T.prim_func
    def main(DQKT: T.Tensor([B, H, C, C], dtype), Q: T.Tensor([B, H, C, DK], dtype),  # type: ignore
             DKOUT: T.Tensor([B, H, C, DK], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)    # dqkt, A operand (transposed)
            q_l1 = T.alloc_L1([C, DK], dtype)   # Q, B operand
            o_l0 = T.alloc_L0C([C, DK], accum)
            T.copy(DQKT[b, h, 0, 0], a_l1)
            T.copy(Q[b, h, 0, 0], q_l1)
            T.gemm_v0(a_l1, q_l1, o_l0, transpose_A=True, init=True)
            T.copy(o_l0, DKOUT[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_dv(B, H, C, DV, dtype="float16", accum="float"):
    """cube: t = A^T dVnew"""

    @T.prim_func
    def main(A: T.Tensor([B, H, C, C], dtype), DVNEW: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             T1: T.Tensor([B, H, C, DV], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, C], dtype)     # A operand (transposed)
            v_l1 = T.alloc_L1([C, DV], dtype)    # B operand
            o_l0 = T.alloc_L0C([C, DV], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(DVNEW[b, h, 0, 0], v_l1)
            T.gemm_v0(a_l1, v_l1, o_l0, transpose_A=True, init=True)
            T.copy(o_l0, T1[b, h, 0, 0])

    return main


def main():
    from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd, _golden_item

    B, H, T_len, DK, DV, C = 1, 1, 256, 64, 64, 64
    NC = T_len // C
    (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T_len, DK, DV, "float16", "zero", False)
    Qn, Kn, Vn = q.double().numpy()[0, 0], k.double().numpy()[0, 0], v.double().numpy()[0, 0]
    Bn, Gn, Hn, DOn = (beta.double().numpy()[0, 0], g.double().numpy()[0, 0],
                       h0.double().numpy()[0, 0], do.double().numpy()[0, 0])
    _, cache = _np_chunk_fwd(T_len, DK, DV, Qn, Kn, Vn, Bn, Gn, Hn, C)
    dk_g = np.asarray(_golden_item(gold, "dk", 0), np.float64)[0, 0]
    dv_g = np.asarray(_golden_item(gold, "dv", 0), np.float64)[0, 0]

    def f16(x):
        return torch.from_numpy(np.asarray(x, np.float64).astype(np.float16))[None, None].to("npu")

    kdk, kdv = k_dk(B, H, C, DK), k_dv(B, H, C, DV)
    ok = True
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        ca = cache[c]
        # host references (golden chunk quantities)
        dp_g = DOn[sl] @ np.asarray(ca["vnew"], np.float64).T
        dqkt_g = (dp_g * np.asarray(ca["gam_2d"], np.float64)
                  * np.asarray(ca["msk_m"], np.float64)).astype(np.float16)
        dk_ref = np.asarray(ca["p_mat"], np.float64)  # unused, keeps flake quiet
        dk_ref = dqkt_g.astype(np.float64).T @ Qn[sl]
        out_dk = kdk(f16(dqkt_g), f16(Qn[sl])).float().cpu().numpy()[0, 0].astype(np.float64)
        rel_dk = float(np.abs(out_dk - dk_g[sl]).max() / (np.abs(dk_g[sl]).max() + 1e-30))

        t = kdv(f16(np.asarray(ca["a_mat"])), f16(np.asarray(ca["vnew"]))).float().cpu().numpy()[0, 0].astype(np.float64)
        out_dv = Bn[sl][:, None] * t
        rel_dv = float(np.abs(out_dv - dv_g[sl]).max() / (np.abs(dv_g[sl]).max() + 1e-30))
        print(f"DKDV cc={c} rel_dk={rel_dk:.3e} rel_dv={rel_dv:.3e} "
              f"dkmax={float(np.abs(dk_g[sl]).max()):.3e} dvmax={float(np.abs(dv_g[sl]).max()):.3e}")
        # dk's golden carries the cross-chunk Kt^T Delta' term, which this probe
        # does not model -- only the last chunk has Delta' = 0.
        if c == NC - 1:
            ok &= (rel_dk < 5e-3)
        ok &= (rel_dv < 5e-3)
    print("DKDV verdict=%s" % ("PASS" if ok else "FAIL"))
    print("DKDV-DONE")


if __name__ == "__main__":
    main()
