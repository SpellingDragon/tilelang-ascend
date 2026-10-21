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


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_tat(B, H, C, M, N, dtype="float16", accum="float"):
    """cube: one transposed-A gemm, O = A^T B.  One gemm per kernel (A route)."""

    @T.prim_func
    def main(A: T.Tensor([B, H, C, M], dtype), Bm: T.Tensor([B, H, C, N], dtype),  # type: ignore
             O: T.Tensor([B, H, M, N], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            a_l1 = T.alloc_L1([C, M], dtype)   # A operand (transposed)
            b_l1 = T.alloc_L1([C, N], dtype)   # B operand
            o_l0 = T.alloc_L0C([M, N], accum)
            T.copy(A[b, h, 0, 0], a_l1)
            T.copy(Bm[b, h, 0, 0], b_l1)
            T.gemm_v0(a_l1, b_l1, o_l0, transpose_A=True, init=True)
            T.copy(o_l0, O[b, h, 0, 0])

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
    ker = k_tat(B, H, C, DV, DK)
    t1 = ker(f16(e_do), f16(Qn[sl])).float().cpu().numpy()[0, 0].astype(np.float64)
    t2 = ker(f16(dvn), f16(wm)).float().cpu().numpy()[0, 0].astype(np.float64)
    got = t1 - t2
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
