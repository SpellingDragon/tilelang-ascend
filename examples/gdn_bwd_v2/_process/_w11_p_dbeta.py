# _w11_p_dbeta.py — 第 13 个验证点：dbeta 的单项 rowsum(V .* t1)
#
#   t1 = A^T dVnew   （cube，已验证形态）
#   db = rowsum(V .* t1)  （vec 乘法 + cube 的 ones-GEMM）
# 两步各一个 kernel（A 路纪律）。判据：对同一单项的 numpy。
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
def k_mul(B, H, C, DV, dtype="float16", accum="float"):
    """vec: O = X .* Y (elementwise, both [C,DV])."""

    @T.prim_func
    def main(X: T.Tensor([B, H, C, DV], dtype), Y: T.Tensor([B, H, C, DV], dtype),  # type: ignore
             O: T.Tensor([B, H, C, DV], dtype)):  # type: ignore
        with T.Kernel(B * H, threads=1, is_npu=True) as (cid):
            b = cid // H
            h = cid % H
            x16 = T.alloc_ub([C, DV], dtype)
            x32 = T.alloc_ub([C, DV], accum)
            y16 = T.alloc_ub([C, DV], dtype)
            o16 = T.alloc_ub([C, DV], dtype)
            T.copy(X[b, h, 0, 0], x16)   # same-dtype load, then cast in UB
            T.copy(x16, x32)
            T.copy(Y[b, h, 0, 0], y16)
            for i, j in T.Parallel(C, DV):
                x32[i, j] = x32[i, j] * T.cast(y16[i, j], accum)
            T.copy(x32, o16)
            T.copy(o16, O[b, h, 0, 0])

    return main


@tilelang.jit(out_idx=[2], pass_configs=_pass)
def k_rowsum(B, H, C, DV, dtype="float16", accum="float"):
    """cube: rows = X @ ones(DV,16)."""

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
    from test_gdn_bwd_v2 import _case_inputs, _np_chunk_fwd

    B, H, T_len, DK, DV, C = 1, 1, 256, 64, 64, 64
    NC = T_len // C
    (q, k, v, g, beta, h0, do), _ = _case_inputs(B, H, T_len, DK, DV, "float16", "zero", False)
    Qn, Kn, Vn = q.double().numpy()[0, 0], k.double().numpy()[0, 0], v.double().numpy()[0, 0]
    Bn, Gn, Hn, DOn = (beta.double().numpy()[0, 0], g.double().numpy()[0, 0],
                       h0.double().numpy()[0, 0], do.double().numpy()[0, 0])
    _, cache = _np_chunk_fwd(T_len, DK, DV, Qn, Kn, Vn, Bn, Gn, Hn, C)

    def f16(x):
        return torch.from_numpy(np.asarray(x, np.float64).astype(np.float16))[None, None].to("npu")

    km, kr = k_mul(B, H, C, DV), k_rowsum(B, H, C, DV)
    ones = torch.ones(DV, 16, dtype=torch.float16).npu()
    ok = True
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        ca = cache[c]
        t1 = np.asarray(ca["a_mat"], np.float64).T @ np.asarray(ca["vnew"], np.float64)  # [C,DV]
        vt = km(f16(Vn[sl]), f16(t1))
        rows = kr(vt, ones).float().cpu().numpy()[0, 0][:, 0].astype(np.float64)  # [C]
        ref = (Vn[sl] * t1).sum(-1)
        rel = float(np.abs(rows - ref).max() / (np.abs(ref).max() + 1e-30))
        print(f"DB cc={c} rel={rel:.3e} goldmax={float(np.abs(ref).max()):.3e}")
        ok &= rel < 5e-3
    print("DB verdict=%s" % ("PASS" if ok else "FAIL"))
    print("DB-DONE")


if __name__ == "__main__":
    main()
