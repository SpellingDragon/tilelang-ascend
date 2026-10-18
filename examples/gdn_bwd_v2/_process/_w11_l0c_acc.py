# _w11_l0c_acc.py — does an init=False GEMM accumulate into a partially-written L0C?
#   C = A1@B1 (init=True)  then  C += A2@B2 (init=False)   [64^3, fp16 in, fp32 L0C]
#   Verdict: golden = A1@B1 + A2@B2.
#     rel ~1e-3            => accumulation is fine (D10 dismissed)
#     non-zero only in some columns / rel ~1 => the 2nd GEMM writes partial L0C
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

    @tilelang.jit(out_idx=[4], pass_configs=pass_configs())
    def acc_probe(dtype="float16"):
        M = N = K = 64

        @T.prim_func
        def main(
            A1: T.Tensor([M, K], dtype),  # type: ignore
            B1: T.Tensor([K, N], dtype),  # type: ignore
            A2: T.Tensor([M, K], dtype),  # type: ignore
            B2: T.Tensor([K, N], dtype),  # type: ignore
            C: T.Tensor([M, N], "float"),  # type: ignore
        ):
            with T.Kernel(1, threads=1, is_npu=True) as (cid):
                a1 = T.alloc_L1([M, K], dtype)
                b1 = T.alloc_L1([K, N], dtype)
                a2 = T.alloc_L1([M, K], dtype)
                b2 = T.alloc_L1([K, N], dtype)
                o = T.alloc_L0C([M, N], "float")
                T.copy(A1[0, 0], a1)
                T.copy(B1[0, 0], b1)
                T.copy(A2[0, 0], a2)
                T.copy(B2[0, 0], b2)
                T.gemm_v0(a1, b1, o, init=True)
                T.gemm_v0(a2, b2, o, init=False)   # <- the suspect
                T.copy(o, C[0, 0])

        return main


def main():
    # zero1: emulate scan2a's h0=0 case, where the first GEMM (B5b, t1@hin)
    # produces an all-zero L0C before the accumulating B4 GEMM.
    zero1 = "zero1" in sys.argv
    torch.manual_seed(11)
    dt = torch.float16
    a1 = torch.randn(64, 64).to(dt)
    if zero1:
        a1 = torch.zeros(64, 64).to(dt)
    b1 = torch.randn(64, 64).to(dt)
    a2 = torch.randn(64, 64).to(dt)
    b2 = torch.randn(64, 64).to(dt)
    n1, n2 = a1.double().numpy(), a2.double().numpy()
    m1, m2 = b1.double().numpy(), b2.double().numpy()
    gold = n1 @ m1 + n2 @ m2
    print(f"GOLD acc max={float(np.abs(gold).max()):.3e} "
          f"part1_max={float(np.abs(n1 @ m1).max()):.3e} part2_max={float(np.abs(n2 @ m2).max()):.3e} "
          f"zero1={zero1}")
    if _GOLDEN_ONLY:
        print("L0CACC-GOLDEN-ONLY-DONE")
        return
    out = acc_probe()(a1.npu(), b1.npu(), a2.npu(), b2.npu()).float().cpu().numpy().reshape(64, 64)
    r = float(np.abs(out - gold).max() / (np.abs(gold).max() + 1e-30))
    nz = np.abs(out).max(axis=0)
    print(f"ACCRES rel={r:.3e} nz_cols={int((nz > 0).sum())}/64 head_cols={np.round(nz[:8], 3).tolist()}")
    print(f"ACCRES card_rows={np.round(np.abs(out).max(axis=1), 2).tolist()}")
    print("L0CACC-DONE")


if __name__ == "__main__":
    main()
