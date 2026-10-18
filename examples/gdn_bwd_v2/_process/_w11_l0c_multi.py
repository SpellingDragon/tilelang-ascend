# _w11_l0c_multi.py — direct proof of D11: does L0C over-subscription corrupt GEMMs?
#   One kernel allocates 13 x [64,64] fp32 L0C (208KB) -- the same static size as
#   scan2a -- runs 13 identical A@B GEMMs into them and dumps all 13.
#   Correct kernel: all 13 match A@B. If over-subscription is real, later L0Cs
#   (past the 128KB budget) overlap each other and their results get corrupted.
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402

_GOLDEN_ONLY = "--golden-only" in sys.argv
N = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 13
assert 1 <= N <= 13

if not _GOLDEN_ONLY:
    import tilelang  # noqa: E402
    from tilelang import language as T  # noqa: E402

    def pass_configs():
        from gdn_bwd_v2 import pass_configs as _pc

        return _pc

    @tilelang.jit(out_idx=list(range(2, 2 + N)), pass_configs=pass_configs())
    def multi_l0c(dtype="float16"):
        @T.prim_func
        def main(
            A: T.Tensor([64, 64], dtype),  # type: ignore
            B: T.Tensor([64, 64], dtype),  # type: ignore
            C0: T.Tensor([64, 64], "float"), C1: T.Tensor([64, 64], "float"),  # type: ignore
            C2: T.Tensor([64, 64], "float"), C3: T.Tensor([64, 64], "float"),  # type: ignore
            C4: T.Tensor([64, 64], "float"), C5: T.Tensor([64, 64], "float"),  # type: ignore
            C6: T.Tensor([64, 64], "float"), C7: T.Tensor([64, 64], "float"),  # type: ignore
            C8: T.Tensor([64, 64], "float"), C9: T.Tensor([64, 64], "float"),  # type: ignore
            C10: T.Tensor([64, 64], "float"), C11: T.Tensor([64, 64], "float"),  # type: ignore
            C12: T.Tensor([64, 64], "float"),  # type: ignore
        ):
            with T.Kernel(1, threads=1, is_npu=True) as (cid):
                a = T.alloc_L1([64, 64], dtype)
                b = T.alloc_L1([64, 64], dtype)
                os_ = [T.alloc_L0C([64, 64], "float") for _ in range(N)]
                T.copy(A[0, 0], a)
                T.copy(B[0, 0], b)
                for o in os_:
                    T.gemm_v0(a, b, o, init=True)
                outs_ = [C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12]
                for i, o in enumerate(os_):
                    T.copy(o, outs_[i][0, 0])

        return main


def main():
    torch.manual_seed(5)
    a = torch.randn(64, 64).half()
    b = torch.randn(64, 64).half()
    gold = a.double().numpy() @ b.double().numpy()
    print(f"GOLD gemm max={float(np.abs(gold).max()):.3e} L0C_static={N * 16384}B vs 128KB limit")
    if _GOLDEN_ONLY:
        print("L0CMULTI-GOLDEN-ONLY-DONE")
        return
    args = [a.npu(), b.npu()] + [torch.empty(64, 64, dtype=torch.float32).npu()] * 13
    outs = multi_l0c()(*args)
    torch.npu.synchronize()
    bad = []
    for i in range(N):
        v = outs[i].float().cpu().numpy().reshape(64, 64)
        r = float(np.abs(v - gold).max() / (np.abs(gold).max() + 1e-30))
        tag = "OK " if r < 5e-3 else "BAD"
        if r >= 5e-3:
            bad.append(i)
        print(f"L0C[{i:2d}] rel={r:.3e} {tag}")
    print(f"L0CMULTI bad_slots={bad}")
    print("L0CMULTI-DONE")


if __name__ == "__main__":
    main()
