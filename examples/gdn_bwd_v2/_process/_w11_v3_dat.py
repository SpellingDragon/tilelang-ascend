# _w11_v3_dat.py — dg 的关键一步：A^T dA A^T 的两段 GEMM（B11 的求逆导数）
#
#   t = A^T dA          (gdn_v3_ta: A:[C,C], dA:[C,C] -> [C,C])
#   o = t  A^T          (gdn_v3_tb: A:[C,C], B:[C,C] -> [C,C])
# 中间 t 落 GM（两段各自单 GEMM；L0C→L1 的中转由此避开）。
# dL = msk_j .* (-o)；dg 与 dbeta 的 B11 行都从它来。
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from gdn_bwd_v3 import gdn_v3_ta, gdn_v3_tb  # noqa: E402


def main():
    B, H, C = 1, 2, 64
    torch.manual_seed(7)
    A = torch.randn(B, H, C, C, dtype=torch.float16) * 0.1
    dA = torch.randn(B, H, C, C, dtype=torch.float16)
    kta = gdn_v3_ta(B, H, C, C, C)
    ktb = gdn_v3_tb(B, H, C, C, C)
    t = kta(A.npu(), dA.npu())
    o = ktb(t, A.npu())
    torch.npu.synchronize()
    got = o.float().cpu().numpy().astype(np.float64)
    Ad = A.double().numpy()
    ref = Ad.transpose(0, 1, 3, 2) @ dA.double().numpy() @ Ad.transpose(0, 1, 3, 2)
    rel = float(np.abs(got - ref).max() / np.abs(ref).max())
    print(f"DAT rel={rel:.3e} goldmax={float(np.abs(ref).max()):.3e}")
    print("DAT verdict=%s" % ("PASS" if rel < 5e-3 else "FAIL"))
    print("DAT-DONE")


if __name__ == "__main__":
    main()
