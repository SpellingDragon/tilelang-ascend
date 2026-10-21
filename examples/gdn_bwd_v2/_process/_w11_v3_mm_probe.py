# _w11_v3_mm_probe.py — 单验 gdn_v3_mm（O = A @ X，per-batch-head）
#
# 这是 dk 里唯一没有独立探针的 kernel（`dkt = mm(vnew, Delta')`），也是 dk 2.529
# 的头号嫌疑。两种形状都测：
#   M == C（v3 实际用法：DV=64, C=64）
#   M != C（128 vs 64，排除"只在方阵下正确"的可能）
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

from gdn_bwd_v3 import gdn_v3_mm  # noqa: E402


def run(B, H, C, M, N):
    torch.manual_seed(7)
    a = torch.randn(B, H, C, M, dtype=torch.float16)
    x = torch.randn(B, H, M, N, dtype=torch.float16)
    ker = gdn_v3_mm(B, H, C, M, N)
    o = ker(a.npu(), x.npu())
    torch.npu.synchronize()
    got = o.float().cpu().numpy().astype(np.float64)
    ref = a.double().numpy() @ x.double().numpy()
    rel = float(np.abs(got - ref).max() / np.abs(ref).max())
    print(f"MM C={C} M={M} N={N} rel={rel:.3e} goldmax={float(np.abs(ref).max()):.3e} "
          f"verdict={'PASS' if rel < 5e-3 else 'FAIL'}")
    return rel < 5e-3


def main():
    ok = run(1, 2, 64, 64, 64)     # v3 的实际形状
    ok &= run(1, 2, 64, 128, 64)   # M != C
    ok &= run(1, 2, 64, 64, 128)   # N != C
    print("MM verdict=%s" % ("PASS" if ok else "FAIL"))
    print("MM-DONE")


if __name__ == "__main__":
    main()
