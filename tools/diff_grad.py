#!/usr/bin/env python3
"""diff_grad.py -- 把卡上 dump 的梯度与 golden 做逐 chunk / 逐元素比对。

为什么需要它：中间量（q1..q4）的 `abs.max` 在卡上与本地一致，但 `dk` 仍差 8e-2。
最大值相同而总和不同，只可能是**小元素上的绝对误差**——必须看逐元素。

用法：
    # 1) 车会把 /tmp/w11dump_dk_T256.npy 拉回 artifacts（见 w11_car.sh 的 [4] pull）
    # 2) 本地跑：
    python3 tools/diff_grad.py <dump.npy> dk 256

输出：
    * 每个 chunk 的 max/mean/rel
    * 误差最大的 8 个元素（token、维度、卡上值、golden、绝对/相对误差）
"""

import sys
import types

import numpy as np

_REPO = "/Users/pengweiye/Documents/codes/tilelang-ascend"
sys.path.insert(0, f"{_REPO}/examples/gdn_bwd_v2")
try:
    import torch_npu  # noqa: F401
except ImportError:
    _tm = types.ModuleType("torch_npu")
    _tm.npu = types.SimpleNamespace(synchronize=lambda *a, **k: None)
    sys.modules["torch_npu"] = _tm

from test_gdn_bwd_v2 import _case_inputs, _golden_item  # noqa: E402

C = 64


def main():
    path, item = sys.argv[1], sys.argv[2]
    T_len = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    got = np.load(path).astype(np.float64)          # [B,H,T,D]
    if got.ndim == 4:
        got = got[0, 0]
    (q, k, v, g, beta, h0, do), gold = _case_inputs(1, 1, T_len, 64, 64, "float16", "zero", False)
    ref = np.asarray(_golden_item(gold, item, 0), np.float64)[0, 0]

    print(f"== {item} T={T_len} 卡上 {got.shape} vs golden {ref.shape} ==")
    for c in range(T_len // C):
        sl = slice(c * C, (c + 1) * C)
        d = np.abs(got[sl] - ref[sl])
        print(f"  c={c} max={d.max():.3e} mean={d.mean():.3e} "
              f"rel={d.max() / (np.abs(ref[sl]).max() + 1e-30):.3e}")

    diff = np.abs(got - ref)
    idx = np.dstack(np.unravel_index(np.argsort(diff.ravel())[-8:][::-1], diff.shape))[0]
    print("  top-8 误差元素（t, d, 卡上, golden, abs, rel）:")
    for t, dd in idx:
        g_val, r_val = got[t, dd], ref[t, dd]
        rel = abs(g_val - r_val) / (abs(r_val) + 1e-12)
        print(f"    t={t:4d} c={t // C} d={dd:3d}  {g_val:+.6f}  {r_val:+.6f}  "
              f"abs={abs(g_val - r_val):.3e} rel={rel:.2e}")


if __name__ == "__main__":
    main()
