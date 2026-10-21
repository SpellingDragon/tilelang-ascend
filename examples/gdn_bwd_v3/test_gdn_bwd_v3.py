"""gdn_bwd_v3 的 L0 测试：单 chunk（T=C=64）下 dq / dk / dv 全对 golden。

为什么只测单 chunk：v3 的 dk/dv 目前只含 chunk 内主项（见 gdn_bwd_v3.py 头部的
能力边界）。单 chunk 时没有前一个 chunk，Delta' 恒为 0，跨 chunk 项本就为零，
因此三项都应落在 fp16 的容差内。多 chunk 的 dk/dv 在补齐 dKt 项后再纳入。

运行（从仓库根）：
    python examples/gdn_bwd_v3/test_gdn_bwd_v3.py
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.join(os.path.dirname(_HERE), "gdn_bwd_v2")
# 卡上：/tmp（本脚本与 v2/v3 模块）+ /tmp/ta_unpack/ta_pkg（tilelang 本体）。
# 不依赖外部 PYTHONPATH —— 显式设置会把它覆盖掉。
for p in (_HERE, _V2, "/tmp", "/tmp/ta_unpack/ta_pkg"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch_npu  # noqa: F401,E402

from test_gdn_bwd_v2 import _case_inputs, _golden_item, check_precision  # noqa: E402

from gdn_bwd_v3 import gdn_bwd_v3  # noqa: E402


def run_case(B, H, T_len, DK, DV, C, check_dkdv):
    torch.manual_seed(7)
    dtype = "float16"
    (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T_len, DK, DV, dtype, "zero", False)

    dev = "npu"
    dt = torch.float16
    dq, dk, dv = gdn_bwd_v3(
        q.to(dev).to(dt), k.to(dev).to(dt), v.to(dev).to(dt), g.to(dev),
        beta.to(dev), h0.to(dev).to(dt), do.to(dev).to(dt), chunk_size=C)
    torch.npu.synchronize()

    got = {"dq": dq.cpu().float(), "dk": dk.cpu().float(), "dv": dv.cpu().float()}
    ok = True
    items = ("dq", "dk", "dv") if check_dkdv else ("dq",)
    for item in items:
        ref = np.concatenate([_golden_item(gold, item, b) for b in range(B)], axis=0)
        passed, ratio, max_abs = check_precision(got[item], ref, dtype)
        print(f"V3 {item} rel= {max_abs:.3e} verdict= {'PASS' if passed else 'FAIL'} ratio={ratio:.4f}")
        ok &= passed
    print("V3-DONE all=%s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    # 单 chunk：dq/dk/dv 都可对 golden（Delta' 恒为 0）
    ok = run_case(1, 1, 64, 64, 64, 64, check_dkdv=True)
    # 多 chunk：只有 dq 不含跨 chunk 项，dk/dv 待补 dKt 后再纳入
    ok &= run_case(1, 1, 256, 64, 64, 64, check_dkdv=False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
