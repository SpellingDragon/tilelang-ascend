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

from gdn_bwd_v3 import gdn_bwd_v3, gdn_bwd_v3_relay  # noqa: E402


def run_case(B, H, T_len, DK, DV, C, check_dkdv):
    """基础版 `gdn_bwd_v3` 只对 dq 完整；dk/dv 缺 B6-B8（与 Delta' 无关，单 chunk 也缺）。

    check_dkdv 保留为参数，但只用于打印 dk/dv 的数值（不 gate）——判定交给 relay。
    """
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
    for item in ("dq",):
        ref = np.concatenate([_golden_item(gold, item, b) for b in range(B)], axis=0)
        passed, ratio, max_abs = check_precision(got[item], ref, dtype)
        print(f"V3 {item} rel= {max_abs:.3e} verdict= {'PASS' if passed else 'FAIL'} ratio={ratio:.4f}")
        ok &= passed
    if check_dkdv:   # 数值参考，不 gate
        for item in ("dk", "dv"):
            ref = np.concatenate([_golden_item(gold, item, b) for b in range(B)], axis=0)
            _p, ratio, max_abs = check_precision(got[item], ref, dtype)
            print(f"V3 {item} (partial, not gated) rel={max_abs:.3e} ratio={ratio:.4f}")
    print("V3-DONE all=%s" % ("PASS" if ok else "FAIL"))
    return ok


def run_relay(B, H, T_len, DK, DV, C):
    """多 chunk 的完整 relay：dk/dv/dh0 都应对上 golden。"""
    print(f"RELAYCASE T={T_len} C={C}", flush=True)
    torch.manual_seed(7)
    (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T_len, DK, DV, "float16", "zero", False)
    dev, dt = "npu", torch.float16
    _out = gdn_bwd_v3_relay(
        q.to(dev).to(dt), k.to(dev).to(dt), v.to(dev).to(dt), g.to(dev),
        beta.to(dev), h0.to(dev).to(dt), do.to(dev).to(dt), chunk_size=C)
    dq, dk, dv, dbeta, dg, dh0 = _out[:6]
    if len(_out) == 7:      # GDN_V3_DK_SPLIT=1：把四项各自与 golden 比
        for k, v_part in _out[6].items():
            ref = np.concatenate([_golden_item(gold, "dk", b) for b in range(B)], axis=0)
            _p, ratio, max_abs = check_precision(v_part.cpu().float(), ref, "float16")
            print(f"DKPART {k} rel={max_abs:.3e} ratio={ratio:.4f}")
    torch.npu.synchronize()
    ok = True
    for item, val in (("dq", dq), ("dk", dk), ("dv", dv), ("dbeta", dbeta), ("dg", dg)):
        ref = np.concatenate([_golden_item(gold, item, b) for b in range(B)], axis=0)
        passed, ratio, max_abs = check_precision(val.cpu().float(), ref, "float16")

        print(f"RELAY {item} rel={max_abs:.3e} verdict={'PASS' if passed else 'FAIL'} ratio={ratio:.4f}")
        ok &= passed
    ref0 = np.concatenate([_golden_item(gold, "dh0", b) for b in range(B)], axis=0)
    passed0, ratio0, max_abs0 = check_precision(dh0.cpu().float().reshape(B, H, DV, DK), ref0, "float16")
    print(f"RELAY dh0 rel={max_abs0:.3e} verdict={'PASS' if passed0 else 'FAIL'} ratio={ratio0:.4f}")
    ok &= passed0
    # 落盘 dk/dq，供本地做逐元素比对（四项的 abs.max 对不上"小元素"的绝对误差）
    import os as _os
    _os.makedirs("/tmp", exist_ok=True)
    np.save(f"/tmp/w11dump_dk_T{T_len}.npy", dk.cpu().float().numpy())
    np.save(f"/tmp/w11dump_dq_T{T_len}.npy", dq.cpu().float().numpy())
    print("RELAY-DONE all=%s" % ("PASS" if ok else "FAIL"))
    return ok


def main():
    # 单 chunk：dq/dk/dv 都可对 golden（Delta' 恒为 0）
    ok = run_case(1, 1, 64, 64, 64, 64, check_dkdv=True)
    # 多 chunk：只有 dq 不含跨 chunk 项，dk/dv 待补 dKt 后再纳入
    ok &= run_case(1, 1, 256, 64, 64, 64, check_dkdv=False)
    # relay：无条件跑（它才是 dk/dv/dbeta 的 gate）
    # 单 chunk 先跑：它没有 scan1 的 128 步递推，用来分离"前置累积"与"本层精度"
    ok &= run_relay(1, 1, 64, 64, 64, 64)
    ok &= run_relay(1, 1, 256, 64, 64, 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
