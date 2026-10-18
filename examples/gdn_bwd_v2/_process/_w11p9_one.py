# _w11p9_one.py — 单 case 快检（L0 的任意一档），六梯度对 golden
#   用法: python _w11p9_one.py [basic|basic32|typical|h0|tail]
#   走 wrapper 路径（与 L0 测试同一入口），每项打印 rel/verdict/ratio。
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    import torch_npu  # noqa: F401,E402
except ImportError:
    pass

from test_gdn_bwd_v2 import (  # noqa: E402
    _ITEMS, _case_inputs, _golden_item, _npu, check_precision,
)
from gdn_bwd_v2 import gdn_bwd_v2  # noqa: E402

CASES = {
    "basic": (1, 1, 64, 64, 64, "float16", "zero", False),
    "basic32": (1, 1, 64, 64, 64, "float32", "zero", False),
    "typical": (1, 2, 256, 64, 64, "float16", "zero", False),   # H=2, 4 chunks
    "h0": (1, 2, 256, 64, 64, "float32", "on", False),
    "tail": (1, 1, 130, 64, 64, "float32", "on", False),
}


def main(case="basic"):
    B, H, T, DK, DV, dtype, h0_mode, special = CASES[case]
    (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T, DK, DV, dtype, h0_mode, special)
    print(f"ONE-CASE {case} B={B} H={H} T={T} dt={dtype} h0={h0_mode}", flush=True)
    dq, dk, dv, dbeta, dg, dh0 = gdn_bwd_v2(
        _npu(q), _npu(k), _npu(v), _npu(g), _npu(beta), _npu(h0), _npu(do))
    torch.npu.synchronize()
    got = {"dq": dq.cpu().float(), "dk": dk.cpu().float(), "dv": dv.cpu().float(),
           "dbeta": dbeta.cpu().float(), "dg": dg.cpu().float(), "dh0": dh0.cpu().float()}
    idts = {"dq": dtype, "dk": dtype, "dv": dtype, "dbeta": "float32",
            "dg": "float32", "dh0": dtype}
    allok = True
    for item in _ITEMS:
        ref = np.concatenate([_golden_item(gold, item, b) for b in range(B)], axis=0)
        if special:
            from test_gdn_bwd_v2 import check_precision_special
            passed, ratio, max_abs = check_precision_special(got[item], ref, idts[item])
        else:
            passed, ratio, max_abs = check_precision(got[item], ref, idts[item])
        print(f"ONE {item} rel= {max_abs:.3e} verdict= {'PASS' if passed else 'FAIL'} ratio={ratio:.4f}",
              flush=True)
        allok &= passed
    print(f"ONE-DONE {case} all={'PASS' if allok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "basic")
