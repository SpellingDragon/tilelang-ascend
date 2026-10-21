# _w11p9_one.py — plan-9b 单 case 快检（L0 basic, fp16）：先确认不再挂死 + dq 落点
import sys

sys.path.insert(0, "/tmp")
import numpy as np  # noqa: E402
import torch  # noqa: E402
# fixed seed: _case_inputs has none, and without it every run draws
# different data -- cross-car comparisons become meaningless (10-21 finding).
torch.manual_seed(7)

try:
    import torch_npu  # noqa: F401,E402
except ImportError:
    pass

from test_gdn_bwd_v2 import (  # noqa: E402
    _ITEMS, _case_inputs, _golden_item, _npu, check_precision,
)
from gdn_bwd_v2 import gdn_bwd_v2  # noqa: E402


def main(dtype="float16", case="basic"):
    shapes = {
        "basic": (1, 1, 64, 64, 64, "zero", False),
        "typical": (1, 2, 256, 64, 64, "zero", False),
        "h0": (1, 2, 256, 64, 64, "on", False),
        # single batch, four chunks: separates per-chunk relay problems from
        # the B*H>1 cid-offset problems that l0_typical (H=2) mixes together.
        "multi": (1, 1, 256, 64, 64, "zero", False),
    }[case]
    B, H, T, DK, DV, h0m, sp = shapes
    (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T, DK, DV, dtype, h0m, sp)
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
        passed, ratio, max_abs = check_precision(got[item], ref, idts[item])
        print(f"ONE[{case}] {item} rel= {max_abs:.3e} verdict= {'PASS' if passed else 'FAIL'} ratio={ratio:.4f}")
        if item == "dq" and T > 64:
            # per-chunk split: a single bad chunk hides inside the global rel,
            # and the reverse scan means chunk 0 is processed last.
            g = np.asarray(got[item].double().numpy(), np.float64).reshape(-1, 64, DK)
            r = np.asarray(ref, np.float64).reshape(-1, 64, DK)
            for ci in range(g.shape[0]):
                d = float(np.abs(g[ci] - r[ci]).max())
                print(f"DQCHUNK[{case}] t{ci:02d} maxdiff={d:.3e} goldmax={float(np.abs(r[ci]).max()):.3e}")
        allok &= passed
    print(f"ONE[{case}]-DONE all={'PASS' if allok else 'FAIL'}")
    return allok


if __name__ == "__main__":
    ok = main(sys.argv[1] if len(sys.argv) > 1 else "float16",
              sys.argv[2] if len(sys.argv) > 2 else "basic")
    sys.exit(0 if ok else 1)
