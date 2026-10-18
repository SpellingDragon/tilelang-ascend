# _w11p9_step.py — 逐个 kernel 步进 + 每次 synchronize，定位 plan-9b 后哪一段挂死
import sys

sys.path.insert(0, "/tmp")
import torch  # noqa: E402

try:
    import torch_npu  # noqa: F401,E402
except ImportError:
    pass

from test_gdn_bwd_v2 import _case_inputs, _npu  # noqa: E402
from gdn_bwd_v2 import (  # noqa: E402
    gdn_bwd_v2_cumsum, gdn_bwd_v2_scan1, gdn_bwd_v2_scan2a, gdn_bwd_v2_scan2b,
    gdn_bwd_v2_solve_a,
)


def main():
    dt_s = "float16"
    (q, k, v, g, beta, h0, do), _ = _case_inputs(1, 1, 64, 64, 64, dt_s, "zero", False)
    q, k, v, g, beta, h0, do = (_npu(q), _npu(k), _npu(v), _npu(g), _npu(beta), _npu(h0), _npu(do))
    dt = torch.float16
    msk_m = torch.tril(torch.ones(64, 64, dtype=torch.float32)).npu()
    msk_j = torch.tril(torch.ones(64, 64, dtype=torch.float32), -1).npu()
    idt = torch.eye(64, dtype=torch.float32).npu()
    ones_dv = torch.ones(64, 16, dtype=dt).npu()
    ones_dk = torch.ones(64, 16, dtype=dt).npu()
    ones_c = torch.ones(64, 16, dtype=dt).npu()
    print("STEP 0 inputs ready", flush=True)

    ws_g, ws_gam = gdn_bwd_v2_cumsum(1, 1, 64, 64)(g)
    torch.npu.synchronize()
    print("STEP 1 cumsum OK", flush=True)

    ws_a = gdn_bwd_v2_solve_a(1, 1, 64, 64, 64, 64, dtype=dt_s)(k, beta, ws_g, msk_j, idt)
    torch.npu.synchronize()
    print("STEP 2 solve_a OK", flush=True)

    ws_h = gdn_bwd_v2_scan1(1, 1, 64, 64, 64, 64, 64, 64, dtype=dt_s)(k, v, beta, ws_g, ws_a, h0)
    torch.npu.synchronize()
    print("STEP 3 scan1 OK", flush=True)

    outs_a = gdn_bwd_v2_scan2a(1, 1, 64, 64, 64, 64, 64, 64, dtype=dt_s)(
        q, k, v, do, beta, ws_g, ws_a, msk_m, ones_dv, ws_h, ws_gam)
    torch.npu.synchronize()
    print("STEP 4 scan2a OK", flush=True)

    outs_b = gdn_bwd_v2_scan2b(1, 1, 64, 64, 64, 64, 64, 64, dtype=dt_s)(
        k, v, beta, ws_g, ws_a, msk_j, ones_dv, ones_dk, ones_c,
        outs_a[2], outs_a[3], outs_a[4], outs_a[5], outs_a[6], outs_a[7], outs_a[8])
    torch.npu.synchronize()
    print("STEP 5 scan2b OK", flush=True)
    print("STEP-DONE", flush=True)


if __name__ == "__main__":
    main()
