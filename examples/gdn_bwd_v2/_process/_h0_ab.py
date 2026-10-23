# _h0_ab.py -- 同一条比较路径下，只改 h0（0 vs 0.1·randn），钉住"10% 是 h0 带的还是探针自己的"
#
# 事实支撑：
#   * `test_gdn_bwd_v3.run_relay` → `_case_inputs(..., h0_mode="zero")`，实测六项 ~1e-3；
#   * 我的三支探针（`_realshape_acc` / `_relay_attr` / `_ws_cmp`）全部用 h0 = 0.1·randn，
#     且**每一档**都是 5%~11%（含 H=1、含 4 chunk），golden H_in 也救不回来。
#   * 两边 golden 实现**逐字相同**（`example_gdn_bwd._np_chunk_fwd` vs
#     `test_gdn_bwd_v2._np_chunk_fwd`，ast 取出后 diff 0 行），所以不是 golden 漂移。
#
# 既然如此，就用**测试自己的输入与比较函数**跑两遍，只改 h0：
#   ① `_case_inputs(..., h0_mode="zero")`  —— 与 `v3t` 完全一致，应当复现 ~1e-3
#   ② 同批输入但 h0 换成 0.1·randn          —— 若跳到 ~1e-1，则 h0 路径确实是差的那个
#
# 两条都走 `_golden_item` + 同一个 rel 定义，排除"我的比较代码写错"。
#
# 打印：H0AB mode=zero|randn item=.. rel=.. max_abs=..
import numpy as np
import torch
import torch_npu  # noqa: F401

from gdn_bwd_v3 import gdn_bwd_v3_relay
from test_gdn_bwd_v2 import _case_inputs, _golden_item

ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
GATE = 5e-3


def run(h0_mode, T_len=64, B=1, H=1, DK=64, DV=64, C=64):
    # 与 run_relay 完全同参（fp16 / zero / 无特殊值）；h0_mode="randn" 时下面再替换
    (q, k, v, g, beta, h0, do), gold = _case_inputs(B, H, T_len, DK, DV, "float16", "zero", False)
    if h0_mode == "randn":
        torch.manual_seed(7)
        h0 = torch.randn(B, H, DV, DK, dtype=torch.float32) * 0.1

    dev, dt = "npu", torch.float16
    out = gdn_bwd_v3_relay(
        q.to(dev).to(dt), k.to(dev).to(dt), v.to(dev).to(dt), g.to(dev),
        beta.to(dev), h0.to(dev).to(dt), do.to(dev).to(dt), chunk_size=C)
    torch.npu.synchronize()

    worst = 0.0
    for i, item in enumerate(ITEMS):
        got = out[i].float().cpu()
        ref = np.concatenate([_golden_item(gold, item, b) for b in range(B)], axis=0)
        gotn = got.numpy()
        if gotn.ndim == 4:              # dh0: [B,H,DV,DK] -> 单 head 的 [DV,DK]
            gotn = gotn[0, 0]
        elif gotn.ndim == 3:            # dq/dk/dv: [B,H,T,*] -> [T,*]
            gotn = gotn[0, 0]
        else:                           # dbeta/dg: [B,H,T] -> [T]
            gotn = gotn[0, 0]
        num = float(np.abs(gotn.astype(np.float64) - ref).max())
        den = float(np.abs(ref).max()) + 1e-30
        rel = num / den
        worst = max(worst, rel)
        print(f"H0AB mode={h0_mode:<5} item={item:<5} rel={rel:.3e} max_abs={num:.3e} "
              f"verdict={'PASS' if rel < GATE else 'FAIL'}", flush=True)
    print(f"H0AB mode={h0_mode:<5} worst={worst:.3e} "
          f"verdict={'ALL_PASS' if worst < GATE else 'SOME_FAIL'}", flush=True)
    return worst


def main():
    print("H0AB start（同 _case_inputs 同 _golden_item，只差 h0）", flush=True)
    for mode in ("zero", "randn"):
        try:
            run(mode)
        except Exception as exc:                            # noqa: BLE001
            print(f"H0AB mode={mode} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}", flush=True)
    print("H0AB-DONE", flush=True)


if __name__ == "__main__":
    main()
