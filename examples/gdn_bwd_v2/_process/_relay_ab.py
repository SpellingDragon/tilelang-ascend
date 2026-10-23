# _relay_ab.py -- 不依赖 golden 的判定：relay 在 H=N 与"逐 head 各跑 H=1"是否一致
#
# 为什么需要这一支：golden 对比里若我们自己的比较代码有错（索引、dtype、逐 head 拼接），
# 也会表现成"H>1 不准"。这一支把 golden 完全剔掉：同一份数据，
#   A) 一次 H=16 的 relay
#   B) 16 次 H=1 的 relay（每次喂 [1,1,T,*] 的切片）
# 数学上 A 与 B 必须逐元素相同。若不同 ⇒ relay/kernel 自身有 H 相关缺陷，与 golden 无关。
# 若相同而 A 仍与 golden 差 10% ⇒ 那 10% 在 H=1 就存在，问题不在 H（要回头查比较链路）。
#
# 打印：RELAYAB T=.. H=.. item=.. rel=.. max_abs=..   （rel 以 A 的 |out|max 为分母）
import time

import torch
import torch_npu  # noqa: F401

from gdn_bwd_v3 import gdn_bwd_v3_relay

ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")


def _build(H, T_len, B, DK, DV):
    torch.manual_seed(7)
    dt = torch.float16
    rq = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    q32 = rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rk = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    k32 = rk / rk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v32 = torch.randn(B, H, T_len, DV, dtype=torch.float32)
    g32 = -torch.randn(B, H, T_len).abs() * 0.5 - 0.05
    beta32 = torch.sigmoid(torch.randn(B, H, T_len))
    h032 = torch.randn(B, H, DV, DK, dtype=torch.float32) * 0.1
    do32 = torch.randn(B, H, T_len, DV, dtype=torch.float32)
    # dtype 门与 `_realshape_probe.py` 对齐：q/k/v/h0/do=fp16；g/beta=fp32
    return [x.npu() for x in (q32.to(dt), k32.to(dt), v32.to(dt), g32,
                              beta32, h032.to(dt), do32.to(dt))]


def one(H, T_len, B=1, DK=128, DV=128, C=64):
    args = _build(H, T_len, B, DK, DV)

    t0 = time.perf_counter()
    out_full = [x.float().cpu() for x in gdn_bwd_v3_relay(*args, chunk_size=C)[:6]]
    torch.npu.synchronize()
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    per_head = []
    for hh in range(H):
        # 单 head 切片：q/k/v/do/h0 是 4 维、g/beta 是 3 维，都是第 2 维切；
        # JIT 要求入参连续，故显式 .contiguous()
        sl = [x[:, hh:hh + 1].contiguous() for x in args]
        per_head.append([x.float().cpu() for x in gdn_bwd_v3_relay(*sl, chunk_size=C)[:6]])
    torch.npu.synchronize()
    t_heads = time.perf_counter() - t0

    print(f"RELAYAB T={T_len} H={H} chunks={T_len // C} "
          f"t_full={t_full:.2f}s t_heads={t_heads:.2f}s", flush=True)
    worst = 0.0
    for i, n in enumerate(ITEMS):
        den = float(out_full[i].abs().max()) + 1e-30
        num = 0.0
        for hh in range(H):
            a = out_full[i][0, hh].numpy()
            b = per_head[hh][i][0, 0].numpy()
            num = max(num, float(abs(a - b).max()))
        rel = num / den
        worst = max(worst, rel)
        print(f"RELAYAB T={T_len} item={n:<5} rel={rel:.3e} max_abs={num:.3e} "
              f"verdict={'SAME' if rel < 5e-3 else 'DIFF'}", flush=True)
    print(f"RELAYAB T={T_len} worst={worst:.3e} "
          f"verdict={'H_CONSISTENT' if worst < 5e-3 else 'H_DEPENDENT'}", flush=True)
    return worst


def main():
    print("RELAYAB start（A=一次 H=16；B=逐 head H=1；不涉及 golden）", flush=True)
    for T_len in (512,):
        try:
            one(16, T_len)
        except Exception as exc:                            # noqa: BLE001
            print(f"RELAYAB T={T_len} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}", flush=True)
    print("RELAYAB-DONE", flush=True)


if __name__ == "__main__":
    main()
