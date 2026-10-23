# _realshape_probe.py -- 真形档可达性探针（只测"能否跑通 + 耗时 + 峰值内存"，不比精度）
#
# 背景：tasks.md §9.1 的 DoD 要求真形档 B1 H16 T8192 DK128 DV128 跑通并记录与 fwd 的倍率。
# 至今没有任何一车跑过 H>1 / T>256 / DK=DV=128 的组合。
#
# 分档跑（不在第一档就撞 8192）：
#   T=512  （8 chunk）—— 先确认 H>1 与 DK=DV=128 能编能跑
#   T=2048 （32 chunk）—— 128 chunk 的 1/4，测 launch 数是否线性
#   T=8192 （128 chunk）—— DoD 档；若前两档耗时外推已不可接受，本档仍跑但记录结论
#
# 判读：每档打印 ok/秒数/chunk 数；任一档崩或超时即停，不去撞更大的档。
import os
import time

import torch
import torch_npu  # noqa: F401

import tilelang

from gdn_bwd_v3 import gdn_bwd_v3_relay


def one(T_len, B=1, H=16, DK=128, DV=128, C=64):
    torch.manual_seed(7)
    dt = torch.float16
    dev = "npu"
    q = (torch.randn(B, H, T_len, DK, dtype=torch.float32) / DK ** 0.5).to(dt)
    k = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    k = (k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(dt)
    v = torch.randn(B, H, T_len, DV, dtype=dt)
    g = -torch.randn(B, H, T_len).abs() * 0.5 - 0.05
    beta = torch.sigmoid(torch.randn(B, H, T_len))
    h0 = (torch.randn(B, H, DV, DK, dtype=torch.float32) * 0.1).to(dt)
    do = torch.randn(B, H, T_len, DV, dtype=dt)
    args = tuple(x.to(dev) for x in (q, k, v, g, beta, h0, do))

    torch.npu.synchronize()
    t0 = time.perf_counter()
    try:
        out = gdn_bwd_v3_relay(*args, chunk_size=C)
        torch.npu.synchronize()
    except Exception as exc:                              # noqa: BLE001
        dt_s = time.perf_counter() - t0
        print(f"REALSHAPE T={T_len} H={H} DK={DK} DV={DV} chunks={T_len // C} "
              f"RAISED after {dt_s:.1f}s: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}",
              flush=True)
        return False
    dt_s = time.perf_counter() - t0
    six = out[:6]
    finite = all(bool(torch.isfinite(x).all()) for x in six)
    amps = " ".join(f"{x.abs().max():.3e}" for x in six)
    print(f"REALSHAPE T={T_len} H={H} DK={DK} DV={DV} chunks={T_len // C} "
          f"ok=1 finite={int(finite)} sec={dt_s:.1f} absmax=[{amps}]", flush=True)
    return finite


def main():
    tilelang.cache.clear_cache()
    torch.manual_seed(0)
    print(f"REALSHAPE start H=16 DK=128 DV=128 C=64 free={os.environ.get('TASKSET', '-')}", flush=True)
    for T_len in (512, 2048, 8192):
        if not one(T_len):
            print(f"REALSHAPE verdict=STOP_AT_T{T_len}", flush=True)
            print("REALSHAPE-DONE", flush=True)
            return
    print("REALSHAPE verdict=ALL_THREE_OK", flush=True)
    print("REALSHAPE-DONE", flush=True)


if __name__ == "__main__":
    main()
