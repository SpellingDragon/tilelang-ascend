# _realshape_acc.py -- 真形档（H=16、DK=DV=128、多 chunk）的**数值**对拍
#
# 已有证据的边界：`_realshape_probe.py` 只证"跑得通 + 六项有限"；六项 rel<5e-3 只在
# H=1 的 T=64/T=256 上测过。H=16 且 DK=DV=128 时六项数值是否还成立，**此前零证据**。
#
# 分母（golden）直接 import example 里的 `_np_chunk_fwd/_np_chunk_bwd`：
# 那是 fp64 分块实现，与 v3 反向同源（notes [91] 的 B1-B15），不另写一份以免漂移。
# 它是**单 head** 的实现，所以这里对 16 个 head 逐个跑再取全局 max_abs。
#
# 口径：rel = max_abs / |golden|max，与 P3 出口一致（阈值 5e-3）。逐项打印，不做聚合。
#
# 打印：REALSHAPE-ACC T=… H=… item=… rel=… max_abs=… gmax=… verdict=PASS|FAIL
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

from gdn_bwd_v3 import gdn_bwd_v3_relay
from example_gdn_bwd import _np_chunk_bwd, _np_chunk_fwd

ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
GATE = 5e-3


def one(T_len, dim=128, H=16, B=1, C=64):
    DK = DV = dim
    torch.manual_seed(7)
    dt = torch.float16
    # 先在 CPU 造 fp32 再 cast 到 fp16：golden 与 kernel 拿到的是**同一批 fp16 位型**
    rq = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    q32 = rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rk = torch.randn(B, H, T_len, DK, dtype=torch.float32)
    k32 = rk / rk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v32 = torch.randn(B, H, T_len, DV, dtype=torch.float32)
    g32 = -torch.randn(B, H, T_len).abs() * 0.5 - 0.05
    beta32 = torch.sigmoid(torch.randn(B, H, T_len))
    h032 = torch.randn(B, H, DV, DK, dtype=torch.float32) * 0.1
    do32 = torch.randn(B, H, T_len, DV, dtype=torch.float32)

    # dtype 门与 `_realshape_probe.py` 逐项对齐（那一支已实测能跑）：
    # q/k/v/h0/do = fp16；g 与 beta = fp32（cumsum 入口声明 G 为 float32）
    cpu = [q32.to(dt), k32.to(dt), v32.to(dt), g32, beta32, h032.to(dt), do32.to(dt)]
    npc = [x.numpy() for x in cpu]                          # golden 吃同一批位型
    gpu = [x.npu() for x in cpu]

    t0 = time.perf_counter()
    out = gdn_bwd_v3_relay(*gpu, chunk_size=C)
    torch.npu.synchronize()
    card_s = time.perf_counter() - t0

    got = [x.float().cpu() for x in out[:6]]

    # V08 硬门：非有限值必须在聚合前逐项抓住。旧路径把逐元素 diff 交给
    # `max(prev, float(nan))` —— Python 的 max 对 nan 比较恒为 False，NaN 被吞掉，
    # 该 case 反而打印出"看起来正常"的 rel（max_abs=0）。这里先逐项
    # `torch.isfinite(...).all()`，非有限直接判该 case FAIL（数值复验待上卡）。
    finite = {n: bool(torch.isfinite(got[i]).all().item()) for i, n in enumerate(ITEMS)}
    for n in ITEMS:
        if not finite[n]:
            print(f"REALSHAPE-ACC NONFINITE item={n} T={T_len}", flush=True)

    qn, kn, vn, gn, betan, h0n, don = npc
    max_abs = {n: 0.0 for n in ITEMS}
    gmax = {n: 0.0 for n in ITEMS}
    tg0 = time.perf_counter()
    for b in range(B):
        for h in range(H):
            _o, cache = _np_chunk_fwd(T_len, DK, DV,
                                      qn[b, h].astype(np.float64),
                                      kn[b, h].astype(np.float64),
                                      vn[b, h].astype(np.float64),
                                      betan[b, h].astype(np.float64),
                                      gn[b, h].astype(np.float64),
                                      h0n[b, h].astype(np.float64), C)
            ref = _np_chunk_bwd(T_len, DK, DV, cache, don[b, h].astype(np.float64))
            for i, n in enumerate(ITEMS):
                r = np.abs(ref[i])
                gmax[n] = max(gmax[n], float(r.max()) if r.size else 0.0)
                diff = np.abs(got[i][b, h].numpy().astype(np.float64) - ref[i])
                max_abs[n] = max(max_abs[n], float(diff.max()) if diff.size else 0.0)
    gold_s = time.perf_counter() - tg0

    print(f"REALSHAPE-ACC T={T_len} H={H} DK={DK} DV={DV} chunks={T_len // C} "
          f"card_s={card_s:.1f} gold_s={gold_s:.1f}", flush=True)
    ok_all = True
    for n in ITEMS:
        # 非有限项不报 rel（nan 不参与比较）；硬门已由上面的 finite 记下
        rel = max_abs[n] / (gmax[n] + 1e-30) if finite[n] else float("nan")
        good = finite[n] and rel < GATE
        ok_all &= good
        print(f"REALSHAPE-ACC T={T_len} item={n:<5} rel={rel:.3e} "
              f"max_abs={max_abs[n]:.3e} gmax={gmax[n]:.3e} "
              f"verdict={'PASS' if good else 'FAIL'}", flush=True)
    # dim=64 是 CONTROL：这里历史上就是 ~1e-3；它不过则本支失效，128 的数字不采信
    tag = "CONTROL" if dim == 64 else "ACC"
    print(f"REALSHAPE-ACC T={T_len} {tag} dim={dim} "
          f"verdict={'ALL_PASS' if ok_all else 'SOME_FAIL'}", flush=True)
    return ok_all


def main():
    all_ok = True
    # CONTROL = 已实测 known-good 的配置：H=1、DK=DV=64、T=256（example 原生路径 4.1e-04~2.4e-03）。
    # T2：`dk` 在 8~16 chunk 有波动（T=512/DK=64 实测 9.97e-03，T=8192 却 4.29e-03），
    # 这里用 H=1 扫一条 T 曲线把它定性（H=1 使 golden 便宜 16×，而 dk 无 head 耦合）。
    for T_len, dim, H in (
        (8192, 128, 16),   # 目标形（B1 H16 T8192 DK128 DV128，128 chunk）
        (256, 64, 1),      # CONTROL
        (128, 128, 1),     # 2 chunk
        (512, 128, 1),     # 8 chunk
        (2048, 128, 1),    # 32 chunk
        (8192, 128, 1),    # 128 chunk（DoD 形）
    ):
        try:
            all_ok &= one(T_len, dim, H)
        except Exception as exc:                            # noqa: BLE001
            print(f"REALSHAPE-ACC T={T_len} dim={dim} H={H} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}", flush=True)
            all_ok = False
            break
    print(f"REALSHAPE-ACC verdict={'ALL_PASS' if all_ok else 'SOME_FAIL'}", flush=True)
    print("REALSHAPE-ACC-DONE", flush=True)


if __name__ == "__main__":
    main()
