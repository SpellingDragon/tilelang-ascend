# _realshape_time2.py -- 同形档的**暖机**计时（bwd 与 fwd 同法测，才谈得上倍率）
#
# 为什么另开一支：`_realshape_probe.py` 的 sec 是"首次触达"耗时，里面混着 20 个 v3 kernel
# 与 3 个 v2 kernel 的编译（T=512 那档 267s 里绝大部分是编译）。DoD 要的是"与 fwd 的倍率"，
# 分子分母若一边含编译一边不含，比值无意义。这里对同一 T 连跑 N 次，取 min/median，
# 并且 bwd / fwd 用同一套重复次数、同一份输入规模。
#
# 前提：同一个车里先跑过 `rs`（缓存已热）。单独发本 leg 时第一次迭代仍含编译，看 min 即可。
#
# 打印：REALSHAPE2 T=… bwd=[…] bwd_med=… fwd=[…] fwd_med=… ratio=…
import os
import statistics
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

# 故意不 import tilelang、也不 clear_cache：本次要的就是**热缓存**下的耗时
from gdn_bwd_v3 import gdn_bwd_v3_relay


def _sync():
    torch.npu.synchronize()


def _build(T_len, B, H, DK, DV):
    torch.manual_seed(7)
    dt = torch.float16
    q = F.normalize(torch.randn(B, H, T_len, DK, dtype=torch.float32), dim=-1).to(dt)
    k = F.normalize(torch.randn(B, H, T_len, DK, dtype=torch.float32), dim=-1).to(dt)
    v = torch.randn(B, H, T_len, DV, dtype=torch.float32).to(dt)
    g = F.logsigmoid(torch.randn(B, H, T_len, dtype=torch.float32))
    # beta 留 fp32：kernels 声明 Beta 为 float32（实测 fp16 时报 Buffer dtype mismatch）
    beta = torch.sigmoid(torch.randn(B, H, T_len, dtype=torch.float32))
    h0 = (torch.randn(B, H, DV, DK, dtype=torch.float32) * 0.1).to(dt)
    do = torch.randn(B, H, T_len, DV, dtype=torch.float32).to(dt)
    return tuple(x.npu() for x in (q, k, v, g, beta, h0, do))


def _bwd_once(args, C):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out = gdn_bwd_v3_relay(*args, chunk_size=C)
    torch.npu.synchronize()
    return time.perf_counter() - t0, out


def _fwd_once(args, C, BK, BV):
    from gdn_chunk_cumsum import chunk_cumsum
    from gdn_chunk_h import chunk_h
    from gdn_chunk_o import chunk_o
    from gdn_chunk_scaled_dot_kkt import kkt
    from gdn_solve_tril import solve_tril
    from gdn_wy_fast import wy_fast

    q, k, v, g, beta = args
    # fwd 六段链的 Beta 声明是 fp16（`Beta: T.Tensor([B,H,L], dtype)`，dtype=float16），
    # 而 bwd 侧要 fp32；同一份输入喂两边时必须在这里转，不动 bwd 那份。
    beta = beta.to(torch.float16)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    gc = chunk_cumsum(g, C)
    a = solve_tril(kkt(k, beta, gc, C, BK))
    w, u = wy_fast(k, v, beta, gc, a, C, BK, BV)
    s, nv, _fs = chunk_h(k, w, u, gc, C, BK, BV)
    o = chunk_o(q, k, nv, s, gc, C, BK, BV)
    torch.npu.synchronize()
    return time.perf_counter() - t0, o


def one(T_len, B=1, H=16, DK=128, DV=128, C=64, BK=64, BV=64, reps=3):
    args7 = _build(T_len, B, H, DK, DV)
    q, k, v, g, beta, h0, do = args7

    bt, out = [], None
    for _ in range(reps):
        try:
            el, out = _bwd_once(args7, C)
        except Exception as exc:                          # noqa: BLE001
            print(f"REALSHAPE2 T={T_len} BWD-RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}", flush=True)
            return False
        bt.append(el)
    finite = all(bool(torch.isfinite(x).all()) for x in out[:6])

    ft = None
    try:
        ft = []
        for _ in range(reps):
            el, _o = _fwd_once((q, k, v, g, beta), C, BK, BV)
            ft.append(el)
    except Exception as exc:                              # noqa: BLE001
        print(f"REALSHAPE2 T={T_len} FWD-RAISED {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:110]}", flush=True)
        ft = None

    bs = " ".join(f"{x:.2f}" for x in bt)
    msg = (f"REALSHAPE2 T={T_len} H={H} chunks={T_len // C} "
           f"bwd=[{bs}] bwd_min={min(bt):.2f} bwd_med={statistics.median(bt):.2f} finite={int(finite)}")
    if ft:
        fs = " ".join(f"{x:.2f}" for x in ft)
        med = statistics.median(ft)
        msg += (f" fwd=[{fs}] fwd_min={min(ft):.2f} fwd_med={med:.2f}"
                f" ratio_bwd_over_fwd={statistics.median(bt) / med:.2f}")
    else:
        msg += " fwd=UNAVAILABLE"
    print(msg, flush=True)
    return finite


def main():
    reps = int(os.environ.get("RS_REPS", "3"))
    print(f"REALSHAPE2 start H=16 DK=128 DV=128 C=64 reps={reps}", flush=True)
    for T_len in (512, 2048, 8192):
        if not one(T_len, reps=reps):
            print(f"REALSHAPE2 verdict=STOP_AT_T{T_len}", flush=True)
            print("REALSHAPE2-DONE", flush=True)
            return
    print("REALSHAPE2 verdict=ALL_THREE_OK", flush=True)
    print("REALSHAPE2-DONE", flush=True)


if __name__ == "__main__":
    main()
