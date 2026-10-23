# _realshape_fwd.py -- 同形档的**前向**计时（给 DoD 的"与 fwd 的倍率"提供分母）
#
# tasks.md §9.1 的性能判据是"真形档 B1 H16 T8192 DK128 DV128 跑通并记录与 fwd 的倍率"。
# 分母必须是同一个 operator 家族的前向：`examples/linear_attention_and_rnn/gdn/*`
# 的六段链（cumsum -> kkt -> solve_tril -> wy_fast -> chunk_h -> chunk_o），
# 它和 v3 反向共用同一套 A/W/U/H/S 结构，golden 也同源。
#
# 不 import `gdn_full.py`：那个文件在模块级就跑 ref_seq_gdn（对 L 做 Python 循环），
# L=8192 会先把时间花在 golden 上。这里只复制它的六段调用顺序，逐段计时。
#
# 判读：每档打印总秒数与六段分解；任一档崩即停，不去撞更大的档。
import os
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import tilelang

from gdn_chunk_cumsum import chunk_cumsum
from gdn_chunk_h import chunk_h
from gdn_chunk_o import chunk_o
from gdn_chunk_scaled_dot_kkt import kkt
from gdn_solve_tril import solve_tril
from gdn_wy_fast import wy_fast


def _sync():
    torch.npu.synchronize()


def one(T_len, B=1, H=16, DK=128, DV=128, C=64, BK=64, BV=64):
    torch.manual_seed(7)
    dt = torch.float16
    # 输入全部在 CPU 造好再搬一次，不在 NPU 上随机（与 _realshape_probe.py 同规矩）
    q = F.normalize(torch.randn(B, H, T_len, DK, dtype=torch.float32), dim=-1).to(dt).npu()
    k = F.normalize(torch.randn(B, H, T_len, DK, dtype=torch.float32), dim=-1).to(dt).npu()
    v = torch.randn(B, H, T_len, DV, dtype=torch.float32).to(dt).npu()
    g = F.logsigmoid(torch.randn(B, H, T_len, dtype=torch.float32)).npu()
    beta = torch.sigmoid(torch.randn(B, H, T_len, dtype=torch.float32)).to(dt).npu()

    stage = {}
    _sync()
    t_all = time.perf_counter()
    try:
        _sync(); t0 = time.perf_counter()
        gc = chunk_cumsum(g, C)
        _sync(); stage["cumsum"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        a = kkt(k, beta, gc, C, BK)
        _sync(); stage["kkt"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        a = solve_tril(a)
        _sync(); stage["solve"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        w, u = wy_fast(k, v, beta, gc, a, C, BK, BV)
        _sync(); stage["wy"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        s, nv, _fs = chunk_h(k, w, u, gc, C, BK, BV)
        _sync(); stage["h"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        o = chunk_o(q, k, nv, s, gc, C, BK, BV)
        _sync(); stage["o"] = time.perf_counter() - t0
    except Exception as exc:                              # noqa: BLE001
        dt_s = time.perf_counter() - t_all
        done = ",".join(f"{n}={s:.1f}" for n, s in stage.items()) or "-"
        print(f"REALSHAPE-FWD T={T_len} H={H} DK={DK} DV={DV} chunks={T_len // C} "
              f"RAISED after {dt_s:.1f}s (done: {done}): "
              f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}", flush=True)
        return False
    dt_s = time.perf_counter() - t_all
    amp = float(o.abs().max())
    finite = int(bool(torch.isfinite(o).all()))
    parts = " ".join(f"{n}={s:.2f}" for n, s in stage.items())
    print(f"REALSHAPE-FWD T={T_len} H={H} DK={DK} DV={DV} chunks={T_len // C} "
          f"ok=1 finite={finite} sec={dt_s:.1f} absmax={amp:.3e} [{parts}]", flush=True)
    return finite == 1


def main():
    tilelang.cache.clear_cache()
    torch.manual_seed(0)
    print(f"REALSHAPE-FWD start H=16 DK=128 DV=128 C=64 BK=64 BV=64 "
          f"free={os.environ.get('TASKSET', '-')}", flush=True)
    for T_len in (512, 2048, 8192):
        if not one(T_len):
            print(f"REALSHAPE-FWD verdict=STOP_AT_T{T_len}", flush=True)
            print("REALSHAPE-FWD-DONE", flush=True)
            return
    print("REALSHAPE-FWD verdict=ALL_THREE_OK", flush=True)
    print("REALSHAPE-FWD-DONE", flush=True)


if __name__ == "__main__":
    main()
