# _w11_fwd_ab.py — 训练端净收益 A/B（真形档，只量 fwd 状态链）
#
# 问题：GDN 的 fwd 已内核化、bwd 仍走参考实现。训练一步的净收益
#       = (标量 fwd − TA fwd)，因为 bwd 两边都一样。
# 本脚本只量 TA 的 cumsum + solve_a + scan1（fwd 的状态链，等价于 fwd 的
# 大部分搬运/GEMM 成本），与窗五的标量基线（fwd 293 ms / bwd 431 ms）对照。
#
# 判读：
#   fwd_saving = 293 − ta_fwd_ms
#   train_saving_pct = fwd_saving / (293 + 431)      # 一步训练的总占比
#   若 fwd_saving <= 0 ⇒ "TA fwd + eager bwd" 对训练无正收益，训练必须等 bwd。
import sys
import time

sys.path.insert(0, "/tmp")
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

torch.manual_seed(7)

from gdn_bwd_v2 import (  # noqa: E402
    gdn_bwd_v2_cumsum,
    gdn_bwd_v2_scan1,
    gdn_bwd_v2_solve_a,
)

B, H, T, DK, DV, C, BK = 1, 16, 8192, 128, 128, 64, 64
dt = torch.float16
dev = "npu"

k = torch.randn(B, H, T, DK, dtype=dt, device=dev)
v = torch.randn(B, H, T, DV, dtype=dt, device=dev)
g = -torch.rand(B, H, T, dtype=torch.float32, device=dev) * 0.1
beta = torch.rand(B, H, T, dtype=torch.float32, device=dev)
h0 = torch.zeros(B, H, DV, DK, dtype=dt, device=dev)
msk_j = torch.tril(torch.ones(C, C, dtype=torch.float32, device=dev), -1)
idt = torch.eye(C, dtype=torch.float32, device=dev)


def fwd_chain():
    ws_g, _ws_gam = gdn_bwd_v2_cumsum(B, H, T, C)(g)
    ws_a = gdn_bwd_v2_solve_a(B, H, T, DK, C, BK, dtype="float16")(k, beta, ws_g, msk_j, idt)
    ws_h = gdn_bwd_v2_scan1(B, H, T, DK, DV, C, BV, BK, dtype="float16")(
        k, v, beta, ws_g, ws_a, h0)
    return ws_g, ws_h


# warmup (also compiles)
t0 = time.perf_counter()
fwd_chain()
torch.npu.synchronize()
print(f"AB warmup_s={time.perf_counter() - t0:.1f}")

ts = []
for _ in range(3):
    t0 = time.perf_counter()
    fwd_chain()
    torch.npu.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)

med = sorted(ts)[1]
scalar_fwd, scalar_bwd = 293.0, 431.0   # window-5 scalar baseline (gdn fwd/bwd, ms)
saving = scalar_fwd - med
step = scalar_fwd + scalar_bwd
print(f"AB ta_fwd_ms={med:.1f} runs={['%.1f' % x for x in ts]}")
print(f"AB scalar_fwd_ms={scalar_fwd:.0f} scalar_bwd_ms={scalar_bwd:.0f}")
print(f"AB fwd_saving_ms={saving:.1f} train_step_saving_pct={100.0 * saving / step:.2f}")
print(f"AB verdict={'TA_FWD_WINS' if saving > 0 else 'NO_GAIN'}")
print("AB-DONE")
