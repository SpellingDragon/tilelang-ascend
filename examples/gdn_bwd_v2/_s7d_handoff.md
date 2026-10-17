# S7d handoff — gdn_bwd_v2 编译门修复（熔断待裁）

状态：**部分完成 / 受阻待裁**。错 A 已修复（编译门 L0-fp32 scan 转绿）；错 B 已定量定因（本文档 §2）；
**第 3 种新形态错（错 C：bisheng gemm_v0 签名失配）触发熔断条款，未修，待裁**（§3）。
预授权拆分（scan→scan1+scan2）**未执行**，理由见 §4 耦合分析，拆分方案见 §5。

---

## 1. 任务对照

| # | 任务 | 状态 |
|---|------|------|
| 1 | 修错 A（exp codegen） | ✅ 完成（fwd 三循环绿形，L0-fp32/TSC 两档已过 codegen） |
| 2 | 错 B 定因（IR 摘录+判定写 handoff） | ✅ 完成（本文档 §2，双模型三档对照闭环） |
| 2b | 错 B 修复（预授权拆 kernel） | ⏸ 未执行——第 3 错熔断 + §4 接口耦合，建议与错 C 一并裁决 |
| 3 | design.md :19/:265 勘误 | ✅ 完成（见 design.md，与 :193 口径一致） |
| 4 | 编译门 4 档全绿 | ❌ 未达：L0-fp16/TSC-fp16 卡错 C；TSC 两档卡错 B（未拆分）；仅 L0-fp32 全绿 |

## 2. 错 B 定因（必须交付项）

**错情（实测）**：`ascend_memory_planning.cc:767` LinearScanAllocator FATAL：
- TSC-fp16：`Memory allocation failed for: dhp_ub required: 32768, new memory available: 5344`
- TSC-fp32：`Memory allocation failed for: io_hv required: 32768, new memory available: 5344`
（S7c 时为 5600；本版新增 zero_ub(256B) 后 5344，差额与新 buffer 一致。）

**定因方法**：`_s7d_mp_model.py` monkeypatch 捕获 AscendMemoryPlanning 入参真 IR（pre-planning），
对 C++（`/tlasc/src/transform/ascend_memory_planning.cc`，存档 `_s7d_mp_src_*.txt`）做忠实 Python 移植
（touch-based linear_seq + LivenessAnalysis + **ReorderKillPoints** + ExtendKillIndex + LinearScanAllocator bump 语义），
与"语句级近似"（原始 kill 点、无 reorder/extend）双模型对照。三档结果：

| case | modelA 忠实（含 reorder+extend） | modelB 语句级（无 reorder/extend） | 实测编译门 |
|---|---|---|---|
| TSC-fp16 | FAIL `dhp_ub required: 32768, new memory available: 5344` | **OK usage=196288/196352** | FAIL 同 modelA **逐字吻合** |
| TSC-fp32 | FAIL `io_hv required: 32768, new memory available: 5344` | **OK usage=196160/196352** | FAIL 同 modelA **逐字吻合** |
| L0-fp16 | OK usage=196288/196352 | OK usage=196288/196352 | 内存门通过（卡在错 C） |

**校准**：modelA 两 TSC 档的 FATAL 文案（buffer 名/required/available 三个字段）与实测一字不差 → 移植可信。

**判定：模型失配 / allocator 建模问题（ReorderKillPoints kill 点错位 + ExtendKillIndex 拉伸），非拆分不足。**
依据：
1. 语句级近似（真实峰值）三档全过（196288/196160/196288 ≤ 196352）——**UB 真实峰值在预算内**；
2. 失败完全由 kill 错位/拉伸造成的虚高区间导致：以下 buffer 的**声称 live 区间右端 > 语句可达范围**（TSC-fp16 interval 表摘录，完整表见 `_s7d_mp_out_tscfp16.txt`）：

```
hin_ub:  [114, 430], size=32768, stmt-reach=[114, 289] KILL-DISPLACED old=289 new=429 genL=1 killL=2 claimed-kill 430 > stmt-reach 289
io_hv1:  [115, 429], size=16384, stmt-reach=[115, 180] KILL-DISPLACED old=180 new=429 genL=1 killL=2 claimed-kill 429 > stmt-reach 180
be_ub:   [122, 430], size=256,   stmt-reach=[122, 368] KILL-DISPLACED old=368 new=429 genL=1 killL=2
glast_ub:[129, 430], size=32,    stmt-reach=[129, 285] KILL-DISPLACED old=285 new=429 genL=1 killL=3
bee_ub:  [135, 430], size=256,   stmt-reach=[135, 324] KILL-DISPLACED old=324 new=429 genL=1 killL=2
k2_ub:   [204, 429], size=32768, stmt-reach=[204, 213] KILL-DISPLACED old=213 new=429 genL=1 killL=2
dgam_ub: [234, 429], size=256,   stmt-reach=[234, 417] KILL-DISPLACED old=417 new=429 genL=1 killL=2
```

即 7 个 buffer 的 kill 点被 ReorderKillPoints 从语句可达末端（如 io_hv1 的 180）后移到 gen_kill_seq 尾（429≈函数尾），
区间横跨整个 pass2（io_hv1 语句可达仅 pass1）；另 g_ub/zero_ub/e_ub/th_ub 等经 ExtendKillIndex 拉到 430。
这些虚高区间在 dhp_ub/io_hv 的分配点（[246,288] / [242,250]）上抬高 bump 占用至 191008B，
available=196352-191008=5344，32768B 的 dhp_ub 无法落位 → FATAL。

**C++ 判据（一手源 `_s7d_mp_src_300_620.txt` / `_s7d_mp_src_620_1059.txt`）**：
- ReorderKillPoints：`kill_level > gen_level` 时擦 kill 点、后移到"下一个 entry level==gen_level-1"处；
  gen_level∈{0,1} 时搜索落空 → kill 推到 gen_kill_seq 尾 = 函数尾（本例 7 个 buffer 全部命中此分支，genL=1）。
- ExtendKillIndex：gen 在循环前且 kill 在循环内 → kill 延到 loop_end marker。
- LinearScanAllocator FATAL 文案的 `available = memory_limit_ - next_new_offset_`（bump 尾空间，非 live-sum）。

## 3. 错 C（第 3 种新形态，熔断待裁）

**错情**：bisheng 期 `no matching function for call to 'gemm_v0'` @ 生成 cpp :353（L0-fp16 scan；日志 `/tmp/s7b_compile.log` 头 4 行）。
```cpp
// /tmp/tmphll4f10q.cpp:353 (L0-fp16)
tl::ascend::gemm_v0<half, float, 64, 16, 64, false, false, 128>(t1_l1[0], onesbv_l1[0], rs_l0[0], ...);
// :216  onesbv_l1 的来源是 float：
tl::ascend::copy_gm_to_l1<float, 64, 16>(onesbv_l1[0], OnesDV[0], 16, 64, 16, true);
```

**根因**：ones-GEMM 的 B 操作数 dtype 与 A 操作数不一致。`tl_templates/ascend/common.h:1204` 的
`gemm_v0(LocalTensor<T1> A, LocalTensor<T1> B, LocalTensor<T2> C, ...)` 要求 **A、B 同 T1**；
fp16 kernel 中 A（t1_l1）为 half，而 ones 系 B 为 float → 实参 `LocalTensor<float>` 对不上 `T1=half`。
fp32 档两侧同为 float，故绿（`gemm_v0<float,float,...>` 34 处全匹配）。

**源码证据（gdn_bwd_v2.py）**：
- `:244-246` `OnesDV/OnesDK/OnesC: T.Tensor([D,16], "float")`——参数显式 "float"，不随 kernel dtype；
- `:285-287` `onesbv_l1/onesdk_l1/onesc_l1 = T.alloc_L1(..., "float")`；
- `:909-911` host wrapper `torch.ones(..., dtype=torch.float32)`。
- 附加病灶：fp16 生成码 :485 孤例 `gemm_v0<float, float, 64, 64, 64, false, true>(dsum_l0[0], a_l1[0], dx_l0[0], ...)`
  —— A 源为 L0C float（dsum_l0）而 B（a_l1）为 half，第二处 dtype 漂移（修 :353 后可能继续爆）。

**影响面**：fp16 全档（L0-fp16 已实证；TSC-fp16 未来同样命中，现被错 B 提前拦截）；fp32 全档不受影响。
**S7c 为何未见**：错 A（Unresolved call tir.exp）在 codegen 期先爆，bisheng 从未被调起；错 A 修复后才暴露。

**候选修复方向（供裁决，均未动手）**：
- (a) ones 参数/L1 副本 dtype 跟随 kernel `dtype`（fp16 档传 fp16 ones；数值恒 1 无精度损失）——动 kernel 签名 + wrapper；
- (b) 放宽 `gemm_v0` helper 支持 A/B 异 dtype（改 C++ 头，影响面大）；
- (c) :485 处 dtype 漂移需单独定因（dsum_l0 当 gemm A 源的形态）。

## 4. 拆分暂缓的理由（与预授权的接口耦合）

预授权拆分（scan→scan1+scan2）在错 B 判定分支（模型失配）已命中，**本可执行**；暂缓原因是与错 C 修复存在接口耦合：
错 C 无论走 (a) 还是 (b)，都会改变 `OnesDV/OnesDK/OnesC` 的 dtype/签名（方向 (a)）或 gemm 调用形态（方向 (b)），
而 Ones 系参数位于 scan 参数表 idx 11-13、落在拆分后的 scan2 参数表中——**先拆分会按旧签名固化 scan1/scan2 接口，错 C 裁决后需返工**。
建议裁决次序：先定错 C 方案 → 再一次性落拆分（§5 已备好设计）。若编排者裁定拆分先行，本设计可直接执行。

## 5. 拆分方案（已设计未落码，预授权路径）

- `gdn_bwd_v2_scan1(B,H,L,DK,DV,C,BV,BK,dtype,accum_dtype)`：pass1 replay → WS_H。
  参数：K,V,Beta,S(≡WS_G),A(≡WS_A),H0,WS_H(out_idx=[6]) + workspace_{vdk,ldk,lcb,vcb,lbd,vbd}(workspace_idx=[7..12])。
  host：`ws_h = gdn_bwd_v2_scan1(...)(k, v, beta, ws_g, ws_a, h0)`。
- `gdn_bwd_v2_scan2(...)`：pass2 reverse adjoint，读 WS_H/WS_A/WS_G 写六梯度。
  参数：Q,K,V,Beta,S,A,MskM,MskJ,OnesDV,OnesDK,OnesC,WS_H(入),WS_D(workspace),DQ,DKG,DVG,DBeta,DG,DH0(out_idx=[13..18]) + 9 个 workspace_*（workspace_idx=[12,19..27]）。
  host：`outs = gdn_bwd_v2_scan2(...)(q, k, v, beta, ws_g, ws_a, msk_m, msk_j, ones_dv, ones_dk, ones_c, ws_h)`。
- host wrapper `gdn_bwd_v2()` 内部两段链路、对外仍返 6 元组 → `test_gdn_bwd_v2.py` 零改动（自裁申报）。
- 跨 kernel 中间态走 GM workspace（H3 原生支持）；拆分后 scan2 仅剩 glast_ub 32B 残余错位（无害）。

## 6. 复跑命令

```bash
# 编译门（S7c 同款）：容器 /tmp/s7b_compile_probe.py，日志 /tmp/s7b_compile.log
docker exec cann910b-l bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
  PYTHONPATH=/tlasc python3 /tmp/s7b_compile_probe.py > /tmp/s7b_compile.log 2>&1; \
  grep -E "S7B-(COMPILE|ALL|FAIL)" /tmp/s7b_compile.log'
# 错 B 定因探针（三档）
docker cp examples/gdn_bwd_v2/_s7d_mp_model.py cann910b-l:/tmp/s7d_mp_model.py
docker exec cann910b-l bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
  PYTHONPATH=/tlasc python3 /tmp/s7d_mp_model.py scan TSC-fp16'
# 质量门
python3 -m py_compile examples/gdn_bwd_v2/gdn_bwd_v2.py
uvx ruff check examples/gdn_bwd_v2/gdn_bwd_v2.py
grep -n "H1\|H2\|H3" examples/gdn_bwd_v2/design.md
# golden 自检（容器，无 NPU 自动跳过 kernel stage）
docker exec cann910b-l bash -lc 'cd /tlasc/examples/gdn_bwd_v2 2>/dev/null; ... test_gdn_bwd_v2.py --level l0'
```

## 7. 过程件清单（前缀 _s7d_，均在本目录）

- `_s7d_mp_model.py` — 错 B 定因探针（忠实移植 + 双模型对照 + interval 表 + IR 摘录）
- `_s7d_mp_out_tscfp16.txt` / `_s7d_mp_out_tscfp32.txt` / `_s7d_mp_out_l0fp16.txt` — 三档定因输出（含完整 interval 表）
- `_s7d_mp_src_1_300.txt` / `_s7d_mp_src_300_620.txt` / `_s7d_mp_src_620_1059.txt` — ascend_memory_planning.cc 判据源存档
- `_s7d_handoff.md` — 本文档
