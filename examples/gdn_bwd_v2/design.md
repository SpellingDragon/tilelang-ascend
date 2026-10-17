# gdn_bwd_v2 算子设计文档（chunked 二版）

> **本文件是 P1-1zj 的设计产出（纯设计波，不含实现）**。v1（per-token adjoint）真机撞 TA 编译器缺陷
> （上游 #1885：j 轴广播收缩 `sum_j A[j,i]*v[j]` 触发芯片异常 507015，判据矩阵见
> `system-one/release/runs/1013-p2-13-oncard-window9-ta-bwd/notes.md`），二版改为 **chunked 伴随分解**。
> v1 目录 `examples/gdn_bwd/` 冻结不动；本设计的数学口径以 `examples_experiment/chunk_gated_delta_rule/`
> 与 fla `chunk_bwd` 分解为权威（与 v1 §1.3.3 同构，逐式核对无冲突）。
>
> **设计期数学实证**：chunked 伴随分解（本文 §3 的 B1–B15）已用 numpy 对拍 v1 golden 逐 token 伴随
> （`examples/gdn_bwd/test_gdn_bwd.py` 的 `_np_fwd_exact/_np_bwd_adjoint` 口径），4 组测试形
> （含尾块 T=130/300）六梯度全对（max rel ≤ 2.6e-15，fwd rel ≤ 1.4e-15），过程件 `/tmp/q6_math_check.py`。

## 三条硬约束（不可妥协，全文逐条遵守）

| # | 约束 | 来历 | 落法 |
|---|------|------|------|
| H1 | **j 收缩全部走 `T.gemm_v0`**（GEMM 域＝窗五 fwd 六件真机 rel=0 实证绿区；禁 vector 广播收缩） | #1885 判据矩阵 #13/#14/#15/#17/#18/#19 | §5 的 j 收缩规避矩阵：所有 `Σ_j` 项要么是 `gemm_v0`，要么是 ones-GEMM（`X @ ones(D,16)`）；无一例外 |
| H2 | **无 per-token 标量系数**（chunk 形态系数为整块 tile op） | v1 踩雷链源头（`g_row[t]`/`beta_row[t]` 逐 token 标量） | 全部系数以 UB 向量/矩阵形态（整块）存在，经 `T.tile.mul` 乘入；唯一 chunk 级标量 `exp(γ_{C-1})` 用编译期常量偏移 `g_ub[C-1]`（`gdn_chunk_h.py:112` 绿形） |
| H3 | **同步硬约束沿用 v1 design §7**：AUTO_SYNC 族全开 + 跨 chunk 递推态只走 GM workspace + 照 fwd 六件先例显式管理跨核 flag | fork G-F3 / M4 证伪 / #1851 对接 | §7 全文沿用，pass_configs 五 True，新增件照 fwd 六件先例显式管理跨核 flag |

另两条 v1 遗留约束：**bf16 不纳入**（裁决③）、**变长 seq_lens 不支持**（裁决④）。

## 1. 概述

### 1.1 算子名称与目标

`gdn_bwd_v2` — GDN（gated delta rule）反向算子，chunked 伴随口径。输入前向张量与上游梯度
`dO`，输出六梯度 `dq, dk, dv, dbeta, dg, dh0`，语义与 v1 golden（fp64 逐 token 伴随）逐项一致。

### 1.2 布局约定

与 v1 golden 同口径（`test_gdn_bwd.py:42-45`）：`S ∈ R^{DV×DK}`，`HIST[h,t] = S_{t-1}`，
`h0/dh0: (B,H,DV,DK)`。fla 参考用 `H ∈ (DK,DV)`（本文的 `S = H^T`），§3 公式已折算到
S 约定；引用 fla 公式处标注 `[fla]`。

### 1.3 语义真值（与 v1 §1.3.1/§1.3.2 完全一致，此处只列要点）

前向逐 token：`S_t = a_t S_{t-1} + u_t k_t^T`，`u_t = β_t (v_t − a_t S_{t-1} k_t)`，
`O_t = S_t q_t`，`a_t = exp(g_t)`。反向为 v1 §1.3.2 的伴随递推（golden `order="ub"` 分支）。
**本设计不改变语义，只改变分解方式**；六梯度对拍口径与 golden 完全复用（§9）。

## 2. chunked 前向分解（反向的 replay 基准）

序列按 chunk `C=64` 切 `NT = ceil(T/C)` 块，块内 token 指标 `r,c ∈ [0,n)`（尾块 `n = T − (NT−1)C`）。
块入口状态 `H_in ∈ R^{DV×DK}`（首块 = `h0`）。以下为块内分解（fla 同构，S 约定）：

```
γ_r   = Σ_{m≤r} g_m                        # cumsum（gdn_chunk_cumsum 绿形）
Γ_rc  = exp(γ_r − γ_c)                     # kkt 系数（gdn_chunk_scaled_dot_kkt 绿形）
KKᵀ_rc = k_r · k_c                          # gram GEMM
L_rc  = β_r · Γ_rc · KKᵀ_rc  (r > c)        # 严格下三角；L = J ⊙ (diag(β)(Γ ⊙ KKᵀ))
A     = (I + L)^{-1}                        # solve_tril 绿形
E_r   = exp(γ_r)                            # ⚠️ cumsum 指数（非 exp(g_r)）
θ_r   = exp(γ_{n-1} − γ_r)                  # K̃ = diag(θ) K
U0    = A (β ⊙ V)                           # 行 r：A 的行 × (β_c v_c)
W     = A (β ⊙ E ⊙ K)
Vnew  = U0 − W H_inᵀ     (= u，逐 token 值)   # (n,DV)
H_out = exp(γ_{n-1}) · H_in + Vnewᵀ K̃       # (DV,DK)，K̃ = diag(θ)K
P_rc  = Γ_rc · M_rc · (q_r · k_c)            # M = causal mask
O     = diag(E)(Q H_inᵀ) + P · Vnew          # (n,DV)
```

**设计期实证**（`/tmp/q6_math_check.py`）：`Vnew == golden u`、`H_out == golden S_{n-1}`、
`O == golden O` 均 rel ≤ 1.4e-15；`(I+L)U = Ē`（`Ē = β⊙V − β⊙E⊙(H_in k)`）逐 c 闭合 1e-16 级。

## 3. chunked 伴随分解（B1–B15，本算子的数学规格）

反向输入：`dO ∈ R^{n×DV}`、`Δ' = ∂L/∂H_out ∈ R^{DV×DK}`（来自下一块；末块 `Δ' = 0`）。
反向输出：`dQ,dK (n,DK)`、`dV (n,DV)`、`dβ,dg (n,)`、`Δ = ∂L/∂H_in`（回传上一块）。
下列每步右侧为实现映射（详见 §5）；`⊙` 为 element-wise，行广播乘标注 `[行广播]`。

| # | 公式（S 约定） | 形态 | 输出 |
|---|---------------|------|------|
| B1 | `dVnew = Pᵀ dO + K̃ Δ'ᵀ` | 2 GEMM（`Pᵀ@dO`、`K̃@Δ'ᵀ`） | dVnew (n,DV) |
| B2 | `dP = dO Vnewᵀ` | 1 GEMM | dP (n,n) |
| B3 | `d(QKᵀ) = dP ⊙ Γ ⊙ M`；`dΓ += dP ⊙ M ⊙ (QKᵀ)` | 2D×2D tile op ×3 | dQKt, dGam |
| B4 | `dQ += d(QKᵀ) K`；`dK += d(QKᵀ)ᵀ Q` | 2 GEMM | dQ, dK |
| B5 | `dE += rowsum(dO ⊙ QH)`；`dQ += (E⊙dO) H_inᵀ`；`dH += (E⊙dO)ᵀ Q` | ones-GEMM + 2 GEMM + [行广播] | dE, dQ, dH |
| B6 | `dU0 = dVnew`；`dW = −dVnew H_in`；`dH += −dVnewᵀ W` | 2 GEMM | dW, dH |
| B7 | `dA += dU0 (β⊙V)ᵀ`；`t1 = Aᵀ dU0`；`dV += β ⊙ t1`；`dβ += rowsum(V ⊙ t1)` | 2 GEMM + [行广播] + ones-GEMM | dA, dV, dβ |
| B8 | `dA += dW (β⊙E⊙K)ᵀ`；`t2 = Aᵀ dW`；`dK += (β⊙E) ⊙ t2`；`d(βE) = rowsum(K ⊙ t2)`；`dβ += E ⊙ d(βE)`；`dE += β ⊙ d(βE)` | 2 GEMM + [行广播] + ones-GEMM | dA, dK, dβ, dE |
| B9 | `dH += exp(γ_{n-1}) · Δ'`；`dK̃ = Vnew Δ'`；`dγ_{n-1} += exp(γ_{n-1})·⟨H_in, Δ'⟩` | chunk 级标量 + 1 GEMM + reduce→[1] | dH, dKt, dγ |
| B10 | `dK += θ ⊙ dK̃`；`dθ = rowsum(K ⊙ dK̃)`；`dγ += −dθ ⊙ θ`；`dγ_{n-1} += Σ_r (dθ ⊙ θ)_r` | [行广播] + ones-GEMM + 1D tile op | dK, dγ |
| B11 | `dL = J ⊙ (−Aᵀ dA Aᵀ)`；`d(KKᵀ) = dL ⊙ β ⊙ Γ`；`dβ += rowsum(dL ⊙ Γ ⊙ KKᵀ)`；`dΓ += dL ⊙ β ⊙ KKᵀ`；`dK += (dX + dXᵀ) K` | 2 GEMM + 2D tile op + ones-GEMM + 1 GEMM | dL, dK, dβ, dΓ |
| B12 | `dγ += rowsum(dΓ ⊙ Γ) − colsum(dΓ ⊙ Γ)` | 2 ones-GEMM | dγ |
| B13 | `dγ += E ⊙ dE` | 1D tile op | dγ |
| B14 | `dg = 后缀和(dγ)`（`dg_m = Σ_{r≥m} dγ_r`） | serial 链式扫描（cumsum 镜像绿形） | dg (n,) |
| B15 | `Δ = dH` 落 GM relay（`WS_D`），`dh0 = Δ`（首块，转回 `(B,H,DV,DK)`） | GM store（`T.Scope("V")`） | Δ, dh0 |

**符号审计备注**：① `E = exp(γ)`（cumsum 指数）——B5/B8/B13 的 exp 链导数 `dγ += E ⊙ dE`；
② B10 的 `θ = exp(γ_{n-1} − γ)` 导数 `dγ += −dθ ⊙ θ`、`dγ_{n-1} += Σ(dθ ⊙ θ)`；③ B12 的 `dΓ → dγ`
须乘 `Γ`（exp 导数）。三处均为设计期对拍踩过的坑（初版漏乘 exp 导数，`dg` rel 1e+1 量级），实现时**照抄本表**。

**伴随正确性证据**：`/tmp/q6_math_check.py`（`MATH_CHECK: PASS`，4 形 24 项 rel ≤ 2.6e-15）；
等价锚：上游 #1828 `wy_fast_bwd_split` golden 的 dA/dk/dv/dbeta/dg 结构与 B7/B8/B10/B11 同构
（其 `dA_frag@A.T`、`A.T@dA_frag`、`×(−J·exp_gate)`、dg row/col-sum 结构互证；符号/转置差异以本表为准）。

## 4. 六梯度承载矩阵（覆盖断言，逐项有承载步骤）

| 梯度 | 承载步骤 | 形状契约 |
|------|---------|---------|
| `dq` | B5（`dQ += (E⊙dO)H_inᵀ`）+ B4（`dQ += d(QKᵀ)K`） | (B,H,T,DK) |
| `dk` | B4（`d(QKᵀ)ᵀQ`）+ B8（`(β⊙E)⊙t2`）+ B10（`θ⊙dK̃`）+ B11（`(dX+dXᵀ)K`） | (B,H,T,DK) |
| `dv` | B7（`β ⊙ t1`） | (B,H,T,DV) |
| `dbeta` | B7（`rowsum(V⊙t1)`）+ B8（`E ⊙ d(βE)`）+ B11（`rowsum(dL⊙Γ⊙KKᵀ)`） | (B,H,T) |
| `dg` | dγ 源＝B5/B3（dΓ⊙Γ）/B12/B8（dE）/B9/B10 → B14 后缀和 | (B,H,T) |
| `dh0` | B15（首块 `Δ`） | (B,H,DV,DK) |

## 5. API 映射与 #1885 规避矩阵（H1 落地账）

### 5.1 j 收缩规避矩阵（逐条：每个 `Σ_j` 的落法与依据）

| 收缩项 | 落法 | #1885 依据 |
|--------|------|-----------|
| `Pᵀ dO`（B1） | `T.gemm_v0(p_l1, do_l1, dvnew_l0, transpose_A=True, init=True)` | GEMM 域＝窗五六件真机绿区；无 Broadcast 展开 |
| `K̃ Δ'ᵀ`（B1） | `T.gemm_v0(kt_l1, dp_l1, dvnew_l0, transpose_B=True, init=False)` | 同上 |
| `dO Vnewᵀ`（B2） | `T.gemm_v0(do_l1, vnew_l1, dp_l0, transpose_B=True, init=True)` | 同上 |
| `d(QKᵀ)K` / `d(QKᵀ)ᵀQ`（B4） | 2× `T.gemm_v0` | 同上 |
| `rowsum(dO⊙QH)`（B5） | ones-GEMM：`T.gemm_v0(t1_l1, ones_l1, de_l0, init=True)`（`ones: (DV,16)` 常量） | 行/列和改写为 GEMM 收缩，保持 GEMM 域（避 #13/#14 广播收缩家族） |
| `(E⊙dO)H_inᵀ`、`(E⊙dO)ᵀQ`（B5） | 2× `T.gemm_v0` | GEMM 域 |
| `dVnew H_in`、`dVnewᵀW`（B6） | 2× `T.gemm_v0` | GEMM 域 |
| `dU0(β⊙V)ᵀ`、`AᵀdU0`（B7） | 2× `T.gemm_v0` | GEMM 域 |
| `rowsum(V⊙t1)`（B7） | ones-GEMM | 同 rowsum |
| `dW(β⊙E⊙K)ᵀ`、`AᵀdW`（B8） | 2× `T.gemm_v0` | GEMM 域 |
| `rowsum(K⊙t2)`、`rowsum(K⊙dK̃)`（B8/B10） | ones-GEMM | 同 rowsum |
| `⟨H_in,Δ'⟩`（B9） | ones-GEMM 两段：`rowsum(Hin⊙Dp)` → `colsum` → `[1]`（`gemm_v0(ones(DK,16))` ×2） | 严格 H1：内积也不出 GEMM 域；备选 `T.reduce_sum(..., dim=(0,1)) → [1]`（判据矩阵 #3 PASS），实现波可任选 |
| `Vnew Δ'`（B9） | `T.gemm_v0(vnew_l1, dp_l1, dkt_l0, init=True)` | GEMM 域 |
| `Σ_r(dθ⊙θ)`（B10） | ones-GEMM：`gemm_v0((dθ⊙θ)_l1, ones(C,16), sum_l0)` → `[1]` | 严格 H1；备选 `T.reduce_sum(dim=0) → [1]`（#3 PASS） |
| `−Aᵀ dA Aᵀ`（B11） | 2× `T.gemm_v0`（`Aᵀ@dA` → `@Aᵀ`） | GEMM 域 |
| `rowsum(dL⊙Γ⊙KKᵀ)`（B11） | ones-GEMM | 同 rowsum |
| `(dX+dXᵀ)K`（B11） | `T.gemm_v0(dsum_l1, k_l1, dk_l0, init=False)` | GEMM 域 |
| `rowsum/colsum(dΓ⊙Γ)`（B12） | 2× ones-GEMM（colsum 用 `onesᵀ @ X`） | 同 rowsum |
| `后缀和(dγ)`（B14） | serial 链式扫描 `dg[i] = dg[i+1] + dgam[i]` | 判据矩阵 #7 serial 密度 PASS；`gdn_chunk_cumsum.py` 同构链式扫描真机绿 |

**无 per-token 标量系数（H2）核对**：B5/B7/B8/B10 的 `⊙β`、`⊙E`、`⊙θ` 全部是 `[行广播]` 的
`T.tile.mul(dst, src2d, coeff1d)` 整块形态（`gdn_wy_fast.py` 的 `k_ub[i,j] *= beta_ub[i]` 绿形；
判据矩阵 #12 纯 i 广播乘 PASS）；**无一** `beta_row[t]` 式逐 token 标量读。B9 的 `exp(γ_{n-1})`
为 chunk 级标量，用 `g_ub[C-1]` 编译期常量偏移（`gdn_chunk_h.py:112` 绿形）。

### 5.2 element-wise tile op 映射（无收缩，全部 2D×2D / 1D×1D / 行广播）

| 步骤 | API |
|------|-----|
| B3 `dQKt = dP⊙Γ⊙M` | `T.tile.mul(t, dp, gam)` → `T.tile.mul(dqkt, t, msk)` |
| B3 `dGam += dP⊙M⊙QKᵀ` | 2× `T.tile.mul` + `T.tile.add` |
| B5/B7/B8/B10 行广播 | `T.tile.mul(dst, src2d, coeff1d)`（`ascend_tile.py:1152` src1 接 BufferLoad） |
| B11 `dL = J⊙(−AᵀdA Aᵀ)` | `T.tile.neg` + `T.tile.mul(dlx, dx, jmask)` |
| B11 `dX + dXᵀ` | `T.tile.transpose(tmp, dXkk)` + `T.tile.add` |
| B13 `dγ += E⊙dE` | 1D `T.tile.mul` + `T.tile.add` |
| `Γ`/`L` 构造（前置） | `T.tile.sub`/`neg`/`exp` 两趟（`gdn_chunk_scaled_dot_kkt.py` 绿形） |

### 5.3 GM 侧（host 约束 §5 合规）

host 侧仅 metadata view（`reshape/permute`）+ kernel 调用 + 结果 reshape；无 `.contiguous()`、
无 padding、无 buffer 内容改写。尾块 `n < C` 用 kernel 内 `valid extent`（`TAIL_MASK=True` 自动掩码），
两侧均不 host-pad。GM 标量 store（`dh0` 标量位）包 `T.Scope("V")`（判据矩阵 #6 PASS）。

### 5.4 实现接口勘误（S7e 收口波，编排者裁决已定）

实现落盘（`gdn_bwd_v2.py`）与设计期描述的接口级偏差，以本节为准：

1. **ones-GEMM 常量 dtype 跟随主 dtype**（错 C 方案 a）：`OnesDV/OnesDK/OnesC: T.Tensor(..., dtype)`
   （fp16 档为 half），wrapper `torch.ones(..., dtype=<主 dtype>)`。依据 `tl_templates/ascend/common.h:1204`
   `gemm_v0<T1,T2,...>(LocalTensor<T1> A, LocalTensor<T1> B, LocalTensor<T2> C)`：A/B 必须同 T1 且位于
   L1（内部 `copy_l1_to_l0a/l0b` 取操作数），C 必须 L0C；ones 常量是 GEMM 的 B 操作数，故与主 dtype 一致。
   §5.1 表中 `ones: (DV,16)` 等常量的 dtype 按本条解读。
2. **B11 `dX+dXᵀ` staging**（错 C 第二处，原孤例 `gemm_v0(dsum_l0, a_l1, dx_l0)`）：将 L0C fp32 累加器
   直接作 A 源违反 common.h 签名（fp16 档 T1 失配；fp32 档属编译可过但非法的 L0C→L0A 搬运）。实现改为
   `dsum` 经 `workspace_lcc → UB → io_cc → workspace_vcc → da_l1` 链搬为主 dtype L1 后
   `gemm_v0(da_l1, a_l1, dx_l0)`，与全 kernel「A/B=主 dtype L1、C=fp32 L0C」纪律一致；fp16 档 dsum
   多一次 half 舍入（与既有 rounding profile 一致，§9 rtol=2⁻⁹ 覆盖）。
3. **scan 拆分两 kernel**（错 B memory-planning KILL-DISPLACED 消解，编排者裁决 2）：
   `gdn_bwd_v2_scan1`（pass1 replay → WS_H）+ `gdn_bwd_v2_scan2`（reverse，读 WS_H/WS_A/WS_G 写六梯度），
   跨 kernel 走 GM workspace。参数表（实现口径；含计划表漏列的 `DO`——B1/B2/B5 必需输入，补录于语义位
   `V` 之后，后续索引顺移 +1）：

   | kernel | 参数（序） | out_idx | workspace_idx |
   |--------|-----------|---------|---------------|
   | scan1 | K,V,Beta,S,A,H0,WS_H + workspace_{vdk,ldk,lcb,vcb,lbd,vbd} | [6]（WS_H） | [7..12]（CV 交接） |
   | scan2 | Q,K,V,DO,Beta,S,A,MskM,MskJ,OnesDV,OnesDK,OnesC,WS_H,WS_D,DQ,DKG,DVG,DBeta,DG,DH0 + workspace_{ldk,lcc,lcb,lbd,lc16,vdk,vcc,vcb,vbd} | [14..19]（六梯度） | [13,20..28]（WS_D+CV 交接） |

   wrapper 对外仍返 6 元组（`dq,dk,dv,dbeta,dg,dh0`），test 调用面零改动；「test 调用链改两段」落在
   wrapper 内部两段 kernel 链（scan1 → ws_h → scan2）。
4. **checklist #11/#12 表述勘误**：实现以 `workspace_idx`（`tilelang.jit` 参数）承载 CV 交接 buffer
   （`workspace_*`）的自动分配，与 AUTO_CV_COMBINE 自动分离并存；#11/#12 的「无 workspace_idx」应读作
   「无手工 vid 偏移」，workspace_idx 本身照 fwd 六件先例使用。
5. **S7f-D/E：跨 chunk 累加箱 GM 化（H3 兑现）+ reverse scan 二拆**（编排者裁决 D/E，S7f 波）：
   - **D**：reverse scan 的跨 chunk 逐块累加梯度中间箱全部挪 GM workspace 累加（dA 记录 WS_DA、dGam
     记录 WS_DG2/WS_DGAM2、dH 半箱经 WS_D relay）——H3「跨 chunk 递推态只走 GM workspace」的合规修正
     （放 UB 是实现期违规），沿 WS_D `:750` 直拷先例形态。
   - **E**（条件触发已命中）：D 后 scan2 UB 静态和仍 983.6KiB ≫ 192KiB（`_s7f_ub_budget.py` 实测）→
     reverse scan 二拆为 `gdn_bwd_v2_scan2a`（B1-B6 + B3/B4 + B9 + B15）与 `gdn_bwd_v2_scan2b`（B7/B8 +
     B10-B15），与 scan1/scan2 拆分同构；跨段经 7 个 [B,H,L,*] 形态 GM 边界箱（WS_DV/WS_DW/WS_DKT/
     WS_DE/WS_DGAM2/WS_DGAM/WS_DK）+ WS_DA；DK 宽输出按 BK=64 半分块（`init=(hh==0)` 半积）。本条
     取代条目 3 的 scan2 参数表（对外 6 元组不变，test 调用面零改动）。
   - **容量**：UB 复算 scan2a 182.8KiB / scan2b 191.1KiB 均 ≤192KiB（修复前 scan2 983.6KiB）；真形档
     GM 增量 +225.0MiB（边界箱）→ workspace 总计 307.5MiB（仍 GM-only 状态，H3 完整）。
   - **收口状态（S7g 选项 A 已落，编译门 20/20 全绿）**：S7f 的 5 站点 sync 失配已按「一箱一写一读或
     严格 W-R 交替」纪律全部重排（甲类读侧对称化 ×3 站 + 乙类 vbd 整消/w 保活合并），`_s7f_sync_audit.py`
     复跑 SYNC-AUDIT-ALL-OK。附带修平三类编译阻塞（均 S7f 时代已埋、被 CombineCV 墙挡住）：ones-GEMM
     半宽 ones（`onesdk_l1[BK,16]`，dbe 逐半累加语义不变）；io_* 过路箱单箱化（fp32 档 UB 容量）；scan2b
     13 站 `T.Parallel` 跨维引用改嵌套 `T.serial`（broadcast 物化临时 `*_broadcast_*_data` 消除，逐元素
     语义等价）。UB 复算 scan2a 158.8KiB(fp16)/166.8KiB(fp32 上界)、scan2b 175.1/183.1 ≤192KiB。
     golden 双门复跑绿（adjoint-vs-FD 6.2e-10 / chunk-math 2.8e-15）。**错 B 动态终验**（`_s7d_mp_model.py`
     六档 scan2a/scan2b × L0-fp16/TSC-fp16/TSC-fp32）：modelA-faithful 六档全 OK（最高 195936/196352），
     错 B 病理（ALLOC FAIL）消除，与编译门 20/20 互证；但 KILL-DISPLACED 机制痕迹仍在（小箱 kill 被
     ReorderKillPoints/ExtendKillIndex 后移，属 C++ 保守释放设计），「形态不再出现」字面判据未达成，
     建议判据修正为「modelA 六档 OK + 编译门全绿」待裁。真机数值窗（含错 B 运行时面）留上卡验收窗。

## 6. fwd 六件复用表（窗五真机 rel=0 事实标准，`examples/linear_attention_and_rnn/gdn/`）

| 件 | fwd 功能（绿形要点） | bwd 复用面 |
|----|---------------------|-----------|
| `gdn_chunk_cumsum.py` | `s_ub[i]=s_ub[i-1]+g_ub[i]` serial 链式扫描 | **直接复用**（前置 `γ=cumsum(g)`）；B14 后缀和镜像此形态 |
| `gdn_chunk_scaled_dot_kkt.py` | `gemm_v0(k_l1,k_l1,a_l0,transpose_B=True)` + `coeff=g_ub[j]-g_ub[i]` 两趟 exp | **直接复用**（前置 `Γ⊙KKᵀ`） |
| `gdn_solve_tril.py` | 前代 `mul_ub[j,k]=o_ub[j,k]*o_ub[i,j]` + `reduce_sum` | **直接复用**（对 `C` 个 RHS 前代解出 `A=(I+L)^{-1}` 落 `WS_A`） |
| `gdn_wy_fast.py` | `gemm_v0`×2 + `k_ub[i,j]*=beta_ub[i]` 行广播乘 | **形态复用**：fwd replay 趟的 `U0/W` 段照抄其 GEMM+广播乘结构 |
| `gdn_chunk_h.py` | `s_ub[i,j]*=g_ub[C-1]` chunk 级标量 + 状态递推 | **形态复用**：`H_out` 递推段同构；⚠️ 其 `fill 0.0` 不支持 `h0≠0`，bwd 的 `H_in` 一律走 GM 读（含 h0） |
| `gdn_chunk_o.py` | 三 GEMM（QKᵀ/QSᵀ/QKV）+ `qk_ub[i,j]*=coeff_ub[i,j]*=msk_ub[i,j]` | **形态复用**：bwd replay 的 `P`/`QH` 段照抄；伴随 `dP/dQKt` 为新增 |

> 上游迁移 PR 侦察（GitHub API / gh CLI，2026-10-14）：#1802 `chunk_o_bwd`（OPEN）、#1798
> `chunk_delta_bwd`（**CLOSED 未合并**）、#1828 `wy_fast_bwd_split`（OPEN）。三件数学覆盖面与
> B1–B15 同构，但**契约错位**（bf16-only 入口 vs 本方 fp32 L0 两阶段、#1802 `dg` 输出需 host sum
> merge 违反 host 约束 §5、禁 tail chunk、布局/符号约定差异、内部广播收缩形态未经我方 #1885 复验），
> **不采用适配层方案**；其 golden 结构留作 §3 的互证锚（差异以本设计为准）。

## 7. 同步策略（硬约束节，全文沿用 v1 design §7）

**模式：全自动同步**，pass_configs 五 True，与 v1 §7.1 逐字一致：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC:    True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC:       True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK:       True,
}
```

三条不可协商条款（H3）：
1. **AUTO_SYNC 族全开**（含 `AUTO_SYNC_VS` 配对 `cce_auto_sync=False`，#1346 保留语义）。
2. **跨 chunk 递推态只走 GM workspace**：`H_in/H_out` 链（`WS_H`）、`Δ` relay（`WS_D`）落地 GM
   后再读回；禁止同一 UB buffer 跨 trip 就地读写（G-F3 根因约束）。
3. **照 fwd 六件先例显式管理跨核 flag**（`T.set_cross_flag/wait_cross_flag` 配对显式管理）；
   `T.copy` 依赖由 pass 推导。
   > **勘误（S7c，编排者已裁）**：原条款为「新增件禁手写 cross flag」，按 examples 优先原则改为
   > 「照 fwd 六件先例显式管理跨核 flag」（真机绿先例：linear_attention_causal.py 的
   > `set_cross_flag("MTE3"/"FIX", id)` / `wait_cross_flag(id)` 配对形态）。
reverse scan 迭代间 `Δ` 落 `WS_D` 后下一 trip 读回（#1851 跨 loop 生命周期同步依赖保留机制对接）。

## 8. GM workspace 规格与真形档容量复算（MB 级，替代 v1 的 16GiB）

### 8.1 workspace 清单（跨 chunk 递推态 + 前置产物，全部 GM）

| 张量 | 形状 | dtype | 用途 |
|------|------|-------|------|
| `WS_H` | `(B,H,NT+1,DV,DK)` | fp16 | fwd replay 的 `H_in/H_out` 链（bwd 趟读入口状态） |
| `WS_A` | `(B,H,NT,C,C)` | fp16 | 前置 solve 的 `A=(I+L)^{-1}`（bwd 的 dA 链） |
| `WS_G` | `(B,H,T)` | fp32 | 前置 cumsum 的 `γ`（B5/B8/B9/B10/B13 的 `E/θ/g_l` 源） |
| `WS_D` | `(B,H,DV,DK)` | fp32 | reverse scan 的 `Δ` relay（硬约束 H3-2） |

**不存** `Vnew/P/QH`（bwd 趟内以 2–3 次 GEMM 重算，换 32MiB+ 空间；fwd replay 趟仅产 `WS_H`）。

### 8.2 真形档复算（B1·H16·DK128·DV128·C64，可复算算式）

`per-chunk 元素价：DK·DV = 128·128 = 16384 elem`

| 张量 | 算式 | L=4096（NT=64） | L=8192（NT=128） |
|------|------|----------------|-----------------|
| `WS_H` | `B·H·(NT+1)·DK·DV·2B` = `1·16·65·16384·2B = 34,078,720B` / `1·16·129·16384·2B = 67,633,152B` | **32.50 MiB**（主体 NT·512KiB=32MiB + 一块） | **64.50 MiB** |
| `WS_A` | `B·H·NT·C·C·2B` = `1·16·NT·64·64·2B` = `NT·128KiB` | **8.0 MiB** | **16.0 MiB** |
| `WS_G` | `B·H·T·4B` = `1·16·4096·4B` / `1·16·8192·4B` | **0.25 MiB** | **0.5 MiB** |
| `WS_D` | `B·H·DK·DV·4B` = `1·16·128·128·4B` | **1.0 MiB** | **1.0 MiB** |
| **合计** | | **41.75 MiB** | **82.00 MiB** |

> 单位口径 MiB（2²⁰ B）；复算式逐项可验（`python3 -c "print(1*16*65*128*128*2/1048576)"` → `32.5`）。

对照 v1（`examples/gdn_bwd/gdn_bwd.py` 的 `SWS (B,H,L,DV,DK)` + `LAM (B,H,L+1,DV,DK)`，均 fp32）：
`2·B·H·L·DK·DV·4B = 2·1·16·8192·128·128·4B = 17,179,869,184B = 16.0 GiB @L8192`（L4096 = 8.0 GiB，不可行）。
二版 **82.00 MiB @L8192**（−200×），且随 `L` 线性项仅 `WS_H/WS_A/WS_G`。

### 8.3 UB 预算（典型 `C=64, DK=128, DV=128`，BV 分块）

L0C 约束 `block_M·block_N·4B ≤ 128KiB ⇒ ≤16384`：`DK·DV = 16384` **卡界** ⇒ `V` 按 `BV=64` 分块
（v1 §5.3 同款）。UB 主力缓冲（fp32）：`A(64·64)=16KiB`、`dA=16KiB`、`Vnew(64·128)=32KiB`、
`dO(64·128)=32KiB`、`P/dP(64·64)=16KiB×2`、`dQKt=16KiB`、`t1/t2(64·128)×2=64KiB`、
系数件 `Γ/β/E/θ/M/J` ≈ 24KiB —— 合计 ≈ 232KiB（fp32）/ ≈ 116KiB（fp16 staging），< UB 容量（192KiB/核），
fp32 档需按 `BV=64` 复用两片 32KiB 缓冲（实现波核验，见 §12）。

## 9. 验证方案（沿用 v1 §9，golden 与 chunk 化无关）

三级锚照搬 v1 §9.1（fp64 解析递推 `_np_bwd_adjoint` / 中心差分 `gradcheck_fd` / torch autograd）；
六项逐项判（`dq/dk/dv/dbeta/dg/dh0` + `adjoint`，不早退）；判据行 `ITEM <name> rel= <v> verdict=`；
L0 门槛四用例（`l0_basic/l0_typical/l0_h0/l0_special`）与 NaN/Inf 位置敏感比较照搬 v1 §9.2；
精度表（dtype= `[float16, float32]`）照搬 v1 §9.3：fp16 `atol=2⁻¹⁴, rtol=2⁻⁹, max_abs=1e-1, ratio=0.99`；
fp32 `atol=2⁻¹⁶, rtol=2⁻¹⁰, max_abs=1e-2, ratio=0.99`；`adjoint` 锚 `rel ≤ 1e-8`（TOL_ADJ）。
**六梯度 fp32 目标 `rel ≤ 2e-4`（TOL_F32）**，dbeta 为 fork 唯一败项（窗十 P1-1zi 真判决），重点盯防。

### 9.1 性能哨兵（checklist #22，不可因 large/L1 跳过）

| 用例 | Shape | 路径 | timeout |
|------|-------|------|---------|
| `perf_large` | `(16,16,16384,128,128)` C=64 | 最大 DMA/最多 chunk/六梯度全写 | 300s |
| `perf_crosschunk` | `(1,1,8192,128,128)` C=64 | reverse scan 最长（128 chunk）+ workspace 82.0MiB 档 | 120s |
| `perf_vblock` | `(1,2,2048,128,128)` BV=64 | L0C 卡界路径 | 120s |

计时纪律与「先数值自证后计时」硬闸沿用 v1 §9.4；**本波不出性能数值承诺**（待真机标定）。

## 10. 编程模式选型

**结论：混合模式 —— Expert 内存层级（显式 `T.alloc_L1/ub/L0C` + `T.gemm_v0`）+ Developer 自动同步（pass_configs 五 True，零手写 barrier）**，与 v1 §2/§7 同款结论。

理由：
1. 含 GEMM 收缩（H1 要求全部走 `gemm_v0`）⇒ 必须显式管理 L0A/L0B/L0C 与 BV 分块（Expert 面）；
2. 3 复用前置 kernel（cumsum/kkt/solve_tril）与 fwd 六件同构，六件本身即此混合形态（显式 buffer + `gemm_v0` + 自动同步）——真机 rel=0 绿区正是这一组合；
3. 同步硬约束 H3 指定 AUTO_SYNC 全开 + 照 fwd 六件先例显式管理跨核 flag ⇒ 同步面走 Developer 自动推导、跨核 flag 照先例显式配对；
4. threads 拆分：`gemm_v0` 落 Cube 侧、`T.tile.*` 落 Vector 侧，由 `AUTO_CV_COMBINE` 自动分离（消 vid 偏移，checklist #12 无 workspace_idx/vid 手工偏移）。

## 11. 数据重排性能可行性（checklist #21，逐路径量化）

单位：真形档 B1·H16·C64·DK128·DV128（fp16 输入 / fp32 accumulate）。

| 结构路径 | GM pass | DMA transaction（平均字节） | GM 标量访问 | 地址 div/mod | 有效并行度 |
|---------|---------|------------------------------|-------------|--------------|------------|
| 输入读入（q,k,v,β,g,dO） | 1 读 | 6 张量 × NT 次块拷贝，每次 ≤ `C·128·2B=16KiB`（q/k/dO 级），β/g `C·4B=256B` | 0 | `chunk_id·C`（编译期 C=64 ⇒ 移位） | NT 次串行 trip（B·H 并行于 task 轴） |
| `WS_H` 写+读 | 1 写 + 1 读 | 每 chunk `DV·DK·2B=32KiB` ×(NT+1) | 0 | `chunk_id·16384`（常量步长） | 双趟各 NT trip |
| `WS_A` 写+读 | 1 写 + 1 读 | 每 chunk `C·C·2B=8KiB` ×NT | 0 | `chunk_id·4096` | 前置 kernel + bwd 趟 |
| `WS_G` 写+读 | 1 写 + 1 读 | `T·4B` 整块（≤32KiB） | 0 | 连续 | 1 次 |
| `WS_D` Δ relay | NT 次写 + NT 次读 | 每 chunk `DV·DK·4B=64KiB` | **0**（GM 标量零访问——H2/H3 达标） | 连续（单 tensor 复用） | reverse scan 串行（语义决定） |
| 梯度写回（dq,dk,dv,dbeta,dg,dh0） | 1 写 | 块内连续 `C·128·2B=16KiB` 级 ×6 张量 | 0 | 同输入 | 同 trip 内直写 |

- **无逐元素 strided GM 主路径**：所有 GM 访问均为块级连续拷贝（`T.copy` region，块长 C 对齐）；
- **GM 标量访问 = 0**（对照 v1 的 per-token 标量读是踩雷链源头）；`dh0` 标量位仅尾部一次 `Scope("V")` store；
- 地址计算全部为编译期常量步长乘 `chunk_id`（无运行期 div/mod）；
- 有效并行度：`(B·H)` task 轴并行 + 每 trip 内 BV=64 双缓冲；reverse scan 串行链为语义下界（不可并行化）。

## 12. 技术约束确认（checklist #8）

1. **三维 kernel**：`(b, h)` 合并为单 task 轴 `bh = b·H + h`（H 编译期常量档）；chunk 轴 `T.serial(NT)` 双趟（fwd replay 正序 + bwd 逆序），NT 动态 ⇒ `NT_MAX` 编译期 + `if chunk < NT` 保护。
2. **尾块**：`n = T − (NT−1)C < C` 时用 `TAIL_MASK=True` 自动掩码 + kernel 内 valid extent；`C=64` 固定（fwd 六件同款）。
3. **L0C 卡界**：`DK·DV=128·128=16384 elem = 128KiB` 恰卡界 ⇒ `BV=64` 分块两趟（v1 §5.3 同款结论）。
4. **动态轴**：`T` 符号维（`B,H,DK,DV` 静态档）；`NT=ceil(T/64)`；workspace 按 `T` 上界分配（`WS_H` 需 `NT_MAX+1` 档）。
5. **dtype**：输入 fp16（主干）/fp32（对拍），accum fp32；bf16 不纳入（裁决③）；变长 `seq_lens` 不支持（裁决④）。
6. **h0 非零**原生支持（`H_in` 首块从 GM `h0` 读；fwd `gdn_chunk_h.py` 的 `fill 0.0` 缺口在本设计中不存在）。

## 13. 本项目同类实现参考（checklist #13/#16/#17）

| 参考 | 路径 | 复用/差异 |
|------|------|-----------|
| fwd 六件（主参考） | `examples/linear_attention_and_rnn/gdn/gdn_{chunk_cumsum,chunk_scaled_dot_kkt,solve_tril,wy_fast,chunk_h,chunk_o}.py` | §6 复用表；输出形状 `(B,H,T,*)`，无需 transpose |
| v1 bwd（冻结） | `examples/gdn_bwd/gdn_bwd.py` + `test_gdn_bwd.py` | golden 口径直接复用（§9）；workspace/分解方式弃用 |
| chunked fwd 参考 | `examples_experiment/chunk_gated_delta_rule/design.md` | §2 数学口径权威；其 H∈(DK,DV) 与本设计 S∈(DV,DK) 差转置（§1.2 已折算） |
| 上游 PR | tile-ai/tilelang-ascend #1802/#1798/#1828 | §6 备注：数学同构、契约错位，不采用；golden 结构互证 |

## 14. proto.yaml

同目录 `proto.yaml`（接口规格），`inputs[].dtype` 全集 `[float16, float32]` 与 §9 精度表一致；
`attrs` 覆盖 `C`（chunk size，固定 64）与 `NT_MAX`；输出六梯度形状按 §4 承载矩阵。

## 15. 风险点

1. **实现波的 UB 预算核验**（§8.3 fp32 档 ≈232KiB 超单核 UB，需 BV=64 双缓冲复用落地后实测；若仍超，退 fp16 staging 全线）。
2. **#1885 复发面**：全部 j 收缩主路径已关进 GEMM 域（§5.1）；残余依赖判据矩阵的面为 serial 链式扫描
   （B14，#7 PASS）与（若实现波选备选形态）`reduce_sum→[1]`（#3 PASS）。上卡首验若复发，退路：B14 改
   `T.copy`+逐元素加扫描、内积全走 ones-GEMM（已在主路径）。
3. **dbeta 病灶**：fork 侧 dbeta 单项败（窗十 8.002e-01）与 TA 侧 #1885 是两个独立缺陷；本设计的 dbeta
   三链（B7/B8/B11）全在 GEMM 域，但真机首验需优先盯 dbeta。
4. **workspace 随 T 线性**：`WS_H/WS_A` 需 `NT_MAX` 上界档；超长序列（T>16384）需分段调度（本波不展开）。

## 16. quality-checklist 自检表（`tilelang-op-design/references/quality-checklist.md`）

| # | 检查项 | 判定 | 证据 |
|---|--------|------|------|
| 0 | Host 侧 Buffer 操作合规 | ✅ | §5.3：仅 metadata view + kernel 调用 + 结果 reshape，无 contiguous/padding/内容改写 |
| 1 | 编程模式明确结论 | ✅ | §10：混合模式（Expert 内存 + Developer 同步）+ 理由 4 条 |
| 2 | API 映射具体到函数名参数 | ✅ | §5.1/§5.2：`T.gemm_v0(...)` 签名 + `T.tile.*` 逐条 |
| 3 | 内存搬运路径完整 | ✅ | §11 表：GM pass/DMA/标量/div-mod/并行度逐路径 |
| 4 | Tiling 约束分析 | ✅ | §8.3/§12-3：L0C 128KiB 卡界 ⇒ BV=64 |
| 5 | 同步策略与模式匹配 | ✅ | §7：AUTO_SYNC 全开 + H3 三条款 |
| 6 | Golden + L0 门槛测试计划 | ✅ | §9：三级锚 + L0 四用例；L1/L2 交 test-design |
| 7 | 无占位符/模糊描述 | ✅ | 全文无 TODO/待补充（实现波核验项已标注归属） |
| 8 | 技术约束已确认 | ✅ | §12 六条 |
| 9 | GEMM 分形限制 | ✅ | block 64×64 ≥ 16×16；BV=64 |
| 10 | L0C 容量验证 | ✅ | §8.3：64·64·4B=16KiB/块，(64)·128·4B=32KiB ≤ 128KiB |
| 11 | CV 融合场景 | ✅ | 混合模式：AUTO_CV_COMBINE 自动分离，无 workspace_idx（Developer 面）；显式 workspace 为 GM 跨 chunk 态（Expert 面）§8（S7e 勘误见 §5.4-4） |
| 12 | workspace_idx/vid 一致性 | ✅ | 无 workspace_idx 参数；无手工 vid 偏移（AUTO_CV_COMBINE 消 vid）（S7e 勘误见 §5.4-4） |
| 13 | 同类实现已列出 | ✅ | §13 表 |
| 14 | 参考实现差异说明 | ✅ | §6/§13：上游 PR 契约错位 6 条 |
| 15 | 参考实现分析完整 | ✅ | §6 复用表（内存层级/同步/pass_configs） |
| 16 | 参考实现标注原算子路径 | ✅ | §13 路径列 |
| 17 | 参考实现输出形状 transpose | ✅ | §1.2/§13：S=H^T 折算说明 |
| 18 | 函数无全局变量依赖 | ✅ | §12-4：B/H/NT 从 shape/参数取 |
| 19 | 精度标准已定义 | ✅ | §9 精度表（fp16/fp32 各行 + adjoint 锚） |
| 20 | proto.yaml 已产出 | ✅ | 同目录 `proto.yaml`，dtype 与 §9 一致 |
| 21 | 数据重排性能可行性量化 | ✅ | §11 |
| 22 | 最大/关键 case 超时门禁 | ✅ | §9.1 三哨兵（300s/120s/120s） |

**通过条件满足**：0 项 + 必须项（1,2,3,7,8,9,13,19,20,21,22）全过，推荐项 9/9 ≥ 4。

## 17. 波次边界

本波纯设计：不写 kernel 实现（下一波）、不出性能数值承诺（待真机）、不做 bf16/变长（裁决③④）。
§2/§3 数学与 `chunk_gated_delta_rule` 参考逐式核对**无冲突**（fla H∈(DK,DV) ↔ 本设计 S∈(DV,DK)
转置折算），golden 对拍实证 `MATH_CHECK: PASS`。
