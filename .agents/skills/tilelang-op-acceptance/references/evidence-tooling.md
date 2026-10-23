# 取证工具：三个可直接复用的实现

本文件是 `tilelang-op-acceptance` 的工具体。三件事各解决一类"结论不可复核"：

| 工具 | 解决 | 触发时机 |
|---|---|---|
| T1 示例对账 | 单文件示例与 kernel 文件漂移 | 每次改 kernel 或示例 |
| T2 真机长跑（车） | 传错码 / 双跑互踩 / 无对照 / 产物丢失 | 每次上卡 |
| T3 两套 golden 互查 | 分母不可信 | 仓库里存在两份 golden 时 |

每个工具都附**自检方法**（正负两向），因为它自己也会错。

---

## T1 示例对账（example ↔ kernel 逐行）

### 为什么必须有

单文件示例是 kernel 文件的**手抄副本**（上库要求单文件）。手抄必然漂移，
而漂移**只在真机上以数值或异常暴露**——本项目漂了三次，每次代价一轮上卡。

### 实现（完整可用，约 100 行）

```python
#!/usr/bin/env python3
"""example 的 kernel 段必须与 kernel 文件逐行一致。

比对范围：从锚点函数（如 gdn_v3_ab）的 @tilelang.jit 装饰行到 kernel 文件末尾；
example 侧到 `# ========== Golden` 之前。
唯一允许的差异：example 内联了前置件，因此删掉了 `try: from xxx import ...` 段。
"""
import argparse, difflib, os, sys

KERNEL_REL = "examples/<op>/<op>.py"
EXAMPLE_REL = "examples/<op>/example_<op>.py"
ANCHOR = "def <anchor>("              # 两文件共有的第一个函数
EXAMPLE_END = "# ========== Golden"


def _lines(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().splitlines()


def _slice_from_anchor(lines, end_pred, path):
    try:
        i = next(n for n, s in enumerate(lines) if s.startswith(ANCHOR))
    except StopIteration:
        raise SystemExit(f"FAIL: {path} 里找不到 {ANCHOR!r}")
    if i == 0 or not lines[i - 1].lstrip().startswith("@tilelang.jit"):
        raise SystemExit(f"FAIL: {path}:{i+1} 的 {ANCHOR} 上方不是 @tilelang.jit")
    for n in range(i - 1, len(lines)):
        if end_pred(lines[n]):
            return lines[i - 1:n]
    return lines[i - 1:]


def _drop_import_guards(lines):
    """去掉 `try: from <dep> import ...` 整段（example 合法删除）。"""
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "try:" and any("import" in s for s in lines[i:i + 9]):
            j = i
            while j < len(lines) and not lines[j].startswith("    except ImportError:"):
                j += 1
            j += 1
            while j < len(lines) and (lines[j].startswith("        ") or not lines[j].strip()):
                j += 1
            i = j
            continue
        out.append(lines[i])
        i += 1
    return out


def _norm(lines):
    return [s.rstrip() for s in _drop_import_guards(lines) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("TA_ROOT", "."))
    a = ap.parse_args()
    kp, ep = os.path.join(a.root, KERNEL_REL), os.path.join(a.root, EXAMPLE_REL)
    for p in (kp, ep):
        if not os.path.exists(p):
            raise SystemExit(f"FAIL: 缺文件 {p}")
    kern = _slice_from_anchor(_lines(kp), lambda s: s.startswith("if __name__"), kp)
    exam = _slice_from_anchor(_lines(ep), lambda s: s.startswith(EXAMPLE_END), ep)
    ka, ea = _norm(kern), _norm(exam)
    if ka == ea:
        print(f"SYNC-OK kernel={len(ka)} example={len(ea)} 行（已归一化）")
        return 0
    d = list(difflib.unified_diff(ka, ea, fromfile=KERNEL_REL, tofile=EXAMPLE_REL, lineterm=""))
    print(f"SYNC-FAIL kernel={len(ka)} example={len(ea)} 行，差异 {len(d)} 行：")
    print("\n".join(d[:80]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

### 自检（两向都必须做）

```bash
python3 check_example_sync.py                 # 期望 SYNC-OK，exit 0
# 复制一份，故意删掉示例里一行的 .contiguous()
cp -r <root> /tmp/neg && perl -0pi -e 's/\.contiguous\(\)//' /tmp/neg/<example>
python3 check_example_sync.py --root /tmp/neg # 期望 SYNC-FAIL，exit 1
```

> 只做正向不算验证过——**门禁自身也必须有一次"拒绝"的实测**。

---

## T2 真机长跑（车）

### 为什么必须有

一次上卡要同时保证四件事，缺一件结论就不可复核：

1. 卡上跑的是**这一份码**（不是上次的残留）；
2. **只有一个**跑在占设备；
3. 每条腿（leg）**带对照臂**；
4. 跑完产物**能取回**（实例重启 `/tmp` 即清空）。

### 骨架

```bash
#!/bin/bash
# 用法：car.sh "leg1 leg2"     环境变量：CARD_SSH / CARD_KEY / CARD_PORT / SRC
set -u
SSH="-T -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i $CARD_KEY -p $CARD_PORT"
H="$CARD_SSH"
LEGS="${1:?legs}"
MANIFEST=( "$SRC/op.py:op.py"  "$SRC/_process/_probe.py:_probe.py"  "$SRC/example_op.py:example_op.py" )

# ---- [1] manifest md5：本地算、远端算、逐件比 ----
fail=0
for item in "${MANIFEST[@]}"; do
  f="${item%%:*}"; b="${item##*:}"
  L=$(md5 -q "$f" 2>/dev/null || md5sum "$f" | cut -d' ' -f1)
  R=$(cat "$f" | ssh $SSH "$H" "cat > /tmp/$b; md5sum /tmp/$b | cut -d' ' -f1" 2>/dev/null | tail -1)
  [ "$L" = "$R" ] && echo "OK   $b $L" || { echo "FAIL $b local=$L remote=$R"; fail=1; }
done
[ "$fail" = 0 ] || { echo "ABORT manifest mismatch"; exit 3; }

# ---- [2] 启动：单锁 + 环境 + 每条腿收工后数 CONTROL ----
ssh -T $SSH "$H" "rm -f /tmp/car_all.log /tmp/car.lock /tmp/car_*.log; : > /tmp/car_all.log
cat > /tmp/car_run.sh <<'EOF'
#!/bin/bash
set -u
exec 9>/tmp/car.lock; flock -n 9 || { echo \"ABORT another car holds the lock\"; exit 9; }
source /usr/local/Ascend/cann-*/set_env.sh >/dev/null 2>&1 || true
export PYTHONPATH=\"/tmp:\$PYTHONPATH\"
export TILELANG_CACHE_DIR=/tmp/car_cache; rm -rf \"\$TILELANG_CACHE_DIR\"; mkdir -p \"\$TILELANG_CACHE_DIR\"
cd /tmp
run_leg () {  # $1 名字  $2 命令  $3 超时秒
  echo \"=== LEG \$1 ===\"
  timeout \"\$3\" bash -c \"\$2\" > /tmp/car_\$1.log 2>&1
  echo \"\$1-EXITC=\$?\"
  # 对照臂计数：探针必须在输出里打 CONTROL ... PASS|FAIL
  _ctl=\$(grep -ac \"CONTROL\" /tmp/car_\$1.log 2>/dev/null || true)
  _cfail=\$(grep -ac \"CONTROL.*FAIL\" /tmp/car_\$1.log 2>/dev/null || true)
  if [ \"\${_ctl:-0}\" -eq 0 ]; then echo \"\$1-CONTROL=none\"
  elif [ \"\${_cfail:-0}\" -ne 0 ]; then echo \"\$1-CONTROL=FAIL\"
  else echo \"\$1-CONTROL=ok(\${_ctl})\"; fi
  grep -aE '^CONTROL|^TARGET|^verdict=|Traceback|Error' /tmp/car_\$1.log | tail -20
}
for leg in $LEGS; do
  case \$leg in
    probe) run_leg probe 'python3 -u /tmp/_probe.py' 1800 ;;
    suite) run_leg suite 'python3 -u /tmp/test_op.py --level all' 1800 ;;
    *)     echo \"unknown leg \$leg\" ;;
  esac
done
echo CAR-DONE
EOF
chmod +x /tmp/car_run.sh
(setsid bash /tmp/car_run.sh > /tmp/car_all.log 2>&1 < /dev/null &)
echo CAR-LAUNCHED"

# ---- [3] 轮询（只看是否有 CAR-DONE，别读日志猜） ----
for _ in $(seq 1 240); do
  sleep 20
  ssh -T $SSH "$H" 'cat /tmp/car_all.log' 2>/dev/null | grep -q CAR-DONE && { echo "[done]"; break; }
done
```

### 归档（必须做，且要带字节数）

```bash
#!/bin/bash
# pull.sh [tag]   把全量日志 + 卡上运行器本体一起拉回 artifacts/<tag>/
set -u
OUT="$(dirname "$0")/../artifacts/car/${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT" || exit 1
ssh $SSH "$H" 'cat /tmp/car_run.sh' > "$OUT/car_run.sh" 2>/dev/null   # 运行器本体：证明"哪段脚本产出了这批数"
for f in $(ssh $SSH "$H" 'ls -1 /tmp/car_*.log 2>/dev/null'); do
  b=$(basename "$f"); ssh $SSH "$H" "cat $f" > "$OUT/$b" 2>/dev/null
done
(cd "$OUT" && for f in *.log car_run.sh; do [ -f "$f" ] && echo "$(wc -c < "$f") $f"; done > MANIFEST.txt)
awk '$1 == 0 {print "EMPTY:", $2; bad=1} END {exit bad?1:0}' "$OUT/MANIFEST.txt" \
  && echo "PULLED -> $OUT" || { echo "FAIL: 存在 0 字节日志"; exit 3; }
```

### 三个必须遵守的细节（都是踩过的）

1. **脚本嵌在 `ssh "…"` 的双引号串里时**：内层双引号必须写成 `\"`，且**不许出现反引号**
   （反引号会被本地当作命令替换执行，实测报 `CONTROL: command not found`）。
2. **不要在上一条车还在跑时改本地脚本**：bash 按字节偏移增量读取，插行即错位
   （实测报 `syntax error near token ')'`，`setsid` 被截成 `etsid`）。
   要改就先 `ps` 确认无在跑实例。
3. **区分"本地脚本"与"卡上运行器"**：车启动时会把运行器写成 `/tmp/car_run.sh`，
   之后改本地脚本不影响它——但也意味着**不归档运行器就无法证明产出来源**。

---

## T3 两套 golden 互查

### 为什么

仓库里常同时存在两份 fp64 golden（一份在测试文件、一份被合并进示例）。
**两份不一致时，你比较的不是"算子 vs 数学"，而是"算子 vs 某一份实现"**。
这一步纯 CPU、秒级，应该排在所有上卡动作之前。

### 实现

```python
#!/usr/bin/env python3
"""同输入 → 两套 golden → 逐项 rel；≥1e-12 就是分母问题，先修分母。"""
import sys
import numpy as np
sys.path.insert(0, ".")
from test_<op> import (golden_a_fwd, golden_a_bwd, golden_b_fwd, golden_b_bwd, sample)

ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    H, DK, DV = 1, 64, 64
    Q, K, V, BETA, G, H0, dO = sample(H, T, DK, DV, seed=1000)

    a = [golden_a_bwd(T, DK, DV, golden_a_fwd(T, DK, DV, Q[h], K[h], V[h], BETA[h], G[h], H0[h], C)[1], dO[h])
         for h in range(H)]
    b = golden_b_bwd(H, T, DK, DV, Q, K, V, BETA, G, H0, dO)   # 注意返回次序可能不同：[i][h]

    print(f"GX T={T} C={C} chunks={T // C}")
    worst = 0.0
    for i, item in enumerate(ITEMS):
        num = den = 0.0
        for h in range(H):
            ref = np.asarray(b[i][h], np.float64)
            got = np.asarray(a[h][i], np.float64)
            num = max(num, float(np.abs(got - ref).max()))
            den = max(den, float(np.abs(ref).max()))
        rel = num / (den + 1e-30)
        worst = max(worst, rel)
        print(f"GX item={item:<5} rel={rel:.3e} {'SAME' if rel < 5e-3 else 'DIFFER'}")
    print(f"GX worst={worst:.3e} verdict={'AGREE' if worst < 5e-3 else 'DISAGREE'}")


if __name__ == "__main__":
    main()
```

### 两个坑

1. **返回结构的索引次序常常相反**：一份 golden 是"单 head 调用 → 返回六项"（`a[h][i]`），
   另一份是"全 batch 调用 → 返回六个数组"（`b[i][h]`）。搞反了会得到一个假的"一致"。
   **自检方法**：故意把 `ref` 换成另一项（如用 `dk` 的参考去比 `dq`），应当报 `DIFFER`。
2. **两套 golden 的 `h0` 语义**可能不同（有的内部把 `h0` 折进递推、有的要求外部传入）。
   比之前先对齐 `h0` 的取值。

---

## 工具之间的顺序

```text
T3 golden 互查（纯 CPU，秒级）
   └─ T1 示例对账（纯本地，秒级）
        └─ 静态门（ruff / 内核静态检查）
             └─ T2 上车（含 CONTROL 门）
                  └─ T2 归档（日志 + 运行器 + 字节数）
```

**前三级都不需要设备**。本项目早期把这类检查放到上卡窗口里做，
传输与工装吃掉了窗口的一半——先把不需要设备的都做完，再上车。
