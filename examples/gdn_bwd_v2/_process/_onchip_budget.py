# _onchip_budget.py -- 按核内实际 alloc_* 算 v3 各 kernel 的片上占用（纯静态，不碰卡）
#
# 动机：DK=DV=128 时六项从 ~1e-3 跳到 ~1e-1（notes [148]），而 64 完全正常。
# 128 与 64 的差别之一是**片上 buffer 体积翻倍**。本支把 `gdn_bwd_v3.py` 里每个
# `@T.alloc_ub / alloc_L1 / alloc_L0C` 按 (C, DK, DV) 的真实尺寸累加，
# 超过预算就标 OVER——那种情况后端可能不报错而是编出错的行为。
#
# 预算（910B）：UB 192 KiB / L1 512 KiB / L0C 128 KiB。
# 说明：TileLang 会做复用与对齐（所以这是**上界**），但上界都超了就一定有问题。
#
# 用法：python3 _process/_onchip_budget.py [DK] [DV] [C]
import ast
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parents[2] / "gdn_bwd_v3" / "gdn_bwd_v3.py"
BUDGET = {"ub": 192 * 1024, "L1": 512 * 1024, "L0C": 128 * 1024}
DT_BYTES = {"float16": 2, "float32": 4, "float": 4, "bfloat16": 2}


def _size(expr, env):
    """解析 [C, DK] 这类维度表达式。"""
    dims = [d.strip() for d in expr.strip("[]").split(",") if d.strip()]
    n = 1
    for d in dims:
        n *= int(env[d]) if d in env else 1
    return n


def main():
    DK = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    DV = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    C = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    env = {"C": C, "DK": DK, "DV": DV, "BK": 64, "BV": 64, "M": DK, "N": DK}
    src = SRC.read_text()
    tree = ast.parse(src)

    print(f"BUDGET DK={DK} DV={DV} C={C}  (UB<=192K L1<=512K L0C<=128K)")
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        # 只看带 @tilelang.jit 的 kernel 工厂
        decs = [ast.get_source_segment(src, d) or "" for d in fn.decorator_list]
        if not any("tilelang.jit" in d for d in decs):
            continue
        # kernel 工厂的参数名 -> 值（优先用命令行给的 DK/DV/C）
        params = [a.arg for a in fn.args.args]
        loc = dict(env)
        for i, p in enumerate(params[:4]):        # 前几个是 B,H,... 与形状相关
            if p not in loc:
                loc[p] = 1
        tot = {"ub": 0, "L1": 0, "L0C": 0}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            key = {"alloc_ub": "ub", "alloc_L1": "L1", "alloc_L0C": "L0C"}.get(name)
            if key is None or len(node.args) < 2:
                continue
            try:
                shape = _size(ast.get_source_segment(src, node.args[0]), loc)
            except Exception:                       # noqa: BLE001
                continue
            dt = ast.get_source_segment(src, node.args[1]) or ""
            nb = DT_BYTES.get(dt.strip(), 4)
            tot[key] += shape * nb
        flag = "".join(f" {k.upper()}-OVER" for k, v in tot.items() if v > BUDGET[k])
        print(f"  {fn.name:<18} ub={tot['ub'] / 1024:7.1f}K  L1={tot['L1'] / 1024:7.1f}K  "
              f"L0C={tot['L0C'] / 1024:7.1f}K{flag or '  ok'}")


if __name__ == "__main__":
    main()
