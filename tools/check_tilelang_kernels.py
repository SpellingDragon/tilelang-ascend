#!/usr/bin/env python3
"""check_tilelang_kernels.py -- 发车前 30 秒的静态自查（任何 @tilelang.jit kernel 通用）。

检查项（编号对应 notes [90]/[94] 的事故）：
  S1  out_idx 指向的参数必须是输出：若指向本表里的"输入名"直接报错
  S2  T.copy(src, dst) 的 dtype 必须一致（GM 参数 vs alloc 声明）
  S3  同一 2-D T.Parallel 里不得读两个以上的 1-D UB buffer（507015）
  S4  kernel 体里引用的名字必须已定义（_step 那类 NameError）
  S5  torch.zeros_like(x) 的 x 不能在之前被赋 None（末 chunk 的 Delta'）
  S6  T.copy 的实参个数必须是 2
  S7  host 侧 f16/f32 包装的实参 dtype 必须与 kernel 声明一致
  S8  k_* 调用的实参个数 = 参数个数 - 输出个数
  S9  T.Parallel 最多两维（3-D 是编译期 FATAL）

用法：python3 tools/check_tilelang_kernels.py examples/gdn_bwd_v3/gdn_bwd_v3.py [more.py ...]
"""

from __future__ import annotations

import ast
import sys

INPUT_NAMES = {
    "A", "Bm", "BV", "Beta", "Carry", "DO", "DVNEW", "DVN", "DP", "E", "EDO",
    "Gam", "G", "H", "Hh", "H0", "K", "Msk", "Q", "Raw", "S", "V", "WH",
    "WN", "X", "DG", "Ones", "IDT", "MskJ", "Beta",
}
OUTPUT_HINTS = {"O", "U", "VN", "R", "Y", "WH", "DQ", "DKT", "EDO", "DQKT", "T1", "P"}


def kernel_entries(tree: ast.Module):
    """yield (fn_name, out_idx list or None, main FuncDef) for every @tilelang.jit kernel."""
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        out_idx = None
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "jit"):
                for kw in dec.keywords:
                    if kw.arg == "out_idx":
                        out_idx = [e.value for e in kw.value.elts]
        inner = None
        for sub in node.body:
            if isinstance(sub, ast.FunctionDef) and sub.name == "main":
                inner = sub
        if inner is not None:
            yield node.name, out_idx, inner, node


def param_dtypes(main: ast.FunctionDef) -> dict[str, str]:
    d = {}
    for a in main.args.args:
        ann = a.annotation
        if isinstance(ann, ast.Call) and isinstance(ann.func, ast.Attribute) and ann.func.attr == "Tensor":
            dt = None
            for kw in ann.keywords:
                if kw.arg == "dtype" and isinstance(kw.value, ast.Constant):
                    dt = kw.value.value
            if dt is None and len(ann.args) >= 2 and isinstance(ann.args[1], ast.Constant):
                dt = ann.args[1].value
            if dt:
                d[a.arg] = dt
    return d


def alloc_dtypes(main: ast.FunctionDef) -> dict[str, str]:
    d = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                    and v.func.attr in ("alloc_ub", "alloc_L1", "alloc_L0C", "alloc_fragment")):
                dt = None
                if len(v.args) >= 2 and isinstance(v.args[1], ast.Constant):
                    dt = v.args[1].value
                for kw in v.keywords:
                    if kw.arg == "dtype" and isinstance(kw.value, ast.Constant):
                        dt = kw.value.value
                if dt:
                    d[node.targets[0].id] = dt
    return d


def check_file(path: str) -> list[str]:
    src = open(path).read()
    tree = ast.parse(src)
    problems: list[str] = []

    for fname, out_idx, main, outer in kernel_entries(tree):
        params = [a.arg for a in main.args.args]
        pd = param_dtypes(main)
        ad = alloc_dtypes(main)

        # S1 out_idx 指向的参数必须是"只写不读"：用 T.copy 的 dst 位置判定
        written, read = set(), set()
        copies = []
        for node in ast.walk(main):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "copy"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "T"
                    and len(node.args) == 2):
                copies.append((_base_name(node.args[0]), _base_name(node.args[1])))
        for src_base, dst_base in copies:
            if src_base:
                read.add(src_base)
            if dst_base:
                written.add(dst_base)
        # 其它位置的 Subscript 才算读（dst 位置已被 written 覆盖）
        for node in ast.walk(main):
            if isinstance(node, ast.Subscript):
                b = _base_name(node)
                if b in params and b not in written:
                    read.add(b)
        if out_idx:
            for i in out_idx:
                if i >= len(params):
                    problems.append(f"S1 {fname}: out_idx {i} 越界（只有 {len(params)} 个参数）")
                elif params[i] in read:
                    problems.append(
                        f"S1 {fname}: out_idx[{i}] 指向 '{params[i]}'，但它在函数体里被读取"
                        f"（输入不能当输出）—— 参数序 {list(enumerate(params))}")
                elif params[i] not in written:
                    problems.append(
                        f"S1 {fname}: out_idx[{i}] 指向 '{params[i]}'，但它从未作为 T.copy 的目标出现")

        # S2 T.copy 的 dtype
        for node in ast.walk(main):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "copy"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "T"
                    and len(node.args) == 2):
                a, b = node.args
                an = a.id if isinstance(a, ast.Name) else None
                bn = b.id if isinstance(b, ast.Name) else None
                da = pd.get(an) or ad.get(an)
                db = pd.get(bn) or ad.get(bn)
                if da and db and da != db:
                    problems.append(f"S2 {fname}: T.copy({an}:{da} -> {bn}:{db}) 跨 dtype")

        # S3 一个 2-D Parallel 里读多个 1-D buffer
        one_d = {n for n in ad if _is_one_d(main, n)}
        for node in ast.walk(main):
            if (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Attribute)
                    and node.iter.func.attr == "Parallel"
                    and len(node.iter.args) >= 2):
                used = {n.id for sub in ast.walk(node) if isinstance(sub, ast.Name)
                        for n in [sub] if n.id in one_d}
                if len(used) > 1:
                    problems.append(f"S3 {fname}: 2-D T.Parallel 里读了 {sorted(used)}（507015 风险）")

        # S4 未定义名字（只查 Name 读取）
        # 闭包：main 在外层 gdn_v3_xxx(B,H,C,...) 里，外层的参数也是已定义
        outer_names = {a.arg for a in outer.args.args}
        for _n in ast.walk(outer):                     # 外层函数体内的局部赋值/循环变量
            if isinstance(_n, ast.Assign):
                for _t in _n.targets:
                    for _sub in ast.walk(_t):
                        if isinstance(_sub, ast.Name):
                            outer_names.add(_sub.id)
            if isinstance(_n, ast.For):
                if isinstance(_n.target, ast.Tuple):      # for a, b in ...
                    for _e in _n.target.elts:
                        if isinstance(_e, ast.Name):
                            outer_names.add(_e.id)
                elif isinstance(_n.target, ast.Name):     # for c in ...
                    outer_names.add(_n.target.id)
        defined = set(params) | set(ad) | outer_names | {"T", "range", "torch"}
        for node in ast.walk(main):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    for sub in ast.walk(t):
                        if isinstance(sub, ast.Name):
                            defined.add(sub.id)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple):
                for e in node.target.elts:
                    if isinstance(e, ast.Name):
                        defined.add(e.id)
            if isinstance(node, ast.With):
                for item in node.items:
                    tgt = item.optional_vars
                    if isinstance(tgt, ast.Name):
                        defined.add(tgt.id)
                    elif isinstance(tgt, ast.Tuple):
                        for e in tgt.elts:
                            if isinstance(e, ast.Name):
                                defined.add(e.id)
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
                defined |= {a.arg for a in node.args.args}
        for node in ast.walk(main):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined:
                problems.append(f"S4 {fname}: 未定义名字 '{node.id}'")

    # S7/S8：host 侧调用与 kernel 声明的一致性
    kdefs = {}
    for fname, out_idx, main, outer in kernel_entries(tree):
        kdefs[fname] = ([a.arg for a in main.args.args], param_dtypes(main), out_idx or [])
    inst = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            v = node.value
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in kdefs):
                inst[node.targets[0].id] = v.func.id
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in inst):
            kname = inst[node.func.id]
            params, dtypes, outs = kdefs[kname]
            expect = len(params) - len(outs)
            if len(node.args) != expect:
                problems.append(
                    f"S8 第 {node.lineno} 行: {node.func.id}({kname}) 传 {len(node.args)} 个实参，"
                    f"应为 {expect}（{len(params)} 参数 - {len(outs)} 输出）")
            for idx, arg in enumerate(node.args):
                if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)):
                    continue
                declared = dtypes.get(params[idx]) if idx < len(params) else None
                if arg.func.id == "f16" and declared == "float":
                    problems.append(
                        f"S7 第 {node.lineno} 行: {node.func.id} 第 {idx} 个实参用 f16 传给 "
                        f"{kname}.{params[idx]}（声明 float）")
                if arg.func.id == "f32" and declared not in (None, "float"):
                    problems.append(
                        f"S7 第 {node.lineno} 行: {node.func.id} 第 {idx} 个实参用 f32 传给 "
                        f"{kname}.{params[idx]}（声明 {declared}）")

    # S9：T.Parallel 最多两维（3-D 在 ascend_lower_parallel_to_vector.cc:298 是 LOG(FATAL)）
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Parallel" and len(node.args) >= 3):
            problems.append(
                f"S9 第 {node.lineno} 行: T.Parallel 有 {len(node.args)} 维，"
                f"超过 2 维会在 pass 里 LOG(FATAL)")

    # S5 zeros_like(None)
    none_vars = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is None:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    none_vars.add(t.id)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "zeros_like" and node.args
                and isinstance(node.args[0], ast.Name) and node.args[0].id in none_vars):
            problems.append(f"S5 第 {node.lineno} 行: zeros_like('{node.args[0].id}')，该变量可能为 None")

    # S6 T.copy 实参个数
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "copy"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "T"
                and len(node.args) != 2):
            problems.append(f"S6 第 {node.lineno} 行: T.copy 有 {len(node.args)} 个实参")

    return problems


def _base_name(node: ast.AST):
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_one_d(main: ast.FunctionDef, name: str) -> bool:
    for node in ast.walk(main):
        if (isinstance(node, ast.Assign) and node.targets
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name and isinstance(node.value, ast.Call)):
                v = node.value
                if v.args and isinstance(v.args[0], ast.List) and len(v.args[0].elts) == 1:
                    return True
    return False


if __name__ == "__main__":
    files = sys.argv[1:] or ["examples/gdn_bwd_v3/gdn_bwd_v3.py"]
    total = 0
    for f in files:
        probs = check_file(f)
        for p in probs:
            print(p)
        total += len(probs)
    print(f"STATIC issues={total}")
    sys.exit(0 if total == 0 else 1)
