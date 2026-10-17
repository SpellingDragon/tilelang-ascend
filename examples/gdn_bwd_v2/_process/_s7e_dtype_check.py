# S7e process artifact: copy dtype-discipline scan for gdn_bwd_v2.py.
# Rule (tl_templates/ascend/common.h: gemm_v0 takes A/B from L1, and GM<->onchip
# copies must be dtype-matched; ub<->ub and l1<->l1 casts are allowed):
#   gm<->ub / gm<->l1 / gm<->l0c / ub<->l0c  must connect buffers of the same
#   dtype class. FP32 class = {"float", accum_dtype}; FP16 class = {dtype}.
# Run: python3 examples/gdn_bwd_v2/_s7e_dtype_check.py
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gdn_bwd_v2.py")

tree = ast.parse(open(SRC).read())
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

FP32 = {"float", "accum_dtype"}
FP16 = {"dtype"}


def dtype_of(node):
    """Return the dtype symbol of a T.Tensor / T.alloc_* call's last arg."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    return "?"


def base_of(node):
    """Base name of a buffer expression (Name or Subscript)."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def klass(sym):
    if sym in FP32:
        return "fp32"
    if sym in FP16:
        return "fp16"
    return "unknown"


fail = False
for kname in ("gdn_bwd_v2_cumsum", "gdn_bwd_v2_solve_a", "gdn_bwd_v2_scan1", "gdn_bwd_v2_scan2a", "gdn_bwd_v2_scan2b"):
    fn = funcs.get(kname)
    if fn is None:
        continue
    main = next(s for s in fn.body if isinstance(s, ast.FunctionDef) and s.name == "main")
    gm = {}  # name -> dtype symbol
    for a in main.args.args:
        ann = a.annotation
        if isinstance(ann, ast.Call) and isinstance(ann.func, ast.Subscript):
            targs = ann.func.value  # T.Tensor
            gm[a.arg] = dtype_of(ann.args[1]) if len(ann.args) > 1 else None
        elif isinstance(ann, ast.Call):
            gm[a.arg] = dtype_of(ann.args[1]) if len(ann.args) > 1 else None
    chip = {}  # alloc name -> (scope, dtype symbol)
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr.startswith("alloc_"):
                scope = call.func.attr.split("_", 1)[1]  # ub / L1 / L0C
                tgt = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                if tgt:
                    chip[tgt] = (scope, dtype_of(call.args[-1]))
    # S7g: resolve simple name aliases (io_* single transit box)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(main):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Name)
                and isinstance(node.targets[0], ast.Name)
            ):
                tgt, src = node.targets[0].id, node.value.id
                if tgt not in chip and src in chip:
                    chip[tgt] = chip[src]
                    changed = True
    bad = []
    for node in ast.walk(main):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "copy" or len(node.args) < 2:
            continue
        s, d = node.args[0], node.args[1]
        sb, db = base_of(s), base_of(d)
        sk = klass(gm[sb]) if sb in gm else (klass(chip[sb][1]) + ":" + chip[sb][0] if sb in chip else None)
        dk = klass(gm[db]) if db in gm else (klass(chip[db][1]) + ":" + chip[db][0] if db in chip else None)
        if sk is None or dk is None:
            continue
        s_kind = "gm" if sb in gm else chip[sb][0].lower()
        d_kind = "gm" if db in gm else chip[db][0].lower()
        cross = {"gm:ub", "ub:gm", "gm:l1", "l1:gm", "gm:l0c", "l0c:gm", "ub:l0c", "l0c:ub"}
        key = f"{s_kind}:{d_kind}"
        if key in cross:
            scls, dcls = klass(gm[sb]) if sb in gm else klass(chip[sb][1]), klass(gm[db]) if db in gm else klass(chip[db][1])
            if scls != dcls:
                bad.append((node.lineno, sb, sk, db, dk))
    status = "OK" if not bad else "FAIL"
    if bad:
        fail = True
    print(f"DTYPE-CHECK {kname} {status} mismatched_copies={bad}")

print("DTYPE-ALL-OK" if not fail else "DTYPE-FAIL")
sys.exit(1 if fail else 0)
