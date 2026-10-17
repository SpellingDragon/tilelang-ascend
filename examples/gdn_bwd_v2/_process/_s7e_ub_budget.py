# S7e process artifact: UB/L1/L0C allocation budget for gdn_bwd_v2.py kernels.
# Static sum of declared T.alloc_* buffers at the TSC profile
# (C=64, BV=64, DK=DV=128; fp32 accum = 4B, main dtype = 2B) -- the reference
# for the TSC scan2 "Memory allocation failed" over-commit analysis.
# Note: the AscendMemoryPlanner linear-scan allocator reuses non-overlapping
# intervals; this is the ALLOCATED total, the live peak is lower but the
# whole-func one-interval-per-buffer model keeps long-lane buffers resident.
# Run: python3 examples/gdn_bwd_v2/_s7e_ub_budget.py
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gdn_bwd_v2.py")

C, BV, DK, DV = 64, 64, 128, 128
SIZES = {"C": C, "BV": BV, "BK": 64, "DK": DK, "DV": DV, "B": 1, "H": 16, "16": 16, "8": 8, "1": 1}
BPE = {"dtype": 2, "accum_dtype": 4, "float": 4}

tree = ast.parse(open(SRC).read())
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def shape_elems(node):
    """Evaluate a literal list shape like [C, DK] or [B * H, C, 16]."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    total = 1
    for el in node.elts:
        v = eval(compile(ast.Expression(el), "<shape>", "eval"), {"__builtins__": {}}, SIZES)  # noqa: S307
        total *= int(v)
    return total


for kname in ("gdn_bwd_v2_cumsum", "gdn_bwd_v2_solve_a", "gdn_bwd_v2_scan1", "gdn_bwd_v2_scan2a", "gdn_bwd_v2_scan2b"):
    fn = funcs.get(kname)
    if fn is None:
        continue
    main = next(s for s in fn.body if isinstance(s, ast.FunctionDef) and s.name == "main")
    sums = {"ub": 0, "L1": 0, "L0C": 0}
    counts = {"ub": 0, "L1": 0, "L0C": 0}
    long32 = []
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr.startswith("alloc_"):
                scope = call.func.attr.split("_", 1)[1]
                n_el = shape_elems(call.args[0])
                dsym = call.args[-1]
                ds = dsym.id if isinstance(dsym, ast.Name) else dsym.value
                bpe = BPE.get(ds, 0)
                if n_el is None or scope not in sums:
                    continue
                nbytes = n_el * bpe
                sums[scope] += nbytes
                counts[scope] += 1
                if scope == "ub" and nbytes >= 16384:
                    long32.append((node.targets[0].id, nbytes))
    print(
        f"UB-BUDGET {kname} ub={sums['ub']}B({counts['ub']}) "
        f"L1={sums['L1']}B({counts['L1']}) L0C={sums['L0C']}B({counts['L0C']})"
    )
    if kname.startswith("gdn_bwd_v2_scan2"):
        print(f"UB-BUDGET {kname} ub_ge16KiB={sorted(long32, key=lambda x: -x[1])}")
        print(f"UB-BUDGET {kname} ub_total_KiB={sums['ub'] / 1024:.1f} (UB capacity 192KiB/core)")
sys.exit(0)
