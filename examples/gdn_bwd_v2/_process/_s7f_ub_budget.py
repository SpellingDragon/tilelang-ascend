# S7f process artifact: UB/L1/L0C allocation budget, path-parameterized.
# Same static-sum model as _s7e_ub_budget.py (TSC profile C=64, BV=64,
# DK=DV=128; fp32 accum = 4B, main dtype = 2B), but:
#   - source path is argv[1] (default gdn_bwd_v2.py) so stage snapshots
#     (_s7f_stage_s7e_*, _s7f_stage_d_*) stay re-runnable ("no git" rule);
#   - kernel list is dynamic (every gdn_bwd_v2_* factory in the file), so it
#     covers the E-split scan2a/scan2b naming without edits.
# Run: python3 examples/gdn_bwd_v2/_s7f_ub_budget.py [path/to/src.py]
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gdn_bwd_v2.py")

C, BV, DK, DV = 64, 64, 128, 128
SIZES = {"C": C, "BV": BV, "BK": 64, "DK": DK, "DV": DV, "B": 1, "H": 16,
         "16": 16, "8": 8, "1": 1, "bk_num": DK // 64}  # bk_num: S7g keep-alive box
BPE = {"dtype": 2, "accum_dtype": 4, "float": 4}

tree = ast.parse(open(SRC).read())
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
knames = sorted(k for k in funcs if k.startswith("gdn_bwd_v2_"))


def shape_elems(node):
    """Evaluate a literal list shape like [C, DK] or [B * H, C, 16]."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    total = 1
    for el in node.elts:
        v = eval(compile(ast.Expression(el), "<shape>", "eval"), {"__builtins__": {}}, SIZES)  # noqa: S307
        total *= int(v)
    return total


over = False
for kname in knames:
    fn = funcs.get(kname)
    main = next((s for s in fn.body if isinstance(s, ast.FunctionDef) and s.name == "main"), None)
    if main is None:
        continue
    sums = {"ub": 0, "L1": 0, "L0C": 0}
    counts = {"ub": 0, "L1": 0, "L0C": 0}
    dt_el = 0  # S7g: dtype-class ub elements (fp32-profile capacity audit)
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
                if scope == "ub" and ds == "dtype":
                    dt_el += n_el
                if scope == "ub" and nbytes >= 16384:
                    long32.append((node.targets[0].id, nbytes))
    ub32 = sums["ub"] + 2 * dt_el  # fp32 profile: dtype boxes grow 2B -> 4B
    # Hard verdict stays on the fp16-mode sum (the S7f gate). ub32 is an
    # ADVISORY no-reuse upper bound only: real memory planning reuses dead
    # boxes, so e.g. scan1 fp32 compiles green in-container while the static
    # ub32 sum exceeds 192KiB (model false positive). The authoritative fp32
    # capacity check is the container compile gate.
    verdict = "OK" if sums["ub"] <= 192 * 1024 else "OVER"
    ub32_note = "" if ub32 <= 192 * 1024 else " (advisory: static no-reuse bound)"
    if verdict == "OVER":
        over = True
    print(
        f"UB-BUDGET {os.path.basename(SRC)}::{kname} ub={sums['ub']}B({counts['ub']}) "
        f"L1={sums['L1']}B({counts['L1']}) L0C={sums['L0C']}B({counts['L0C']}) "
        f"ub_KiB={sums['ub'] / 1024:.1f}/192.0 ub32_KiB={ub32 / 1024:.1f}/192.0{ub32_note} {verdict}"
    )
    if long32:
        print(f"UB-BUDGET {kname} ub_ge16KiB={sorted(long32, key=lambda x: -x[1])}")
print("UB-BUDGET-ALL-OK" if not over else "UB-BUDGET-OVER")
sys.exit(1 if over else 0)
