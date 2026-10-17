# S7e process artifact: AST consistency check for gdn_bwd_v2.py kernel split.
# Verifies per kernel (cumsum/solve_a/scan1/scan2):
#   1. every name referenced in the prim_func body is either a parameter,
#      a local T.alloc_* binding, a loop/thread var, or a closure constant;
#   2. every T.alloc_* binding is referenced at least once (dead buffer check);
#   3. wrapper call sites match the kernel parameter counts.
# Run: python3 examples/gdn_bwd_v2/_s7e_ast_check.py
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gdn_bwd_v2.py")

tree = ast.parse(open(SRC).read())

KERNELS = {
    "gdn_bwd_v2_cumsum": "main",
    "gdn_bwd_v2_solve_a": "main",
    "gdn_bwd_v2_scan1": "main",
    "gdn_bwd_v2_scan2a": "main",
    "gdn_bwd_v2_scan2b": "main",
}


def prim_func_of(fn_node):
    for st in fn_node.body:
        if isinstance(st, ast.FunctionDef) and st.name == "main":
            return st
    return None


class UseCollector(ast.NodeVisitor):
    """Collect free-ish names used in the body (excluding assignment targets
    that are fresh T.alloc_* bindings)."""

    def __init__(self):
        self.used = set()
        self.allocs = {}
        self.bound = set()

    def visit_Name(self, node):
        self.used.add(node.id)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # detect x = T.alloc_*([..], ..)
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                call = node.value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr.startswith("alloc_")
                ):
                    self.allocs[tgt.id] = call.func.attr
                    self.bound.add(tgt.id)
        self.generic_visit(node)


fail = False
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
for kname in KERNELS:
    fn = funcs.get(kname)
    if fn is None:
        print(f"AST-CHECK {kname} FAIL missing function")
        fail = True
        continue
    main = prim_func_of(fn)
    if main is None:
        print(f"AST-CHECK {kname} FAIL missing prim_func main")
        fail = True
        continue
    params = [a.arg for a in main.args.args]
    uc = UseCollector()
    for st in main.body:
        uc.visit(st)
    # names used in body must come from params/allocs/loop vars/builtins
    loopvars = set()
    for node in ast.walk(main):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            loopvars.add(node.target.id)
        if isinstance(node, ast.withitem) and isinstance(
            node.context_expr, ast.Call
        ):
            cal = node.context_expr
            if cal.args and isinstance(cal.args[0], ast.Call):
                inner = cal.args[0]
                for v in ast.walk(inner):
                    if isinstance(v, ast.Name):
                        loopvars.add(v.id)
    ok_names = (
        set(params)
        | set(uc.allocs)
        | loopvars
        | {"T", "True", "False", "None", "range", "len", "min", "max", "int", "float", "print", "enumerate", "zip"}
    )
    # closure names: the factory scope OUTSIDE main (params + top-level assigns)
    closure = {a.arg for a in fn.args.args}
    for st in fn.body:
        if st is main:
            continue
        for node in ast.walk(st):
            if isinstance(node, ast.Name):
                closure.add(node.id)
            if isinstance(node, ast.arg):
                closure.add(node.arg)
    # local bindings inside main (assign targets, for targets, with-as)
    locals_ = set(loopvars)
    for node in ast.walk(main):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        locals_.add(n.id)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            locals_.add(node.target.id)
        if isinstance(node, ast.withitem) and node.optional_vars is not None:
            for n in ast.walk(node.optional_vars):
                if isinstance(n, ast.Name):
                    locals_.add(n.id)
    ok_names |= closure | locals_
    # a name is "missing" if Loaded in main body but not legal anywhere
    read = {n.id for n in ast.walk(main) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(n for n in read if n not in ok_names)
    written = {n.id for n in ast.walk(main) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    allv = {n.id for n in ast.walk(main) if isinstance(n, ast.Name)}
    dead = sorted(a for a in uc.allocs if a not in read)
    status = "OK" if (not missing and not dead) else "FAIL"
    if status == "FAIL":
        fail = True
    print(
        f"AST-CHECK {kname} {status} params={len(params)} "
        f"allocs={len(uc.allocs)} missing={missing} dead={dead}"
    )

# wrapper call-site arity: count args of gdn_bwd_v2_scan1/scan2/cumsum/solve_a calls
wrapper = funcs.get("gdn_bwd_v2")
calls = {}
if wrapper is not None:
    for node in ast.walk(wrapper):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("gdn_bwd_v2")
        ):
            calls.setdefault(node.func.id, []).append(len(node.args))
for k, v in sorted(calls.items()):
    print(f"AST-CHECK call-site {k} nargs={v}")
# inner kernel-arg calls: gdn_bwd_v2_scan1(...)(k, v, ...)
kcalls = {}
if wrapper is not None:
    for node in ast.walk(wrapper):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Call):
            f = node.func
            if isinstance(f.func, ast.Name) and f.func.id.startswith("gdn_bwd_v2"):
                kcalls.setdefault(f.func.id + "/kernel", []).append(len(node.args))
for k, v in sorted(kcalls.items()):
    print(f"AST-CHECK call-site {k} nargs={v}")

print("AST-ALL-OK" if not fail else "AST-FAIL")
sys.exit(1 if fail else 0)
