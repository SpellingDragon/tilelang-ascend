#!/usr/bin/env python3
"""compare_relay.py -- 把本地 relay 的中间量与卡上日志逐项对齐。

用法：
    python3 examples/gdn_bwd_v3/test_gdn_bwd_v3_host.py --dump > /tmp/local.json
    python3 tools/compare_relay.py /tmp/local.json /tmp/w11car_v3t.log

卡上日志的形状（gdn_bwd_v3.py 里打印）：
    RELAYDBG c=3 |da|=7.168e+00 |dl|=4.391e+00 ... |r1|=... |r2|=... |r3|=...
    RELAYQ c=3 q1=... q2=... q3=... q4=...
"""

import json
import re
import sys

KEYMAP = {"da": "da", "dl": "dl", "de": "de", "dgam": "dgam",
          "r1": "r1", "r2": "r2", "r3": "r3",
          "q1": "q1", "q2": "q2", "q3": "q3", "q4": "q4"}


def parse_log(path):
    got = {}
    pat = re.compile(r"RELAY(?:DBG|Q) c=(\d+)(.*)")
    kv = re.compile(r"(\w+)=([-+0-9.eE]+)")
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if not m:
            continue
        c = "c" + m.group(1)
        d = got.setdefault(c, {})
        for k, v in kv.findall(m.group(2)):
            if k in KEYMAP:
                d[k] = float(v)
    return got


def main():
    local = json.load(open(sys.argv[1]))
    remote = parse_log(sys.argv[2])
    worst = 0.0
    for c in sorted(local, reverse=True):
        if c not in remote:
            print(f"{c}: 卡上缺这一 chunk")
            continue
        for k in KEYMAP:
            if k not in remote[c] or k not in local[c]:
                continue
            a, b = remote[c][k], local[c][k]
            rel = abs(a - b) / (abs(b) + 1e-30)
            worst = max(worst, rel)
            flag = "" if rel < 5e-3 else "   <-- 差"
            print(f"{c} {k:5s} 卡={a:.4e} 本地={b:.4e} rel={rel:.2e}{flag}")
    print(f"WORST rel={worst:.3e}")


if __name__ == "__main__":
    main()
