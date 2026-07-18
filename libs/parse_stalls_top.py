#!/usr/bin/env python3
"""
Parse NCu --page source for dK kernel.

Goal: separate mio_throttle vs short_scoreboard sources.
For each stall reason: top-5 SASS instructions by stall sample count.
Intersection: are top mio and top short_sb the SAME instructions?

Stall column layout (verified in B2.2.4 against ncu_b21_source.txt):
  After address + SASS + leading numeric counters,
  last 34 ints = stall block (17 Issued + 17 Not-Issued):
  Order: barrier, branch_resolving, dispatch, drain, lg_throttle,
         long_scoreboard, math_pipe, membar, mio_throttle, misc,
         no_instructions, not_selected, selected, short_scoreboard,
         sleeping, tex_throttle, wait.
"""
import re
import sys
from collections import defaultdict

STALL_NAMES = [
    "barrier", "branch_resolving", "dispatch", "drain", "lg_throttle",
    "long_scoreboard", "math_pipe", "membar", "mio_throttle", "misc",
    "no_instructions", "not_selected", "selected", "short_scoreboard",
    "sleeping", "tex_throttle", "wait",
]
LSB_IDX   = STALL_NAMES.index("long_scoreboard")
MIO_IDX   = STALL_NAMES.index("mio_throttle")
SHORT_IDX = STALL_NAMES.index("short_scoreboard")
WAIT_IDX  = STALL_NAMES.index("wait")
BARRIER_IDX = STALL_NAMES.index("barrier")
MATH_IDX  = STALL_NAMES.index("math_pipe")

_NUM_TOKEN = re.compile(r"^-?\d+$")


def classify(source: str) -> str:
    s = source.strip()
    if "LDG" in s:    return "LDG (global load)"
    if "STG" in s:    return "STG (global store)"
    if "LDS" in s and ".U16" in s: return "LDS.U16"
    if "LDS" in s and ".U32" not in s and ".U16" not in s and ".U8" not in s: return "LDS (.U32 default)"
    if "LDS" in s and ".U8" in s:  return "LDS.U8"
    if "LDS" in s:    return "LDS (other)"
    if "STS.U16" in s: return "STS.U16"
    if "STS.U8" in s:  return "STS.U8"
    if "STS" in s:    return "STS (other)"
    if "HMMA" in s or "MMA" in s: return "MMA"
    if "F2FP" in s or "FRND" in s: return "cvt FP→FP"
    if "I2F" in s or "F2I" in s:  return "cvt int↔FP"
    if "I2I" in s or "F2F" in s:  return "cvt other"
    if "FFMA" in s:   return "FFMA"
    if "FMUL" in s:   return "FMUL"
    if "FADD" in s:   return "FADD"
    if "HMUL2" in s or "HFMA2" in s or "HADD2" in s: return "FP16 math"
    if "MUFU" in s:   return "MUFU/transcendental"
    if "LDC" in s:    return "LDC (constant)"
    if "IADD" in s or "ISETP" in s or "IMAD" in s or "SHF" in s or "LEA" in s:
        return "INT arith"
    if "PRMT" in s:   return "PRMT (byte perm)"
    if "BAR" in s:    return "BAR (barrier)"
    return "other"


def parse_line(line: str):
    line = line.rstrip()
    if not line:
        return None
    m = re.match(r"^(0x[0-9a-f]+)\s+(.*)$", line)
    if not m:
        return None
    addr = m.group(1)
    tokens = m.group(2).split()
    i = 0
    while i < len(tokens) and not _NUM_TOKEN.match(tokens[i]):
        i += 1
    source_str = " ".join(tokens[:i]).strip()
    nums = []
    for tok in tokens[i:]:
        if _NUM_TOKEN.match(tok):
            nums.append(int(tok))
    return addr, source_str, nums


def main(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("0x"):
                continue
            parsed = parse_line(line)
            if parsed is None:
                continue
            addr, source, nums = parsed
            if len(nums) < 34:
                continue
            stalls = nums[-34:]
            issued = stalls[:17]
            ni     = stalls[17:]
            rows.append({
                "addr": addr,
                "source": source,
                "class": classify(source),
                "mio_iss": issued[MIO_IDX],
                "mio_ni":  ni[MIO_IDX],
                "short_iss": issued[SHORT_IDX],
                "short_ni":  ni[SHORT_IDX],
                "long_iss": issued[LSB_IDX],
                "long_ni":  ni[LSB_IDX],
                "wait_ni":  ni[WAIT_IDX],
                "barrier_ni": ni[BARRIER_IDX],
                "math_iss": issued[MATH_IDX],
            })

    # === Aggregate per class ===
    cls_agg = defaultdict(lambda: dict(mio=0, short=0, long=0, count=0))
    for r in rows:
        c = r["class"]
        cls_agg[c]["mio"]   += r["mio_iss"] + r["mio_ni"]
        cls_agg[c]["short"] += r["short_iss"] + r["short_ni"]
        cls_agg[c]["long"]  += r["long_iss"] + r["long_ni"]
        cls_agg[c]["count"] += 1

    def show_class_table(stall_key, title):
        tot = sum(v[stall_key] for v in cls_agg.values())
        if tot == 0:
            print(f"{title}: no samples")
            return
        sorted_cls = sorted(cls_agg.items(), key=lambda kv: -kv[1][stall_key])
        print(f"\n=== {title} by instruction class (sum issued+ni) ===")
        print(f"{'Class':28s} {'# instr':>8s} {'samples':>10s} {'% of total':>10s}")
        print("-" * 60)
        for c, v in sorted_cls[:10]:
            pct = 100.0 * v[stall_key] / tot
            print(f"{c:28s} {v['count']:>8d} {v[stall_key]:>10d} {pct:>9.1f}%")

    show_class_table("mio", "mio_throttle")
    show_class_table("short", "short_scoreboard")
    show_class_table("long", "long_scoreboard")

    # === Top-N individual SASS instructions per stall ===
    def show_top(stall_iss, stall_ni, title, N=10):
        ranked = sorted(rows,
                        key=lambda r: -(r[stall_iss] + r[stall_ni]))
        print(f"\n=== Top-{N} SASS lines by {title} ===")
        print(f"{'addr':18s} {'cls':24s} {'iss':>7s} {'ni':>7s} {'sum':>7s}  source (truncated)")
        print("-" * 110)
        for r in ranked[:N]:
            tot = r[stall_iss] + r[stall_ni]
            if tot == 0:
                break
            src = r["source"][:48]
            print(f"{r['addr']:18s} {r['class']:24s} {r[stall_iss]:>7d} {r[stall_ni]:>7d} {tot:>7d}  {src}")

    show_top("mio_iss",  "mio_ni",  "mio_throttle")
    show_top("short_iss","short_ni","short_scoreboard")
    show_top("long_iss", "long_ni", "long_scoreboard")

    # === Intersection: are top-mio lines ALSO top-short_sb lines? ===
    top_n = 20
    top_mio = set(r["addr"] for r in sorted(rows,
                  key=lambda r: -(r["mio_iss"] + r["mio_ni"]))[:top_n])
    top_short = set(r["addr"] for r in sorted(rows,
                    key=lambda r: -(r["short_iss"] + r["short_ni"]))[:top_n])
    intersect = top_mio & top_short
    print(f"\n=== Intersection top-{top_n} mio ∩ top-{top_n} short_sb ===")
    print(f"  |top mio|      = {len(top_mio)}")
    print(f"  |top short_sb| = {len(top_short)}")
    print(f"  |intersection| = {len(intersect)}")
    if intersect:
        print(f"\n  Common addresses (instructions seen in BOTH top stalls):")
        for addr in sorted(intersect):
            r = next(x for x in rows if x["addr"] == addr)
            mio_sum = r["mio_iss"] + r["mio_ni"]
            short_sum = r["short_iss"] + r["short_ni"]
            print(f"    {addr}  {r['class']:24s}  mio={mio_sum:>6d}  short_sb={short_sum:>6d}  src={r['source'][:50]}")
    else:
        print(f"\n  EMPTY: top mio and top short_sb are DISJOINT instruction sets.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/lib/podman-data/projects/goml/runs/ncu_dk_source.txt"
    main(path)
