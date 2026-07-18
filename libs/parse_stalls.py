#!/usr/bin/env python3
"""
Parse NCu --page source output, attribute stall samples per source-instruction
type (LDG global vs LDS shared vs MMA vs FFMA/MUFU/other).

Column layout (parsed from header in ncu_b21_source.txt):
  Col 0:  Address
  Col 1:  Source (SASS instruction)
  Col 2:  Warp Stall Sampling (All)              — total samples per PC
  Col 3:  Warp Stall Sampling (Not Issued)
  Col 4:  # Samples
  Col 5..28: misc counters (instructions executed, L1/L2 traffic, etc.)
  Col 29..45: stall_*  Issued    (17 stall reasons)
  Col 46..62: stall_*  Not Issued (17 stall reasons)
Order of stall reasons (per header text):
  barrier, branch_resolving, dispatch, drain, lg_throttle, long_scoreboard,
  math_pipe, membar, mio_throttle, misc, no_instructions, not_selected,
  selected, short_scoreboard, sleeping, tex_throttle, wait

We aggregate stall_long_scoreboard and stall_mio_throttle (both Issued and
Not-Issued groups) per instruction class.
"""
import sys
import re
from collections import defaultdict

STALL_NAMES = [
    "barrier", "branch_resolving", "dispatch", "drain", "lg_throttle",
    "long_scoreboard", "math_pipe", "membar", "mio_throttle", "misc",
    "no_instructions", "not_selected", "selected", "short_scoreboard",
    "sleeping", "tex_throttle", "wait",
]
LSB_IDX = STALL_NAMES.index("long_scoreboard")
MIO_IDX = STALL_NAMES.index("mio_throttle")
WAIT_IDX = STALL_NAMES.index("wait")
SHORT_IDX = STALL_NAMES.index("short_scoreboard")


def classify(source: str) -> str:
    s = source.strip()
    if "LDG" in s:
        return "LDG (global load)"
    if "STG" in s:
        return "STG (global store)"
    if "LDS.U16" in s:
        return "LDS.U16 (path β smdO 16-bit reads)"
    if re.match(r"^\s*(@!?P\d+\s+)?LDS\b", s):
        return "LDS.U32 (smPT/smK/smQ 32-bit reads)"
    if "STS" in s:
        return "STS (shared store)"
    if "HMMA" in s or "QMMA" in s or "MMA" in s:
        return "MMA"
    if "MUFU" in s or "FMUL" in s and "EXP" in s.upper():
        return "MUFU/exp"
    if "FFMA" in s or "FADD" in s or "FMUL" in s:
        return "FP math"
    if "I2F" in s or "F2I" in s or "F2F" in s or "I2I" in s:
        return "conversion"
    if "LDC" in s:
        return "LDC (constant)"
    return "other"


_NUM_TOKEN = re.compile(r"^-?\d+$")
_ADDR_TAGS = ("Global", "Shared")     # parts of "Global Load" / "Shared Load" / etc.


def parse_line(line: str):
    # Layout: addr  SASS-src...  NUM NUM NUM ... [Global|Shared] [Load|Store]  NUM NUM ...
    # SASS source contains no pure-numeric tokens. All numerics in the rest of
    # the line are the data columns (in order: samples_all, samples_not_issued,
    # # samples, exec counts, ..., L1/L2 traffic, stall_* issued×17, stall_*
    # not-issued×17). The text tokens "Global Load" / "Shared Load" / "-" are
    # placeholders we skip.
    line = line.rstrip()
    if not line:
        return None
    m = re.match(r"^(0x[0-9a-f]+)\s+(.*)$", line)
    if not m:
        return None
    addr = m.group(1)
    tokens = m.group(2).split()
    # Find first pure-numeric token → SASS source ends just before it.
    i = 0
    while i < len(tokens) and not _NUM_TOKEN.match(tokens[i]):
        i += 1
    source_str = " ".join(tokens[:i]).strip()
    nums = []
    for tok in tokens[i:]:
        if _NUM_TOKEN.match(tok):
            nums.append(int(tok))
        # skip "Global", "Load", "Shared", "Store", "-" and other text
    return addr, source_str, nums


def main(path: str):
    agg = defaultdict(lambda: {
        "samples_all": 0,
        "samples_not_issued": 0,
        "long_sb_issued": 0,
        "long_sb_ni": 0,
        "mio_issued": 0,
        "mio_ni": 0,
        "wait_issued": 0,
        "wait_ni": 0,
        "short_sb_issued": 0,
        "short_sb_ni": 0,
        "count": 0,
    })

    with open(path) as f:
        for line in f:
            if not line.startswith("0x"):
                continue
            parsed = parse_line(line)
            if parsed is None:
                continue
            addr, source, nums = parsed
            if len(nums) < 30:
                # Some rows have only a few counters (cold lines). Still aggregate
                # by class but skip stall extraction.
                cls = classify(source)
                agg[cls]["count"] += 1
                continue
            # Stall block: last 34 ints (17 issued + 17 not-issued)
            stalls = nums[-34:]
            issued = stalls[:17]
            ni = stalls[17:]
            cls = classify(source)
            a = agg[cls]
            a["count"] += 1
            a["samples_all"] += nums[0]
            a["samples_not_issued"] += nums[1]
            a["long_sb_issued"] += issued[LSB_IDX]
            a["long_sb_ni"]     += ni[LSB_IDX]
            a["mio_issued"]     += issued[MIO_IDX]
            a["mio_ni"]         += ni[MIO_IDX]
            a["wait_issued"]    += issued[WAIT_IDX]
            a["wait_ni"]        += ni[WAIT_IDX]
            a["short_sb_issued"] += issued[SHORT_IDX]
            a["short_sb_ni"]     += ni[SHORT_IDX]

    print(f"{'Instruction class':40s} {'# insn':>7s} {'samples':>10s} "
          f"{'long_sb_NI':>11s} {'mio_iss':>9s} {'wait_NI':>9s} "
          f"{'short_NI':>9s}")
    print("-" * 105)

    # Sort by long_sb_ni descending.
    ranked = sorted(agg.items(), key=lambda kv: -kv[1]["long_sb_ni"])
    tot_lsb_ni = sum(v["long_sb_ni"] for _, v in ranked)
    tot_mio    = sum(v["mio_issued"] for _, v in ranked)
    tot_wait   = sum(v["wait_ni"] for _, v in ranked)
    tot_short  = sum(v["short_sb_ni"] for _, v in ranked)
    for cls, v in ranked:
        pct_lsb = 100.0 * v["long_sb_ni"] / tot_lsb_ni if tot_lsb_ni else 0
        print(f"{cls:40s} {v['count']:>7d} {v['samples_all']:>10d} "
              f"{v['long_sb_ni']:>11d}({pct_lsb:5.1f}%) "
              f"{v['mio_issued']:>9d} {v['wait_ni']:>9d} "
              f"{v['short_sb_ni']:>9d}")
    print("-" * 105)
    print(f"{'TOTAL':40s} {'':>7s} {'':>10s} "
          f"{tot_lsb_ni:>11d} {tot_mio:>9d} {tot_wait:>9d} {tot_short:>9d}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/lib/podman-data/projects/goml/runs/ncu_b21_source.txt"
    main(path)
