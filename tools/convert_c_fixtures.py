"""Writes tests/cross_derived.json from the fixture headers of tandem-c and tandem-cuda.

    python tools/convert_c_fixtures.py ../tandem-c/tests ../tandem-cuda/tests > tests/cross_derived.json
"""

import ast
import json
import re
import sys
from pathlib import Path


def balanced(text, start):
    depth = 0
    for i in range(start, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[start : i + 1]


def literal(body):
    """A C initialiser as nested lists, with the integer and float suffixes dropped."""
    body = re.sub(r"(?<=[0-9.])(ull|u|f)\b", "", body)
    return ast.literal_eval(body.replace("{", "[").replace("}", "]"))


def array(text, name):
    m = re.search(rf"\b{name}\[[^\]]*\]\s*=\s*", text)
    return literal(balanced(text, m.end()))


def scalar(text, name):
    return int(re.search(rf"\b{name}\s*=\s*(\d+)", text).group(1))


c_tests, cuda_tests = Path(sys.argv[1]), Path(sys.argv[2])
scalar_below, fill, normal, exponential = (
    (c_tests / f).read_text() for f in ("cross_below.h", "cross_fill_below.h", "cross_normal.h", "cross_exponential.h")
)
device_below, device_normal, device_exponential = (
    (cuda_tests / f).read_text() for f in ("cross_fill_below.h", "cross_fill_normal.h", "cross_fill_exponential.h")
)
out = {
    # Sequential scalar draws after one Bool, so from bit position 1. Rejected draws consume the stream.
    "scalar_below32": [{"range": n, "out": o, "end_pos": e} for n, o, e in array(scalar_below, "CROSS_U32")],
    "scalar_below64": [{"range": n, "out": o, "end_pos": e} for n, o, e in array(scalar_below, "CROSS_U64")],
    # Seed 42, fills from bit positions 0, 1 and 12345, which align up to the draw width.
    "fill_below32": [{"start": s, "range": n, "out": o, "end_pos": e} for s, n, o, e in array(fill, "CROSS_FILL_U32")],
    "fill_below64": [{"start": s, "range": n, "out": o, "end_pos": e} for s, n, o, e in array(fill, "CROSS_FILL_U64")],
    # Seed 42, 64 scalar pairs from bit position 1, after one Bool.
    "pairs32": array(normal, "CROSS_NORMALF"),
    "pairs32_end_pos": scalar(normal, "CROSS_NORMALF_END_POS"),
    # Seed 42, fills of 64 from several starts.
    "exponential32": [{"start": s, "out": o, "end_pos": e} for s, o, e in array(exponential, "CROSS_EXPONENTIALF")],
    # Key of seed 42, K = 32, from tandem-cuda. `rejected` counts elements on the fallback.
    "device_below32": [{"range": n, "rejected": r, "out": o} for n, r, o in array(device_below, "CROSS_BELOW32")],
    "device_below64": [{"range": n, "rejected": r, "out": o} for n, r, o in array(device_below, "CROSS_BELOW64")],
    # Fills from bit positions 1 and 12345, where the global draw index differs from the element index.
    "device_below32_at": [{"start": s, "range": n, "rejected": r, "out": o} for s, n, r, o in array(device_below, "CROSS_BELOW32_AT")],
    "device_below64_at": [{"start": s, "range": n, "rejected": r, "out": o} for s, n, r, o in array(device_below, "CROSS_BELOW64_AT")],
    "device_normal32": [{"pos": p, "n": n, "out": o[:n]} for p, n, o in array(device_normal, "CROSS_NORMAL32")],
    "device_exponential32": [{"pos": p, "n": n, "out": o[:n]} for p, n, o in array(device_exponential, "CROSS_EXP32")],
}
json.dump(out, sys.stdout, separators=(",", ":"))
print()
