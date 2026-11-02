#!/usr/bin/env python3
"""Headroom Meter — should you combine two systems? Answer in seconds, no training.

A diagnostic built on the Law of Routing Existence:
    routing_advantage = headroom(error_orthogonality) x efficiency(identifiability)
    -> a router/ensemble/hybrid creates value IFF the systems fail on orthogonal
       inputs. If headroom ~ 0, no router, attention, or fusion will help.

Input: per-item correctness of two systems (A, B) on the SAME eval set — exactly
what any team already has from running their own evaluation. No model internals,
no weights, no training. Optionally an item-level cue/feature to estimate how much
headroom a cheap router could actually capture.

Output: headroom, the failure-overlap breakdown, the dominance check, and a plain
verdict: COMBINE / DON'T-COMBINE / DEPLOY-DOMINANT — plus the projected best-case
ensemble accuracy (the oracle).

Usage:
    headroom_meter.py --a a_correct.json --b b_correct.json
    headroom_meter.py --a a.csv --b b.csv --labels-a A --labels-b B   # CSV 0/1 col
    cat results.json | headroom_meter.py --stdin   # {"a":[0,1,...],"b":[1,1,...]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str) -> list[int]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            # accept {"correct":[...]} or first list value
            data = data.get("correct") or next(v for v in data.values() if isinstance(v, list))
        return [int(bool(x)) for x in data]
    # plain text / csv: one 0/1 per line (last field if comma-separated)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tok = line.split(",")[-1].strip()
        try:
            out.append(int(float(tok) > 0.5))
        except ValueError:
            pass
    return out


def analyze(a: list[int], b: list[int], cue: list[float] | None = None) -> dict:
    n = len(a)
    if len(b) != n:
        raise SystemExit(f"length mismatch: A has {n}, B has {len(b)}")
    if n == 0:
        raise SystemExit("empty input")
    acc_a = sum(a) / n
    acc_b = sum(b) / n
    both = sum(1 for i in range(n) if a[i] and b[i]) / n
    only_a = sum(1 for i in range(n) if a[i] and not b[i]) / n
    only_b = sum(1 for i in range(n) if b[i] and not a[i]) / n
    neither = sum(1 for i in range(n) if not a[i] and not b[i]) / n
    oracle = both + only_a + only_b
    best = max(acc_a, acc_b)
    headroom = oracle - best
    # error orthogonality: of the items at least one gets right, what fraction is
    # exclusive to one system (1.0 = fully orthogonal, 0 = fully redundant)
    solved = both + only_a + only_b
    orthogonality = (only_a + only_b) / solved if solved > 1e-9 else 0.0

    # optional: can a cheap cue identify which system to trust? (identifiability)
    captured = None
    if cue is not None and len(cue) == n:
        # threshold the cue to predict "use B"; pick the better of {<=t,>t} sweep
        order = sorted(set(cue))
        best_router = best
        for t in order:
            use_b = [1 if cue[i] > t else 0 for i in range(n)]
            acc = sum((b[i] if use_b[i] else a[i]) for i in range(n)) / n
            best_router = max(best_router, acc)
            use_b = [1 if cue[i] <= t else 0 for i in range(n)]
            acc = sum((b[i] if use_b[i] else a[i]) for i in range(n)) / n
            best_router = max(best_router, acc)
        captured = (best_router - best) / headroom if headroom > 1e-9 else 0.0

    # verdict
    if best > 0.97:
        verdict = "DEPLOY-DOMINANT"
        why = (f"one system is near-perfect (best={best:.3f}); combining cannot help. "
               "Ship the dominant system alone.")
    elif headroom < 0.03:
        verdict = "DON'T-COMBINE"
        why = (f"headroom={headroom:.3f} ~ 0: the systems fail on the SAME inputs "
               f"(orthogonality={orthogonality:.2f}). No router/ensemble/fusion will "
               "help — the problem is geometry, not architecture.")
    else:
        verdict = "COMBINE"
        why = (f"headroom={headroom:.3f} with orthogonality={orthogonality:.2f}: the "
               f"systems fail on different inputs. A router could lift accuracy "
               f"from {best:.3f} toward the {oracle:.3f} oracle.")
        if captured is not None:
            why += (f" A cheap cue captures ~{captured*100:.0f}% of that headroom "
                    "(identifiability).")

    return {"n": n, "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
            "both_correct": round(both, 4), "only_a": round(only_a, 4),
            "only_b": round(only_b, 4), "neither": round(neither, 4),
            "best_single": round(best, 4), "oracle": round(oracle, 4),
            "headroom": round(headroom, 4), "error_orthogonality": round(orthogonality, 4),
            "cue_captured_fraction": (round(captured, 4) if captured is not None else None),
            "verdict": verdict, "explanation": why}


def main():
    ap = argparse.ArgumentParser(description="Headroom Meter — should you combine two systems?")
    ap.add_argument("--a", help="per-item correctness of system A (json/csv/txt of 0/1)")
    ap.add_argument("--b", help="per-item correctness of system B")
    ap.add_argument("--cue", default=None, help="optional per-item cue (one float/line)")
    ap.add_argument("--stdin", action="store_true", help='read {"a":[...],"b":[...]} from stdin')
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    if args.stdin:
        payload = json.loads(sys.stdin.read())
        a = [int(bool(x)) for x in payload["a"]]
        b = [int(bool(x)) for x in payload["b"]]
        cue = [float(x) for x in payload["cue"]] if "cue" in payload else None
    else:
        if not (args.a and args.b):
            raise SystemExit("provide --a and --b (or --stdin)")
        a, b = _load(args.a), _load(args.b)
        cue = None
        if args.cue:
            cue = [float(x.strip().split(",")[-1]) for x in
                   Path(args.cue).read_text().splitlines() if x.strip()]

    result = analyze(a, b, cue)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    r = result
    print("=== Headroom Meter ===")
    print(f"  items: {r['n']}   acc(A)={r['acc_a']}  acc(B)={r['acc_b']}")
    print(f"  failure overlap:  both={r['both_correct']}  onlyA={r['only_a']}  "
          f"onlyB={r['only_b']}  neither={r['neither']}")
    print(f"  best_single={r['best_single']}   oracle={r['oracle']}   "
          f"HEADROOM={r['headroom']}")
    print(f"  error_orthogonality={r['error_orthogonality']}"
          + (f"   cue_captured={r['cue_captured_fraction']}" if r['cue_captured_fraction'] is not None else ""))
    print(f"  >>> VERDICT: {r['verdict']}")
    print(f"      {r['explanation']}")


if __name__ == "__main__":
    main()
