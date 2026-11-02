#!/usr/bin/env python3
"""Re-run seed 7 evolution EXACTLY as the background task did, and check the output."""
import os, subprocess, sys

cmd = [
    sys.executable, "evolve_tape_output.py",
    "--target", "swap_if_gt2",
    "--program-length", "10",
    "--population", "32768",
    "--generations", "300",
    "--random-reset-fraction", "0.10",
    "--shape-write", "0.3",
    "--shape-move", "0.2",
    "--shape-read", "0.15",
    "--shape-conditional", "0.25",
    "--seed", "7",
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:")
print(result.stdout)
print("STDERR (filtered):")
for line in result.stderr.splitlines():
    if not any(w in line.lower() for w in ['warn', 'metal', 'mps', 'xla', 'systemmem', 'maxcache', 'i0000', 'w0000', 'absl', 'init']):
        print(line)
print("exit:", result.returncode)
