#!/usr/bin/env python3
"""Compare the CPU FP-Growth executable on main and the current branch."""

import csv
import datetime
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = {
    "iris": ["iris/iris.data", "0.2", "0.7"],
    "bank": ["bank+marketing/bank_clean.csv", "0.01", "0.6", "--class", "yes", "--max-length", "3", "--top", "50"],
}
REPETITIONS = 5


def build(root):
    subprocess.run(["make", "fp_growth"], cwd=root, check=True)


def command(root, args):
    return [str(root / "fp_growth"), str(root / args[0]), *args[1:]]


def output(root, args):
    return subprocess.run(command(root, args), check=True, capture_output=True).stdout


def run(command_args):
    start = time.perf_counter()
    pid = os.fork()
    if pid == 0:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.execv(command_args[0], command_args)
    _, status, usage = os.wait4(pid, 0)
    if os.waitstatus_to_exitcode(status):
        raise RuntimeError(f"command failed: {command_args}")
    return {
        "elapsed_s": time.perf_counter() - start,
        "user_s": usage.ru_utime,
        "system_s": usage.ru_stime,
        "peak_rss_mib": usage.ru_maxrss / 1024,
        "major_faults": usage.ru_majflt,
    }


def write_report(rows, baseline_ref):
    results_path = ROOT / "benchmarks/results_cpu.csv"
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CPU FP-Growth benchmark",
        "",
        f"Date (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"Platform: {platform.platform()}",
        f"Baseline: {baseline_ref}",
        "",
        "Five fresh-process measurements per implementation and case, after one warm-up. "
        "Each case's stdout was verified byte-for-byte before timing. Elapsed time includes "
        "startup, loading, preprocessing, mining, CPU rule counting, and formatting.",
        "",
        "| Dataset | Implementation | Median elapsed (s) | Min–max (s) | Median peak RSS (MiB) |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in CASES:
        for implementation in ("main", "integer_ids"):
            group = [row for row in rows if row["dataset"] == dataset and row["implementation"] == implementation]
            elapsed = [row["elapsed_s"] for row in group]
            memory = [row["peak_rss_mib"] for row in group]
            lines.append(
                f"| {dataset} | {implementation} | {statistics.median(elapsed):.4f} | "
                f"{min(elapsed):.4f}–{max(elapsed):.4f} | {statistics.median(memory):.2f} |"
            )
    report_path = ROOT / "benchmarks/report_cpu.md"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    baseline_ref = "origin/main"
    with tempfile.TemporaryDirectory(prefix="fp-growth-main-") as temporary:
        baseline = Path(temporary) / "main"
        subprocess.run(["git", "worktree", "add", "--detach", str(baseline), baseline_ref], cwd=ROOT, check=True)
        try:
            build(baseline)
            build(ROOT)
            roots = {"main": baseline, "integer_ids": ROOT}
            for args in CASES.values():
                if output(baseline, args) != output(ROOT, args):
                    raise RuntimeError(f"output mismatch for: {' '.join(args)}")

            rows = []
            for args in CASES.values():
                for root in roots.values():
                    run(command(root, args))
                for repetition in range(1, REPETITIONS + 1):
                    order = ("main", "integer_ids") if repetition % 2 else ("integer_ids", "main")
                    for implementation in order:
                        result = run(command(roots[implementation], args))
                        row = {"dataset": next(name for name, case in CASES.items() if case == args),
                               "implementation": implementation, "repetition": repetition, **result}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
            write_report(rows, baseline_ref)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=ROOT, check=False)
            shutil.rmtree(baseline, ignore_errors=True)


if __name__ == "__main__":
    main()
