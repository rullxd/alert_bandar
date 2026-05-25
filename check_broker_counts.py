#!/usr/bin/env python3
"""Check broker file counts in BROKER_ACTIVITY_DAILY.

Usage:
  python check_broker_counts.py [--validate]

--validate : attempt to parse JSON files to ensure they are valid (slower)

Generates `BROKER_ACTIVITY_DAILY/counts_report.json` with summary.
"""
from pathlib import Path
import json
import argparse
from collections import Counter

ROOT = Path(".")
ACTIVITY_DIR = ROOT / "BROKER_ACTIVITY_DAILY"


def is_parseable(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def main(validate: bool = False) -> int:
    if not ACTIVITY_DIR.exists():
        print(f"Directory not found: {ACTIVITY_DIR.resolve()}")
        return 2

    brokers = [p for p in ACTIVITY_DIR.iterdir() if p.is_dir()]
    if not brokers:
        print(f"No broker subdirectories found in {ACTIVITY_DIR}")
        return 0

    counts = {}
    parse_errors = {}

    for b in sorted(brokers):
        files = list(b.glob("broker_activity_*.json"))
        if validate:
            good = 0
            bad_files = []
            for f in files:
                if is_parseable(f):
                    good += 1
                else:
                    bad_files.append(str(f))
            counts[b.name] = good
            if bad_files:
                parse_errors[b.name] = bad_files
        else:
            counts[b.name] = len(files)

    counter = Counter(counts.values())
    most_common_count, most_common_freq = counter.most_common(1)[0]

    anomalies = {b: c for b, c in counts.items() if c != most_common_count}

    report = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total_brokers": len(brokers),
        "most_common_count": most_common_count,
        "most_common_freq": most_common_freq,
        "counts": counts,
        "anomalies": anomalies,
        "parse_errors": parse_errors,
    }

    out_path = ACTIVITY_DIR / "counts_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Brokers checked: {len(brokers)}")
    print(f"Most common file count: {most_common_count} (occurs {most_common_freq} brokers)")
    if anomalies:
        print("Brokers with non-matching counts:")
        for b, c in sorted(anomalies.items()):
            print(f"  - {b}: {c}")
    else:
        print("All brokers have the same file count.")

    print(f"Report written to: {out_path.resolve()}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Parse JSON files to ensure validity")
    args = parser.parse_args()
    raise SystemExit(main(args.validate))
