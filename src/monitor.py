import json
import re
from collections import Counter

from src import config


def parse_log_line(line: str) -> dict | None:
    match = re.search(
        r"tenure=(\d+) MonthlyCharges=([\d.]+) Contract=(.+)$", line
    )
    if not match:
        return None
    return {
        "tenure": int(match.group(1)),
        "monthly_charges": float(match.group(2)),
        "contract": match.group(3),
    }


def main() -> None:
    log_path = config.MODEL_DIR / "predictions.log"
    if not log_path.exists():
        print("No prediction log found yet. Send some requests first.")
        return

    records = []
    with open(log_path) as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                records.append(parsed)

    if not records:
        print("Log file exists but has no parseable predictions yet.")
        return

    tenures = [r["tenure"] for r in records]
    avg_tenure = sum(tenures) / len(tenures)

    contract_counts = Counter(r["contract"] for r in records)

    print(f"Total logged predictions: {len(records)}")
    print(f"Average tenure (live traffic): {avg_tenure:.1f} months")
    print("Contract mix (live traffic):")
    for contract, count in contract_counts.most_common():
        pct = 100 * count / len(records)
        print(f"  {contract}: {count} ({pct:.1f}%)")

    print()
    print("Compare against training data averages in docs/problem-definition.md")
    print("or docs/experiments.md to spot drift manually.")


if __name__ == "__main__":
    main()