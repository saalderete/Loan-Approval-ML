"""Kept from the original scaffold. Equivalent to `loan-approval predict --scenario <id>`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loan_approval.cli import main

if __name__ == "__main__":
    args = sys.argv[1:] or ["--scenario", "steady-first-home"]
    sys.exit(main(["predict", *args]))
