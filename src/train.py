"""Kept from the original scaffold. Equivalent to `loan-approval train` (train, calibrate, evaluate, register)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loan_approval.cli import main

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
