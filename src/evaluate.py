"""Kept from the original scaffold. Evaluation runs inside `loan-approval train`; this lists the results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loan_approval.cli import main

if __name__ == "__main__":
    sys.exit(main(["list", *sys.argv[1:]]))
