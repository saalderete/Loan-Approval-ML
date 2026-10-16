"""Kept from the original scaffold. Equivalent to `loan-approval prepare` (validate + filter the HMDA file)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loan_approval.cli import main

if __name__ == "__main__":
    sys.exit(main(["prepare", *sys.argv[1:]]))
