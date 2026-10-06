"""Paper CLI: ``python -m paper tables --run-dir runs/main [--data-tier FREE]``."""
from __future__ import annotations

import argparse
import sys

from .make_tables import make_tables


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m paper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--data-tier", default="FREE", choices=["FREE", "LOW-COST", "PAID"])
    a = ap.parse_args(argv)
    print(f"wrote {make_tables(a.run_dir, a.data_tier)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
