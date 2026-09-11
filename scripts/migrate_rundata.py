#!/usr/bin/env python3
"""
CLI README

Usage:
  python scripts/migrate_rundata.py [--subdir SUBDIR] [--dry-run] [--overwrite]

Arguments:
  --subdir SUBDIR   Data subdirectory under data/ to migrate. Defaults to
                    nonbatch.
  --dry-run         Report planned migrations without writing v2 files.
  --overwrite       Replace existing v2 run directories when present.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.export.path_utils import DEFAULT_RUN_SUBDIR
from src.export.run_store import migrate_legacy_subdir


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy .rundata files to v2 JSON/NPZ storage.")
    parser.add_argument(
        "--subdir",
        default=DEFAULT_RUN_SUBDIR,
        help=f"Data subdirectory under data/ (default: {DEFAULT_RUN_SUBDIR}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would migrate without writing v2 files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing v2 run directories.")
    args = parser.parse_args()

    report = migrate_legacy_subdir(
        args.subdir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

