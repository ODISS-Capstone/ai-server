#!/usr/bin/env python3
"""Remove demo speaker memory so story #1 (registration) can be replayed."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset ODISS demo speaker data.")
    parser.add_argument(
        "--speaker-id",
        default="demo_kimyoungsu",
        help="Speaker id used in demo (default: demo_kimyoungsu)",
    )
    parser.add_argument(
        "--md-root",
        type=Path,
        default=Path("data/md_database"),
        help="MD database root (default: data/md_database)",
    )
    parser.add_argument(
        "--structured-root",
        type=Path,
        default=Path("data/md_database/structured_memory"),
    )
    parser.add_argument(
        "--clear-flash",
        action="store_true",
        help="Also clear shared flash/*.md session cache (recommended for live demo)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    speaker = args.speaker_id
    targets: list[Path] = [
        args.structured_root / "speakers" / speaker,
        args.md_root / "patients" / speaker,
        args.md_root / "permanent" / "patients" / speaker,
        args.md_root / "flash" / f"current_user_profile_{speaker}.md",
    ]
    # Legacy jetson test id
    if speaker == "demo_kimyoungsu":
        targets.extend(
            [
                args.structured_root / "speakers" / "jetson_live",
                args.structured_root / "speakers" / "demo_kimyoungsu2",
                args.md_root / "permanent" / "patients" / "jetson_live",
            ]
        )
        if args.clear_flash:
            flash_dir = args.md_root / "flash"
            targets.extend(
                path
                for path in flash_dir.glob("*.md")
                if path.is_file()
            )

    removed = 0
    for path in targets:
        if not path.exists():
            continue
        print(f"remove: {path}")
        if not args.dry_run:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        removed += 1

    if removed == 0:
        print(f"No data found for speaker_id={speaker!r}")
    else:
        action = "would remove" if args.dry_run else "removed"
        print(f"Done ({action} {removed} path(s)).")


if __name__ == "__main__":
    main()
