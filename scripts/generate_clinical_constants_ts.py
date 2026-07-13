#!/usr/bin/env python3
"""Sync KDOQI daily limits from Python into frontend TypeScript mirrors.

Source of truth: backend/clinical_constants.py → KDOQI_DAILY_LIMITS

Writes (surgical replace — preserves surrounding hand-written code):
  - frontend/src/data/foodDatabase.ts        → STAGE_THRESHOLDS
  - frontend/src/utils/clinicalConstants.ts → KDOQI_DAILY_LIMITS

Usage:
  python3 scripts/generate_clinical_constants_ts.py --check   # exit 1 if drift
  python3 scripts/generate_clinical_constants_ts.py           # write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.clinical_constants import KDOQI_DAILY_LIMITS  # noqa: E402

FOOD_DB_TS = ROOT / "frontend" / "src" / "data" / "foodDatabase.ts"
CLINICAL_TS = ROOT / "frontend" / "src" / "utils" / "clinicalConstants.ts"

GENERATED_COMMENT = (
    "/* GENERATED from backend/clinical_constants.py — "
    "do not hand-edit; run scripts/generate_clinical_constants_ts.py */\n"
)

STAGE_ORDER = ("G2", "G3a", "G3b", "G4")


def fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def stage_object_lines() -> list[str]:
    lines: list[str] = []
    for stage in STAGE_ORDER:
        lim = KDOQI_DAILY_LIMITS[stage]
        lines.append(
            f"  {stage}: {{ potassium: {fmt_num(lim['potassium'])}, "
            f"phosphorus: {fmt_num(lim['phosphorus'])}, "
            f"protein: {fmt_num(lim['protein_per_kg'])}, "
            f"sodium: {fmt_num(lim['sodium'])} }},"
        )
    return lines


def stage_thresholds_block(*, with_comment: bool) -> str:
    body = (
        "export const STAGE_THRESHOLDS = {\n"
        + "\n".join(stage_object_lines())
        + "\n};"
    )
    return (GENERATED_COMMENT + body) if with_comment else body


def kdoqi_daily_limits_block(*, with_comment: bool) -> str:
    body = (
        "export const KDOQI_DAILY_LIMITS = {\n"
        + "\n".join(stage_object_lines())
        + "\n} as const;"
    )
    return (GENERATED_COMMENT + body) if with_comment else body


def find_export_block(content: str, export_prefix: str) -> tuple[int, int]:
    """Return [start, end) of optional GENERATED comment + export const …."""
    # Only swallow a preceding comment if it is our generator marker —
    # never eat an unrelated file-level docstring.
    pattern = re.compile(
        rf"(?:/\*[^*]*GENERATED from backend/clinical_constants\.py[^*]*\*/\s*)?"
        rf"{re.escape(export_prefix)}",
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        raise ValueError(f"Could not find export starting with {export_prefix!r}")

    start = match.start()
    brace_start = content.index("{", match.end() - 1)
    depth = 0
    i = brace_start
    while i < len(content):
        ch = content[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                if content.startswith(" as const", end):
                    end += len(" as const")
                if end < len(content) and content[end] == ";":
                    end += 1
                return start, end
        i += 1
    raise ValueError(f"Unbalanced braces for {export_prefix!r}")


def extract_data_export(block: str, export_prefix: str) -> str:
    """Strip a leading generated comment, return export const … only."""
    idx = block.find(export_prefix)
    if idx < 0:
        raise ValueError(f"Block missing {export_prefix!r}")
    return block[idx:]


def sync_file(
    path: Path,
    export_prefix: str,
    new_block_with_comment: str,
    expected_data_only: str,
    *,
    check_only: bool,
) -> bool:
    """Return True if file already matches expected data (no write needed for values)."""
    content = path.read_text(encoding="utf-8")
    start, end = find_export_block(content, export_prefix)
    current_block = content[start:end]
    current_data = extract_data_export(current_block, export_prefix)

    data_ok = current_data == expected_data_only
    comment_ok = current_block == new_block_with_comment

    if check_only:
        if data_ok:
            print(f"CHECK OK  {path.relative_to(ROOT)} — values match Python")
            return True
        print(f"CHECK FAIL {path.relative_to(ROOT)} — values drifted from Python")
        print("--- expected ---")
        print(expected_data_only)
        print("--- found ---")
        print(current_data)
        return False

    if comment_ok:
        print(f"WRITE SKIP {path.relative_to(ROOT)} — already up to date")
        return True

    new_content = content[:start] + new_block_with_comment + content[end:]
    # Sanity: data portion must still match Python after write
    assert expected_data_only in new_block_with_comment
    path.write_text(new_content, encoding="utf-8")
    if data_ok:
        print(
            f"WRITE OK   {path.relative_to(ROOT)} — "
            "values unchanged; added generated comment"
        )
    else:
        print(
            f"WRITE OK   {path.relative_to(ROOT)} — "
            "values updated from Python + generated comment"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync KDOQI daily limits into frontend TypeScript mirrors."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if TS values drift from Python (does not write).",
    )
    args = parser.parse_args()

    for stage in STAGE_ORDER:
        if stage not in KDOQI_DAILY_LIMITS:
            raise SystemExit(f"Missing stage {stage} in KDOQI_DAILY_LIMITS")
        lim = KDOQI_DAILY_LIMITS[stage]
        for key in ("potassium", "phosphorus", "protein_per_kg", "sodium"):
            if key not in lim:
                raise SystemExit(f"Missing {key} for {stage}")

    stage_data = stage_thresholds_block(with_comment=False)
    stage_full = stage_thresholds_block(with_comment=True)
    kdoqi_data = kdoqi_daily_limits_block(with_comment=False)
    kdoqi_full = kdoqi_daily_limits_block(with_comment=True)

    print(f"Source: backend/clinical_constants.py ({len(STAGE_ORDER)} stages)")
    print("Remap: protein_per_kg → protein (TS key preserved for call sites)")

    ok_food = sync_file(
        FOOD_DB_TS,
        "export const STAGE_THRESHOLDS",
        stage_full,
        stage_data,
        check_only=args.check,
    )
    ok_clin = sync_file(
        CLINICAL_TS,
        "export const KDOQI_DAILY_LIMITS",
        kdoqi_full,
        kdoqi_data,
        check_only=args.check,
    )

    if args.check:
        if ok_food and ok_clin:
            print("All checks passed — zero value drift.")
            return 0
        print("Drift detected.")
        return 1

    # Post-write verification: values still match
    food_after = FOOD_DB_TS.read_text(encoding="utf-8")
    clin_after = CLINICAL_TS.read_text(encoding="utf-8")
    assert stage_data in food_after, "foodDatabase.ts missing expected STAGE_THRESHOLDS"
    assert kdoqi_data in clin_after, "clinicalConstants.ts missing expected KDOQI block"
    assert "protein_per_kg" not in stage_data
    assert "protein:" in stage_data
    print("Post-write asserts: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
