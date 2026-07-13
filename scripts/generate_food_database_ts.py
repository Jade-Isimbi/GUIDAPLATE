#!/usr/bin/env python3
"""Regenerate frontend/src/data/foodDatabase.ts from CSV.

Syncs:
  - FOODS array (all rows)
  - RWANDAN_FOOD_IDS (food_id where is_rwandan == 1)

Usage:
  python3 scripts/generate_food_database_ts.py --check
  python3 scripts/generate_food_database_ts.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "backend" / "data" / "food_database.csv"
TS_PATH = ROOT / "frontend" / "src" / "data" / "foodDatabase.ts"

RWANDAN_COMMENT = (
    "/* GENERATED from food_database.csv is_rwandan=1 — "
    "do not hand-edit; run scripts/generate_food_database_ts.py */\n"
)

# Prior RiskAssessment hardcoded Set (pre-codegen) — bootstrap --check only.
PRIOR_HARDCODED_RWANDAN_IDS = list(range(1, 51)) + [387, 388, 389, 390]


def escape_ts_string(value: str | None) -> str:
    if value is None:
        return '""'
    text = str(value)
    text = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{text}"'


def format_number(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return "0"
    num = float(value)
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    rounded = round(num, 2)
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def row_to_ts_literal(row: dict[str, str]) -> str:
    parts = [
        f"id: {int(float(row['food_id']))}",
        f"english: {escape_ts_string(row.get('english'))}",
        f"french: {escape_ts_string(row.get('french') or '')}",
        f"kinyarwanda: {escape_ts_string(row.get('kinyarwanda') or '')}",
        f"category: {escape_ts_string(row.get('category'))}",
        f"meal_type: {escape_ts_string(row.get('meal_type'))}",
        f"protein_g: {format_number(row.get('protein_g'))}",
        f"potassium_mg: {format_number(row.get('potassium_mg'))}",
        f"phosphorus_mg: {format_number(row.get('phosphorus_mg'))}",
        f"sodium_mg: {format_number(row.get('sodium_mg'))}",
        f"energy_kcal: {format_number(row.get('energy_kcal'))}",
        f"preparation_method: {escape_ts_string(row.get('preparation_method'))}",
        f"source: {escape_ts_string(row.get('source'))}",
        f"ckd_stage_safe: {escape_ts_string(row.get('ckd_stage_safe'))}",
        f"notes: {escape_ts_string(row.get('notes'))}",
    ]
    return "  { " + ", ".join(parts) + " },"


def load_csv_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_rwandan_ids(rows: list[dict[str, str]] | None = None) -> list[int]:
    rows = rows if rows is not None else load_csv_rows()
    ids: list[int] = []
    for row in rows:
        flag = str(row.get("is_rwandan", "0")).strip()
        if flag in ("1", "1.0", "True", "true", "yes"):
            ids.append(int(float(row["food_id"])))
    return sorted(ids)


def format_id_lines(ids: list[int], per_line: int = 20) -> str:
    lines: list[str] = []
    for i in range(0, len(ids), per_line):
        chunk = ids[i : i + per_line]
        lines.append("  " + ", ".join(str(n) for n in chunk) + ",")
    return "\n".join(lines)


def rwandan_ids_block(ids: list[int], *, with_comment: bool) -> str:
    body = (
        "export const RWANDAN_FOOD_IDS: ReadonlySet<number> = new Set([\n"
        + format_id_lines(ids)
        + "\n]);"
    )
    return (RWANDAN_COMMENT + body) if with_comment else body


def strip_rwandan_block(content: str) -> str:
    pattern = re.compile(
        r"(?:/\*[^*]*GENERATED from food_database\.csv is_rwandan=1[^*]*\*/\s*)?"
        r"export const RWANDAN_FOOD_IDS\b[\s\S]*?\];\s*",
        re.MULTILINE,
    )
    return pattern.sub("", content)


def extract_preserved_sections(ts_content: str) -> tuple[str, str]:
    foods_marker = "export const FOODS: Food[] = ["
    categories_marker = "export const CATEGORIES"

    foods_start = ts_content.index(foods_marker)
    categories_start = ts_content.index(categories_marker, foods_start)

    interface_block = ts_content[:foods_start].rstrip() + "\n\n"
    tail_block = strip_rwandan_block(ts_content[categories_start:])
    return interface_block, tail_block


def find_rwandan_block(content: str) -> tuple[int, int] | None:
    pattern = re.compile(
        r"(?:/\*[^*]*GENERATED from food_database\.csv is_rwandan=1[^*]*\*/\s*)?"
        r"export const RWANDAN_FOOD_IDS\b",
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        return None
    start = match.start()
    end_marker = content.find("]);", match.end())
    if end_marker < 0:
        raise ValueError("Unclosed RWANDAN_FOOD_IDS Set")
    end = end_marker + len("]);")
    return start, end


def extract_set_ids_from_block(block: str) -> list[int]:
    inner = re.search(r"new Set\(\[(.*)\]\)", block, re.S)
    if not inner:
        raise ValueError("Could not parse RWANDAN_FOOD_IDS Set literal")
    return [int(x) for x in re.findall(r"\d+", inner.group(1))]


def insert_rwandan_into_tail(tail_block: str, rwandan_full: str) -> str:
    stage_markers = (
        "/* GENERATED from backend/clinical_constants.py",
        "export const STAGE_THRESHOLDS",
    )
    insert_at = None
    for marker in stage_markers:
        idx = tail_block.find(marker)
        if idx >= 0:
            insert_at = idx
            break
    if insert_at is None:
        return tail_block.rstrip() + "\n\n" + rwandan_full + "\n"

    return (
        tail_block[:insert_at].rstrip()
        + "\n\n"
        + rwandan_full
        + "\n\n"
        + tail_block[insert_at:].lstrip()
    )


def check_rwandan(ts_content: str, expected: list[int]) -> bool:
    found = find_rwandan_block(ts_content)
    if found is None:
        if expected == PRIOR_HARDCODED_RWANDAN_IDS:
            print(
                "CHECK OK  frontend/src/data/foodDatabase.ts — "
                "RWANDAN_FOOD_IDS export not yet present, but CSV matches "
                f"prior hardcoded list ({len(expected)} ids)"
            )
            return True
        print(
            "CHECK FAIL — RWANDAN_FOOD_IDS missing and CSV != prior hardcoded list"
        )
        print(f"  CSV:    {expected}")
        print(f"  prior:  {PRIOR_HARDCODED_RWANDAN_IDS}")
        return False

    start, end = found
    actual = extract_set_ids_from_block(ts_content[start:end])
    if actual == expected:
        print(
            f"CHECK OK  frontend/src/data/foodDatabase.ts — "
            f"RWANDAN_FOOD_IDS matches CSV ({len(expected)} ids)"
        )
        if expected == PRIOR_HARDCODED_RWANDAN_IDS:
            print(
                "         byte-identical to prior RiskAssessment hardcoded Set"
            )
        return True

    print("CHECK FAIL RWANDAN_FOOD_IDS drifted from CSV")
    print(f"  expected ({len(expected)}): {expected}")
    print(f"  found    ({len(actual)}): {actual}")
    print(f"  missing: {sorted(set(expected) - set(actual))}")
    print(f"  extra:   {sorted(set(actual) - set(expected))}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync FOODS + RWANDAN_FOOD_IDS into foodDatabase.ts from CSV."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if RWANDAN_FOOD_IDS drifts from CSV (does not write).",
    )
    args = parser.parse_args()

    rows = load_csv_rows()
    expected_ids = csv_rwandan_ids(rows)
    print(f"CSV is_rwandan=1: {len(expected_ids)} ids")

    ts_content = TS_PATH.read_text(encoding="utf-8")

    if args.check:
        return 0 if check_rwandan(ts_content, expected_ids) else 1

    interface_block, tail_block = extract_preserved_sections(ts_content)
    literals = [row_to_ts_literal(row) for row in rows]
    foods_block = "export const FOODS: Food[] = [\n" + "\n".join(literals) + "\n];\n\n"

    rwandan_full = rwandan_ids_block(expected_ids, with_comment=True)
    tail_with_rwandan = insert_rwandan_into_tail(tail_block, rwandan_full)
    new_content = interface_block + foods_block + tail_with_rwandan

    assert "export interface Food" in interface_block
    assert "export const CATEGORIES" in tail_with_rwandan
    assert "export const STAGE_THRESHOLDS" in tail_with_rwandan
    assert "export const RWANDAN_FOOD_IDS" in tail_with_rwandan
    assert "export function potassiumColor" in tail_with_rwandan

    TS_PATH.write_text(new_content, encoding="utf-8")

    written = extract_set_ids_from_block(rwandan_full)
    assert written == expected_ids
    if written == PRIOR_HARDCODED_RWANDAN_IDS:
        print(
            f"RWANDAN_FOOD_IDS written: {len(written)} — "
            "identical to prior RiskAssessment hardcoded Set"
        )
    else:
        print(
            f"RWANDAN_FOOD_IDS written: {len(written)} — "
            "UPDATED vs prior hardcoded "
            f"(missing={sorted(set(PRIOR_HARDCODED_RWANDAN_IDS) - set(written))} "
            f"extra={sorted(set(written) - set(PRIOR_HARDCODED_RWANDAN_IDS))})"
        )

    print(f"Total foods written: {len(rows)}")
    print("Preserved: export interface Food — yes")
    print("Preserved: export const CATEGORIES — yes")
    print(
        "Preserved: STAGE_THRESHOLDS, potassiumColor, "
        "isSafeForStage, getDefaultGrams — yes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
