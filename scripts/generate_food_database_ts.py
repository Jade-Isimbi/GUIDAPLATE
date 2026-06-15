#!/usr/bin/env python3
"""Regenerate frontend/src/data/foodDatabase.ts FOODS array from CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "backend" / "data" / "food_database.csv"
TS_PATH = ROOT / "frontend" / "src" / "data" / "foodDatabase.ts"


def escape_ts_string(value) -> str:
    if pd.isna(value):
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


def format_number(value) -> str:
    if pd.isna(value):
        return "0"
    num = float(value)
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    rounded = round(num, 2)
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def row_to_ts_literal(row: pd.Series) -> str:
    parts = [
        f"id: {int(row['food_id'])}",
        f"english: {escape_ts_string(row['english'])}",
        f"french: {escape_ts_string(row['french'])}",
        f"kinyarwanda: {escape_ts_string(row['kinyarwanda'])}",
        f"category: {escape_ts_string(row['category'])}",
        f"meal_type: {escape_ts_string(row['meal_type'])}",
        f"protein_g: {format_number(row['protein_g'])}",
        f"potassium_mg: {format_number(row['potassium_mg'])}",
        f"phosphorus_mg: {format_number(row['phosphorus_mg'])}",
        f"sodium_mg: {format_number(row['sodium_mg'])}",
        f"energy_kcal: {format_number(row['energy_kcal'])}",
        f"preparation_method: {escape_ts_string(row['preparation_method'])}",
        f"source: {escape_ts_string(row['source'])}",
        f"ckd_stage_safe: {escape_ts_string(row['ckd_stage_safe'])}",
        f"notes: {escape_ts_string(row['notes'])}",
    ]
    return "  { " + ", ".join(parts) + " },"


def extract_preserved_sections(ts_content: str) -> tuple[str, str]:
    foods_marker = "export const FOODS: Food[] = ["
    categories_marker = "export const CATEGORIES"

    foods_start = ts_content.index(foods_marker)
    categories_start = ts_content.index(categories_marker, foods_start)

    interface_block = ts_content[:foods_start].rstrip() + "\n\n"
    tail_block = ts_content[categories_start:]
    return interface_block, tail_block


def main() -> None:
    ts_content = TS_PATH.read_text(encoding="utf-8")
    interface_block, tail_block = extract_preserved_sections(ts_content)

    df = pd.read_csv(CSV_PATH)
    literals = [row_to_ts_literal(row) for _, row in df.iterrows()]
    foods_block = "export const FOODS: Food[] = [\n" + "\n".join(literals) + "\n];\n\n"

    new_content = interface_block + foods_block + tail_block

    assert "export interface Food" in interface_block
    assert "export const CATEGORIES" in tail_block
    assert "export const STAGE_THRESHOLDS" in tail_block
    assert "export function potassiumColor" in tail_block
    assert "export function isSafeForStage" in tail_block
    assert "export function getDefaultGrams" in tail_block

    TS_PATH.write_text(new_content, encoding="utf-8")

    print(f"Total foods written: {len(df)}")
    print("Preserved: export interface Food — yes")
    print("Preserved: export const CATEGORIES — yes")
    print(
        "Preserved: STAGE_THRESHOLDS, potassiumColor, "
        "isSafeForStage, getDefaultGrams — yes"
    )


if __name__ == "__main__":
    main()
