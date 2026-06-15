"""
recommender.py
GuidaPlate — Food recommendation engine using Rwanda-specific food database
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.config import DIETARY_RISK_THRESHOLDS, FOOD_DATABASE_CSV

_recommender: FoodRecommender | None = None

NUTRIENT_COLUMNS: dict[str, str] = {
    "potassium": "potassium_mg",
    "phosphorus": "phosphorus_mg",
    "protein": "protein_g",
    "protein_per_kg": "protein_g",
    "sodium": "sodium_mg",
}

SUBSTITUTE_OUTPUT_KEYS = [
    "english",
    "french",
    "kinyarwanda",
    "category",
    "potassium_mg",
    "phosphorus_mg",
    "protein_g",
    "sodium_mg",
    "ckd_stage_safe",
    "notes",
    "reason",
]


class FoodRecommender:
    """Food substitute and lookup engine backed by the GuidaPlate food database."""

    def __init__(self) -> None:
        try:
            if not Path(FOOD_DATABASE_CSV).exists():
                raise FileNotFoundError(
                    f"Food database not found at {FOOD_DATABASE_CSV}. "
                    "Ensure backend/data/food_database.csv exists."
                )
            self.foods = pd.read_csv(FOOD_DATABASE_CSV)
            self.thresholds = DIETARY_RISK_THRESHOLDS
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            raise

    @staticmethod
    def _stage_to_number(ckd_stage: str) -> int:
        mapping = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4, "G5": 5}
        if ckd_stage not in mapping:
            raise ValueError(f"Unknown CKD stage: {ckd_stage!r}")
        return mapping[ckd_stage]

    @staticmethod
    def _parse_stage_safe(ckd_stage_safe: str, stage_number: int) -> bool:
        if pd.isna(ckd_stage_safe) or not str(ckd_stage_safe).strip():
            return False
        text = str(ckd_stage_safe).strip()
        if "-" in text:
            parts = text.split("-", 1)
            low, high = int(parts[0]), int(parts[1])
            return low <= stage_number <= high
        return stage_number == int(text)

    @staticmethod
    def _sanitize_value(value: object) -> object:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, (pd.Int64Dtype,)) or (
            hasattr(value, "dtype") and str(getattr(value, "dtype", "")) == "Int64"
        ):
            return int(value)
        if isinstance(value, (int,)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, (float,)):
            return float(value)
        return value

    def _row_to_dict(self, row: pd.Series, extra: dict | None = None) -> dict:
        result = {
            col: self._sanitize_value(row[col]) for col in row.index
        }
        if extra:
            result.update(extra)
        return result

    def _find_food_by_english(self, food_name: str) -> pd.Series | None:
        name = food_name.strip().lower()
        if not name:
            return None

        english = self.foods["english"].fillna("").str.lower()
        exact = self.foods[english == name]
        if not exact.empty:
            return exact.iloc[0]

        partial = self.foods[english.str.contains(name, regex=False, na=False)]
        if not partial.empty:
            return partial.iloc[0]
        return None

    def _nutrient_column(self, nutrient: str) -> str | None:
        return NUTRIENT_COLUMNS.get(nutrient.lower())

    def _build_reason(
        self,
        queried: pd.Series,
        candidate: pd.Series,
        exceeded_nutrients: list[str],
        category: str,
    ) -> str:
        parts: list[str] = []
        for nutrient in exceeded_nutrients:
            col = self._nutrient_column(nutrient)
            if col is None:
                continue
            queried_val = float(queried[col])
            candidate_val = float(candidate[col])
            label = nutrient.replace("_", " ")
            unit = "mg" if col.endswith("_mg") else "g"
            parts.append(
                f"Lower {label} ({candidate_val:g}{unit} vs {queried_val:g}{unit})"
            )
        improvements = ", ".join(parts)
        return f"{improvements} — same category ({category})"

    def _improvement_score(
        self,
        queried: pd.Series,
        candidate: pd.Series,
        exceeded_nutrients: list[str],
    ) -> float:
        total = 0.0
        for nutrient in exceeded_nutrients:
            col = self._nutrient_column(nutrient)
            if col is None:
                continue
            queried_val = float(queried[col])
            candidate_val = float(candidate[col])
            if queried_val > 0:
                total += (queried_val - candidate_val) / queried_val * 100.0
            elif candidate_val < queried_val:
                total += 100.0
        return total

    def get_substitutes(
        self,
        food_name: str,
        ckd_stage: str,
        risk_label: str,
        exceeded_nutrients: list[str],
    ) -> list[dict]:
        if risk_label == "LOW":
            return []

        if not exceeded_nutrients:
            return []

        queried = self._find_food_by_english(food_name)
        if queried is None:
            return []

        category = str(queried["category"]) if pd.notna(queried["category"]) else ""
        if category == "Other":
            return []

        stage_number = self._stage_to_number(ckd_stage)
        queried_id = queried["food_id"]

        candidates = self.foods[
            (self.foods["category"] == category) & (self.foods["food_id"] != queried_id)
        ].copy()

        if candidates.empty:
            return []

        safe_mask = candidates["ckd_stage_safe"].apply(
            lambda s: self._parse_stage_safe(s, stage_number)
        )
        candidates = candidates[safe_mask]

        for nutrient in exceeded_nutrients:
            col = self._nutrient_column(nutrient)
            if col is None:
                return []
            queried_val = float(queried[col])
            candidates = candidates[candidates[col] < queried_val]

        if candidates.empty:
            return []

        candidates = candidates.copy()
        candidates["_score"] = candidates.apply(
            lambda row: self._improvement_score(queried, row, exceeded_nutrients),
            axis=1,
        )
        candidates = candidates.sort_values("_score", ascending=False).head(3)

        results: list[dict] = []
        for _, row in candidates.iterrows():
            reason = self._build_reason(queried, row, exceeded_nutrients, category)
            item = {
                key: self._sanitize_value(row[key])
                for key in SUBSTITUTE_OUTPUT_KEYS
                if key != "reason"
            }
            item["reason"] = reason
            results.append(item)
        return results

    def get_all_foods(
        self,
        stage: str | None = None,
        category: str | None = None,
        search: str | None = None,
    ) -> list[dict]:
        df = self.foods.copy()

        if stage is not None:
            stage_number = self._stage_to_number(stage)
            df = df[
                df["ckd_stage_safe"].apply(
                    lambda s: self._parse_stage_safe(s, stage_number)
                )
            ]

        if category is not None:
            df = df[df["category"] == category]

        if search is not None and search.strip():
            term = search.strip().lower()
            english = df["english"].fillna("").str.lower()
            french = df["french"].fillna("").str.lower()
            kinyarwanda = df["kinyarwanda"].fillna("").str.lower()
            mask = (
                english.str.contains(term, regex=False, na=False)
                | french.str.contains(term, regex=False, na=False)
                | kinyarwanda.str.contains(term, regex=False, na=False)
            )
            df = df[mask]

        return [self._row_to_dict(row) for _, row in df.iterrows()]

    def get_food_by_name(self, name: str) -> dict | None:
        term = name.strip().lower()
        if not term:
            return None

        for col in ("english", "french", "kinyarwanda"):
            series = self.foods[col].fillna("").astype(str).str.lower()
            matches = self.foods[series == term]
            if not matches.empty:
                return self._row_to_dict(matches.iloc[0])

            contains = self.foods[series.str.contains(term, regex=False, na=False)]
            if not contains.empty:
                return self._row_to_dict(contains.iloc[0])

        return None


def get_recommender() -> FoodRecommender:
    global _recommender
    if _recommender is None:
        _recommender = FoodRecommender()
    return _recommender
