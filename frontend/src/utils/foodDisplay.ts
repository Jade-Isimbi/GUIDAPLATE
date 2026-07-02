type FoodLike = {
  english: string;
  french?: string | null;
  kinyarwanda?: string | null;
};

export function foodTranslation(value: string | null | undefined): string | null {
  if (value == null) return null;
  const trimmed = value.trim();
  if (trimmed === '' || trimmed.toLowerCase() === 'null') return null;
  return trimmed;
}

export function displayName(food: FoodLike): string {
  return foodTranslation(food.kinyarwanda) ||
    foodTranslation(food.french) ||
    food.english;
}

export function matchesFoodQuery(food: FoodLike, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  if (food.english.toLowerCase().includes(q)) return true;
  const french = foodTranslation(food.french);
  if (french?.toLowerCase().includes(q)) return true;
  const kinyarwanda = foodTranslation(food.kinyarwanda);
  if (kinyarwanda?.toLowerCase().includes(q)) return true;
  return false;
}
