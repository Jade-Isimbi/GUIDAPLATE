export interface Food {
  id: number;
  english: string;
  french: string;
  kinyarwanda: string;
  category: string;
  meal_type: string;
  protein_g: number;
  potassium_mg: number;
  phosphorus_mg: number;
  sodium_mg: number;
  energy_kcal: number;
  preparation_method: string;
  source: string;
  ckd_stage_safe: string;
  notes: string;
}

export const FOODS: Food[] = [
  { id: 1, english: "sweet potatoes", french: "Patate douce", kinyarwanda: "Ibijumba", category: "Starch", meal_type: "Any", protein_g: 1.1, potassium_mg: 311, phosphorus_mg: 25, sodium_mg: 4, energy_kcal: 101, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-3", notes: "High potassium 311mg — restrict stages 4,5" },
  { id: 2, english: "cassava", french: "Manioc", kinyarwanda: "Imyumbati", category: "Starch", meal_type: "Lunch/Dinner", protein_g: 0.6, potassium_mg: 171, phosphorus_mg: 17, sodium_mg: 2, energy_kcal: 148, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 171mg — limit at stage 5" },
  { id: 3, english: "Irish potatoes", french: "Pomme de terre", kinyarwanda: "Ibirayi", category: "Starch", meal_type: "Any", protein_g: 2, potassium_mg: 544, phosphorus_mg: 44, sodium_mg: 7, energy_kcal: 77, preparation_method: "boiled/fried", source: "USDA FDC", ckd_stage_safe: "1-2", notes: "Very high potassium 544mg — restrict stages 3,4,5" },
  { id: 4, english: "banana", french: "Banane", kinyarwanda: "Umuneke", category: "Starch", meal_type: "Any", protein_g: 0.73, potassium_mg: 358, phosphorus_mg: 22, sodium_mg: 1, energy_kcal: 89, preparation_method: "raw", source: "USDA FDC", ckd_stage_safe: "1-3", notes: "High potassium 358mg — restrict stages 4,5" },
  { id: 5, english: "maize", french: "Maïs", kinyarwanda: "Ibigori", category: "Grain", meal_type: "Breakfast/Any", protein_g: 2.9, potassium_mg: 226, phosphorus_mg: 73, sodium_mg: 2, energy_kcal: 146, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 226mg — limit at stage 5" },
  { id: 6, english: "rice", french: "Riz", kinyarwanda: "Umuceri", category: "Grain", meal_type: "Lunch/Dinner", protein_g: 2.3, potassium_mg: 20, phosphorus_mg: 48, sodium_mg: 0, energy_kcal: 119, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 7, english: "sorghum", french: "Sorgho", kinyarwanda: "Amasaka", category: "Grain", meal_type: "Breakfast/Any", protein_g: 3.4, potassium_mg: 129, phosphorus_mg: 117, sodium_mg: 2, energy_kcal: 140, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 8, english: "wheat", french: "Blé", kinyarwanda: "Ingano", category: "Grain", meal_type: "Any", protein_g: 12.2, potassium_mg: 358, phosphorus_mg: 346, sodium_mg: 2, energy_kcal: 328, preparation_method: "ground", source: "Kenya FCT 2018", ckd_stage_safe: "1-3", notes: "High potassium 358mg and very high phosphorus 346mg — restrict stages 4,5" },
  { id: 9, english: "milk", french: "Lait", kinyarwanda: "Amata", category: "Dairy", meal_type: "Any", protein_g: 3.2, potassium_mg: 136, phosphorus_mg: 96, sodium_mg: 46, energy_kcal: 70, preparation_method: "raw", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 10, english: "beef meat", french: "Bœuf", kinyarwanda: "Inyama y'inka", category: "Meat", meal_type: "Lunch/Dinner", protein_g: 28.7, potassium_mg: 230, phosphorus_mg: 194, sodium_mg: 67, energy_kcal: 197, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 230mg — limit at stage 5" },
  { id: 11, english: "fish", french: "Poisson", kinyarwanda: "Ifi", category: "Fish", meal_type: "Lunch/Dinner", protein_g: 20.5, potassium_mg: 366, phosphorus_mg: 211, sodium_mg: 56, energy_kcal: 92, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-3", notes: "High potassium 366mg — restrict stages 4,5" },
  { id: 12, english: "ghee", french: "Beurre clarifié", kinyarwanda: "Amavuta y'inka", category: "Fat/Oil", meal_type: "Cooking", protein_g: 0.3, potassium_mg: 5, phosphorus_mg: 3, sodium_mg: 2, energy_kcal: 900, preparation_method: "refined", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 13, english: "eggs", french: "Oeufs", kinyarwanda: "Amagi", category: "Egg", meal_type: "Any", protein_g: 12.5, potassium_mg: 120, phosphorus_mg: 171, sodium_mg: 147, energy_kcal: 134, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate phosphorus 171mg and sodium 147mg — limit at stage 5" },
  { id: 14, english: "chicken meat", french: "Poulet", kinyarwanda: "Inkoko", category: "Meat", meal_type: "Lunch/Dinner", protein_g: 25.5, potassium_mg: 221, phosphorus_mg: 189, sodium_mg: 68, energy_kcal: 226, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 221mg — limit at stage 5" },
  { id: 15, english: "goat meat", french: "Viande de chèvre", kinyarwanda: "Inyama y'ihene", category: "Meat", meal_type: "Lunch/Dinner", protein_g: 28.4, potassium_mg: 335, phosphorus_mg: 211, sodium_mg: 73, energy_kcal: 174, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-3", notes: "High potassium 335mg — restrict stages 4,5" },
  { id: 16, english: "cassava leaves", french: "Feuilles de manioc", kinyarwanda: "Isombe", category: "Vegetable", meal_type: "Lunch/Dinner", protein_g: 5.8, potassium_mg: 321, phosphorus_mg: 66, sodium_mg: 7, energy_kcal: 36, preparation_method: "boiled", source: "USDA FDC", ckd_stage_safe: "1-3", notes: "High potassium 321mg — restrict stages 4,5" },
  { id: 17, english: "beans", french: "Haricots secs", kinyarwanda: "Ibishyimbo", category: "Legume", meal_type: "Lunch/Dinner", protein_g: 7.9, potassium_mg: 319, phosphorus_mg: 138, sodium_mg: 3, energy_kcal: 117, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-3", notes: "High potassium 319mg — restrict stages 4,5" },
  { id: 18, english: "peas", french: "Pois secs", kinyarwanda: "Amashaza", category: "Legume", meal_type: "Lunch/Dinner", protein_g: 7.9, potassium_mg: 249, phosphorus_mg: 108, sodium_mg: 4, energy_kcal: 93, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 249mg — limit at stage 5" },
  { id: 19, english: "yams", french: "Ignames", kinyarwanda: "Amateke", category: "Starch", meal_type: "Lunch/Dinner", protein_g: 1.5, potassium_mg: 248, phosphorus_mg: 60, sodium_mg: 6, energy_kcal: 118, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 249mg — limit at stage 5" },
  { id: 20, english: "groundnuts", french: "Arachides", kinyarwanda: "Ubunyobwa", category: "Legume", meal_type: "Any", protein_g: 26.2, potassium_mg: 680, phosphorus_mg: 370, sodium_mg: 6, energy_kcal: 593, preparation_method: "roasted", source: "Kenya FCT 2018", ckd_stage_safe: "1-2", notes: "Very high potassium 680mg and phosphorus 370mg — restrict stages 3,4,5" },
  { id: 21, english: "sugar cane", french: "Canne à sucre", kinyarwanda: "Igisheke", category: "Starch", meal_type: "Any", protein_g: 0.1, potassium_mg: 150, phosphorus_mg: 22, sodium_mg: 1, energy_kcal: 37, preparation_method: "raw", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 150mg — limit at stage 5" },
  { id: 22, english: "soybeans", french: "Soja", kinyarwanda: "Soya", category: "Legume", meal_type: "Lunch/Dinner", protein_g: 34, potassium_mg: 1730, phosphorus_mg: 384, sodium_mg: 22, energy_kcal: 386, preparation_method: "boiled", source: "USDA FDC", ckd_stage_safe: "1", notes: "Extremely high potassium 1730mg and phosphorus 384mg — avoid all CKD stages" },
  { id: 23, english: "sunflower oil", french: "Huile de tournesol", kinyarwanda: "Amavuta y'igihwagari", category: "Fat/Oil", meal_type: "Cooking", protein_g: 0, potassium_mg: 0, phosphorus_mg: 0, sodium_mg: 0, energy_kcal: 884, preparation_method: "refined", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 24, english: "palm oil", french: "Huile de palme", kinyarwanda: "Amavuta ya palmiye", category: "Fat/Oil", meal_type: "Cooking", protein_g: 0, potassium_mg: 0, phosphorus_mg: 0, sodium_mg: 0, energy_kcal: 891, preparation_method: "refined", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 25, english: "tomatoes", french: "Tomates", kinyarwanda: "Inyanya", category: "Vegetable", meal_type: "Any", protein_g: 1.1, potassium_mg: 221, phosphorus_mg: 28, sodium_mg: 4, energy_kcal: 22, preparation_method: "raw", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 221mg — limit at stage 5" },
  { id: 26, english: "onions", french: "Oignons", kinyarwanda: "Ibitunguru", category: "Vegetable", meal_type: "Any", protein_g: 1.2, potassium_mg: 149, phosphorus_mg: 32, sodium_mg: 4, energy_kcal: 42, preparation_method: "raw", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 27, english: "spinach", french: "Épinard", kinyarwanda: "Epinari", category: "Vegetable", meal_type: "Lunch/Dinner", protein_g: 2.4, potassium_mg: 300, phosphorus_mg: 27, sodium_mg: 18, energy_kcal: 27, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 300mg — limit at stage 5" },
  { id: 28, english: "cabbage", french: "Chou", kinyarwanda: "Ishu", category: "Vegetable", meal_type: "Any", protein_g: 1.1, potassium_mg: 148, phosphorus_mg: 21, sodium_mg: 6, energy_kcal: 18, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 29, english: "pumpkin", french: "Citrouille", kinyarwanda: "Ibihaza", category: "Vegetable", meal_type: "Lunch/Dinner", protein_g: 0.7, potassium_mg: 178, phosphorus_mg: 31, sodium_mg: 2, energy_kcal: 32, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 30, english: "plantains", french: "Banane plantain", kinyarwanda: "Imizuzu", category: "Starch", meal_type: "Lunch/Dinner", protein_g: 0.8, potassium_mg: 260, phosphorus_mg: 24, sodium_mg: 4, energy_kcal: 97, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 260mg — limit at stage 5" },
  { id: 31, english: "mangoes", french: "Mangues", kinyarwanda: "Umwembe", category: "Fruit", meal_type: "Any", protein_g: 0.5, potassium_mg: 150, phosphorus_mg: 12, sodium_mg: 2, energy_kcal: 64, preparation_method: "raw", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 32, english: "avocados", french: "Avocats", kinyarwanda: "Avoka", category: "Fruit", meal_type: "Any", protein_g: 1.6, potassium_mg: 400, phosphorus_mg: 37, sodium_mg: 5, energy_kcal: 185, preparation_method: "raw", source: "Kenya FCT 2018", ckd_stage_safe: "1-3", notes: "High potassium 400mg — restrict stages 4,5" },
  { id: 33, english: "pineapples", french: "Ananas", kinyarwanda: "Inanasi", category: "Fruit", meal_type: "Any", protein_g: 0.5, potassium_mg: 114, phosphorus_mg: 9, sodium_mg: 1, energy_kcal: 41, preparation_method: "raw", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 34, english: "passion fruit", french: "Fruit de la passion", kinyarwanda: "Amatunda", category: "Fruit", meal_type: "Any", protein_g: 0.9, potassium_mg: 165, phosphorus_mg: 13, sodium_mg: 5, energy_kcal: 42, preparation_method: "raw", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 35, english: "oranges", french: "Oranges", kinyarwanda: "Amaronje", category: "Fruit", meal_type: "Any", protein_g: 0.7, potassium_mg: 148, phosphorus_mg: 16, sodium_mg: 3, energy_kcal: 42, preparation_method: "raw", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 36, english: "papaya", french: "Papaye", kinyarwanda: "Ipapayi", category: "Fruit", meal_type: "Any", protein_g: 0.5, potassium_mg: 210, phosphorus_mg: 11, sodium_mg: 3, energy_kcal: 36, preparation_method: "raw", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 210mg — limit at stage 5" },
  { id: 37, english: "coffee", french: "Café", kinyarwanda: "Ikawa", category: "Beverage", meal_type: "Breakfast/Any", protein_g: 0.1, potassium_mg: 37, phosphorus_mg: 3, sodium_mg: 2, energy_kcal: 2, preparation_method: "brewed", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 38, english: "tea", french: "Thé", kinyarwanda: "Icyayi", category: "Beverage", meal_type: "Breakfast/Any", protein_g: 0, potassium_mg: 37, phosphorus_mg: 1, sodium_mg: 1, energy_kcal: 1, preparation_method: "brewed", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 39, english: "green beans", french: "Haricots verts", kinyarwanda: "Imiteja", category: "Vegetable", meal_type: "Lunch/Dinner", protein_g: 1.8, potassium_mg: 209, phosphorus_mg: 38, sodium_mg: 6, energy_kcal: 31, preparation_method: "boiled", source: "USDA FDC", ckd_stage_safe: "1-4", notes: "Moderate potassium 209mg — limit at stage 5" },
  { id: 40, english: "millet", french: "Mil", kinyarwanda: "Uburo", category: "Grain", meal_type: "Breakfast/Any", protein_g: 11, potassium_mg: 195, phosphorus_mg: 285, sodium_mg: 5, energy_kcal: 378, preparation_method: "ground/boiled", source: "USDA FDC", ckd_stage_safe: "1-3", notes: "High phosphorus 285mg — restrict stages 4,5" },
  { id: 41, english: "honey", french: "Miel", kinyarwanda: "Ubuki", category: "Other", meal_type: "Any", protein_g: 0.4, potassium_mg: 52, phosphorus_mg: 4, sodium_mg: 4, energy_kcal: 304, preparation_method: "raw", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 42, english: "sour milk", french: "Lait fermenté", kinyarwanda: "Ikivuguto", category: "Dairy", meal_type: "Any", protein_g: 3.3, potassium_mg: 140, phosphorus_mg: 87, sodium_mg: 40, energy_kcal: 61, preparation_method: "fermented", source: "Kenya FCT 2018", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 43, english: "eggplant", french: "Aubergine", kinyarwanda: "Ikibiringanya", category: "Vegetable", meal_type: "Lunch/Dinner", protein_g: 0.8, potassium_mg: 211, phosphorus_mg: 21, sodium_mg: 2, energy_kcal: 27, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 211mg — limit at stage 5" },
  { id: 44, english: "carrot", french: "Carotte", kinyarwanda: "Karoti", category: "Vegetable", meal_type: "Any", protein_g: 0.7, potassium_mg: 231, phosphorus_mg: 30, sodium_mg: 31, energy_kcal: 30, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 231mg — limit at stage 5" },
  { id: 45, english: "pork", french: "Porc", kinyarwanda: "Inyama y'ingurube", category: "Meat", meal_type: "Lunch/Dinner", protein_g: 28.3, potassium_mg: 321, phosphorus_mg: 220, sodium_mg: 62, energy_kcal: 374, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-3", notes: "High potassium 321mg — restrict stages 4,5" },
  { id: 46, english: "amaranth leaves", french: "Feuilles d'amarante", kinyarwanda: "Dodo", category: "Vegetable", meal_type: "Lunch/Dinner", protein_g: 3.1, potassium_mg: 251, phosphorus_mg: 40, sodium_mg: 10, energy_kcal: 38, preparation_method: "boiled", source: "Kenya FCT 2018", ckd_stage_safe: "1-4", notes: "Moderate potassium 251mg — limit at stage 5" },
  { id: 47, english: "watermelon", french: "Pastèque", kinyarwanda: "Inkangaza", category: "Fruit", meal_type: "Any", protein_g: 0.61, potassium_mg: 112, phosphorus_mg: 11, sodium_mg: 1, energy_kcal: 30, preparation_method: "raw", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 48, english: "olive oil", french: "Huile d'olive", kinyarwanda: "Amavuta ya elayo", category: "Fat/Oil", meal_type: "Cooking", protein_g: 0, potassium_mg: 0, phosphorus_mg: 0, sodium_mg: 0, energy_kcal: 884, preparation_method: "refined", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 49, english: "lemons", french: "Citrons", kinyarwanda: "Indimu", category: "Fruit", meal_type: "Any", protein_g: 1.1, potassium_mg: 138, phosphorus_mg: 16, sodium_mg: 2, energy_kcal: 29, preparation_method: "raw", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
  { id: 50, english: "apples", french: "Pomme", kinyarwanda: "Amapome", category: "Fruit", meal_type: "Any", protein_g: 0.3, potassium_mg: 107, phosphorus_mg: 11, sodium_mg: 1, energy_kcal: 52, preparation_method: "raw", source: "USDA FDC", ckd_stage_safe: "1-5", notes: "Safe all stages — low nutrient content" },
];

export const CATEGORIES = [
  "All", "Starch", "Grain", "Meat",
  "Fish", "Dairy", "Egg", "Legume",
  "Vegetable", "Fruit", "Fat/Oil",
  "Beverage", "Other",
];

export const STAGE_THRESHOLDS = {
  G2: { potassium: 3500, phosphorus: 1000, protein: 0.8, sodium: 2300 },
  G3a: { potassium: 3000, phosphorus: 800, protein: 0.6, sodium: 2300 },
  G3b: { potassium: 3000, phosphorus: 800, protein: 0.6, sodium: 2300 },
  G4: { potassium: 2500, phosphorus: 700, protein: 0.55, sodium: 2300 },
};

export type CKDStage = keyof typeof STAGE_THRESHOLDS;

export function potassiumColor(mg: number): string {
  if (mg > 300) return "#E74C3C";
  if (mg >= 200) return "#F39C12";
  return "#27AE60";
}

export function isSafeForStage(ckdStageSafe: string, stageNum: number): boolean {
  if (!ckdStageSafe.includes("-")) {
    return parseInt(ckdStageSafe, 10) === stageNum;
  }
  const [low, high] = ckdStageSafe.split("-").map(Number);
  return stageNum >= low && stageNum <= high;
}

export function getDefaultGrams(category: string): number {
  const map: Record<string, number> = {
    Grain: 150,
    Starch: 150,
    Meat: 100,
    Fish: 100,
    Dairy: 240,
    Egg: 55,
    Legume: 150,
    Vegetable: 100,
    Fruit: 150,
    "Fat/Oil": 15,
    Beverage: 240,
    Other: 50,
  };
  return map[category] ?? 100;
}
