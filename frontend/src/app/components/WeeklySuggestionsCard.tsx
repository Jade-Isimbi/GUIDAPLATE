import { useEffect, useState } from 'react';
import { authFetch, getAuthToken } from '../../utils/auth';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const VISIBLE_FOOD_LIMIT = 6;

interface FoodSuggestion {
  english: string;
  french: string | null;
  kinyarwanda: string | null;
  category: string | null;
  potassium_mg: number | null;
  phosphorus_mg: number | null;
  protein_g: number | null;
  sodium_mg: number | null;
  reason: string;
}

interface MealOccasionSuggestions {
  occasion: string;
  suggestions: FoodSuggestion[];
}

interface WeeklySuggestionsResponse {
  trajectory_risk: string;
  trajectory_confidence: number;
  flagged_nutrient: string | null;
  flagged_reason: string;
  remaining_days: number;
  suggestions_by_meal: MealOccasionSuggestions[];
  clinical_note: string;
  analysis_available: boolean;
}

const OCCASION_LABELS: Record<string, string> = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  dinner: 'Dinner',
  snack: 'Snack',
};

function FoodSuggestionItem({ food, isLast }: { food: FoodSuggestion; isLast: boolean }) {
  return (
    <li className={`min-w-0 ${isLast ? '' : 'border-b border-gray-100 pb-2 mb-2'}`}>
      <p className="text-sm font-medium text-foreground leading-snug capitalize">{food.english}</p>
      {food.category && (
        <p className="text-[11px] text-muted-foreground leading-tight">{food.category}</p>
      )}
    </li>
  );
}

export function WeeklySuggestionsCard() {
  const [data, setData] = useState<WeeklySuggestionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedOccasions, setExpandedOccasions] = useState<Set<string>>(new Set());

  useEffect(() => {
    const token = getAuthToken();
    if (!token) {
      setLoading(false);
      return;
    }

    authFetch(`${API_BASE}/api/next-meal/weekly-suggestions`)
      .then(async (r) => {
        if (r.status === 401) return null;
        if (!r.ok) throw new Error(await r.text());
        return r.json() as Promise<WeeklySuggestionsResponse>;
      })
      .then((result) => {
        setData(result);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Weekly suggestions error:', err);
        setError('Could not load food suggestions.');
        setLoading(false);
      });
  }, []);

  const toggleExpanded = (occasion: string) => {
    setExpandedOccasions((prev) => {
      const next = new Set(prev);
      if (next.has(occasion)) {
        next.delete(occasion);
      } else {
        next.add(occasion);
      }
      return next;
    });
  };

  if (loading) {
    return (
      <div className="flex items-center gap-3 py-2">
        <div className="w-4 h-4 rounded-full border-2 border-teal-600 border-t-transparent animate-spin" />
        <p className="text-sm text-muted-foreground">Loading food suggestions...</p>
      </div>
    );
  }

  if (error || !data) {
    return null;
  }

  return (
    <div className="space-y-3 min-w-0">
      <div>
        <p className="text-sm font-semibold text-foreground">
          Suggested for the rest of your week
        </p>
        <p className="text-xs text-muted-foreground mt-0.5">{data.flagged_reason}</p>
      </div>

      {!data.analysis_available && (
        <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 mb-3 flex items-center gap-2">
          <span className="text-amber-600 text-sm font-medium">
            ⚠ Pattern analysis temporarily unavailable
          </span>
          <span className="text-amber-500 text-xs">
            Showing general safe foods for your stage
          </span>
        </div>
      )}

      <Tabs defaultValue="breakfast">
        <TabsList className="w-full grid grid-cols-4 h-auto bg-transparent p-0 gap-0 rounded-none border-b border-border">
          {data.suggestions_by_meal.map((group) => (
            <TabsTrigger
              key={group.occasion}
              value={group.occasion}
              className="text-xs sm:text-sm rounded-none border-b-2 border-transparent data-[state=active]:border-teal-600 data-[state=active]:bg-transparent data-[state=active]:shadow-none"
            >
              {OCCASION_LABELS[group.occasion] || group.occasion}
            </TabsTrigger>
          ))}
        </TabsList>

        {data.suggestions_by_meal.map((group) => {
          const isExpanded = expandedOccasions.has(group.occasion);
          const hasMore = group.suggestions.length > VISIBLE_FOOD_LIMIT;
          const visibleFoods = isExpanded
            ? group.suggestions
            : group.suggestions.slice(0, VISIBLE_FOOD_LIMIT);

          return (
            <TabsContent key={group.occasion} value={group.occasion} className="mt-3">
              {group.suggestions.length === 0 ? (
                <p className="text-sm text-muted-foreground py-2">
                  No suggestions available for this occasion.
                </p>
              ) : (
                <>
                  <ul className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3">
                    {visibleFoods.map((food, index) => (
                      <FoodSuggestionItem
                        key={`${group.occasion}-${food.english}`}
                        food={food}
                        isLast={index === visibleFoods.length - 1}
                      />
                    ))}
                  </ul>
                  {hasMore && (
                    <button
                      type="button"
                      onClick={() => toggleExpanded(group.occasion)}
                      className="mt-2 text-xs font-medium text-teal-700 dark:text-teal-300 hover:underline"
                    >
                      {isExpanded
                        ? 'Show less'
                        : `Show more (${group.suggestions.length - VISIBLE_FOOD_LIMIT} more)`}
                    </button>
                  )}
                </>
              )}
            </TabsContent>
          );
        })}
      </Tabs>

      <p className="text-xs text-muted-foreground pt-1">{data.clinical_note}</p>
    </div>
  );
}
