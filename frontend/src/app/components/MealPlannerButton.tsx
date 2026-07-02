import { useEffect, useState } from 'react';
import { ChefHat } from 'lucide-react';

const LABELS = [
  'Ask me anything',
  'What should I eat?',
  'Build a meal plan',
  'Foods to avoid?',
  'Help me eat safely',
];

export function MealPlannerButton({
  onClick,
  fixedLabel,
}: {
  onClick: () => void;
  fixedLabel?: string;
}) {
  const [labelIndex, setLabelIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (fixedLabel) return;
    const interval = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setLabelIndex((i) => (i + 1) % LABELS.length);
        setVisible(true);
      }, 300);
    }, 2200);
    return () => clearInterval(interval);
  }, [fixedLabel]);

  return (
    <button
      type="button"
      onClick={onClick}
      className="fixed bottom-6 left-6 z-50 flex items-center gap-2 px-4 h-11 rounded-full shadow-lg hover:scale-105 transition-all duration-200"
      style={{
        background: 'linear-gradient(135deg, #2E86AB 0%, #1A5F7A 100%)',
      }}
      aria-label="Meal Suggestions"
    >
      <ChefHat className="w-4 h-4 text-white shrink-0" />
      <span
        className="text-white text-sm font-medium whitespace-nowrap"
        style={{
          opacity: visible ? 1 : 0,
          transition: 'opacity 300ms ease',
          minWidth: '140px',
        }}
      >
        {fixedLabel ?? LABELS[labelIndex]}
      </span>
    </button>
  );
}
