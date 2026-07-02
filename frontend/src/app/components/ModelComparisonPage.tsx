import type { ReactNode } from 'react';
import { BarChart3 } from 'lucide-react';
import { formatStageDisplay, getRiskDisplay } from '../../utils/riskDisplay';

const CAUTION_LABEL = getRiskDisplay('MODERATE').label;
const HIGH_INTAKE_LABEL = getRiskDisplay('HIGH').label;

const TEAL = '#2E86AB';

const CARD_CLASS =
  'bg-white dark:bg-card rounded-xl border border-border/50 p-4 sm:p-5 shadow-sm';

type ModelType = 'baseline' | 'production' | 'failed' | 'research';

interface ModelRow {
  name: string;
  subtitle: string;
  type: ModelType;
  accuracy: number;
  f1_macro: number;
  auc: number | null;
  recall_low: number | null;
  recall_mod: number | null;
  recall_high: number | null;
  precision_mod: number | null;
  mcnemar_p: number | null;
  train_acc: number | null;
  test_acc: number | null;
  gap: number | null;
  status: string;
  deployed: boolean;
  note: string;
}

const MODELS: ModelRow[] = [
  {
    name: 'Rule-Based Baseline',
    subtitle: 'Clinical threshold rule',
    type: 'baseline',
    accuracy: 75.0,
    f1_macro: 0.718,
    auc: null,
    recall_low: null,
    recall_mod: 0.357,
    recall_high: 1.0,
    precision_mod: null,
    mcnemar_p: null,
    train_acc: null,
    test_acc: 75.0,
    gap: null,
    status: 'Baseline',
    deployed: false,
    note:
      'Current clinical standard. No learning, no parameters. Misses 64% of caution cases.',
  },
  {
    name: 'XGBoost v1',
    subtitle: 'Data leakage present',
    type: 'failed',
    accuracy: 75.3,
    f1_macro: 0.723,
    auc: 0.941,
    recall_low: null,
    recall_mod: 0.367,
    recall_high: 1.0,
    precision_mod: null,
    mcnemar_p: 0.5,
    train_acc: 74.8,
    test_acc: 75.3,
    gap: -0.5,
    status: 'Deprecated',
    deployed: false,
    note:
      'Ratio features directly encoded label definition. Model learned to replicate the rule. McNemar p=0.50 — no improvement.',
  },
  {
    name: 'XGBoost v3',
    subtitle: 'Production model',
    type: 'production',
    accuracy: 99.0,
    f1_macro: 0.985,
    auc: 0.997,
    recall_low: null,
    recall_mod: 0.969,
    recall_high: 1.0,
    precision_mod: null,
    mcnemar_p: 0.000001,
    train_acc: 100.0,
    test_acc: 99.0,
    gap: 1.0,
    status: 'Production',
    deployed: true,
    note:
      'Raw features + weighted clinical severity labels. McNemar p<0.0001 vs baseline. No overfitting (1% gap).',
  },
  {
    name: 'LSTM v1',
    subtitle: 'Original sequence model',
    type: 'failed',
    accuracy: 81.4,
    f1_macro: 0.765,
    auc: 0.969,
    recall_low: null,
    recall_mod: 0.357,
    recall_high: 1.0,
    precision_mod: null,
    mcnemar_p: 0.137,
    train_acc: null,
    test_acc: 81.4,
    gap: null,
    status: 'Deprecated',
    deployed: false,
    note:
      'Same caution recall as baseline (0.357). Architecture alone does not fix label problem. McNemar p=0.137 — not significant.',
  },
  {
    name: 'LSTM v3',
    subtitle: 'Production sequence model',
    type: 'production',
    accuracy: 91.8,
    f1_macro: 0.915,
    auc: 0.984,
    recall_low: null,
    recall_mod: 0.908,
    recall_high: 0.884,
    precision_mod: null,
    mcnemar_p: 0.000001,
    train_acc: 93.0,
    test_acc: 91.8,
    gap: 1.2,
    status: 'Production',
    deployed: true,
    note:
      'Weighted labels + masking + occasion encoding. 6-meal sequence classifier. McNemar p<0.000001 vs baseline.',
  },
  {
    name: 'HMM Supervised',
    subtitle: 'Hidden Markov Model',
    type: 'research',
    accuracy: 67.8,
    f1_macro: 0.67,
    auc: 0.84,
    recall_low: null,
    recall_mod: 0.602,
    recall_high: 0.626,
    precision_mod: null,
    mcnemar_p: null,
    train_acc: 63.8,
    test_acc: 67.8,
    gap: -4.0,
    status: 'Research only',
    deployed: false,
    note:
      'Too simple for production. Negative gap indicates underfitting. Transition matrix used for next-meal probability forecast.',
  },
];

const MCNEMAR_TESTS = [
  { comparison: 'Baseline vs XGBoost v1', pValue: '0.500', significant: false },
  { comparison: 'Baseline vs XGBoost v3', pValue: '<0.0001', significant: true },
  { comparison: 'Baseline vs LSTM v1', pValue: '0.137', significant: false },
  { comparison: 'Baseline vs LSTM v3', pValue: '<0.000001', significant: true },
  { comparison: 'LSTM v1 vs LSTM v3', pValue: '0.00003', significant: true },
  { comparison: 'XGBoost v3 vs LSTM v3', pValue: '0.041', significant: true },
];

const OVERFITTING_ROWS = [
  { model: 'Rule Baseline', train: null, test: 75.0, gap: null, verdict: 'No params' },
  { model: 'XGBoost v1', train: 74.8, test: 75.3, gap: -0.5, verdict: 'Leakage' },
  { model: 'XGBoost v3', train: 100.0, test: 99.0, gap: 1.0, verdict: '✓ Clean' },
  { model: 'LSTM v3', train: 93.0, test: 91.8, gap: 1.2, verdict: '✓ Clean' },
  { model: 'HMM', train: 63.8, test: 67.8, gap: -4.0, verdict: 'Underfitting' },
];

const STAGE_BREAKDOWN = [
  { stage: 'G2', n: 238, accuracy: 99.6, modRecall: 0.988, highRecall: 1.0 },
  { stage: 'G3a', n: 36, accuracy: 97.2, modRecall: 0.857, highRecall: 1.0 },
  { stage: 'G3b', n: 19, accuracy: 94.7, modRecall: 0.8, highRecall: 1.0 },
  { stage: 'G4', n: 3, accuracy: null, modRecall: null, highRecall: null },
];

interface ModelComparisonProps {
  isDark: boolean;
  theme: Record<string, string>;
}

function formatPct(value: number | null): string {
  if (value === null) return 'N/A';
  return `${value.toFixed(1)}%`;
}

function formatRecall(value: number | null): string {
  if (value === null) return 'N/A';
  return value.toFixed(3);
}

function modRecallClass(value: number | null): string {
  if (value === null) return 'text-muted-foreground';
  if (value >= 0.9) return 'text-green-600 dark:text-green-400 font-bold';
  if (value >= 0.6) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-600 dark:text-red-400 font-bold';
}

function statusBadgeClass(status: string): string {
  if (status === 'Production') {
    return 'bg-teal-50 text-teal-700 border-teal-200 dark:bg-teal-900/30 dark:text-teal-300 dark:border-teal-800';
  }
  if (status === 'Deprecated') {
    return 'bg-gray-100 text-gray-600 border-gray-200 dark:bg-gray-800/50 dark:text-gray-400 dark:border-gray-700';
  }
  if (status === 'Baseline') {
    return 'bg-slate-100 text-slate-700 border-slate-200 dark:bg-slate-800/50 dark:text-slate-300 dark:border-slate-700';
  }
  return 'bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-900/30 dark:text-purple-300 dark:border-purple-800';
}

function rowClass(type: ModelType): string {
  if (type === 'baseline') return 'bg-muted/40 dark:bg-muted/20';
  if (type === 'production') {
    return 'bg-teal-50/80 dark:bg-teal-900/15 font-medium border-l-4 border-l-teal-600';
  }
  if (type === 'failed' || type === 'research') return 'text-muted-foreground';
  return '';
}

function gapVerdictClass(gap: number | null, verdict: string): string {
  if (gap === null) return 'text-muted-foreground';
  if (verdict.includes('Clean')) return 'text-green-600 dark:text-green-400 font-medium';
  if (Math.abs(gap) < 5 && verdict === 'Leakage') return 'text-amber-600 dark:text-amber-400';
  if (gap < -2) return 'text-slate-600 dark:text-slate-400';
  if (Math.abs(gap) < 2) return 'text-green-600 dark:text-green-400 font-medium';
  return 'text-amber-600 dark:text-amber-400';
}

function SectionCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <div className={CARD_CLASS}>
      <div className="mb-4">
        <h2 className="text-base font-semibold text-foreground">{title}</h2>
        {subtitle && <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

export function ModelComparisonPage(_props: ModelComparisonProps) {
  return (
    <div className="max-w-[1600px] mx-auto px-4 py-6 space-y-5 bg-muted/30 dark:bg-transparent rounded-2xl">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <BarChart3 size={22} style={{ color: TEAL }} />
          <h1 className="text-2xl font-semibold text-foreground">Model Comparison</h1>
        </div>
        <p className="text-sm text-muted-foreground">
          Performance of all trained models against the rule-based baseline
        </p>
      </div>

      {/* Section 1 — Key Finding Banner */}
      <div
        className="rounded-xl border p-4 sm:p-5 shadow-sm"
        style={{
          background: 'linear-gradient(135deg, rgba(46,134,171,0.12) 0%, rgba(39,174,96,0.08) 100%)',
          borderColor: 'rgba(46,134,171,0.25)',
        }}
      >
        <p className="text-sm sm:text-base font-semibold text-foreground leading-relaxed">
          The ML Gap: Rule-based baseline misses{' '}
          <span className="text-red-600 dark:text-red-400">64%</span> of {CAUTION_LABEL.toLowerCase()} cases.
          Production meal checker (v3) misses only{' '}
          <span style={{ color: TEAL }}>3%</span>.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
          <div className={`${CARD_CLASS} !p-3`}>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Baseline {CAUTION_LABEL} Recall
            </p>
            <p className="text-2xl font-bold text-red-600 dark:text-red-400 mt-1">0.357</p>
            <p className="text-xs text-muted-foreground mt-1">64% of cases missed</p>
          </div>
          <div className={`${CARD_CLASS} !p-3`}>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Production checker {CAUTION_LABEL} Recall
            </p>
            <p className="text-2xl font-bold mt-1" style={{ color: TEAL }}>
              0.969
            </p>
            <p className="text-xs text-muted-foreground mt-1">Only 3% missed</p>
          </div>
          <div className={`${CARD_CLASS} !p-3`}>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              McNemar p-value
            </p>
            <p className="text-2xl font-bold mt-1" style={{ color: TEAL }}>
              &lt; 0.0001
            </p>
            <p className="text-xs text-muted-foreground mt-1">Statistically significant</p>
          </div>
        </div>
      </div>

      {/* Section 2 — Main Comparison Table */}
      <SectionCard title="Model Performance Comparison">
        <div className="overflow-x-auto -mx-1">
          <table className="w-full text-sm min-w-[880px]">
            <thead>
              <tr className="border-b border-border/60 text-left">
                <th className="py-2.5 pr-3 font-semibold text-muted-foreground">Model</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground">Type</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">Accuracy</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">F1 Macro</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">Discrimination</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">{CAUTION_LABEL} Recall</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">{HIGH_INTAKE_LABEL} Recall</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">Train/Test Gap</th>
                <th className="py-2.5 pl-2 font-semibold text-muted-foreground">Status</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((model) => (
                <tr key={model.name} className={`border-b border-border/40 ${rowClass(model.type)}`}>
                  <td className="py-3 pr-3">
                    <div className="font-semibold text-foreground">{model.name}</div>
                    <div className="text-xs text-muted-foreground">{model.subtitle}</div>
                    {model.deployed && (
                      <span className="inline-flex items-center gap-1 text-xs text-green-600 dark:text-green-400 mt-1 font-medium">
                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-500" />
                        Live
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-2 capitalize text-muted-foreground">{model.type}</td>
                  <td className="py-3 px-2 text-right tabular-nums">{formatPct(model.accuracy)}</td>
                  <td className="py-3 px-2 text-right tabular-nums">{model.f1_macro.toFixed(3)}</td>
                  <td className="py-3 px-2 text-right tabular-nums">
                    {model.auc === null ? (
                      <span className="text-muted-foreground">N/A</span>
                    ) : (
                      model.auc.toFixed(3)
                    )}
                  </td>
                  <td className={`py-3 px-2 text-right tabular-nums ${modRecallClass(model.recall_mod)}`}>
                    {formatRecall(model.recall_mod)}
                  </td>
                  <td className="py-3 px-2 text-right tabular-nums">
                    {formatRecall(model.recall_high)}
                  </td>
                  <td className="py-3 px-2 text-right tabular-nums">
                    {model.gap === null ? (
                      <span className="text-muted-foreground">N/A</span>
                    ) : (
                      `${model.gap > 0 ? '+' : ''}${model.gap.toFixed(1)}%`
                    )}
                  </td>
                  <td className="py-3 pl-2">
                    <span
                      className={`inline-block text-xs font-medium px-2 py-0.5 rounded-full border ${statusBadgeClass(model.status)}`}
                    >
                      {model.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 space-y-2">
          {MODELS.map((model) => (
            <p key={`note-${model.name}`} className="text-xs text-muted-foreground leading-relaxed">
              <span className="font-semibold text-foreground">{model.name}:</span> {model.note}
            </p>
          ))}
        </div>
      </SectionCard>

      {/* Section 3 — McNemar Test Results */}
      <SectionCard
        title="Statistical Significance (McNemar Tests)"
        subtitle="Does the improvement over baseline happen by chance?"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/60 text-left">
                <th className="py-2.5 pr-3 font-semibold text-muted-foreground">Comparison</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">p-value</th>
                <th className="py-2.5 pl-2 font-semibold text-muted-foreground">Significant?</th>
              </tr>
            </thead>
            <tbody>
              {MCNEMAR_TESTS.map((row) => (
                <tr key={row.comparison} className="border-b border-border/40">
                  <td className="py-2.5 pr-3 text-foreground">{row.comparison}</td>
                  <td className="py-2.5 px-2 text-right tabular-nums">{row.pValue}</td>
                  <td
                    className={`py-2.5 pl-2 font-medium ${
                      row.significant
                        ? 'text-green-600 dark:text-green-400'
                        : 'text-red-600 dark:text-red-400'
                    }`}
                  >
                    {row.significant ? '✓ Yes' : '✗ No'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
          v1 models (XGBoost v1, LSTM v1) show no statistically significant improvement over the rule
          baseline — confirming that the labeling methodology, not the architecture, was the root
          cause of {CAUTION_LABEL.toLowerCase()} case detection failure.
        </p>
      </SectionCard>

      {/* Section 4 — Overfitting Analysis */}
      <SectionCard title="Overfitting Analysis">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/60 text-left">
                <th className="py-2.5 pr-3 font-semibold text-muted-foreground">Model</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">Train</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">Test</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">Gap</th>
                <th className="py-2.5 pl-2 font-semibold text-muted-foreground">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {OVERFITTING_ROWS.map((row) => (
                <tr key={row.model} className="border-b border-border/40">
                  <td className="py-2.5 pr-3 text-foreground">{row.model}</td>
                  <td className="py-2.5 px-2 text-right tabular-nums">
                    {row.train === null ? 'N/A' : `${row.train.toFixed(1)}%`}
                  </td>
                  <td className="py-2.5 px-2 text-right tabular-nums">{row.test.toFixed(1)}%</td>
                  <td className="py-2.5 px-2 text-right tabular-nums">
                    {row.gap === null ? 'N/A' : `${row.gap > 0 ? '+' : ''}${row.gap.toFixed(1)}%`}
                  </td>
                  <td className={`py-2.5 pl-2 ${gapVerdictClass(row.gap, row.verdict)}`}>
                    {row.verdict}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
          5-fold cross-validation:
          <br />
          <span className="text-foreground font-medium">XGBoost v3:</span> 99.80% ± 0.17%{' '}
          <span className="text-green-600 dark:text-green-400">✓ Stable</span>
          <br />
          <span className="text-foreground font-medium">LSTM v3:</span> 92.41% ± 1.06%{' '}
          <span className="text-green-600 dark:text-green-400">✓ Stable</span>
        </p>
      </SectionCard>

      {/* Section 5 — Per-stage breakdown */}
      <SectionCard
        title={`Per-Stage Breakdown (Production meal checker v3)`}
        subtitle="Performance varies by stage due to training dataset class imbalance"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/60 text-left">
                <th className="py-2.5 pr-3 font-semibold text-muted-foreground">Stage</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">N (test)</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">Accuracy</th>
                <th className="py-2.5 px-2 font-semibold text-muted-foreground text-right">{CAUTION_LABEL} Recall</th>
                <th className="py-2.5 pl-2 font-semibold text-muted-foreground text-right">{HIGH_INTAKE_LABEL} Recall</th>
              </tr>
            </thead>
            <tbody>
              {STAGE_BREAKDOWN.map((row) => (
                <tr key={row.stage} className="border-b border-border/40">
                  <td className="py-2.5 pr-3 font-semibold text-foreground">{formatStageDisplay(row.stage)}</td>
                  <td className="py-2.5 px-2 text-right tabular-nums">{row.n}</td>
                  <td className="py-2.5 px-2 text-right tabular-nums">
                    {row.accuracy === null ? '—' : `${row.accuracy.toFixed(1)}%`}
                  </td>
                  <td className={`py-2.5 px-2 text-right tabular-nums ${modRecallClass(row.modRecall)}`}>
                    {row.modRecall === null ? '—' : row.modRecall.toFixed(3)}
                  </td>
                  <td className="py-2.5 pl-2 text-right tabular-nums">
                    {row.highRecall === null ? '— (insufficient data)' : row.highRecall.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
          Performance degrades with kidney disease severity reflecting training dataset imbalance — Stage 2 patients are
          substantially overrepresented (238 vs 19 Stage 3b test samples). Stage 4 classification cannot be
          evaluated due to insufficient test samples (n=3).
        </p>
      </SectionCard>
    </div>
  );
}

export default ModelComparisonPage;
