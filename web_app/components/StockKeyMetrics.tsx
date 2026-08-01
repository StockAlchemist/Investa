'use client';
import React from 'react';
import { cn } from '../lib/utils';
import {
    METRICS,
    formatMetric,
    metricTone,
    type MetricDef,
    type MetricGroup,
} from '../lib/metrics';

/**
 * The metric block the backend derives per symbol (`key_metrics` on
 * /fundamentals), keyed exactly like the heatmap payload so both read against
 * the same catalogue in lib/metrics.ts.
 */
export type KeyMetrics = Record<string, number | null | undefined>;

/**
 * Panel labels are deliberately shorter than the heatmap's. The map has a whole
 * dropdown row to name a metric in; a four-column table has about twelve
 * characters before the value starts getting pushed off, and a truncated label
 * is worse than an abbreviated one.
 */
const SHORT_LABELS: Record<string, string> = {
    eps_ttm: 'EPS (TTM)',
    eps_3y: 'EPS Growth 3Y',
    eps_5y: 'EPS Growth 5Y',
    sales_ttm: 'Sales (TTM)',
    sales_3y: 'Sales Growth 3Y',
    sales_5y: 'Sales Growth 5Y',
    op_margin: 'Oper. Margin',
    lt_debt_eq: 'LT Debt/Eq',
    debt_equity: 'Debt/Eq',
    rel_volume: 'Rel. Volume',
    analyst: 'Analyst Consensus',
    earnings_days: 'Next Earnings',
    yield: 'Div Yield',
};

/** Below this share of a metric's clamp, a reading is close enough to typical
 *  that colouring it a verdict says more than the data does. The heatmap
 *  tooltip has no dead zone — it shows one value at a time; thirty-odd rows
 *  need the unremarkable ones to recede. */
const DEAD_ZONE = 0.15;

/** The four groups the detail window shows. Performance is deliberately absent:
 *  the Chart tab plots it over any horizon, which beats eleven more rows. */
const PANEL_GROUPS: MetricGroup[] = ['Valuation', 'Earnings & Sales', 'Profitability', 'Market'];

interface ExtraRow {
    key: string;
    label: string;
    value: number | null | undefined;
    format: (v: number) => string;
}

/** "58.4M" — a share count, which is not money and must not wear a currency
 *  sign the way `formatCompactCap` does. */
function compactCount(v: number): string {
    if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
    return v.toFixed(0);
}

function displayValue(metric: MetricDef, value: number | null | undefined): string {
    const text = formatMetric(value, metric.format, metric.key);
    // A magnitude has no sign to report: "+3.50%" reads as a change in the yield
    // rather than as the yield itself.
    return metric.scale === 'sequential' ? text.replace(/^\+/, '') : text;
}

function MetricRow({ label, value, tone, title }: {
    label: string;
    value: string;
    tone: string;
    title?: string;
}) {
    return (
        <div className="flex items-baseline justify-between gap-2 py-[5px] border-b border-border/30 last:border-b-0">
            <span className="text-[11px] text-muted-foreground truncate" title={title || label}>{label}</span>
            <span className={cn('text-[12.5px] font-semibold tabular-nums whitespace-nowrap', tone)}>
                {value}
            </span>
        </div>
    );
}

/**
 * The valuation / earnings / profitability / market readings for one company,
 * as four dense grouped panels.
 *
 * Deliberately not thirty-five stat cards: the same figures as cards would be
 * three screens of scrolling and no structure. Grouping is what makes the block
 * scannable — a reader looking for leverage goes to one panel rather than
 * scanning everything — and the tight rows fit the whole picture in roughly the
 * space six cards used to take.
 */
export default function StockKeyMetrics({ metrics, beta, averageVolume, className }: {
    metrics?: KeyMetrics;
    /** From the fundamentals blob rather than the metric block; shown beside the
     *  other market readings because that is where a reader looks for it. */
    beta?: number | null;
    averageVolume?: number | null;
    className?: string;
}) {
    if (!metrics) return null;

    // Market gets the two readings that live on the fundamentals blob itself.
    // Neither is tinted: a beta of 1.4 is not "worse" than 0.6, it is a
    // different bet, and average volume is a fact about liquidity, not quality.
    const extraRows: Record<string, ExtraRow[]> = {
        Market: [
            { key: 'beta', label: 'Beta', value: beta, format: (v) => v.toFixed(2) },
            { key: 'avg_volume', label: 'Avg Volume', value: averageVolume, format: compactCount },
        ],
    };

    const panels = PANEL_GROUPS.map(group => {
        const defs = METRICS.filter(m => m.group === group);
        const extras = (extraRows[group] || []).filter(r => r.value != null);
        const present = defs.filter(m => metrics[m.field] != null).length + extras.length;
        return { group, defs, extras, present };
    }).filter(p => p.present > 0);

    if (panels.length === 0) return null;

    return (
        <div className={cn('space-y-3', className)}>
            <div className="flex items-baseline justify-between gap-3 flex-wrap">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                    <span className="w-1.5 h-5 rounded-full bg-indigo-500" />
                    Key Metrics
                </h3>
                <p className="text-[10px] text-muted-foreground">
                    <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Green</span> beats,{' '}
                    <span className="text-rose-600 dark:text-rose-400 font-semibold">red</span> trails a typical
                    S&amp;P 500 company on that measure
                </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                {panels.map(({ group, defs, extras }) => (
                    <div key={group} className="bg-muted/50 rounded-xl px-3 py-2.5">
                        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-1.5">
                            {group}
                        </p>
                        {defs.map(m => {
                            const value = metrics[m.field];
                            return (
                                <MetricRow
                                    key={m.key}
                                    label={SHORT_LABELS[m.key] || m.label}
                                    title={m.label}
                                    // An absent reading is shown as absent rather than
                                    // dropped: which figures a company does not publish
                                    // is itself worth seeing, and a fixed row set keeps
                                    // the four panels comparable.
                                    value={value == null ? '–' : displayValue(m, value)}
                                    tone={value == null ? 'text-muted-foreground/40' : metricTone(value, m, DEAD_ZONE)}
                                />
                            );
                        })}
                        {extras.map(r => (
                            <MetricRow
                                key={r.key}
                                label={r.label}
                                value={r.format(r.value as number)}
                                tone="text-foreground"
                            />
                        ))}
                    </div>
                ))}
            </div>
        </div>
    );
}
