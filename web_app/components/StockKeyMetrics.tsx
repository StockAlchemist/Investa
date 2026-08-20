'use client';
import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { cn } from '../lib/utils';
import {
    METRICS,
    formatMetric,
    metricTone,
    type MetricDef,
    type MetricGroup,
} from '../lib/metrics';
import { fetchRatios, type FinancialRatio } from '../lib/api';
import { StatementPeriod } from '../lib/statement_chart';
import { RatioChart } from './stock-detail/components/RatioChart';
import { Skeleton } from './ui/skeleton';
import { Table, LineChart as ChartIcon } from 'lucide-react';

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

const DEAD_ZONE = 0.15;

const PANEL_GROUPS: MetricGroup[] = ['Valuation', 'Earnings & Sales', 'Profitability', 'Market'];

interface ExtraRow {
    key: string;
    label: string;
    value: number | null | undefined;
    format: (v: number) => string;
}

function compactCount(v: number): string {
    if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
    return v.toFixed(0);
}

function displayValue(metric: MetricDef, value: number | null | undefined): string {
    const text = formatMetric(value, metric.format, metric.key);
    return metric.scale === 'sequential' ? text.replace(/^\+/, '') : text;
}

function MetricRow({ label, value, tone, title, onClick }: {
    label: string;
    value: string;
    tone: string;
    title?: string;
    onClick?: () => void;
}) {
    return (
        <div
            onClick={onClick}
            className={cn(
                "flex items-baseline justify-between gap-2 py-[5px] border-b border-border/30 last:border-b-0 transition-colors",
                onClick && "cursor-pointer hover:bg-muted/60 rounded px-1 -mx-1"
            )}
        >
            <span className="text-[11px] text-muted-foreground truncate" title={title || label}>{label}</span>
            <span className={cn('text-[12.5px] font-semibold tabular-nums whitespace-nowrap', tone)}>
                {value}
            </span>
        </div>
    );
}

interface MetricChartDef {
    dataKey: string;
    title: string;
    color: string;
    suffix?: string;
    compact?: boolean;
}

const METRIC_CHARTS_BY_GROUP: Record<string, MetricChartDef[]> = {
    'Valuation': [
        { dataKey: 'P/E Ratio', title: 'Price to Earnings (P/E)', color: '#10b981' },
        { dataKey: 'P/S Ratio', title: 'Price to Sales (P/S)', color: '#06b6d4' },
        { dataKey: 'P/B Ratio', title: 'Price to Book (P/B)', color: '#8b5cf6' },
        { dataKey: 'EV/EBITDA', title: 'EV / EBITDA', color: '#f59e0b' },
        { dataKey: 'EV/Sales', title: 'EV / Sales', color: '#ec4899' },
        { dataKey: 'P/FCF Ratio', title: 'Price to Free Cash Flow (P/FCF)', color: '#14b8a6' },
        { dataKey: 'Dividend Yield (%)', title: 'Dividend Yield', color: '#10b981', suffix: '%' },
    ],
    'Earnings & Sales': [
        { dataKey: 'Diluted EPS', title: 'Diluted EPS ($)', color: '#10b981' },
        { dataKey: 'Total Revenue', title: 'Total Revenue (Sales)', color: '#06b6d4', compact: true },
        { dataKey: 'Revenue Growth YoY (%)', title: 'Revenue Growth YoY', color: '#8b5cf6', suffix: '%' },
        { dataKey: 'EPS Growth YoY (%)', title: 'EPS Growth YoY', color: '#ec4899', suffix: '%' },
        { dataKey: 'Operating Margin (%)', title: 'Operating Margin', color: '#f59e0b', suffix: '%' },
    ],
    'Profitability': [
        { dataKey: 'Return on Invested Capital (ROIC) (%)', title: 'Return on Invested Capital (ROIC)', color: '#ec4899', suffix: '%' },
        { dataKey: 'Return on Equity (ROE) (%)', title: 'Return on Equity (ROE)', color: '#10b981', suffix: '%' },
        { dataKey: 'Return on Assets (ROA) (%)', title: 'Return on Assets (ROA)', color: '#06b6d4', suffix: '%' },
        { dataKey: 'Gross Profit Margin (%)', title: 'Gross Margin', color: '#8b5cf6', suffix: '%' },
        { dataKey: 'Net Profit Margin (%)', title: 'Net Margin', color: '#f59e0b', suffix: '%' },
        { dataKey: 'Free Cash Flow Margin (%)', title: 'Free Cash Flow Margin', color: '#14b8a6', suffix: '%' },
    ],
    'Balance Sheet': [
        { dataKey: 'Current Ratio', title: 'Current Ratio', color: '#10b981' },
        { dataKey: 'Quick Ratio', title: 'Quick Ratio', color: '#06b6d4' },
        { dataKey: 'Debt-to-Equity Ratio', title: 'Debt to Equity', color: '#f59e0b' },
        { dataKey: 'Long-Term Debt to Equity', title: 'LT Debt to Equity', color: '#8b5cf6' },
        { dataKey: 'Interest Coverage Ratio', title: 'Interest Coverage Ratio', color: '#ec4899' },
        { dataKey: 'Asset Turnover', title: 'Asset Turnover', color: '#06b6d4' },
        { dataKey: 'Diluted Shares Outstanding', title: 'Diluted Shares Outstanding', color: '#64748b', compact: true },
    ],
};

function StockKeyMetricsGraphs({
    symbol,
    selectedGroup,
    onSelectGroup,
    periodType,
    onSelectPeriodType,
}: {
    symbol: string;
    selectedGroup: string;
    onSelectGroup: (grp: string) => void;
    periodType: StatementPeriod;
    onSelectPeriodType: (p: StatementPeriod) => void;
}) {
    const ratiosQuery = useQuery({
        queryKey: ['stock-ratios', symbol, periodType],
        queryFn: () => fetchRatios(symbol, periodType),
        staleTime: 30 * 60 * 1000,
    });

    const chartData: FinancialRatio[] = ratiosQuery.data?.historical ? [...ratiosQuery.data.historical].reverse() : [];
    const activeCharts = METRIC_CHARTS_BY_GROUP[selectedGroup] || METRIC_CHARTS_BY_GROUP['Valuation'];

    return (
        <div className="space-y-4">
            {/* Category tabs for history graphs */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
                {Object.keys(METRIC_CHARTS_BY_GROUP).map(grp => (
                    <button
                        key={grp}
                        type="button"
                        onClick={() => onSelectGroup(grp)}
                        className={cn(
                            "px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer",
                            selectedGroup === grp
                                ? "bg-muted text-foreground border border-border shadow-sm"
                                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                        )}
                    >
                        {grp}
                    </button>
                ))}
            </div>

            {/* Graphs Grid */}
            {ratiosQuery.isLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {[0, 1, 2, 3].map(i => (
                        <Skeleton key={i} className="h-56 rounded-2xl" />
                    ))}
                </div>
            ) : chartData.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground text-sm bg-muted/30 rounded-2xl">
                    No historical statistics available for this symbol.
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {activeCharts.map(c => (
                        <RatioChart
                            key={c.dataKey}
                            periodType={periodType}
                            data={chartData}
                            dataKey={c.dataKey}
                            title={c.title}
                            color={c.color}
                            suffix={c.suffix}
                            compact={c.compact}
                        />
                    ))}
                </div>
            )}
        </div>
    );
}

export default function StockKeyMetrics({
    symbol,
    metrics,
    beta,
    averageVolume,
    className
}: {
    symbol?: string;
    metrics?: KeyMetrics;
    beta?: number | null;
    averageVolume?: number | null;
    className?: string;
}) {
    const [viewMode, setViewMode] = useState<'table' | 'graphs'>('table');
    const [selectedGroup, setSelectedGroup] = useState<string>('Valuation');
    const [periodType, setPeriodType] = useState<StatementPeriod>('quarterly');

    if (!metrics) return null;

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
            <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                        <span className="w-1.5 h-5 rounded-full bg-indigo-500" />
                        Key Metrics
                    </h3>

                    {/* View Switcher: Table vs Graphs */}
                    {symbol && (
                        <div className="flex items-center p-0.5 rounded-lg bg-muted border border-border/50">
                            <button
                                type="button"
                                onClick={() => setViewMode('table')}
                                className={cn(
                                    "flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer",
                                    viewMode === 'table'
                                        ? "bg-indigo-600 text-white shadow-sm"
                                        : "text-muted-foreground hover:text-foreground"
                                )}
                            >
                                <Table className="w-3.5 h-3.5" />
                                <span>Table</span>
                            </button>
                            <button
                                type="button"
                                onClick={() => setViewMode('graphs')}
                                className={cn(
                                    "flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer",
                                    viewMode === 'graphs'
                                        ? "bg-indigo-600 text-white shadow-sm"
                                        : "text-muted-foreground hover:text-foreground"
                                )}
                            >
                                <ChartIcon className="w-3.5 h-3.5" />
                                <span>Graphs</span>
                            </button>
                        </div>
                    )}
                </div>

                {viewMode === 'table' ? (
                    <p className="text-[10px] text-muted-foreground">
                        <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Green</span> beats,{' '}
                        <span className="text-rose-600 dark:text-rose-400 font-semibold">red</span> trails a typical
                        S&amp;P 500 company on that measure
                    </p>
                ) : (
                    <div className="flex items-center p-0.5 rounded-full bg-muted/70 border border-border/50">
                        {([
                            { id: 'quarterly', label: 'Quarterly' },
                            { id: 'annual', label: 'Annual' },
                        ] as const).map(opt => (
                            <button
                                key={opt.id}
                                type="button"
                                onClick={() => setPeriodType(opt.id)}
                                className={cn(
                                    "px-3 py-1 rounded-full text-[10px] sm:text-xs font-bold transition-all cursor-pointer",
                                    periodType === opt.id
                                        ? "bg-indigo-600 text-white shadow-sm"
                                        : "text-muted-foreground hover:text-foreground"
                                )}
                            >
                                {opt.label}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {viewMode === 'table' ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                    {panels.map(({ group, defs, extras }) => (
                        <div key={group} className="bg-muted/50 rounded-xl px-3 py-2.5 h-full">
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
                                        value={value == null ? '–' : displayValue(m, value)}
                                        tone={value == null ? 'text-muted-foreground/40' : metricTone(value, m, DEAD_ZONE)}
                                        onClick={symbol ? () => {
                                            const targetGroup = group === 'Market' ? 'Valuation' : group;
                                            setSelectedGroup(targetGroup);
                                            setViewMode('graphs');
                                        } : undefined}
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
            ) : symbol ? (
                <StockKeyMetricsGraphs
                    symbol={symbol}
                    selectedGroup={selectedGroup}
                    onSelectGroup={setSelectedGroup}
                    periodType={periodType}
                    onSelectPeriodType={setPeriodType}
                />
            ) : null}
        </div>
    );
}
