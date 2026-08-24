import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTheme } from 'next-themes';
import {
    Receipt,
    Scale,
    Wallet,
    Users,
    Info,
    LineChart as LineChartIcon
} from 'lucide-react';
import { fetchFinancials } from '../../../lib/api';
import {
    StatementPeriod,
    StatementRange,
    DEFAULT_CHART_METRICS,
    MAX_CHART_SERIES,
    defaultRange,
    periodsInRange,
    pickDefaultMetrics,
    SERIES_COLORS_DARK,
    SERIES_COLORS_LIGHT,
    groupBySharedScale,
    isFiniteNumber,
    periodAxisLabel,
    toggleSlot
} from '../../../lib/statement_chart';
import { formatCalendarDayMonth } from '../../../lib/market_time';
import { cn } from '../../../lib/utils';
import { Skeleton } from '../../ui/skeleton';
import { RANKING_CONFIG } from '../constants';
import { StatementChart, MetricChangeStrip } from '../components/StatementChart';
import { Sparkline } from '../components/Sparkline';

function fiscalPeriodDay(iso: string): string {
    return formatCalendarDayMonth(iso);
}

function formatCompact(val: number | undefined): string {
    if (val === undefined || val === null) return '-';
    return new Intl.NumberFormat('en-US', {
        notation: 'compact',
        maximumFractionDigits: 2
    }).format(val);
}

interface FinancialsTabProps {
    symbol: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- fundamentals payload
    fundamentals: any;
    isOpen: boolean;
}

export const FinancialsTab: React.FC<FinancialsTabProps> = ({ symbol, fundamentals, isOpen }) => {
    const [finType, setFinType] = useState<'income' | 'balance' | 'cash' | 'equity'>('income');
    const [finPeriod, setFinPeriod] = useState<StatementPeriod>('quarterly');
    const [chartSlots, setChartSlots] = useState<(string | null)[]>([]);
    const [showAllMetrics, setShowAllMetrics] = useState(false);
    const [finRange, setFinRange] = useState<StatementRange | null>(null);

    const { resolvedTheme } = useTheme();
    const isDarkMode = resolvedTheme === 'dark';
    const seriesColors = isDarkMode ? SERIES_COLORS_DARK : SERIES_COLORS_LIGHT;

    const financialsQuery = useQuery({
        queryKey: ['stock-financials', symbol, finPeriod],
        queryFn: () => fetchFinancials(symbol, finPeriod),
        enabled: isOpen && !!symbol,
        staleTime: 30 * 60 * 1000,
    });
    const financials = financialsQuery.data ?? null;

    const handleFinTypeChange = (type: 'income' | 'balance' | 'cash' | 'equity') => {
        setFinType(type);
        setChartSlots([]);
        setShowAllMetrics(false);
    };

    const handleFinPeriodChange = (period: StatementPeriod) => {
        setFinPeriod(period);
        setChartSlots([]);
        setShowAllMetrics(false);
        setFinRange(null);
    };

    const controls = (
        <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-nowrap overflow-x-auto no-scrollbar gap-2 -mx-4 px-4 sm:mx-0 sm:px-0">
                {[
                    { id: 'income', label: 'Income', fullLabel: 'Income Statement', icon: Receipt },
                    { id: 'balance', label: 'Balance', fullLabel: 'Balance Sheet', icon: Scale },
                    { id: 'cash', label: 'Cash Flow', fullLabel: 'Cash Flow', icon: Wallet },
                    { id: 'equity', label: 'Equity', fullLabel: "Shareholders' Equity", icon: Users }
                ].map((btn) => (
                    <button
                        key={btn.id}
                        onClick={() => handleFinTypeChange(btn.id as 'income' | 'balance' | 'cash' | 'equity')}
                        className={cn(
                            "flex items-center gap-2 px-3 sm:px-4 py-2 rounded-full text-[10px] sm:text-xs font-bold transition-all whitespace-nowrap flex-shrink-0 cursor-pointer",
                            finType === btn.id
                                ? "bg-indigo-500 text-white"
                                : "bg-muted/50 text-muted-foreground hover:bg-muted hover:text-foreground"
                        )}
                        title={btn.fullLabel}
                    >
                        <btn.icon className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                        <span className="hidden sm:inline">{btn.label}</span>
                    </button>
                ))}
            </div>
            <div className="flex items-center p-0.5 rounded-full bg-muted/50 flex-shrink-0">
                {([
                    { id: 'quarterly', label: 'Quarterly' },
                    { id: 'annual', label: 'Annual' }
                ] as const).map(opt => (
                    <button
                        key={opt.id}
                        onClick={() => handleFinPeriodChange(opt.id)}
                        aria-pressed={finPeriod === opt.id}
                        className={cn(
                            "px-3 sm:px-4 py-1.5 rounded-full text-[10px] sm:text-xs font-bold transition-all whitespace-nowrap cursor-pointer",
                            finPeriod === opt.id
                                ? "bg-white dark:bg-zinc-800 text-foreground shadow-sm"
                                : "text-muted-foreground hover:text-foreground"
                        )}
                    >
                        {opt.label}
                    </button>
                ))}
            </div>
        </div>
    );

    const frame = (body: React.ReactNode) => (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {controls}
            {body}
        </div>
    );

    if (financialsQuery.isPending) {
        return frame(
            <div className="space-y-6">
                <Skeleton className="h-80 w-full rounded-2xl" />
                <Skeleton className="h-64 w-full rounded-2xl" />
            </div>
        );
    }

    let rawStatement;
    switch (finType) {
        case 'income': rawStatement = financials?.financials; break;
        case 'balance': rawStatement = financials?.balance_sheet; break;
        case 'cash': rawStatement = financials?.cashflow; break;
        case 'equity': rawStatement = financials?.shareholders_equity; break;
        default: rawStatement = financials?.financials;
    }

    if (!rawStatement || !rawStatement.index || !rawStatement.index.length) {
        return frame(
            <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
                <Info className="w-8 h-8 mb-2 opacity-20" />
                <p>No {finPeriod} data available for this statement.</p>
                {finPeriod === 'quarterly' && (
                    <button
                        onClick={() => setFinPeriod('annual')}
                        className="mt-4 text-[11px] font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 hover:underline cursor-pointer"
                    >
                        Show annual instead
                    </button>
                )}
            </div>
        );
    }
    const statement = rawStatement;

    const ranking = RANKING_CONFIG[finType] || [];
    const indexedData = statement.index.map((label, idx) => ({
        label,
        data: statement.data[idx]
    }));

    const sortedData = [...indexedData].sort((a, b) => {
        const idxA = ranking.indexOf(a.label);
        const idxB = ranking.indexOf(b.label);

        if (idxA !== -1 && idxB !== -1) return idxA - idxB;
        if (idxA !== -1) return -1;
        if (idxB !== -1) return 1;
        return 0;
    });

    const currentStatement = {
        ...statement,
        index: sortedData.map(d => d.label),
        data: sortedData.map(d => d.data)
    };

    const rows = currentStatement.index.map((label, idx) => ({
        label,
        values: currentStatement.data[idx] as (number | null)[],
        ranked: ranking.includes(label)
    }));
    const chartable = rows.filter(r => r.values.some(isFiniteNumber));

    const defaultSlots: (string | null)[] = pickDefaultMetrics(
        DEFAULT_CHART_METRICS[finType] ?? [],
        chartable,
    );
    if (!defaultSlots.length && chartable.length) defaultSlots.push(chartable[0].label);

    const slots = chartSlots.length ? chartSlots : defaultSlots;
    const slotFull = slots.filter(Boolean).length >= MAX_CHART_SERIES && !slots.includes(null);

    const toggleMetric = (label: string) => {
        setChartSlots(prev => toggleSlot(prev.length ? prev : defaultSlots, label));
    };

    const activeSeries = slots
        .map((label, slot) => {
            if (!label) return null;
            const row = chartable.find(r => r.label === label);
            if (!row) return null;
            const magnitudes = row.values.filter(isFiniteNumber).map(Math.abs);
            return {
                key: `m${slot}`,
                label,
                color: seriesColors[slot % seriesColors.length],
                values: row.values,
                maxAbs: magnitudes.length ? Math.max(...magnitudes) : 0
            };
        })
        .filter((s): s is NonNullable<typeof s> => s !== null);

    const range = finRange ?? defaultRange(finPeriod);
    const columnOrder = currentStatement.columns
        .map((col, idx) => ({ col, idx }))
        .reverse()
        .slice(-periodsInRange(range, finPeriod));

    const points = columnOrder.map(({ col, idx }) => {
        const point: Record<string, string | number | null> = {
            period: col,
            label: periodAxisLabel(col, finPeriod)
        };
        activeSeries.forEach(s => {
            point[s.key] = isFiniteNumber(s.values[idx]) ? (s.values[idx] as number) : null;
        });
        return point;
    });

    const scaleGroups = groupBySharedScale(activeSeries);
    const primary = activeSeries[0];
    const chartableRanked = chartable.filter(r => r.ranked);
    const chipRows = showAllMetrics || !chartableRanked.length ? chartable : chartableRanked;

    return frame(
        <>
            <div className="bg-muted rounded-2xl p-4 sm:p-6 space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                    <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                        {finPeriod === 'quarterly' ? 'Quarterly' : 'Annual'} trend
                        <span className="ml-2 font-normal normal-case tracking-normal text-[11px] opacity-70">
                            {points.length} {finPeriod === 'quarterly' ? 'quarters' : 'years'}
                        </span>
                    </h4>
                    <div className="flex items-center gap-3">
                        <p className="hidden sm:block text-[11px] text-muted-foreground">
                            {fundamentals?.financialCurrency ? `Figures in ${fundamentals.financialCurrency}. ` : ''}
                            Pick up to {MAX_CHART_SERIES}.
                        </p>
                        <div className="flex items-center p-0.5 rounded-full bg-background/60">
                            {(['5y', '10y', 'max'] as const).map(opt => (
                                <button
                                    key={opt}
                                    onClick={() => setFinRange(opt)}
                                    aria-pressed={range === opt}
                                    className={cn(
                                        "px-2.5 py-1 rounded-full text-[10px] font-bold uppercase transition-all cursor-pointer",
                                        range === opt
                                            ? "bg-white dark:bg-zinc-800 text-foreground shadow-sm"
                                            : "text-muted-foreground hover:text-foreground"
                                    )}
                                >
                                    {opt}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="flex flex-wrap gap-1.5">
                    {chipRows.map(row => {
                        const slot = slots.indexOf(row.label);
                        const selected = slot !== -1;
                        const disabled = !selected && slotFull;
                        return (
                            <button
                                key={row.label}
                                onClick={() => toggleMetric(row.label)}
                                disabled={disabled}
                                aria-pressed={selected}
                                title={disabled ? `Deselect one first — ${MAX_CHART_SERIES} is the limit` : row.label}
                                className={cn(
                                    "flex items-center gap-1.5 pl-2 pr-2.5 py-1 rounded-full text-[11px] font-medium transition-all border cursor-pointer",
                                    selected
                                        ? "bg-background border-black/10 dark:border-white/15 text-foreground"
                                        : "bg-transparent border-transparent text-muted-foreground hover:bg-background/60 hover:text-foreground",
                                    disabled && "opacity-40 cursor-not-allowed hover:bg-transparent"
                                )}
                            >
                                <span
                                    className="w-2 h-2 rounded-full flex-shrink-0"
                                    style={{
                                        backgroundColor: selected
                                            ? seriesColors[slot % seriesColors.length]
                                            : 'currentColor',
                                        opacity: selected ? 1 : 0.25
                                    }}
                                />
                                {row.label}
                            </button>
                        );
                    })}
                    {chartableRanked.length > 0 && chartable.length > chartableRanked.length && (
                        <button
                            onClick={() => setShowAllMetrics(v => !v)}
                            className="px-2.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 hover:underline cursor-pointer"
                        >
                            {showAllMetrics ? 'Show key items' : `+${chartable.length - chartableRanked.length} more`}
                        </button>
                    )}
                </div>

                {scaleGroups.length === 0 ? (
                    <div className="h-64 flex flex-col items-center justify-center text-center text-muted-foreground">
                        <LineChartIcon className="w-8 h-8 mb-2 opacity-20" />
                        <p className="text-sm">Pick a line item above to chart it.</p>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {scaleGroups.map((group, gIdx) => (
                            <StatementChart
                                key={group.map(s => s.key).join('-') || gIdx}
                                points={points}
                                series={group}
                                periodType={finPeriod}
                            />
                        ))}
                    </div>
                )}

                {primary && (
                    <MetricChangeStrip
                        label={primary.label}
                        color={primary.color}
                        points={points}
                        seriesKey={primary.key}
                        periodType={finPeriod}
                    />
                )}
            </div>

            <div>
                <p className="text-[11px] text-muted-foreground mb-2 px-1">
                    {finPeriod === 'quarterly'
                        ? 'Quarterly figures are built from the company’s own 10-Q filings, differenced out of the year-to-date numbers where that is all it tags.'
                        : 'Annual statements are extended with SEC-filed history where the company files one.'}
                    {' '}Click any row to chart it.
                </p>
                <div className="overflow-x-auto bg-muted">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-secondary/50 font-semibold">
                            <tr>
                                <th className="px-6 py-3 font-semibold text-foreground sticky left-0 z-20 bg-white dark:bg-zinc-950 border-r border-black/5 dark:border-white/10"></th>
                                <th className="px-6 py-3 font-semibold text-center text-muted-foreground">Trend</th>
                                {currentStatement.columns.map(col => (
                                    <th key={col} className="px-4 py-3 font-semibold text-center text-muted-foreground tabular-nums whitespace-nowrap">
                                        <div>{periodAxisLabel(col, finPeriod)}</div>
                                        <div className="text-[10px] font-normal opacity-60">{fiscalPeriodDay(col)}</div>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {currentStatement.index.map((item, idx) => {
                                const slot = slots.indexOf(item);
                                const selected = slot !== -1;
                                const chartableRow = chartable.some(r => r.label === item);
                                return (
                                    <tr
                                        key={item}
                                        onClick={() => chartableRow && toggleMetric(item)}
                                        aria-pressed={selected}
                                        className={cn(
                                            "transition-colors",
                                            chartableRow ? "cursor-pointer hover:bg-accent/5" : "opacity-70",
                                            selected && "bg-accent/5"
                                        )}
                                    >
                                        <td className="px-6 py-3 font-medium text-foreground sticky left-0 z-10 bg-white dark:bg-zinc-950 border-r border-black/5 dark:border-white/10 min-w-[200px]">
                                            <span className="flex items-center gap-2">
                                                <span
                                                    className={cn("w-2 h-2 rounded-full flex-shrink-0", !selected && "opacity-0")}
                                                    style={{ backgroundColor: selected ? seriesColors[slot % seriesColors.length] : 'transparent' }}
                                                />
                                                {item}
                                            </span>
                                        </td>
                                        <td className="px-6 py-3 text-center min-w-[100px]">
                                            <Sparkline data={currentStatement.data[idx] as number[]} />
                                        </td>
                                        {currentStatement.data[idx].map((val, vIdx) => (
                                            <td key={vIdx} className="px-6 py-3 text-foreground text-right font-medium tabular-nums">
                                                {formatCompact(val as number)}
                                            </td>
                                        ))}
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </>
    );
};
