'use client';

import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    AreaChart,
    Area,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from 'recharts';
import {
    TrendingUp,
    DollarSign,
    Calendar,
    PieChart,
    Activity,
    Layers,
} from 'lucide-react';
import {
    fetchStockPositionHistory,
    StockPositionHistoryPoint,
} from '@/lib/api';
import { formatCurrency, formatCompactNumber, cn } from '@/lib/utils';
import { Skeleton } from '@/components/ui/skeleton';

interface PositionPerformanceGraphProps {
    symbol: string;
    currency: string;
    accounts?: string[];
    localCurrency?: string;
}

type GraphView = 'value' | 'return';

const PERIOD_OPTIONS = [
    { label: '1M', value: '1m' },
    { label: '3M', value: '3m' },
    { label: '6M', value: '6m' },
    { label: 'YTD', value: 'ytd' },
    { label: '1Y', value: '1y' },
    { label: '3Y', value: '3y' },
    { label: '5Y', value: '5y' },
    { label: 'ALL', value: 'all' },
] as const;

const BENCHMARK_OPTIONS = [
    { name: 'S&P 500', color: '#f59e0b' },
    { name: 'NASDAQ', color: '#8b5cf6' },
    { name: 'Dow Jones', color: '#0ea5e9' },
] as const;

interface TooltipPayloadItem {
    name?: string;
    value?: number;
    color?: string;
    dataKey?: string | number;
    payload?: StockPositionHistoryPoint;
}

interface CustomTooltipProps {
    active?: boolean;
    payload?: TooltipPayloadItem[];
    label?: string;
    view: GraphView;
    currency: string;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({
    active,
    payload,
    label,
    view,
    currency,
}) => {
    if (!active || !payload || !payload.length) return null;

    const dataPoint = payload[0]?.payload;
    if (!dataPoint) return null;

    const mktVal = dataPoint.value || 0;
    const cost = dataPoint.cost_basis || 0;
    const unrealG = dataPoint.unrealized_gain || 0;
    const unrealPct = dataPoint.unrealized_gain_pct || 0;
    const shares = dataPoint.shares || 0;
    const retPct = dataPoint.return_pct || 0;

    return (
        <div className="bg-popover/95 backdrop-blur-md border border-border/80 p-3 rounded-xl shadow-xl text-xs space-y-2 min-w-[190px]">
            <div className="flex items-center justify-between gap-3 pb-1.5 border-b border-border/50">
                <span className="font-semibold text-foreground flex items-center gap-1.5">
                    <Calendar className="w-3.5 h-3.5 text-muted-foreground" />
                    {label}
                </span>
                {shares > 0 ? (
                    <span className="text-[11px] font-medium text-muted-foreground tabular-nums">
                        {shares.toLocaleString(undefined, { maximumFractionDigits: 4 })} sh
                    </span>
                ) : (
                    <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                        Closed
                    </span>
                )}
            </div>

            {view === 'value' ? (
                <div className="space-y-1.5">
                    <div className="flex items-center justify-between gap-3">
                        <span className="text-muted-foreground flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-indigo-500 inline-block" />
                            Market Value:
                        </span>
                        <span className="font-bold tabular-nums text-foreground">
                            {formatCurrency(mktVal, currency)}
                        </span>
                    </div>

                    <div className="flex items-center justify-between gap-3">
                        <span className="text-muted-foreground flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-slate-400 inline-block" />
                            Cost Basis:
                        </span>
                        <span className="font-medium tabular-nums text-muted-foreground">
                            {formatCurrency(cost, currency)}
                        </span>
                    </div>

                    {cost > 0 && (
                        <div className="flex items-center justify-between gap-3 pt-1 border-t border-border/40">
                            <span className="text-muted-foreground">Unrealized G/L:</span>
                            <span
                                className={cn(
                                    'font-semibold tabular-nums',
                                    unrealG >= 0
                                        ? 'text-up'
                                        : 'text-down'
                                )}
                            >
                                {unrealG >= 0 ? '+' : ''}
                                {formatCurrency(unrealG, currency)} ({unrealPct >= 0 ? '+' : ''}
                                {unrealPct.toFixed(2)}%)
                            </span>
                        </div>
                    )}
                </div>
            ) : (
                <div className="space-y-1.5">
                    <div className="flex items-center justify-between gap-3">
                        <span className="text-muted-foreground flex items-center gap-1.5">
                            <span
                                className={cn(
                                    'w-2 h-2 rounded-full inline-block',
                                    retPct >= 0 ? 'bg-emerald-500' : 'bg-rose-500'
                                )}
                            />
                            Position Return:
                        </span>
                        <span
                            className={cn(
                                'font-bold tabular-nums',
                                retPct >= 0
                                    ? 'text-up'
                                    : 'text-down'
                            )}
                        >
                            {retPct >= 0 ? '+' : ''}
                            {retPct.toFixed(2)}%
                        </span>
                    </div>

                    {payload
                        .filter((p) => p.dataKey && p.dataKey !== 'return_pct' && p.value !== undefined)
                        .map((b) => (
                            <div key={String(b.dataKey)} className="flex items-center justify-between gap-3">
                                <span className="text-muted-foreground flex items-center gap-1.5">
                                    <span
                                        className="w-2 h-2 rounded-full inline-block"
                                        style={{ backgroundColor: b.color }}
                                    />
                                    {b.name}:
                                </span>
                                <span
                                    className={cn(
                                        'font-medium tabular-nums',
                                        (b.value ?? 0) >= 0
                                            ? 'text-up'
                                            : 'text-down'
                                    )}
                                >
                                    {(b.value ?? 0) >= 0 ? '+' : ''}
                                    {(b.value ?? 0).toFixed(2)}%
                                </span>
                            </div>
                        ))}
                </div>
            )}
        </div>
    );
};

export const PositionPerformanceGraph: React.FC<PositionPerformanceGraphProps> = ({
    symbol,
    currency,
    accounts = [],
}) => {
    const [view, setView] = useState<GraphView>('value');
    const [period, setPeriod] = useState<string>('1y');
    const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);

    const toggleBenchmark = (name: string) => {
        setSelectedBenchmarks((prev) =>
            prev.includes(name) ? prev.filter((b) => b !== name) : [...prev, name]
        );
    };

    const {
        data: historyData = [],
        isLoading,
        error,
    } = useQuery<StockPositionHistoryPoint[]>({
        queryKey: ['stock-position-history', symbol, currency, period, accounts, selectedBenchmarks],
        queryFn: () =>
            fetchStockPositionHistory(
                symbol,
                currency,
                period,
                accounts,
                selectedBenchmarks
            ),
        staleTime: 60 * 1000,
    });

    const hasData = historyData.length > 0;

    // Determine current/latest point metrics for header badge
    const latestPoint = hasData ? historyData[historyData.length - 1] : null;
    const firstPoint = hasData ? historyData[0] : null;

    const valueDelta =
        latestPoint && firstPoint
            ? latestPoint.value - firstPoint.value
            : 0;
    const valueDeltaPct =
        firstPoint && firstPoint.value > 1e-6
            ? (valueDelta / firstPoint.value) * 100
            : 0;
    const periodReturnPct = latestPoint ? latestPoint.return_pct : 0;

    // Determine min and max for Y-Axis domain
    const yDomain = useMemo(() => {
        if (!historyData.length) return [0, 'auto'];
        if (view === 'value') {
            const maxVal = Math.max(
                ...historyData.map((d) => Math.max(d.value, d.cost_basis))
            );
            return [0, Math.ceil(maxVal * 1.08)];
        } else {
            const allReturns = historyData.flatMap((d) => {
                const vals = [d.return_pct];
                selectedBenchmarks.forEach((bm) => {
                    if (typeof d[bm] === 'number') vals.push(d[bm] as number);
                });
                return vals;
            });
            const minRet = Math.min(0, ...allReturns);
            const maxRet = Math.max(0, ...allReturns);
            const pad = Math.max(5, (maxRet - minRet) * 0.1);
            return [Math.floor(minRet - pad), Math.ceil(maxRet + pad)];
        }
    }, [historyData, view, selectedBenchmarks]);

    const isOverallPositive =
        view === 'value' ? valueDelta >= 0 : periodReturnPct >= 0;

    return (
        <div className="bg-card border border-border/50 rounded-2xl p-4 sm:p-5 space-y-4 shadow-2xs">
            {/* Header: Controls & View Switcher */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-1">
                    <div className="flex items-center gap-2">
                        <h3 className="text-sm sm:text-base font-semibold flex items-center gap-2">
                            {view === 'value' ? (
                                <PieChart className="w-4 h-4 text-indigo-500" />
                            ) : (
                                <TrendingUp className="w-4 h-4 text-up" />
                            )}
                            Position Performance History
                        </h3>

                        {latestPoint && (
                            latestPoint.shares <= 1e-6 ? (
                                view === 'value' ? (
                                    <span className="text-xs font-semibold px-2 py-0.5 rounded-md tabular-nums bg-muted text-muted-foreground border border-border/40">
                                        Closed Position
                                    </span>
                                ) : (
                                    <span
                                        className={cn(
                                            'text-xs font-semibold px-2 py-0.5 rounded-md tabular-nums',
                                            periodReturnPct >= 0
                                                ? 'bg-up/12 text-up'
                                                : 'bg-rose-500/10 text-down'
                                        )}
                                    >
                                        {periodReturnPct >= 0 ? '+' : ''}
                                        {periodReturnPct.toFixed(2)}% in {period.toUpperCase()}
                                    </span>
                                )
                            ) : (
                                <span
                                    className={cn(
                                        'text-xs font-semibold px-2 py-0.5 rounded-md tabular-nums',
                                        isOverallPositive
                                            ? 'bg-up/12 text-up'
                                            : 'bg-rose-500/10 text-down'
                                    )}
                                >
                                    {view === 'value' ? (
                                        <>
                                            {valueDelta >= 0 ? '+' : ''}
                                            {formatCurrency(valueDelta, currency)} (
                                            {valueDeltaPct >= 0 ? '+' : ''}
                                            {valueDeltaPct.toFixed(2)}%)
                                        </>
                                    ) : (
                                        <>
                                            {periodReturnPct >= 0 ? '+' : ''}
                                            {periodReturnPct.toFixed(2)}% in {period.toUpperCase()}
                                        </>
                                    )}
                                </span>
                            )
                        )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                        {view === 'value'
                            ? `Historical market value and open tax lot cost basis over time${latestPoint && latestPoint.shares <= 1e-6 ? ' (Position closed).' : '.'}`
                            : 'Time-weighted return percentage over the selected period.'}
                    </p>
                </div>

                {/* Right controls: View Toggle & Period Buttons */}
                <div className="flex items-center gap-2 flex-wrap">
                    {/* View Switcher */}
                    <div className="inline-flex items-center gap-0.5 bg-muted/60 border border-border/60 rounded-lg p-0.5 text-xs font-medium">
                        <button
                            onClick={() => setView('value')}
                            className={cn(
                                'flex items-center gap-1.5 px-3 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 cursor-pointer whitespace-nowrap',
                                view === 'value'
                                    ? 'bg-indigo-600 text-white font-bold shadow'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
                            )}
                        >
                            <DollarSign className="w-3.5 h-3.5" />
                            Value
                        </button>
                        <button
                            onClick={() => setView('return')}
                            className={cn(
                                'flex items-center gap-1.5 px-3 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 cursor-pointer whitespace-nowrap',
                                view === 'return'
                                    ? 'bg-indigo-600 text-white font-bold shadow'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
                            )}
                        >
                            <Activity className="w-3.5 h-3.5" />
                            Return (%)
                        </button>
                    </div>

                    {/* Period Selector */}
                    <div className="inline-flex items-center gap-0.5 bg-muted/60 border border-border/60 rounded-lg p-0.5 text-xs font-medium">
                        {PERIOD_OPTIONS.map((opt) => {
                            const active = period === opt.value;
                            return (
                                <button
                                    key={opt.value}
                                    onClick={() => setPeriod(opt.value)}
                                    className={cn(
                                        'px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150 cursor-pointer whitespace-nowrap',
                                        active
                                            ? 'bg-indigo-600 text-white font-bold shadow'
                                            : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
                                    )}
                                >
                                    {opt.label}
                                </button>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* Benchmark Overlay Toggles (in Return mode) */}
            {view === 'return' && (
                <div className="flex items-center gap-2 pt-1 pb-0.5 overflow-x-auto text-xs">
                    <span className="text-muted-foreground text-[11px] font-medium flex items-center gap-1 shrink-0">
                        <Layers className="w-3 h-3" />
                        Compare:
                    </span>
                    {BENCHMARK_OPTIONS.map((bm) => {
                        const isSelected = selectedBenchmarks.includes(bm.name);
                        return (
                            <button
                                key={bm.name}
                                onClick={() => toggleBenchmark(bm.name)}
                                className={cn(
                                    'px-2.5 py-1 rounded-lg border text-[11px] font-medium transition-all flex items-center gap-1.5 cursor-pointer whitespace-nowrap',
                                    isSelected
                                        ? 'bg-muted border-foreground/20 text-foreground font-semibold shadow-2xs'
                                        : 'bg-background/50 border-border/40 text-muted-foreground hover:text-foreground'
                                )}
                            >
                                <span
                                    className="w-2 h-2 rounded-full shrink-0"
                                    style={{ backgroundColor: bm.color }}
                                />
                                {bm.name}
                            </button>
                        );
                    })}
                </div>
            )}

            {/* Chart Area */}
            <div className="h-64 sm:h-72 w-full pt-2">
                {isLoading ? (
                    <div className="h-full w-full relative">
                        <Skeleton className="h-full w-full rounded-xl" />
                        <span className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
                            Loading position history…
                        </span>
                    </div>
                ) : error ? (
                    <div className="h-full w-full flex items-center justify-center text-xs text-muted-foreground">
                        Unable to load position history graph.
                    </div>
                ) : !hasData ? (
                    <div className="h-full w-full flex items-center justify-center text-xs text-muted-foreground">
                        No historical position data recorded for this period.
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                        {view === 'value' ? (
                            <AreaChart data={historyData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="posValueGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.15} />
                                <XAxis
                                    dataKey="date"
                                    tickLine={false}
                                    axisLine={false}
                                    tick={{ fontSize: 10, fill: 'currentColor', opacity: 0.6 }}
                                    minTickGap={40}
                                />
                                <YAxis
                                    domain={yDomain as [number, number | string]}
                                    tickLine={false}
                                    axisLine={false}
                                    tick={{ fontSize: 10, fill: 'currentColor', opacity: 0.6 }}
                                    // Compact ticks: a full THB/JPY figure (฿7,361,571.00)
                                    // overflows the axis gutter and gets clipped.
                                    tickFormatter={(v) => formatCompactNumber(v, currency)}
                                    width={62}
                                />
                                <Tooltip
                                    content={<CustomTooltip view={view} currency={currency} />}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    name="Market Value"
                                    stroke="#6366f1"
                                    strokeWidth={2.2}
                                    fillOpacity={1}
                                    fill="url(#posValueGradient)"
                                />
                                <Line
                                    type="monotone"
                                    dataKey="cost_basis"
                                    name="Cost Basis"
                                    stroke="#94a3b8"
                                    strokeWidth={1.5}
                                    strokeDasharray="4 4"
                                    dot={false}
                                />
                            </AreaChart>
                        ) : (
                            <AreaChart data={historyData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="posReturnGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop
                                            offset="5%"
                                            stopColor={isOverallPositive ? '#10b981' : '#ef4444'}
                                            stopOpacity={0.35}
                                        />
                                        <stop
                                            offset="95%"
                                            stopColor={isOverallPositive ? '#10b981' : '#ef4444'}
                                            stopOpacity={0.0}
                                        />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.15} />
                                <XAxis
                                    dataKey="date"
                                    tickLine={false}
                                    axisLine={false}
                                    tick={{ fontSize: 10, fill: 'currentColor', opacity: 0.6 }}
                                    minTickGap={40}
                                />
                                <YAxis
                                    domain={yDomain as [number, number | string]}
                                    tickLine={false}
                                    axisLine={false}
                                    tick={{ fontSize: 10, fill: 'currentColor', opacity: 0.6 }}
                                    tickFormatter={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(0)}%`}
                                    width={55}
                                />
                                <ReferenceLine y={0} stroke="currentColor" strokeOpacity={0.25} strokeDasharray="2 2" />
                                <Tooltip
                                    content={<CustomTooltip view={view} currency={currency} />}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="return_pct"
                                    name="Position Return"
                                    stroke={isOverallPositive ? '#10b981' : '#ef4444'}
                                    strokeWidth={2.2}
                                    fillOpacity={1}
                                    fill="url(#posReturnGradient)"
                                />
                                {selectedBenchmarks.map((bmName) => {
                                    const bmOpt = BENCHMARK_OPTIONS.find((b) => b.name === bmName);
                                    const col = bmOpt ? bmOpt.color : '#f59e0b';
                                    return (
                                        <Line
                                            key={bmName}
                                            type="monotone"
                                            dataKey={bmName}
                                            name={bmName}
                                            stroke={col}
                                            strokeWidth={1.6}
                                            dot={false}
                                        />
                                    );
                                })}
                            </AreaChart>
                        )}
                    </ResponsiveContainer>
                )}
            </div>
        </div>
    );
};
