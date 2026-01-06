import React, { useState, useEffect, useMemo } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area,
    Legend
} from 'recharts';
import PeriodSelector from './PeriodSelector';
import BenchmarkSelector from './BenchmarkSelector';
import { fetchHistory, PerformanceData } from '../lib/api';
import { formatCurrency } from '../lib/utils';



interface CustomTooltipProps {
    active?: boolean;
    payload?: {
        color?: string;
        name?: string;
        value?: number;
        [key: string]: unknown;
    }[];
    label?: string;
}

interface PerformanceGraphProps {
    currency: string;
    accounts?: string[];
    benchmarks: string[];
    onBenchmarksChange: (benchmarks: string[]) => void;
}

const COLORS = [
    "#2563eb", // Portfolio (Blue)
    "#dc2626", // Red
    "#16a34a", // Green
    "#d97706", // Amber
    "#9333ea", // Purple
    "#0891b2", // Cyan
    "#db2777", // Pink
];

export default function PerformanceGraph({ currency, accounts, benchmarks, onBenchmarksChange }: PerformanceGraphProps) {
    const [view, setView] = useState<'return' | 'value' | 'drawdown'>('return');
    const [period, setPeriod] = useState('1y');
    const [data, setData] = useState<PerformanceData[]>([]);
    const [loading, setLoading] = useState(false);

    // Fetch data when period or benchmarks change
    // Fetch data logic
    const fetchData = React.useCallback(async (isBackground = false) => {
        if (!isBackground) setLoading(true);
        try {
            // Determine interval based on period
            let interval = '1d';
            if (period === '1d') {
                interval = '2m'; // Finer granularity for 1D view
            } else if (period === '5d') {
                interval = '15m'; // 15m * 5 days = ~130 points
            } else if (period === '1m') {
                interval = '60m'; // 1h * 30 days = ~200 points
            }

            const newData = await fetchHistory(currency, accounts, period, benchmarks, interval);
            setData(newData);
        } catch (error) {
            console.error("Failed to fetch history:", error);
        } finally {
            if (!isBackground) setLoading(false);
        }
    }, [currency, accounts, period, benchmarks]);

    // Initial load and params change
    useEffect(() => {
        fetchData(false);
    }, [fetchData]);

    // Auto-refresh for 1D view (Real-time updates)
    useEffect(() => {
        if (period === '1d') {
            const intervalId = setInterval(() => {
                fetchData(true); // Background refresh
            }, 60000); // Check every minute
            return () => clearInterval(intervalId);
        }
    }, [period, fetchData]);

    const periodStats = useMemo(() => {
        if (!data || data.length < 2) return null;

        const start = data[0];
        const end = data[data.length - 1];

        if (view === 'return') {
            const twr = end.twr;
            return {
                label: "Period TWR",
                text: `${twr > 0 ? '+' : ''}${twr.toFixed(2)}%`,
                color: twr >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
            };
        } else if (view === 'value') {
            const startVal = start.value;
            const endVal = end.value;
            const change = endVal - startVal;
            const changePct = startVal !== 0 ? (change / startVal) * 100 : 0;

            return {
                label: "Period Change",
                text: `${formatCurrency(change, currency)} (${change > 0 ? '+' : ''}${changePct.toFixed(2)}%)`,
                color: change >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
            };
        }
        return null;
    }, [data, view, currency]);

    const hasFXRate = useMemo(() => data && data.length > 0 && data.some(d => (d as any).fx_rate !== undefined), [data]);

    // Determine which keys to plot
    // Always plot 'twr' or 'value' for portfolio
    // For benchmarks, they are usually only relevant for 'return' view
    // Benchmarks keys in data will be their tickers/names.
    // We need to know which keys correspond to benchmarks.
    // A simple way is to look at keys in the first data point that are not 'date', 'value', 'twr', 'drawdown', 'fx_rate'.
    const benchmarkKeys = useMemo(() => {
        if (!data || data.length === 0) return [];
        return Object.keys(data[0]).filter(k => k !== 'date' && k !== 'value' && k !== 'twr' && k !== 'drawdown' && k !== 'fx_rate');
    }, [data]);

    const processedData = useMemo(() => {
        if (!data || data.length === 0) return [];
        const startFX = (data[0] as any).fx_rate;
        if (startFX === undefined) return data;

        return data.map(d => {
            const currentFX = (d as any).fx_rate;
            return {
                ...d,
                fx_return: currentFX !== undefined ? ((currentFX / startFX) - 1) * 100 : undefined
            };
        });
    }, [data]);

    if (!data || data.length === 0) {
        return (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-100 dark:border-gray-700 h-64 flex items-center justify-center text-gray-500">
                {loading ? 'Loading...' : 'No historical data available.'}
            </div>
        );
    }

    const formatXAxis = (tickItem: string) => {
        const date = new Date(tickItem);
        // For short periods, show time. For 1M (hourly), maybe show Day + Time?
        // Let's simple check:
        if (period === '1d') {
            return date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
        } else if (period === '5d') {
            // Show Day + Time for context
            return date.toLocaleString(undefined, { weekday: 'short', hour: '2-digit' });
        } else if (period === '1m') {
            // Hourly for 1 month. Date + maybe Hour? Too crowded.
            // Just Date is probably fine, or Day.
            return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        }
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    };

    const formatYAxis = (tickItem: number) => {
        if (view === 'return' || view === 'drawdown') {
            return `${tickItem.toFixed(1)}%`;
        } else {
            // Check if we need more precision
            // If the range is small, we might see duplicate ticks like "1.7M", "1.7M", "1.7M"
            // We can check the data range to decide.
            // But formatting happens per tick.

            // Heuristic working with the tick value itself isn't enough, we need context.
            // However, we can just use more localized formatting if the number is large?
            // Or just increase fraction digits to 2 or 3?

            // Let's try 3 fraction digits for compact notation.
            // "1.662M" vs "1.7M"
            return new Intl.NumberFormat('en-US', {
                notation: "compact",
                maximumFractionDigits: 3,
                minimumFractionDigits: 0
            }).format(tickItem);
        }
    };



    const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
        if (active && payload && payload.length) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const dataPoint = payload[0].payload as any;
            const dateObj = new Date(dataPoint.date);

            let dateStr;
            // For intraday periods (1d, 5d), show time
            if (period === '1d' || period === '5d' || period === '1m') { // 1M is hourly, so show time too? Yes.
                // Actually 1M is 60m interval.
                dateStr = dateObj.toLocaleString(undefined, {
                    weekday: 'short',
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
            } else {
                dateStr = dateObj.toLocaleDateString(undefined, {
                    weekday: 'short',
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                });
            }

            // Find benchmark keys present in this data point
            const allKeys = Object.keys(dataPoint);
            const benchKeys = allKeys.filter(k =>
                k !== 'date' && k !== 'value' && k !== 'twr' && k !== 'drawdown' && k !== 'fx_rate' && k !== 'fx_return'
            );

            return (
                <div className="bg-popover/95 backdrop-blur-md p-4 border border-border shadow-xl rounded-xl min-w-[200px]">
                    <p className="text-sm font-semibold text-foreground mb-3 border-b border-border pb-2">
                        {dateStr}
                    </p>

                    <div className="space-y-3">
                        {/* Portfolio Section */}
                        <div className="space-y-1">
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-xs text-muted-foreground font-medium">Portfolio Value</span>
                                <span className="text-sm font-bold text-foreground">
                                    {formatCurrency(dataPoint.value, currency)}
                                </span>
                            </div>
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-xs text-muted-foreground font-medium">Return (TWR)</span>
                                <span className={`text-sm font-bold ${dataPoint.twr >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                    {dataPoint.twr >= 0 ? '+' : ''}{dataPoint.twr.toFixed(2)}%
                                </span>
                            </div>
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-xs text-muted-foreground font-medium">Drawdown</span>
                                <span className="text-sm font-bold text-rose-500">
                                    {dataPoint.drawdown.toFixed(2)}%
                                </span>
                            </div>
                        </div>

                        {/* FX Rate Section */}
                        {dataPoint.fx_rate !== undefined && (
                            <>
                                <div className="h-px bg-border my-2" />
                                <div className="space-y-1">
                                    <div className="flex items-center justify-between gap-4">
                                        <span className="text-xs text-amber-500 font-medium">FX Rate ({currency}/USD)</span>
                                        <span className="text-sm font-bold text-amber-500">
                                            {dataPoint.fx_rate.toFixed(4)}
                                        </span>
                                    </div>
                                    {dataPoint.fx_return !== undefined && (
                                        <div className="flex items-center justify-between gap-4">
                                            <span className="text-xs text-amber-500 font-medium">FX Return</span>
                                            <span className={`text-sm font-bold ${dataPoint.fx_return >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                                {dataPoint.fx_return >= 0 ? '+' : ''}{dataPoint.fx_return.toFixed(2)}%
                                            </span>
                                        </div>
                                    )}
                                </div>
                            </>
                        )}

                        {/* Benchmarks Section */}
                        {benchKeys.length > 0 && (
                            <>
                                <div className="h-px bg-border my-2" />
                                <div className="space-y-1">
                                    {benchKeys.map((bKey, idx) => {
                                        // Use the same color logic as the main chart if possible, or just standard colors
                                        // We need to match the chart color. The chart uses COLORS array cyclicly.
                                        // But we can't easily know the index here without passing it down?
                                        // We can infer it from the mapped lines if we want strict consistency, 
                                        // but for now let's just use the color from payload if available:
                                        const payloadEntry = payload.find(p => p.name === bKey);
                                        const color = payloadEntry?.color || COLORS[(idx + 1) % COLORS.length];

                                        return (
                                            <div key={bKey} className="flex items-center justify-between gap-4">
                                                <span className="text-xs font-medium" style={{ color: color }}>
                                                    {bKey}
                                                </span>
                                                <span className={`text-sm font-bold ${dataPoint[bKey] >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                                    {dataPoint[bKey] >= 0 ? '+' : ''}{Number(dataPoint[bKey]).toFixed(2)}%
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </>
                        )}
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-card backdrop-blur-md rounded-xl p-4 shadow-sm border border-border mb-6">
            <div className="mb-6">
                <div className="flex flex-col gap-1 mb-4">
                    <h3 className="text-lg font-medium text-muted-foreground">
                        {view === 'return' ? 'Time-Weighted Return' : view === 'value' ? 'Portfolio Value' : 'Drawdown'}
                    </h3>
                    {periodStats ? (
                        <div className="flex items-baseline gap-2">
                            <span className="text-sm font-medium text-muted-foreground">
                                {periodStats.label}
                            </span>
                            <span className={`text-xl font-bold tracking-tight ${periodStats.color}`}>
                                {periodStats.text}
                            </span>
                        </div>
                    ) : (
                        <div className="h-9" /> /* Spacer for loading state */
                    )}
                </div>

                <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
                    <div className="flex items-center gap-2 w-full sm:w-auto overflow-x-auto no-scrollbar pb-1">
                        <PeriodSelector selectedPeriod={period} onPeriodChange={setPeriod} />
                    </div>

                    <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto pb-1">
                        {view === 'return' && (
                            <BenchmarkSelector selectedBenchmarks={benchmarks} onBenchmarkChange={onBenchmarksChange} />
                        )}
                        <div className="flex bg-secondary rounded-lg p-1 border border-border shrink-0">
                            <button
                                onClick={() => setView('return')}
                                className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-colors ${view === 'return'
                                    ? 'bg-card text-foreground shadow-sm'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                    }`}
                            >
                                Return %
                            </button>
                            <button
                                onClick={() => setView('value')}
                                className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-colors ${view === 'value'
                                    ? 'bg-card text-foreground shadow-sm'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                    }`}
                            >
                                Value
                            </button>
                            <button
                                onClick={() => setView('drawdown')}
                                className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-colors ${view === 'drawdown'
                                    ? 'bg-card text-foreground shadow-sm'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                    }`}
                            >
                                Drawdown
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div className="h-64 w-full relative">
                {loading && (
                    <div className="absolute inset-0 bg-white/50 dark:bg-gray-800/50 flex items-center justify-center z-10">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    </div>
                )}
                <ResponsiveContainer width="100%" height="100%">
                    {view === 'return' ? (
                        <LineChart syncId="portfolio-sync" data={processedData} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                            <XAxis
                                dataKey="date"
                                tickFormatter={formatXAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                minTickGap={30}
                            />
                            <YAxis
                                tickFormatter={formatYAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                width={60}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend />
                            <Line
                                name="Portfolio"
                                type="monotone"
                                dataKey="twr"
                                stroke={COLORS[0]}
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 6 }}
                            />
                            {benchmarkKeys.map((key, index) => (
                                <Line
                                    key={key}
                                    name={key}
                                    type="monotone"
                                    dataKey={key}
                                    stroke={COLORS[(index + 1) % COLORS.length]}
                                    strokeWidth={2}
                                    dot={false}
                                />
                            ))}
                            {hasFXRate && (
                                <Line
                                    name={`FX (${currency}/USD)`}
                                    type="monotone"
                                    dataKey="fx_return"
                                    stroke="#f59e0b"
                                    strokeWidth={1.5}
                                    strokeDasharray="5 5"
                                    dot={false}
                                />
                            )}
                        </LineChart>
                    ) : view === 'value' ? (
                        <AreaChart syncId="portfolio-sync" data={processedData} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.1} />
                                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                            <XAxis
                                dataKey="date"
                                tickFormatter={formatXAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                minTickGap={30}
                            />
                            <YAxis
                                tickFormatter={formatYAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                width={60}
                                domain={['auto', 'auto']}
                            />
                            {hasFXRate && (
                                <YAxis
                                    yAxisId="right"
                                    orientation="right"
                                    tick={{ fontSize: 10, fill: '#f59e0b' }}
                                    axisLine={false}
                                    tickLine={false}
                                    width={40}
                                    domain={['auto', 'auto']}
                                    tickFormatter={(val) => val.toFixed(2)}
                                />
                            )}
                            <Tooltip content={<CustomTooltip />} />
                            <Area
                                name="Portfolio Value"
                                type="monotone"
                                dataKey="value"
                                stroke="#2563eb"
                                fillOpacity={1}
                                fill="url(#colorValue)"
                                strokeWidth={2}
                            />
                            {hasFXRate && (
                                <Line
                                    yAxisId="right"
                                    name={`FX (${currency}/USD)`}
                                    type="monotone"
                                    dataKey="fx_rate"
                                    stroke="#f59e0b"
                                    strokeWidth={1.5}
                                    strokeDasharray="5 5"
                                    dot={false}
                                />
                            )}
                        </AreaChart>
                    ) : (
                        <AreaChart syncId="portfolio-sync" data={processedData} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.1} />
                                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                            <XAxis
                                dataKey="date"
                                tickFormatter={formatXAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                minTickGap={30}
                            />
                            <YAxis
                                tickFormatter={formatYAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                width={60}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Area
                                name="Drawdown"
                                type="monotone"
                                dataKey="drawdown"
                                stroke="#ef4444"
                                fillOpacity={1}
                                fill="url(#colorDrawdown)"
                                strokeWidth={2}
                            />
                        </AreaChart>
                    )}

                </ResponsiveContainer>
            </div>
        </div>
    );
}
