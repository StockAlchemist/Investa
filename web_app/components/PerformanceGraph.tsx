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
    Legend,
    ReferenceLine
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

    const [customFromDate, setCustomFromDate] = useState(() => {
        const d = new Date();
        d.setFullYear(d.getFullYear() - 1);
        return d.toISOString().split('T')[0];
    });
    const [customToDate, setCustomToDate] = useState(() => {
        return new Date().toISOString().split('T')[0];
    });
    const [isInitialized, setIsInitialized] = useState(false);

    // Initialize state from localStorage
    useEffect(() => {
        const savedPeriod = localStorage.getItem('investa_graph_period');
        if (savedPeriod) setPeriod(savedPeriod);

        const savedView = localStorage.getItem('investa_graph_view');
        if (savedView && ['return', 'value', 'drawdown'].includes(savedView)) {
            setView(savedView as 'return' | 'value' | 'drawdown');
        }
        setIsInitialized(true);
    }, []);

    // Persist state to localStorage
    useEffect(() => {
        if (isInitialized) {
            localStorage.setItem('investa_graph_period', period);
        }
    }, [period, isInitialized]);

    useEffect(() => {
        if (isInitialized) {
            localStorage.setItem('investa_graph_view', view);
        }
    }, [view, isInitialized]);

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
                interval = '1d'; // Use 1d to ensure consistency with 3M+ and capture official adjusted closes
            }

            const from = period === 'custom' ? customFromDate : undefined;
            const to = period === 'custom' ? customToDate : undefined;

            const newData = await fetchHistory(currency, accounts, period, benchmarks, interval, from, to);
            setData(newData);
        } catch (error) {
            console.error("Failed to fetch history:", error);
        } finally {
            if (!isBackground) setLoading(false);
        }
    }, [currency, accounts, period, benchmarks, customFromDate, customToDate]);

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



    const hasFXRate = useMemo(() => data && data.length > 0 && data.some(d => (d as any).fx_rate != null), [data]);

    // Determine which keys to plot
    // Always plot 'twr' or 'value' for portfolio
    // For benchmarks, they are usually only relevant for 'return' view
    // Benchmarks keys in data will be their tickers/names.
    // We need to know which keys correspond to benchmarks.
    // A simple way is to look at keys in the first data point that are not 'date', 'value', 'twr', 'drawdown', 'fx_rate'.
    const benchmarkKeys = useMemo(() => {
        if (!data || data.length === 0) return [];
        return Object.keys(data[0]).filter(k =>
            k !== 'date' && k !== 'value' && k !== 'twr' && k !== 'drawdown' &&
            k !== 'fx_rate' && k !== 'abs_gain' && k !== 'abs_roi' && k !== 'cum_flow' && k !== 'fx_return' &&
            k !== 'is_baseline' && k !== 'timestamp'
        );
    }, [data]);

    const processedData = useMemo(() => {
        if (!data || data.length === 0) return [];

        // Find the first valid fx_rate to use as baseline
        const firstValidFXPoint = data.find(d => (d as any).fx_rate != null);
        const startFX = firstValidFXPoint ? (firstValidFXPoint as any).fx_rate : undefined;

        // TWR Normalization Baseline (First point / Baseline)
        const baseline = data[0];
        const baseTwrFactor = 1 + (baseline.twr / 100);

        return data.map(d => {
            const currentFX = (d as any).fx_rate;

            // Normalize TWR
            const currentTwrFactor = 1 + (d.twr / 100);
            const adjTwr = ((currentTwrFactor / baseTwrFactor) - 1) * 100;

            return {
                ...d,
                twr: adjTwr, // Override with normalized TWR
                fx_return: startFX !== undefined && currentFX != null ? ((currentFX / startFX) - 1) * 100 : undefined
            };
        });
    }, [data]);

    const periodStats = useMemo(() => {
        if (!processedData || processedData.length < 2) return null;

        // Use processedData (normalized)
        const start = processedData[0];
        const end = processedData[processedData.length - 1];

        if (view === 'return') {
            const twr = end.twr; // Now normalized relative to start
            return {
                label: "Period TWR",
                text: `${twr > 0 ? '+' : ''}${twr.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`,
                color: twr >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
            };
        } else if (view === 'value') {
            const startVal = start.value;
            const endVal = end.value;
            const change = endVal - startVal;
            const changePct = startVal !== 0 ? (change / startVal) * 100 : 0;

            return {
                label: "Period Change",
                text: `${formatCurrency(change, currency)} (${change > 0 ? '+' : ''}${changePct.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%)`,
                color: change >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
            };
        }
        return null;
    }, [processedData, view, currency]);

    const chartedData = useMemo(() => {
        const dataToPlot = (processedData as any[]).filter((d: any) => !d.is_baseline);
        return dataToPlot.map(d => ({
            ...d,
            timestamp: new Date(d.date).getTime()
        }));
    }, [processedData]);

    const xDomain = useMemo(() => {
        if (period === '1d' && chartedData.length > 0) {
            // Robustly calculate 9:30 AM ET and 4:00 PM ET for the given day
            try {
                const firstTs = chartedData[0].timestamp;
                // Create a date object in browser local time
                const d = new Date(firstTs);

                // Get the date string for New York
                // Format: "MM/DD/YYYY"
                const nyDateStr = d.toLocaleDateString("en-US", { timeZone: "America/New_York" });

                // We need to find the timestamp corresponding to 9:30 on this NY date.
                // Since we don't have a timezone library, we can find it by iteration or offset guessing.
                // Heuristic:
                // 1. Parse the NY date string as UTC.
                // 2. Add 9.5 hours to get Open, 16 hours to get Close (assuming UTC).
                // 3. Add 4 or 5 hours (offset) to get back to UTC.
                // Wait, simpler:
                // Construct a string that `Date.parse` accepts as absolute ISO? No.

                // Browser-native robust way:
                // Create a date at Noon UTC on that day.
                // Check its NY time. Adjust.

                // Let's use the explicit "EST" fallback but logging error,
                // AND try a secondary parsing method if NaN.

                const parseWithZone = (dateStr: string, timeStr: string, zone: string) => {
                    const s = `${dateStr} ${timeStr} ${zone}`;
                    const ts = Date.parse(s);
                    if (!isNaN(ts)) return ts;
                    return NaN;
                };

                let startT = parseWithZone(nyDateStr, "09:30:00", "EST");
                let endT = parseWithZone(nyDateStr, "16:00:00", "EST");

                if (isNaN(startT)) {
                    // Try EDT
                    startT = parseWithZone(nyDateStr, "09:30:00", "EDT");
                    endT = parseWithZone(nyDateStr, "16:00:00", "EDT");
                }

                // If still NaN, fallback to "Assume first point is 9:30"
                if (isNaN(startT)) {
                    console.warn("Investa: Failed to parse xDomain dates. Using fallback.");
                    startT = firstTs;
                    // 6.5 hours = 23,400,000 ms
                    endT = startT + 23400000;
                } else {
                    // Double check if the parsed time is actually 9:30 in NY
                    // because "EST" might be interpreted as fixed -0500 even if it's summer.
                    const checkD = new Date(startT);
                    const checkTime = checkD.toLocaleTimeString("en-US", { timeZone: "America/New_York", hour12: false });
                    if (!checkTime.startsWith("09:30")) {
                        // Offset mismatch. Flip EST/EDT.
                        // If we used EST and got 10:30, we need EDT (-1h).
                        // If we used EDT and got 08:30, we need EST (+1h).
                        // Actually easier: just adjust the timestamp by the diff.
                        const [h, m] = checkTime.split(':').map(Number);
                        const diffMinutes = (h * 60 + m) - (9 * 60 + 30);
                        startT -= diffMinutes * 60 * 1000;
                        endT -= diffMinutes * 60 * 1000;
                    }
                }

                return [startT, endT];
            } catch (e) {
                console.error("Investa: Error calculating xDomain", e);
                return ['auto', 'auto'];
            }
        }
        return ['auto', 'auto'];
    }, [period, chartedData]);

    const xTicks = useMemo(() => {
        if (period === '1d' && Array.isArray(xDomain) && typeof xDomain[0] === 'number') {
            const start = xDomain[0] as number;
            const end = xDomain[1] as number;
            const ticks = [];
            // Generate ticks every 30 minutes
            let current = start;
            while (current <= end) {
                ticks.push(current);
                current += 30 * 60 * 1000;
            }
            return ticks;
        }
        return undefined;
    }, [xDomain, period]);

    if (!chartedData || chartedData.length === 0) {
        return (
            <div
                className="rounded-xl p-6 shadow-sm border border-border h-80 flex items-center justify-center text-muted-foreground"
                style={{ backgroundColor: 'var(--menu-solid)' }}
            >
                {loading ? 'Loading...' : 'No historical data available.'}
            </div>
        );
    }

    const formatXAxis = (tickItem: string) => {
        const date = new Date(tickItem);
        // For short periods, show time. For 1M (hourly), maybe show Day + Time?
        // Let's simple check:
        if (period === '1d') {
            return date.toLocaleTimeString(undefined, { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit' });
        } else if (period === '5d') {
            // Show Day + Time for context
            return date.toLocaleString(undefined, { timeZone: 'America/New_York', weekday: 'short', hour: '2-digit' });
        } else if (period === '1m') {
            // Hourly for 1 month. Date + maybe Hour? Too crowded.
            // Just Date is probably fine, or Day.
            return date.toLocaleDateString(undefined, { timeZone: 'America/New_York', month: 'short', day: 'numeric' });
        } else if (['3y', '5y', 'all', 'custom'].includes(period)) {
            // Long periods or custom range: Show Month + Year if range is large
            const showYear = period !== 'custom' || (new Date(customToDate).getTime() - new Date(customFromDate).getTime() > 1000 * 60 * 60 * 24 * 365);
            if (showYear) {
                return date.toLocaleDateString(undefined, { timeZone: 'America/New_York', month: 'short', year: 'numeric' });
            }
            return date.toLocaleDateString(undefined, { timeZone: 'America/New_York', month: 'short', day: 'numeric' });
        }
        return date.toLocaleDateString(undefined, { timeZone: 'America/New_York', month: 'short', day: 'numeric' });
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
                    timeZone: 'America/New_York',
                    weekday: 'short',
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });
            } else {
                dateStr = dateObj.toLocaleDateString(undefined, {
                    timeZone: 'America/New_York',
                    weekday: 'short',
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                });
            }

            // Find benchmark keys present in this data point
            const allKeys = Object.keys(dataPoint);
            const benchKeys = allKeys.filter(k =>
                k !== 'date' && k !== 'value' && k !== 'twr' && k !== 'drawdown' &&
                k !== 'fx_rate' && k !== 'fx_return' &&
                k !== 'abs_gain' && k !== 'abs_roi' && k !== 'cum_flow' &&
                k !== 'is_baseline' && k !== 'timestamp'
            );

            return (
                <div className="p-3 border border-border shadow-2xl rounded-xl min-w-[280px] sm:min-w-[320px] max-w-[calc(100vw-32px)] overflow-visible" style={{ backgroundColor: 'var(--menu-solid)' }}>
                    <p className="text-sm font-bold text-foreground mb-2 border-b border-border pb-1">
                        {dateStr}
                    </p>

                    <div className="grid grid-cols-2 gap-x-6 gap-y-0 pb-1">
                        {/* Portfolio Section */}
                        <div className="space-y-0.5">
                            <div className="flex items-center justify-between gap-2">
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-bold">Value</span>
                                <span className="text-[13px] font-bold text-foreground">
                                    {formatCurrency(dataPoint.value, currency)}
                                </span>
                            </div>
                            <div className="flex items-center justify-between gap-2">
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-bold">TWR</span>
                                <span className={`text-[13px] font-bold ${dataPoint.twr >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                    {dataPoint.twr >= 0 ? '+' : ''}{dataPoint.twr.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%
                                </span>
                            </div>
                            <div className="flex items-center justify-between gap-2">
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-bold">Drawdown</span>
                                <span className="text-[13px] font-bold text-rose-500">
                                    {dataPoint.drawdown.toFixed(2)}%
                                </span>
                            </div>
                        </div>

                        {/* Money-Weighted Metrics Section */}
                        <div className="space-y-0.5">
                            <div className="flex items-center justify-between gap-2">
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-bold">Gain</span>
                                <span className={`text-[13px] font-bold ${dataPoint.abs_gain >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                    {formatCurrency(dataPoint.abs_gain, currency)}
                                </span>
                            </div>
                            <div className="flex items-center justify-between gap-2">
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-bold">Cost</span>
                                <span className="text-[13px] font-bold text-foreground">
                                    {formatCurrency(dataPoint.cum_flow, currency)}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* FX Rate & Benchmarks Section (Full Width) */}
                    {(dataPoint.fx_rate != null || benchKeys.length > 0) && (
                        <div className="mt-1 pt-1 border-t border-border space-y-1">
                            {dataPoint.fx_rate != null && (
                                <div className="grid grid-cols-2 gap-x-6">
                                    <div className="flex items-center justify-between gap-2">
                                        <span className="text-[10px] uppercase tracking-wider text-amber-500 font-bold">FX Rate</span>
                                        <span className="text-[12px] font-bold text-amber-500">
                                            {typeof dataPoint.fx_rate === 'number' ? dataPoint.fx_rate.toFixed(4) : dataPoint.fx_rate}
                                        </span>
                                    </div>
                                    {dataPoint.fx_return != null && (
                                        <div className="flex items-center justify-between gap-2">
                                            <span className="text-[10px] uppercase tracking-wider text-amber-500 font-bold">FX Ret</span>
                                            <span className={`text-[12px] font-bold ${dataPoint.fx_return >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                                {dataPoint.fx_return >= 0 ? '+' : ''}{dataPoint.fx_return.toFixed(2)}%
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}

                            {benchKeys.length > 0 && (
                                <div className="grid grid-cols-2 gap-x-6 gap-y-0.5">
                                    {benchKeys.map((bKey, idx) => {
                                        const payloadEntry = payload.find(p => p.name === bKey);
                                        const color = payloadEntry?.color || COLORS[(idx + 1) % COLORS.length];

                                        return (
                                            <div key={bKey} className="flex items-center justify-between gap-2">
                                                <span className="text-[10px] truncate max-w-[80px] font-bold uppercase tracking-wider" style={{ color: color }}>
                                                    {bKey}
                                                </span>
                                                <span className={`text-[12px] font-bold ${dataPoint[bKey] >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                                    {dataPoint[bKey] >= 0 ? '+' : ''}{Number(dataPoint[bKey]).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            );
        }
        return null;
    };

    const isContinuous = period === '1d';

    return (
        <div className="bg-card rounded-xl p-4 shadow-sm border border-border mb-6 overflow-visible">
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

                <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 min-w-0">
                    <div className="flex flex-col gap-2 w-full min-w-0">
                        <div className="flex items-center gap-2 w-full overflow-x-auto no-scrollbar pb-1 -mx-1 px-1">
                            <PeriodSelector selectedPeriod={period} onPeriodChange={setPeriod} />
                        </div>
                        {period === 'custom' && (
                            <div className="flex items-center gap-2 animate-in fade-in slide-in-from-top-1 duration-200">
                                <div className="flex items-center gap-1.5 bg-secondary rounded-lg px-2.5 py-1 border border-border">
                                    <span className="text-[10px] uppercase tracking-wider font-bold text-muted-foreground">From</span>
                                    <input
                                        type="date"
                                        value={customFromDate}
                                        onChange={(e) => setCustomFromDate(e.target.value)}
                                        className="bg-transparent border-none text-xs sm:text-sm font-medium focus:ring-0 p-0 w-[110px] sm:w-auto"
                                    />
                                </div>
                                <div className="flex items-center gap-1.5 bg-secondary rounded-lg px-2.5 py-1 border border-border">
                                    <span className="text-[10px] uppercase tracking-wider font-bold text-muted-foreground">To</span>
                                    <input
                                        type="date"
                                        value={customToDate}
                                        onChange={(e) => setCustomToDate(e.target.value)}
                                        className="bg-transparent border-none text-xs sm:text-sm font-medium focus:ring-0 p-0 w-[110px] sm:w-auto"
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="flex items-center gap-2 w-full sm:w-auto pb-1">
                        {view === 'return' && (
                            <BenchmarkSelector selectedBenchmarks={benchmarks} onBenchmarkChange={onBenchmarksChange} />
                        )}
                        <div className="flex bg-secondary rounded-lg p-1 border border-border shrink-0">
                            <button
                                onClick={() => setView('return')}
                                className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-all ${view === 'return'
                                    ? 'bg-[#0097b2] text-white shadow-sm'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                    }`}
                            >
                                Return %
                            </button>
                            <button
                                onClick={() => setView('value')}
                                className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-all ${view === 'value'
                                    ? 'bg-[#0097b2] text-white shadow-sm'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                    }`}
                            >
                                Value
                            </button>
                            <button
                                onClick={() => setView('drawdown')}
                                className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-all ${view === 'drawdown'
                                    ? 'bg-[#0097b2] text-white shadow-sm'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                    }`}
                            >
                                Drawdown
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div className="h-[400px] w-full relative overflow-visible pb-4">
                {loading && (
                    <div className="absolute inset-0 bg-white/50 dark:bg-gray-800/50 flex items-center justify-center z-10">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    </div>
                )}
                <ResponsiveContainer width="100%" height="100%">
                    {view === 'return' ? (
                        <LineChart syncId="portfolio-sync" data={chartedData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                            <XAxis
                                dataKey="timestamp"
                                domain={isContinuous ? xDomain : undefined}
                                ticks={isContinuous ? xTicks : undefined}
                                type={isContinuous ? "number" : "category"}
                                scale={isContinuous ? "time" : undefined}
                                tickFormatter={formatXAxis}
                                allowDataOverflow={isContinuous}
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
                                domain={['auto', 'auto']}
                                width={50}
                            />
                            <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--border)', strokeWidth: 1, strokeDasharray: '4 4' }} />
                            <Legend />
                            {/* Portfolio Line with Gradient or Solid color */}
                            {/* Time-Weighted Return (TWR) */}
                            <defs>
                                <linearGradient id="colorTwr" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <Line
                                type="monotone"
                                dataKey="twr"
                                name="Portfolio"
                                stroke="#3b82f6"
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 4, strokeWidth: 0 }}
                                animationDuration={800}
                                connectNulls={true}
                            />
                            {/* Benchmarks */}
                            {benchmarkKeys.map((bKey, idx) => (
                                <Line
                                    key={bKey}
                                    type="monotone"
                                    dataKey={bKey}
                                    name={bKey}
                                    stroke={COLORS[(idx + 1) % COLORS.length]}
                                    strokeWidth={2}
                                    dot={false}
                                    activeDot={{ r: 4, strokeWidth: 0 }}
                                    connectNulls={true}
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
                            <ReferenceLine y={0} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="3 3" />
                        </LineChart>
                    ) : view === 'value' ? (
                        <AreaChart syncId="portfolio-sync" data={chartedData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                            <defs>
                                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                            <XAxis
                                dataKey="timestamp"
                                domain={isContinuous ? xDomain : undefined}
                                ticks={isContinuous ? xTicks : undefined}
                                type={isContinuous ? "number" : "category"}
                                scale={isContinuous ? "time" : undefined}
                                tickFormatter={formatXAxis}
                                allowDataOverflow={isContinuous}
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
                            <Tooltip
                                content={<CustomTooltip />}
                                allowEscapeViewBox={{ x: false, y: false }}
                            />
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
                            <ReferenceLine y={0} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="3 3" />
                        </AreaChart>
                    ) : (
                        <AreaChart syncId="portfolio-sync" data={chartedData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                            <defs>
                                <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.1} />
                                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                            <XAxis
                                dataKey="timestamp"
                                domain={isContinuous ? xDomain : undefined}
                                ticks={isContinuous ? xTicks : undefined}
                                type={isContinuous ? "number" : "category"}
                                scale={isContinuous ? "time" : undefined}
                                tickFormatter={formatXAxis}
                                allowDataOverflow={isContinuous}
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
                            <Tooltip
                                content={<CustomTooltip />}
                                allowEscapeViewBox={{ x: false, y: false }}
                            />
                            <Area
                                name="Drawdown"
                                type="monotone"
                                dataKey="drawdown"
                                stroke="#ef4444"
                                fillOpacity={1}
                                fill="url(#colorDrawdown)"
                                strokeWidth={2}
                            />
                            <ReferenceLine y={0} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="3 3" />
                        </AreaChart>
                    )}

                </ResponsiveContainer>
            </div>
        </div>
    );
}
