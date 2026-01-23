import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
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
    ReferenceLine,
    BarChart,
    Bar,
    ComposedChart
} from 'recharts';
import PeriodSelector from './PeriodSelector';

import { fetchStockHistory } from '../lib/api';
import { formatCurrency } from '../lib/utils';

// --- Types ---
interface StockPriceChartProps {
    symbol: string;
    currency: string;
    benchmarks?: string[]; // Optional initial benchmarks
}

interface CustomTooltipProps {
    active?: boolean;
    payload?: any[];
    label?: string;
    view: 'price' | 'return';
    currency: string;
}

// --- Constants ---
const COLORS = [
    "#2563eb", // Primary Blue (Stock)
    "#dc2626", // Red (Benchmark 1)
    "#16a34a", // Green (Benchmark 2)
    "#d97706", // Amber
    "#9333ea", // Purple
];

// --- Helper Functions ---
const formatValue = (val: number, view: 'price' | 'return', currency: string) => {
    if (view === 'return') return `${val > 0 ? '+' : ''}${val.toFixed(2)}%`;
    return formatCurrency(val, currency);
};

const formatVolume = (val: number) => {
    if (val >= 1e9) return `${(val / 1e9).toFixed(2)}B`;
    if (val >= 1e6) return `${(val / 1e6).toFixed(2)}M`;
    if (val >= 1e3) return `${(val / 1e3).toFixed(2)}K`;
    return val.toString();
};

const calculateSMA = (data: any[], period: number) => {
    if (data.length < period) return [];
    // Calculate SMA
    const smaData = data.map((item, index, array) => {
        if (index < period - 1) return { ...item, sma: null };
        let sum = 0;
        for (let i = 0; i < period; i++) {
            sum += array[index - i].value;
        }
        return { ...item, sma: sum / period };
    });
    return smaData;
};

export default function StockPriceChart({ symbol, currency }: StockPriceChartProps) {
    const [view, setView] = useState<'price' | 'return'>('price');
    const [period, setPeriod] = useState('1y');
    const [showSMA50, setShowSMA50] = useState(false);
    const [showSMA200, setShowSMA200] = useState(false);

    const containerRef = useRef<HTMLDivElement>(null);

    // Determine interval base on period
    const interval = useMemo(() => {
        if (period === '1d') return '2m';
        if (period === '5d') return '15m';
        if (period === '1m') return '1d';
        if (period === '3m') return '1d';
        return '1d';
    }, [period]);

    // Determine Fetch Parameters (Fetch more data than shown for SMA)
    const fetchParams = useMemo(() => {
        let fetchPeriod = period;
        let fetchInterval = interval;

        // Map request to longer period for SMA buffer
        // Note: Backend supports: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        switch (period) {
            case '1d': fetchPeriod = '5d'; break;     // Need prev days for intraday SMA? 
            // Actually intraday SMA usually resets or needs 5d worth of minutes.
            case '5d': fetchPeriod = '1mo'; break;
            case '1m': fetchPeriod = '3mo'; break;
            case '3m': fetchPeriod = '6mo'; break;
            case '6m': fetchPeriod = '1y'; break;
            case 'ytd': fetchPeriod = '2y'; break;    // Safe buffer
            case '1y': fetchPeriod = '5y'; break;     // Increased to 5y to ensure clean 200 SMA
            case '3y': fetchPeriod = '5y'; break;
            case '5y': fetchPeriod = '10y'; break;
            case '10y': fetchPeriod = 'max'; break; // 10y -> max is best bet
            case 'max': fetchPeriod = 'max'; break;
        }

        return { period: fetchPeriod, interval: fetchInterval };
    }, [period, interval]);

    // Data Fetching
    const { data: rawData, isLoading } = useQuery({
        queryKey: ['stock_history', symbol, fetchParams.period, fetchParams.interval],
        queryFn: ({ signal }) => fetchStockHistory(symbol, fetchParams.period, fetchParams.interval, [], signal),
        placeholderData: keepPreviousData,
        staleTime: period === '1d' ? 60 * 1000 : 5 * 60 * 1000,
        refetchInterval: period === '1d' ? 60 * 1000 : false,
    });

    const data = rawData || [];
    const isContinuous = period === '1d';

    // Processing Data for Chart
    const chartedData = useMemo(() => {
        if (!data || data.length === 0) return [];

        let processed = data.map(d => ({
            ...d,
            timestamp: new Date(d.date).getTime(),
        }));

        // Calculate SMAs on the FULL fetched dataset (including buffer)
        if (data.length >= 50) {
            const sma50 = calculateSMA(processed, 50);
            processed = processed.map((p, i) => ({ ...p, sma50: sma50[i].sma }));
        }

        if (data.length >= 200) {
            const sma200 = calculateSMA(processed, 200);
            processed = processed.map((p, i) => ({ ...p, sma200: sma200[i].sma }));
        }

        // NOW Filter down to the requested period for display
        // We calculate start cutoff based on the *User's Selected Period*
        const now = new Date();
        let cutoffTime = 0;

        // Simple cutoff logic
        switch (period) {
            case '1d':
                // For 1d, we just want today (or last trading day). 
                // However, our backend 1d logic usually returns strict 9:30-16:00 of *latest* day.
                // If we fetched 5d, we have 5 days. We want only the last day.
                // Let's take the date string of the last point and filter by that?
                if (processed.length > 0) {
                    const lastDate = new Date(processed[processed.length - 1].timestamp);
                    // Create midnight of that day in local or NY?
                    // Safe bet: Last 390 minutes (6.5 hours)?
                    // Better: Same Year-Month-Day as last point.
                    const lastYMD = lastDate.toISOString().split('T')[0];
                    // Filter where date string starts with lastYMD
                    // But date object comparison is safer.
                    const startOfDay = new Date(lastDate);
                    startOfDay.setHours(0, 0, 0, 0);
                    cutoffTime = startOfDay.getTime();
                }
                break;
            case '5d': cutoffTime = now.getTime() - (5 * 24 * 60 * 60 * 1000); break;
            case '1m': cutoffTime = now.getTime() - (30 * 24 * 60 * 60 * 1000); break;
            case '3m': cutoffTime = now.getTime() - (90 * 24 * 60 * 60 * 1000); break;
            case '6m': cutoffTime = now.getTime() - (180 * 24 * 60 * 60 * 1000); break;
            case 'ytd':
                cutoffTime = new Date(now.getFullYear(), 0, 1).getTime();
                break;
            case '1y': cutoffTime = now.getTime() - (365 * 24 * 60 * 60 * 1000); break;
            case '3y': cutoffTime = now.getTime() - (3 * 365 * 24 * 60 * 60 * 1000); break;
            case '5y': cutoffTime = now.getTime() - (5 * 365 * 24 * 60 * 60 * 1000); break;
            case '10y': cutoffTime = now.getTime() - (10 * 365 * 24 * 60 * 60 * 1000); break;
            default: cutoffTime = 0; // max/all
        }

        // Apply visual filter
        if (period !== 'max') {
            processed = processed.filter(d => d.timestamp >= cutoffTime);
        }

        return processed;
    }, [data, period]);

    // Calculate Stats for Header (Based on Displayed Data)
    const stats = useMemo(() => {
        if (!chartedData || chartedData.length < 2) return null;
        const start = chartedData[0];
        const end = chartedData[chartedData.length - 1];

        const currentPrice = end.value;
        const startPrice = start.value;

        const change = currentPrice - startPrice;
        const changePct = startPrice !== 0 ? (change / startPrice) * 100 : 0;

        return {
            change,
            changePct,
            currentPrice
        };
    }, [data]);

    const gradientOffset = useMemo(() => {
        if (!chartedData || chartedData.length === 0) return 0;

        const dataMax = Math.max(...chartedData.map((d) => d.return_pct));
        const dataMin = Math.min(...chartedData.map((d) => d.return_pct));

        if (dataMax <= 0) return 0;
        if (dataMin >= 0) return 1;

        return dataMax / (dataMax - dataMin);
    }, [chartedData]);

    // Domain Calculation (for X Axis)
    const xDomain = useMemo(() => {
        if (period === '1d' && chartedData.length > 0) {
            // Force strict 09:30 - 16:00 EST visual range
            try {
                // Get the date string in NY time from the first data point
                const firstTs = chartedData[0].timestamp;
                const d = new Date(firstTs);
                const nyDateStr = d.toLocaleDateString("en-US", { timeZone: "America/New_York" });

                // Construct 9:30 and 16:00 for *that* day
                // We'll use a heuristic since we can't easily construct Date from NY string in JS without library.
                // But we know the day. 
                // Let's assume the timestamp IS correct UTC.
                // We just need start/end timestamps.

                // Fallback: Use min/max of data if filtering is working correctly backend side.
                // If backend filters correctly, ['auto', 'auto'] is fine?
                // No, we want fixed range even if data starts at 09:31.

                // Let's rely on data extents + buffer or let auto handle it if backend is strict.
                return ['auto', 'auto'];
            } catch (e) {
                return ['auto', 'auto'];
            }
        }
        return ['auto', 'auto'];
    }, [period, chartedData]);

    // Formatting Functions (EST Forced)
    const formatXAxis = (tickItem: number) => {
        const date = new Date(tickItem);
        if (period === '1d' || period === '5d') {
            return date.toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: '2-digit', minute: '2-digit', hour12: true });
        } else if (period === '1m') {
            return date.toLocaleDateString("en-US", { timeZone: "America/New_York", month: 'short', day: 'numeric' });
        } else {
            return date.toLocaleDateString("en-US", { timeZone: "America/New_York", month: 'short', day: 'numeric' });
        }
    };

    const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
        if (active && payload && payload.length) {
            const dataPoint = payload[0].payload;
            const dateStr = new Date(dataPoint.date).toLocaleString("en-US", {
                timeZone: "America/New_York",
                weekday: 'short', month: 'short', day: 'numeric',
                hour: (period === '1d' || period === '5d') ? '2-digit' : undefined,
                minute: (period === '1d' || period === '5d') ? '2-digit' : undefined,
                year: (period !== '1d' && period !== '5d') ? 'numeric' : undefined
            });

            return (
                <div className="p-3 border border-border shadow-2xl rounded-xl min-w-[240px] bg-card/95 backdrop-blur-sm">
                    <p className="text-sm font-bold text-foreground mb-2 border-b border-border pb-1">
                        {dateStr}
                    </p>
                    <div className="space-y-1">
                        {/* Main Stock */}
                        <div className="flex items-center justify-between gap-4">
                            <span className="text-xs font-bold text-blue-500 uppercase">{symbol}</span>
                            <span className="text-sm font-bold text-foreground">
                                {view === 'price' ? formatCurrency(dataPoint.value, currency) :
                                    <span className={dataPoint.return_pct >= 0 ? "text-emerald-500" : "text-red-500"}>
                                        {dataPoint.return_pct >= 0 ? '+' : ''}{dataPoint.return_pct.toFixed(2)}%
                                    </span>
                                }
                            </span>
                        </div>

                        {/* Volume */}
                        <div className="flex items-center justify-between gap-4">
                            <span className="text-xs text-muted-foreground uppercase">Volume</span>
                            <span className="text-sm text-foreground">{formatVolume(dataPoint.volume)}</span>
                        </div>

                        {/* SMAs */}
                        {view === 'price' && showSMA50 && dataPoint.sma50 != null && (
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-xs font-bold text-orange-500 uppercase">SMA 50</span>
                                <span className="text-sm font-medium text-foreground">{formatCurrency(dataPoint.sma50, currency)}</span>
                            </div>
                        )}
                        {view === 'price' && showSMA200 && dataPoint.sma200 != null && (
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-xs font-bold text-purple-600 uppercase">SMA 200</span>
                                <span className="text-sm font-medium text-foreground">{formatCurrency(dataPoint.sma200, currency)}</span>
                            </div>
                        )}


                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div ref={containerRef} className="bg-card rounded-xl p-4 shadow-sm border border-border mb-6 overflow-visible">
            {/* Header Layout (Matches PerformanceGraph) */}
            <div className="mb-6">
                <div className="flex flex-col items-start gap-1 md:flex-row md:justify-between md:items-center md:gap-0 mb-4">
                    <h3 className="text-lg font-medium text-muted-foreground">Price History</h3>
                    {stats ? (
                        <div className="flex items-baseline gap-2">
                            <span className="text-xl font-bold tracking-tight text-foreground">
                                {formatCurrency(stats.currentPrice, currency)}
                            </span>
                            <span className={`text-sm font-medium ${stats.change >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}`}>
                                {stats.change >= 0 ? '+' : ''}{formatCurrency(stats.change, currency)} ({stats.changePct.toFixed(2)}%)
                            </span>
                        </div>
                    ) : (
                        <div className="h-9" />
                    )}
                </div>

                <div className="flex flex-col xl:flex-row justify-between items-end xl:items-center gap-4 min-w-0">

                    {/* Period Selector - Order 2 on mobile (below controls), Order 1 on Desktop */}
                    <div className="w-full xl:w-auto overflow-x-auto no-scrollbar pb-1 -mx-1 px-1 order-2 xl:order-1">
                        <PeriodSelector selectedPeriod={period} onPeriodChange={setPeriod} />
                    </div>

                    {/* Toggles & Benchmark - Order 1 on mobile (top), Order 2 on Desktop */}
                    <div className="flex items-center gap-3 w-full xl:w-auto justify-between xl:justify-end order-1 xl:order-2">
                        {/* SMA Toggles (Only in Price View) */}
                        <div className="flex items-center gap-2">
                            {view === 'price' && (
                                <div className="flex bg-secondary rounded-lg p-1 border border-border shrink-0 gap-1">
                                    <button
                                        onClick={() => setShowSMA50(!showSMA50)}
                                        className={`px-2 py-1 text-[10px] font-bold rounded-md transition-all ${showSMA50
                                            ? 'bg-orange-500 text-white shadow-sm'
                                            : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                            }`}
                                    >
                                        MA50
                                    </button>
                                    <button
                                        onClick={() => setShowSMA200(!showSMA200)}
                                        className={`px-2 py-1 text-[10px] font-bold rounded-md transition-all ${showSMA200
                                            ? 'bg-purple-600 text-white shadow-sm'
                                            : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                            }`}
                                    >
                                        MA200
                                    </button>
                                </div>
                            )}

                            <div className="flex bg-secondary rounded-lg p-1 border border-border shrink-0">
                                <button
                                    onClick={() => setView('price')}
                                    className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-all ${view === 'price'
                                        ? 'bg-[#0097b2] text-white shadow-sm'
                                        : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                        }`}
                                >
                                    Price
                                </button>
                                <button
                                    onClick={() => setView('return')}
                                    className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-all ${view === 'return'
                                        ? 'bg-[#0097b2] text-white shadow-sm'
                                        : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                        }`}
                                >
                                    Return %
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Chart Container */}
            <div className="h-[400px] w-full relative overflow-visible pb-4">
                {isLoading && (
                    <div className="absolute inset-0 bg-white/50 dark:bg-gray-800/50 flex items-center justify-center z-10 rounded-xl">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    </div>
                )}

                {chartedData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartedData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                            <defs>
                                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="splitColorFill" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset={gradientOffset} stopColor="#10b981" stopOpacity={0.15} />
                                    <stop offset={gradientOffset} stopColor="#ef4444" stopOpacity={0.15} />
                                </linearGradient>
                                <linearGradient id="splitColorStroke" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset={gradientOffset} stopColor="#10b981" stopOpacity={1} />
                                    <stop offset={gradientOffset} stopColor="#ef4444" stopOpacity={1} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.5} />
                            <XAxis
                                dataKey="timestamp"
                                domain={isContinuous ? xDomain : undefined}
                                type={isContinuous ? "number" : "category"}
                                scale={isContinuous ? "time" : undefined}
                                tickFormatter={formatXAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                minTickGap={40}
                            />
                            <YAxis
                                yAxisId="main"
                                tickFormatter={(val) => view === 'return' ? `${val.toFixed(1)}%` : new Intl.NumberFormat('en-US', { notation: "compact", maximumFractionDigits: 2 }).format(val)}
                                domain={['auto', 'auto']}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                width={50}
                            />
                            <YAxis
                                yAxisId="vol"
                                orientation="right"
                                tickFormatter={(val) => ""}
                                axisLine={false}
                                tickLine={false}
                                width={0}
                                domain={[0, (dataMax: number) => dataMax * 5]} // Volume bars low
                            />

                            <Tooltip content={<CustomTooltip view={view} currency={currency} />} cursor={{ stroke: 'var(--border)', strokeWidth: 1, strokeDasharray: '4 4' }} />

                            <Bar dataKey="volume" yAxisId="vol" fill="#9ca3af" opacity={0.15} barSize={period === '1d' ? undefined : 6} />

                            {view === 'price' ? (
                                <>
                                    <Area
                                        yAxisId="main"
                                        type="monotone"
                                        dataKey="value"
                                        stroke="#2563eb"
                                        strokeWidth={2}
                                        fillOpacity={1}
                                        fill="url(#colorPrice)"
                                        activeDot={{ r: 4, strokeWidth: 0 }}
                                    />
                                    {showSMA50 && (
                                        <Line
                                            yAxisId="main"
                                            type="monotone"
                                            dataKey="sma50"
                                            stroke="#f97316" // Orange
                                            strokeWidth={1.5}
                                            dot={false}
                                            activeDot={{ r: 4 }}
                                            connectNulls
                                        />
                                    )}
                                    {showSMA200 && (
                                        <Line
                                            yAxisId="main"
                                            type="monotone"
                                            dataKey="sma200"
                                            stroke="#9333ea" // Purple
                                            strokeWidth={1.5}
                                            dot={false}
                                            activeDot={{ r: 4 }}
                                            connectNulls
                                        />
                                    )}
                                </>
                            ) : (
                                <Area
                                    yAxisId="main"
                                    type="monotone"
                                    dataKey="return_pct"
                                    stroke="url(#splitColorStroke)"
                                    fill="url(#splitColorFill)"
                                    strokeWidth={2}
                                    activeDot={{ r: 4, strokeWidth: 0 }}
                                />
                            )}



                            {view === 'return' && <ReferenceLine y={0} yAxisId="main" stroke="#9ca3af" strokeDasharray="3 3" />}
                        </ComposedChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="flex items-center justify-center h-full text-muted-foreground w-full">
                        {!isLoading ? "No data available." : ""}
                    </div>
                )}
            </div>
        </div >
    );
}
