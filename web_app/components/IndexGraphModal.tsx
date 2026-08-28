'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { createPortal } from 'react-dom';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';
import { X, TrendingUp, Info } from 'lucide-react';
import PeriodSelector from './PeriodSelector';
import TradingViewChart from './TradingViewChart';
import { fetchMarketHistory } from '../lib/api';
import { benchmarkYahooSymbol, toTradingViewSymbol } from '../lib/tradingview';
import { Badge } from './ui/badge';
import type { MarketIndex } from './MarketsTab';
import { cn } from '@/lib/utils';

interface IndexGraphModalProps {
    isOpen: boolean;
    onClose: () => void;
    benchmarks: string[];
    currentIndices?: Record<string, MarketIndex>;
    /** Name of the index card that opened this, if any — the TradingView view
     *  starts on that index rather than on the first benchmark. */
    focusIndex?: string | null;
}

const COLORS = [
    "#0097b2", // Investa Cyan
    "#f59e0b", // Amber
    "#8b5cf6", // Violet
    "#e11d48", // Rose
    "#10b981", // Emerald
];

const CustomTooltip = ({ active, payload, label, period }: {
    active?: boolean;
    payload?: Array<{ value: number; name?: string; color?: string; payload?: Record<string, number> }>;
    label?: string | number;
    period?: string;
}) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-background/60 backdrop-blur-xl p-4 rounded-2xl min-w-[280px] border border-border/50 shadow-2xl">
                <p className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em] mb-3 pb-2">
                    {new Date(label as string | number).toLocaleString([], {
                        calendar: 'gregory',
                        timeZone: 'America/New_York',
                        month: 'short',
                        day: 'numeric',
                        year: period === '1d' ? undefined : 'numeric',
                        hour: period === '1d' || period === '5d' ? '2-digit' : undefined,
                        minute: period === '1d' || period === '5d' ? '2-digit' : undefined
                    })}
                </p>
                <div className="space-y-2.5">
                    {payload.map((entry, index: number) => (
                        <div key={index} className="flex items-center justify-between gap-6">
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                                <span className="text-xs font-bold text-foreground/90">{entry.name}</span>
                            </div>
                            <div className="flex flex-col items-end">
                                <span className={`text-xs font-black tabular-nums ${entry.value >= 0 ? 'text-up' : 'text-rose-500'}`}>
                                    {entry.value.toFixed(2)}%
                                </span>
                                {payload[0]?.payload?.[`${entry.name}_price`] !== undefined && (
                                    <span className="text-[10px] font-medium text-muted-foreground tabular-nums">
                                        {payload[0].payload[`${entry.name}_price`].toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }
    return null;
};

export default function IndexGraphModal({ isOpen, onClose, benchmarks, currentIndices, focusIndex }: IndexGraphModalProps) {
    const [period, setPeriod] = useState('1y');
    const [view, setView] = useState<'return' | 'tradingview'>('return');
    // Carries the index it was picked under, so opening a different card drops
    // a stale pick rather than charting the last card's index.
    const [tvPick, setTvPick] = useState<{ focus: string | null; bench: string } | null>(null);
    const [data, setData] = useState<Array<Record<string, number | string | null>>>([]);
    const [loading, setLoading] = useState(false);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    // TradingView charts one instrument at a time, where the return view
    // overlays them all — so that view picks an index, and only among the ones
    // we can resolve to a TradingView listing.
    const tvBenchmarks = useMemo(
        () => benchmarks.filter(b => {
            // Both hops have to land: a benchmark we know (S&P 500 → ^GSPC) and
            // an instrument the free widget will actually draw. Total-return
            // indices clear the first and fail the second.
            const yahoo = benchmarkYahooSymbol(b);
            return yahoo !== null && toTradingViewSymbol(yahoo) !== null;
        }),
        [benchmarks],
    );

    // The clicked card names an index the way the backend does ("Dow"), the
    // benchmark list the way BENCHMARK_MAPPING does ("Dow Jones") — they meet
    // at the Yahoo symbol.
    const focus = focusIndex ?? null;
    const focusBenchmark = useMemo(() => {
        const target = focus ? benchmarkYahooSymbol(focus) : null;
        return target ? tvBenchmarks.find(b => benchmarkYahooSymbol(b) === target) : undefined;
    }, [focus, tvBenchmarks]);

    const picked = tvPick && tvPick.focus === focus && tvBenchmarks.includes(tvPick.bench) ? tvPick.bench : null;
    const tvSelected = picked ?? focusBenchmark ?? tvBenchmarks[0];

    useEffect(() => {
        // TradingView draws from its own feed — don't pull ours behind it.
        if (!isOpen || view === 'tradingview') return;

        let isMounted = true;
        const fetchData = async () => {
            setLoading(true);
            try {
                // Determine correct interval based on period to match main PerformanceGraph
                let interval = '1d';
                if (period === '1d') interval = '2m';
                else if (period === '5d') interval = '15m';

                const result = await fetchMarketHistory(benchmarks, period, interval);
                if (isMounted) setData(result);
            } catch (error) {
                console.error('Failed to fetch market history:', error);
            } finally {
                if (isMounted) setLoading(false);
            }
        };

        fetchData();
        return () => { isMounted = false; };
    }, [isOpen, benchmarks, period, view]);

    const activeBenchmarks = useMemo(() => {
        return benchmarks;
    }, [benchmarks]);

    const currentReturns = useMemo(() => {
        if (data.length === 0) return {};
        // Find the last entry that actually has values for at least one benchmark
        for (let i = data.length - 1; i >= 0; i--) {
            const entry = data[i];
            const hasValue = benchmarks.some(b => entry[b] !== undefined && entry[b] !== null);
            if (hasValue) return entry;
        }
        return data[data.length - 1] || {};
    }, [data, benchmarks]);

    if (!isOpen || !mounted) return null;

    // Use createPortal to match StockDetailModal's behavior if needed, 
    // but standard fixed position is usually fine if we handle z-index correctly.
    return createPortal(
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 transition-all duration-300 isolate">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in" onClick={onClose} />

            {/* Modal */}
            <div
                style={{ backgroundColor: 'var(--menu-solid)' }}
                className="relative w-full max-w-5xl h-[90vh] sm:h-auto sm:max-h-[85vh] rounded-[2.5rem] flex flex-col overflow-hidden animate-in zoom-in-95 slide-in-from-bottom-10 duration-300"
            >
                {/* Header Section */}
                <div className="sticky top-0 z-50 bg-card flex-shrink-0">
                    <div className="p-8 pb-6 flex justify-between items-start">
                        <div className="flex items-center gap-6 text-foreground">
                            {/* Icon Stack/Placeholder */}
                            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#0097b2] to-primary flex items-center justify-center flex-shrink-0 p-3 overflow-hidden">
                                <TrendingUp className="w-full h-full text-white" />
                            </div>

                            <div className="flex flex-col">
                                <div className="flex items-center gap-4 mb-0.5">
                                    <h2 className="text-3xl font-black tracking-tight text-foreground">Markets</h2>
                                    <Badge variant="secondary" className="font-black text-[10px] uppercase tracking-widest px-3">Benchmarks</Badge>
                                </div>
                                <div className="flex items-center gap-2 text-sm text-cyan-500 font-bold">
                                    Indices Performance
                                </div>
                            </div>
                        </div>

                        <div className="flex items-start gap-8">
                            <div className="flex items-center mr-6">
                                {activeBenchmarks.map((bench, idx) => {
                                    const graphLatest = currentReturns[bench];
                                    const graphPrice = currentReturns[`${bench}_price`];

                                    // Normalize names for lookup (Backend uses "Dow" and "Nasdaq")
                                    const lookupName = bench === 'Dow Jones' ? 'Dow' : (bench === 'NASDAQ' ? 'Nasdaq' : bench);

                                    // Find live data by name
                                    const liveIndex = currentIndices ? Object.values(currentIndices).find((i) => i.name === lookupName || i.name === bench) : null;

                                    const displayPrice = liveIndex?.price ?? (graphPrice != null ? Number(graphPrice) : undefined);
                                    const displayPct = liveIndex?.changesPercentage ?? (graphLatest != null ? Number(graphLatest) : undefined);

                                    if (displayPct === undefined) return null;
                                    return (
                                        <React.Fragment key={bench}>
                                            {idx > 0 && <div className="mx-6" />}
                                            <div className="flex flex-col items-end">
                                                <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/80 mb-0.5">{bench}</span>
                                                <span className="text-xl font-bold tracking-tighter tabular-nums text-foreground">
                                                    {displayPrice != null ? displayPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '--'}
                                                </span>
                                                <span className={cn(
                                                    "text-[10px] font-bold tracking-tight tabular-nums",
                                                    displayPct >= 0 ? "text-up" : "text-rose-500"
                                                )}>
                                                    {displayPct.toFixed(2)}%
                                                </span>
                                            </div>
                                        </React.Fragment>
                                    );
                                })}
                            </div>

                            <button
                                onClick={onClose}
                                className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-all duration-200 text-muted-foreground hover:text-foreground group"
                            >
                                <X className="w-7 h-7 group-hover:rotate-90 transition-transform duration-300" />
                            </button>
                        </div>
                    </div>

                    {/* Range Selector Integration */}
                    <div className="px-8 pb-6 flex items-center justify-between gap-4 flex-wrap">
                        {/* TradingView ships its own range tabs and interval picker,
                            so ours would just be a rival — it gets an index picker
                            in that slot instead. */}
                        {view === 'return' ? (
                            <PeriodSelector selectedPeriod={period} onPeriodChange={setPeriod} />
                        ) : (
                            <div className="flex items-center gap-1.5 flex-wrap">
                                {tvBenchmarks.map((bench) => (
                                    <button
                                        key={bench}
                                        onClick={() => setTvPick({ focus, bench })}
                                        className={cn(
                                            'px-3 py-1 text-[11px] font-bold rounded-full border transition-all',
                                            bench === tvSelected
                                                ? 'bg-primary text-primary-foreground border-transparent shadow-sm'
                                                : 'text-muted-foreground border-border bg-secondary hover:text-foreground hover:bg-accent/10',
                                        )}
                                    >
                                        {bench}
                                    </button>
                                ))}
                            </div>
                        )}

                        <div className="flex items-center gap-6">
                            {view === 'return' && (
                                <div className="hidden sm:flex items-center gap-4">
                                    {activeBenchmarks.map((bench, idx) => (
                                        <div key={bench} className="flex items-center gap-2">
                                            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                                            <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">{bench}</span>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {tvBenchmarks.length > 0 && (
                                <div className="flex bg-secondary rounded-lg p-1 border border-border shrink-0">
                                    {([
                                        { key: 'return', label: 'Return %' },
                                        { key: 'tradingview', label: 'TradingView' },
                                    ] as const).map(({ key, label }) => (
                                        <button
                                            key={key}
                                            onClick={() => setView(key)}
                                            className={cn(
                                                'px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-all whitespace-nowrap',
                                                view === key
                                                    ? 'bg-primary text-primary-foreground shadow-sm'
                                                    : 'text-muted-foreground hover:text-foreground hover:bg-accent/10',
                                            )}
                                        >
                                            {label}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Content Area */}
                <div className="flex-1 overflow-y-auto p-8 custom-scrollbar bg-background/30">
                    {view === 'tradingview' && tvSelected ? (
                        <TradingViewChart symbol={benchmarkYahooSymbol(tvSelected) as string} height={450} />
                    ) : (
                    <div className="h-[450px] w-full relative">
                        {loading && (
                            <div className="absolute inset-0 flex items-center justify-center z-10">
                                <div className="flex flex-col items-center gap-4 bg-background/50 backdrop-blur-sm p-8 rounded-[2rem]">
                                    <div className="w-12 h-12 border-4 border-cyan-500/10 border-t-cyan-500 rounded-full animate-spin" />
                                    <span className="text-xs font-black tracking-[0.2em] text-cyan-500 uppercase animate-pulse">Syncing Markets</span>
                                </div>
                            </div>
                        )}

                        {!loading && data.length === 0 && (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="flex flex-col items-center gap-3 opacity-30">
                                    <Info className="w-12 h-12" />
                                    <span className="text-sm font-bold uppercase tracking-widest">No matching benchmark data</span>
                                </div>
                            </div>
                        )}

                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data} margin={{ top: 20, right: 10, left: -20, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border) / 0.3)" />
                                <XAxis
                                    dataKey="date"
                                    tickFormatter={(val) => {
                                        const d = new Date(val);
                                        if (period === '1d' || period === '5d') {
                                            return d.toLocaleTimeString([], { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit' });
                                        }
                                        return d.toLocaleDateString([], { calendar: 'gregory', timeZone: 'America/New_York', month: 'short', day: 'numeric' });
                                    }}
                                    tick={{ fontSize: 10, fontWeight: 700, fill: 'currentColor' }}
                                    axisLine={false}
                                    tickLine={false}
                                    minTickGap={30}
                                    interval="preserveStartEnd"
                                />
                                <YAxis
                                    tickFormatter={(val) => `${val}%`}
                                    tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))', fontWeight: 700 }}
                                    axisLine={false}
                                    tickLine={false}
                                    width={45}
                                />
                                <Tooltip
                                    wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                    contentStyle={{ backgroundColor: 'transparent', border: 'none' }}
                                    content={<CustomTooltip period={period} />}
                                />
                                {activeBenchmarks.map((bench, idx) => (
                                    <Line
                                        key={bench}
                                        type="monotone"
                                        dataKey={bench}
                                        name={bench}
                                        stroke={COLORS[idx % COLORS.length]}
                                        strokeWidth={2}
                                        dot={false}
                                        activeDot={{ r: 6, strokeWidth: 3, stroke: 'var(--background)', fill: COLORS[idx % COLORS.length] }}
                                        animationDuration={1500}
                                        animationEasing="ease-in-out"
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                    )}

                </div>

                {/* Standard Footer */}
                <div className="bg-card/50 px-8 py-4 flex justify-between items-center bg-secondary/10">
                    <div className="flex items-center gap-4">
                        <span className="text-[10px] text-muted-foreground uppercase tracking-[0.15em] font-black opacity-40">
                            {view === 'tradingview'
                                ? `Market Insight • ${tvSelected ?? ''}`
                                : `Market Insight • ${period.toUpperCase()} View`}
                        </span>
                    </div>
                    <div className="text-[10px] text-muted-foreground font-black uppercase tracking-[0.1em] opacity-40 italic">
                        {view === 'tradingview' ? 'Chart & Data by TradingView' : 'Real-time Data by Yahoo Finance'}
                    </div>
                </div>
            </div>
        </div >,
        document.body
    );
}
