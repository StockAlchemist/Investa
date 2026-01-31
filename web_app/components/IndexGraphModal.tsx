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
    ResponsiveContainer,
    Legend
} from 'recharts';
import { X, TrendingUp, Info } from 'lucide-react';
import PeriodSelector from './PeriodSelector';
import { fetchMarketHistory } from '../lib/api';
import { Badge } from './ui/badge';
import StockIcon from './StockIcon';
import { cn } from '@/lib/utils';

interface IndexGraphModalProps {
    isOpen: boolean;
    onClose: () => void;
    benchmarks: string[];
}

const COLORS = [
    "#0097b2", // Investa Cyan
    "#f59e0b", // Amber
    "#8b5cf6", // Violet
    "#e11d48", // Rose
    "#10b981", // Emerald
];

const CustomTooltip = ({ active, payload, label, period }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-background/95 border border-border p-4 rounded-2xl shadow-2xl backdrop-blur-md min-w-[180px]">
                <p className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em] mb-3 border-b border-border/50 pb-2">
                    {new Date(label).toLocaleString([], {
                        timeZone: 'America/New_York',
                        month: 'short',
                        day: 'numeric',
                        year: period === '1d' ? undefined : 'numeric',
                        hour: period === '1d' || period === '5d' ? '2-digit' : undefined,
                        minute: period === '1d' || period === '5d' ? '2-digit' : undefined
                    })}
                </p>
                <div className="space-y-2.5">
                    {payload.map((entry: any, index: number) => (
                        <div key={index} className="flex items-center justify-between gap-6">
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                                <span className="text-xs font-bold text-foreground/90">{entry.name}</span>
                            </div>
                            <span className={`text-xs font-black tabular-nums ${entry.value >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                {entry.value >= 0 ? '+' : ''}{entry.value.toFixed(2)}%
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        );
    }
    return null;
};

export default function IndexGraphModal({ isOpen, onClose, benchmarks }: IndexGraphModalProps) {
    const [period, setPeriod] = useState('1y');
    const [data, setData] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    useEffect(() => {
        if (!isOpen) return;

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
    }, [isOpen, benchmarks, period]);

    const activeBenchmarks = useMemo(() => {
        if (data.length === 0) return [];
        return Object.keys(data[0]).filter(k => k !== 'date');
    }, [data]);

    const currentReturns = useMemo(() => {
        if (data.length === 0) return {};
        const latest = data[data.length - 1];
        return latest;
    }, [data]);

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
                className="relative w-full max-w-5xl h-[90vh] sm:h-auto sm:max-h-[85vh] rounded-[2.5rem] flex flex-col shadow-2xl overflow-hidden animate-in zoom-in-95 slide-in-from-bottom-10 duration-300 border border-border/50"
            >
                {/* Header Section */}
                <div className="sticky top-0 z-50 bg-card border-b border-border flex-shrink-0 shadow-sm">
                    <div className="p-8 pb-6 flex justify-between items-start">
                        <div className="flex items-center gap-6 flex-1 text-foreground">
                            {/* Icon Stack/Placeholder */}
                            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#0097b2] to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 flex-shrink-0 p-3 overflow-hidden">
                                <TrendingUp className="w-full h-full text-white" />
                            </div>

                            <div className="flex-1 min-w-0 pr-4">
                                <div className="flex items-center justify-between gap-4 mb-1">
                                    <div className="flex items-center gap-3 truncate">
                                        <h2 className="text-3xl font-black tracking-tighter text-foreground">Market Comparison</h2>
                                        <Badge variant="secondary" className="font-black text-[10px] uppercase tracking-widest px-3">Benchmarks</Badge>
                                    </div>
                                    <div className="flex items-center gap-6">
                                        {activeBenchmarks.map((bench, idx) => {
                                            const val = currentReturns[bench];
                                            if (val === undefined) return null;
                                            return (
                                                <div key={bench} className="flex flex-col items-end">
                                                    <span className="text-[9px] font-black uppercase tracking-widest text-muted-foreground/60">{bench}</span>
                                                    <span className={cn(
                                                        "text-lg font-bold tracking-tighter tabular-nums",
                                                        val >= 0 ? "text-emerald-500" : "text-rose-500"
                                                    )}>
                                                        {val >= 0 ? '+' : ''}{val.toFixed(2)}%
                                                    </span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 text-sm text-muted-foreground font-medium italic">
                                    <span className="text-cyan-500 font-bold not-italic">Indices Performance</span>
                                    <span>•</span>
                                    <span>Relative return percentage from start of period</span>
                                </div>
                            </div>
                        </div>

                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-all duration-200 text-muted-foreground hover:text-foreground group"
                        >
                            <X className="w-7 h-7 group-hover:rotate-90 transition-transform duration-300" />
                        </button>
                    </div>

                    {/* Range Selector Integration */}
                    <div className="px-8 pb-6 flex items-center justify-between">
                        <PeriodSelector selectedPeriod={period} onPeriodChange={setPeriod} />
                        <div className="hidden sm:flex items-center gap-4">
                            {activeBenchmarks.map((bench, idx) => (
                                <div key={bench} className="flex items-center gap-2">
                                    <div className="w-2.5 h-2.5 rounded-full shadow-sm" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                                    <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">{bench}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Content Area */}
                <div className="flex-1 overflow-y-auto p-8 custom-scrollbar bg-background/30">
                    <div className="h-[450px] w-full relative">
                        {loading && (
                            <div className="absolute inset-0 flex items-center justify-center z-10">
                                <div className="flex flex-col items-center gap-4 bg-background/50 backdrop-blur-sm p-8 rounded-[2rem] border border-border/50">
                                    <div className="w-12 h-12 border-4 border-cyan-500/10 border-t-cyan-500 rounded-full animate-spin shadow-lg shadow-cyan-500/20" />
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
                                        return d.toLocaleDateString([], { timeZone: 'America/New_York', month: 'short', day: 'numeric' });
                                    }}
                                    tick={{ fontSize: 10, fontWeight: 700, fill: 'currentColor' }}
                                    axisLine={false}
                                    tickLine={false}
                                    minTickGap={30}
                                    interval="preserveStartEnd"
                                />
                                <YAxis
                                    tickFormatter={(val) => `${val > 0 ? '+' : ''}${val}%`}
                                    tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))', fontWeight: 700 }}
                                    axisLine={false}
                                    tickLine={false}
                                    width={45}
                                />
                                <Tooltip content={<CustomTooltip period={period} />} />
                                {activeBenchmarks.map((bench, idx) => (
                                    <Line
                                        key={bench}
                                        type="monotone"
                                        dataKey={bench}
                                        name={bench}
                                        stroke={COLORS[idx % COLORS.length]}
                                        strokeWidth={3}
                                        dot={false}
                                        activeDot={{ r: 6, strokeWidth: 4, stroke: 'var(--background)', fill: COLORS[idx % COLORS.length] }}
                                        animationDuration={1500}
                                        animationEasing="ease-in-out"
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Optional Informational Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12 pb-8">
                        {activeBenchmarks.map((bench, idx) => {
                            const val = currentReturns[bench];
                            return (
                                <div key={bench} className="bg-card/40 border border-border/50 p-6 rounded-[2rem] hover:bg-card/60 transition-all duration-300 group">
                                    <div className="flex items-center gap-3 mb-4">
                                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                                        <span className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground group-hover:text-foreground transition-colors">{bench}</span>
                                    </div>
                                    <div className={cn(
                                        "text-2xl font-black tabular-nums",
                                        val >= 0 ? "text-emerald-500" : "text-rose-500"
                                    )}>
                                        {val !== undefined ? `${val >= 0 ? '+' : ''}${val.toFixed(2)}%` : '--'}
                                    </div>
                                    <p className="text-[10px] text-muted-foreground mt-2 font-medium">Period performance return</p>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Standard Footer */}
                <div className="bg-card/50 px-8 py-4 border-t border-border flex justify-between items-center bg-secondary/10">
                    <div className="flex items-center gap-4">
                        <span className="text-[10px] text-muted-foreground uppercase tracking-[0.15em] font-black opacity-40">
                            Market Insight • {period.toUpperCase()} View
                        </span>
                    </div>
                    <div className="text-[10px] text-muted-foreground font-black uppercase tracking-[0.1em] opacity-40 italic">
                        Real-time Data by Yahoo Finance
                    </div>
                </div>
            </div>
        </div>,
        document.body
    );
}
