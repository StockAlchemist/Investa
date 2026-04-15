'use client';

import React from 'react';
import { ResponsiveContainer, AreaChart, Area, YAxis } from 'recharts';
import { Loader2, TrendingUp, TrendingDown } from 'lucide-react';

interface IndexData {
    name: string;
    price: number;
    change: number;
    changesPercentage: number;
    sparkline?: number[];
}

interface MarketIndicesBoxProps {
    indices: Record<string, IndexData>;
    onClick: () => void;
    isFetching?: boolean;
}

const INDEX_COLORS: Record<string, string> = {
    'S&P 500': '#06b6d4',
    'NASDAQ': '#8b5cf6',
    'Dow Jones': '#f59e0b',
    'Russell 2000': '#10b981',
};

function getColor(name: string): string {
    for (const key of Object.keys(INDEX_COLORS)) {
        if (name.toLowerCase().includes(key.toLowerCase())) return INDEX_COLORS[key];
    }
    return '#94a3b8';
}

export default function MarketIndicesBox({ indices, onClick, isFetching = false }: MarketIndicesBoxProps) {
    const indexList = Object.values(indices);

    return (
        <div
            onClick={onClick}
            className="hidden md:flex items-stretch gap-0 rounded-2xl bg-muted/20 dark:bg-white/[0.04] border border-border/60 dark:border-white/[0.06] hover:border-border hover:bg-muted/30 dark:hover:bg-white/[0.07] transition-all duration-300 group cursor-pointer overflow-hidden relative shadow-sm"
        >
            {isFetching && (
                <div className="absolute top-1.5 right-1.5 z-20">
                    <Loader2 className="w-2.5 h-2.5 animate-spin text-muted-foreground/50" />
                </div>
            )}
            {indexList.map((index, idx) => {
                const color = getColor(index.name);
                const isUp = (index.change || 0) >= 0;
                const chartData = (index.sparkline || []).map(v => ({ value: v }));
                const gradientId = `mibGrad-${index.name.replace(/[^a-zA-Z]/g, '')}`;

                return (
                    <div
                        key={index.name}
                        className={`flex flex-col px-3 py-2 min-w-[96px] relative overflow-hidden ${idx > 0 ? 'border-l border-border/40 dark:border-white/[0.05]' : ''}`}
                    >
                        {/* Sparkline background */}
                        {chartData.length > 1 && (
                            <div className="absolute inset-0 opacity-35 group-hover:opacity-55 transition-opacity duration-300 pointer-events-none">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={chartData} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                                        <defs>
                                            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={color} stopOpacity={0.5} />
                                                <stop offset="95%" stopColor={color} stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <YAxis hide domain={['dataMin', 'dataMax']} />
                                        <Area
                                            type="monotone"
                                            dataKey="value"
                                            stroke={color}
                                            fill={`url(#${gradientId})`}
                                            strokeWidth={2}
                                            dot={false}
                                            isAnimationActive={false}
                                        />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        )}

                        {/* Content */}
                        <div className="relative z-10 flex items-center justify-between gap-1 mb-0.5">
                            <span className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground truncate max-w-[60px]">
                                {index.name.replace('Dow Jones', 'Dow').replace('Russell 2000', 'RUT').replace('S&P 500', 'S&P')}
                            </span>
                            <span className={`text-[9px] font-bold tabular-nums flex items-center gap-0.5 ${isUp ? 'text-emerald-500' : 'text-rose-500'}`}>
                                {isUp
                                    ? <TrendingUp className="w-2.5 h-2.5 shrink-0" />
                                    : <TrendingDown className="w-2.5 h-2.5 shrink-0" />
                                }
                                {Math.abs(index.changesPercentage || 0).toFixed(2)}%
                            </span>
                        </div>
                        <div className="relative z-10 text-xs font-bold text-foreground tabular-nums leading-none">
                            {index.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) ?? '0.00'}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
