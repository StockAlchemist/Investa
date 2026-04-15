'use client';

import React from 'react';
import { AreaChart, Area, YAxis, ResponsiveContainer } from 'recharts';

interface MarketIndex {
    name: string;
    price: number;
    change: number;
    changesPercentage: number;
    sparkline?: number[];
}

interface MarketsTabProps {
    indices: Record<string, MarketIndex>;
    onIndexClick: () => void;
}

export default function MarketsTab({ indices, onIndexClick }: MarketsTabProps) {
    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold tracking-tight text-foreground">Market Indices</h2>
                <div className="text-xs text-muted-foreground">7D Trend</div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.values(indices).map((index: MarketIndex) => {
                    const normalized = index.name.toLowerCase();
                    let accentColor = "bg-emerald-500";
                    if (normalized.includes('s&p 500') || normalized.includes('500')) accentColor = "bg-cyan-500";
                    if (normalized.includes('nasdaq')) accentColor = "bg-violet-500";
                    if (normalized.includes('dow jones') || normalized.includes('dow')) accentColor = "bg-amber-500";

                    return (
                        <div
                            key={index.name}
                            onClick={onIndexClick}
                            className="metric-card card-shine p-5 relative overflow-hidden group cursor-pointer"
                        >
                            {/* Standardized accent bar for Markets tab */}
                            <div className="absolute top-0 left-0 right-0 h-[2px] bg-cyan-500 opacity-80" />

                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <span className="text-muted-foreground text-[10px] font-bold uppercase tracking-widest">{index.name}</span>
                                    <h3 className="text-2xl font-bold text-foreground mt-1 tabular-nums">
                                        {index.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                    </h3>
                                </div>
                                <div className={`flex flex-col items-end font-bold tabular-nums ${index.change >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                                    <div className="flex items-center text-lg">
                                        {index.change >= 0 ? "+" : ""}{index.change.toFixed(2)}
                                    </div>
                                    <div className="text-sm">
                                        {index.changesPercentage.toFixed(2)}%
                                    </div>
                                </div>
                            </div>

                            {index.sparkline && index.sparkline.length > 1 && (
                                <div className="h-16 w-full mt-2 filter drop-shadow-sm opacity-90 group-hover:opacity-100 transition-opacity">
                                    <ResponsiveContainer width="100%" height="100%">
                                        {(() => {
                                            const normalized_inner = index.name.toLowerCase();
                                            let color = "#10b981"; // Default Emerald
                                            if (normalized_inner.includes('s&p 500') || normalized_inner.includes('500')) color = "#0097b2"; // Cyan
                                            if (normalized_inner.includes('nasdaq')) color = "#8b5cf6"; // Violet
                                            if (normalized_inner.includes('dow jones') || normalized_inner.includes('dow')) color = "#f59e0b"; // Amber

                                            const gradientId = `splitFill-${index.name.replace(/[^a-zA-Z]/g, '')}`;
                                            return (
                                                <AreaChart data={index.sparkline!.map((v: number) => ({ value: v }))}>
                                                    <defs>
                                                        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                                                            <stop offset="5%" stopColor={color} stopOpacity={0.2} />
                                                            <stop offset="95%" stopColor={color} stopOpacity={0} />
                                                        </linearGradient>
                                                    </defs>
                                                    <YAxis hide domain={['dataMin', 'dataMax']} />
                                                    <Area
                                                        type="monotone"
                                                        dataKey="value"
                                                        stroke={color}
                                                        fill={`url(#${gradientId})`}
                                                        strokeWidth={3}
                                                        dot={false}
                                                        isAnimationActive={false}
                                                    />
                                                </AreaChart>
                                            );
                                        })()}
                                    </ResponsiveContainer>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
