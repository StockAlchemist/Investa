'use client';

import React from 'react';
import { ResponsiveContainer, AreaChart, Area, YAxis } from 'recharts';

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
}

export default function MarketIndicesBox({ indices, onClick }: MarketIndicesBoxProps) {
    const indexList = Object.values(indices);

    return (
        <div
            onClick={onClick}
            className="hidden lg:flex items-center gap-1 p-1 rounded-2xl bg-card/40 border border-border/50 hover:bg-accent/10 hover:border-cyan-500/30 transition-all duration-300 group cursor-pointer overflow-hidden relative"
        >
            {indexList.map((index, idx) => (
                <div
                    key={index.name}
                    className={`flex flex-col items-start px-3 py-1.5 min-w-[110px] ${idx < indexList.length - 1 ? 'border-r border-border/30' : ''}`}
                >
                    <div className="flex items-center justify-between w-full gap-2">
                        <span className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground group-hover:text-foreground transition-colors">
                            {index.name}
                        </span>
                        <span className={`text-[9px] font-bold tabular-nums ${(index.change || 0) >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                            {(index.changesPercentage || 0).toFixed(2)}%
                        </span>
                    </div>
                    <div className="text-xs font-bold text-foreground tabular-nums">
                        {index.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) ?? "0.00"}
                    </div>

                    {index.sparkline && index.sparkline.length > 1 && (
                        <div className="absolute bottom-0 left-0 right-0 h-[2px] pointer-events-none opacity-30 group-hover:opacity-60 transition-opacity">
                            {/* Sparklines are handled via the relative container or ignored for simplicity here since they might overlap. 
                   Actually, let's keep them scoped to the individual index section if possible, but the original UI had them at bottom.
               */}
                        </div>
                    )}
                </div>
            ))}

            {/* Universal hover effect indicator */}
            <div className="absolute top-0 right-0 p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-cyan-500"><path d="M15 3h6v6" /><path d="M10 14 21 3" /><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /></svg>
            </div>
        </div>
    );
}
