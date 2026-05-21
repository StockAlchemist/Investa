'use client';
import React, { useMemo } from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from 'recharts';
import { TrendingDown } from 'lucide-react';
import { PerformanceData } from '../../lib/api';
import { cn } from '../../lib/utils';

interface DrawdownTimelineProps {
    history: PerformanceData[] | null;
    isLoading?: boolean;
}

export default function DrawdownTimeline({ history, isLoading }: DrawdownTimelineProps) {
    const { series, maxDD, maxDDDate, longestDays } = useMemo(() => {
        const rows = (history ?? [])
            .filter(d => typeof d.drawdown === 'number')
            .map(d => ({ date: d.date, drawdown: d.drawdown as number }));

        // Normalize sign: drawdown is reported as a negative % (0 at peaks).
        // Guard against a backend that emits positive magnitudes.
        const norm = rows.map(r => ({ date: r.date, drawdown: r.drawdown > 0 ? -r.drawdown : r.drawdown }));

        let worst = 0;
        let worstDate = '';
        for (const r of norm) {
            if (r.drawdown < worst) { worst = r.drawdown; worstDate = r.date; }
        }

        // Longest underwater stretch (consecutive days with drawdown < 0).
        let longest = 0;
        let run = 0;
        for (const r of norm) {
            if (r.drawdown < -0.01) { run += 1; longest = Math.max(longest, run); }
            else run = 0;
        }

        return { series: norm, maxDD: worst, maxDDDate: worstDate, longestDays: longest };
    }, [history]);

    return (
        <div className="metric-card p-5 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-red-500 opacity-80" />
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <TrendingDown className="w-3.5 h-3.5 text-red-500" />
                    <h3 className="section-label">Drawdown (1Y)</h3>
                </div>
                {series.length > 0 && (
                    <div className="flex items-center gap-4 text-right">
                        <div>
                            <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">Max</div>
                            <div className="text-sm font-bold tabular-nums text-red-600 dark:text-red-400">
                                {maxDD.toFixed(2)}%
                            </div>
                        </div>
                        <div>
                            <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">Longest</div>
                            <div className="text-sm font-bold tabular-nums text-foreground">{longestDays}d</div>
                        </div>
                    </div>
                )}
            </div>

            {isLoading ? (
                <div className="h-56 animate-pulse bg-muted/30 rounded-lg" />
            ) : series.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-12">No drawdown history available.</p>
            ) : (
                <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={series} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                            <defs>
                                <linearGradient id="dd-grad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#ef4444" stopOpacity={0.05} />
                                    <stop offset="100%" stopColor="#ef4444" stopOpacity={0.4} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                            <XAxis
                                dataKey="date"
                                tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                                axisLine={{ stroke: 'hsl(var(--border))' }}
                                tickFormatter={(v) => (typeof v === 'string' ? v.slice(5) : v)}
                                minTickGap={40}
                            />
                            <YAxis
                                tickFormatter={(v) => `${v.toFixed(0)}%`}
                                tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                                axisLine={{ stroke: 'hsl(var(--border))' }}
                                width={40}
                                domain={['dataMin', 0]}
                            />
                            <ReferenceLine y={0} stroke="hsl(var(--border))" />
                            <Area
                                type="monotone"
                                dataKey="drawdown"
                                stroke="#ef4444"
                                strokeWidth={1.5}
                                fill="url(#dd-grad)"
                                isAnimationActive={false}
                                dot={false}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: 'transparent', border: 'none', boxShadow: 'none' }}
                                content={({ active, payload, label }) => {
                                    if (!active || !payload || !payload.length) return null;
                                    return (
                                        <div className="bg-background/98 backdrop-blur-2xl p-3 rounded-xl border border-border/60 shadow-2xl">
                                            <p className="font-medium text-foreground mb-1 text-sm">{label}</p>
                                            <div className="flex items-center gap-2 text-xs">
                                                <span className="w-2 h-2 rounded-full bg-red-500" />
                                                <span className="text-muted-foreground">Drawdown:</span>
                                                <span className="font-medium text-red-600 dark:text-red-400 tabular-nums">
                                                    {Number(payload[0].value).toFixed(2)}%
                                                </span>
                                            </div>
                                        </div>
                                    );
                                }}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            )}
            {maxDDDate && (
                <p className={cn('text-[10px] text-muted-foreground/60 mt-2 tabular-nums')}>
                    Deepest trough on {maxDDDate}
                </p>
            )}
        </div>
    );
}
