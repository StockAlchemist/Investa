'use client';
import React, { useMemo } from 'react';
import { AssetChangeData } from '../../lib/api';
import { cn } from '../../lib/utils';

interface MonthlyHeatmapProps {
    data: AssetChangeData | null;
}

const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// Map a percent return to a Tailwind background class. Symmetric scale,
// saturating at ±10% so a few extreme months don't wash out the rest.
function bucketClass(v: number | null): string {
    if (v == null) return 'bg-muted/30';
    if (v >= 8)   return 'bg-emerald-500/90 text-white';
    if (v >= 5)   return 'bg-emerald-500/70 text-white';
    if (v >= 3)   return 'bg-emerald-500/50 text-emerald-950 dark:text-white';
    if (v >= 1)   return 'bg-emerald-500/30 text-emerald-900 dark:text-emerald-100';
    if (v >  0)   return 'bg-emerald-500/15 text-emerald-900 dark:text-emerald-200';
    if (v === 0)  return 'bg-muted/40 text-muted-foreground';
    if (v > -1)   return 'bg-red-500/15 text-red-900 dark:text-red-200';
    if (v > -3)   return 'bg-red-500/30 text-red-900 dark:text-red-100';
    if (v > -5)   return 'bg-red-500/50 text-red-950 dark:text-white';
    if (v > -8)   return 'bg-red-500/70 text-white';
    return 'bg-red-500/90 text-white';
}

export default function MonthlyHeatmap({ data }: MonthlyHeatmapProps) {
    const { grid, years, yearTotals } = useMemo(() => {
        const monthly = data?.M ?? [];
        // year -> [12] of returns (null if missing)
        const byYear = new Map<number, (number | null)[]>();
        for (const row of monthly) {
            const dateStr = row.Date as string;
            const ret = row['Portfolio M-Return'] as number | undefined;
            if (!dateStr || typeof ret !== 'number' || Number.isNaN(ret)) continue;
            const d = new Date(dateStr);
            const y = d.getFullYear();
            const m = d.getMonth();
            if (!byYear.has(y)) byYear.set(y, Array(12).fill(null));
            byYear.get(y)![m] = ret;
        }
        const yrs = Array.from(byYear.keys()).sort((a, b) => b - a); // newest first
        const totals = new Map<number, number>();
        for (const y of yrs) {
            const months = byYear.get(y)!;
            const compounded = months
                .filter((v): v is number => typeof v === 'number')
                .reduce((acc, v) => acc * (1 + v / 100), 1);
            totals.set(y, (compounded - 1) * 100);
        }
        return { grid: byYear, years: yrs, yearTotals: totals };
    }, [data]);

    if (years.length === 0) {
        return null;
    }

    return (
        <div className="metric-card p-5">
            <div className="flex items-center justify-between mb-4">
                <h3 className="section-label">Monthly Returns</h3>
                <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                    <div className="flex items-center gap-1">
                        <span className="w-2.5 h-2.5 rounded-sm bg-red-500/70" />
                        <span>−</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <span className="w-2.5 h-2.5 rounded-sm bg-muted/40" />
                        <span>0</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/70" />
                        <span>+</span>
                    </div>
                </div>
            </div>

            <div className="overflow-x-auto">
                <table className="w-full border-separate border-spacing-1 text-[11px] tabular-nums">
                    <thead>
                        <tr>
                            <th className="text-left text-[10px] font-semibold text-muted-foreground uppercase tracking-wider w-12">Year</th>
                            {MONTH_LABELS.map(m => (
                                <th key={m} className="text-center text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                                    {m}
                                </th>
                            ))}
                            <th className="text-right text-[10px] font-semibold text-muted-foreground uppercase tracking-wider w-16 pl-2">Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {years.map(y => {
                            const row = grid.get(y)!;
                            const total = yearTotals.get(y);
                            return (
                                <tr key={y}>
                                    <td className="text-left font-bold text-foreground pr-2">{y}</td>
                                    {row.map((v, i) => (
                                        <td
                                            key={i}
                                            className={cn(
                                                'text-center font-medium rounded h-7 min-w-[2.5rem]',
                                                bucketClass(v),
                                            )}
                                            title={v != null ? `${MONTH_LABELS[i]} ${y}: ${v.toFixed(2)}%` : `${MONTH_LABELS[i]} ${y}: —`}
                                        >
                                            {v != null ? v.toFixed(1) : ''}
                                        </td>
                                    ))}
                                    <td className={cn(
                                        'text-right font-bold pl-2',
                                        total == null ? 'text-muted-foreground' :
                                        total >= 0 ? 'text-emerald-600 dark:text-emerald-400'
                                                   : 'text-red-600 dark:text-red-400',
                                    )}>
                                        {total != null ? `${total > 0 ? '+' : ''}${total.toFixed(1)}%` : '—'}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
