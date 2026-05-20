'use client';
import React, { useMemo, useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Sector } from 'recharts';
import { MousePointerClick } from 'lucide-react';
import { Holding } from '../../lib/api';
import { formatCompactNumber, cn } from '../../lib/utils';

export interface AggregatedSlice {
    name: string;
    value: number;
    /**
     * Names of source buckets aggregated into this slice — used by "Other" so the
     * drill-down can map back to the original holdings.
     */
    sourceBuckets?: string[];
    [key: string]: unknown;
}

export type PieBucketKey = 'quoteType' | 'Sector' | 'Industry' | 'Country';

interface AllocationPieChartProps {
    title: string;
    data: AggregatedSlice[];
    currency: string;
    holdings: Holding[];
    bucketKey: PieBucketKey;
}

const PALETTE = [
    '#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ef4444',
    '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#84cc16',
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ActiveSlice = (props: any) => (
    <Sector {...props} outerRadius={props.outerRadius + 7} />
);

// Mirror the unknown-detection used elsewhere so drill-down groups behave the
// same as the donut aggregation.
function isUnknown(v: unknown): boolean {
    if (v == null) return true;
    const s = String(v).trim().toUpperCase();
    return s === '' || s === '-' || s === 'NONE' || s === 'NULL' || s === 'UNKNOWN'
        || s.startsWith('N/A') || s.startsWith('UNKNOWN');
}

function getBucket(h: Holding, key: PieBucketKey): string {
    let raw: unknown;
    if (key === 'Country') {
        raw = (h['geography'] as string) || (h['Country'] as string);
    } else {
        raw = h[key];
    }
    return isUnknown(raw) ? 'Unknown' : (raw as string);
}

export default function AllocationPieChart({ title, data, currency, holdings, bucketKey }: AllocationPieChartProps) {
    const [activeIndex, setActiveIndex] = useState<number | null>(null);
    const [selectedBucket, setSelectedBucket] = useState<string | null>(null);

    const total = data.reduce((s, d) => s + d.value, 0);
    const active = activeIndex !== null ? data[activeIndex] : null;

    const handleSliceClick = (name: string) => {
        setSelectedBucket(prev => (prev === name ? null : name));
    };

    // Drill-down: list holdings that fall into the selected slice.
    const drillRows = useMemo(() => {
        if (!selectedBucket) return [];
        const mvKey = `Market Value (${currency})`;
        const selectedSlice = data.find(d => d.name === selectedBucket);

        // For "Other", expand to all source buckets that were folded into it.
        const isOther = selectedSlice?.sourceBuckets && selectedSlice.sourceBuckets.length > 0;
        const sourceBucketSet = isOther
            ? new Set(selectedSlice!.sourceBuckets)
            : new Set([selectedBucket]);

        const matched = (holdings ?? [])
            .map(h => {
                const value = Math.max(0, (h[mvKey] as number) || 0);
                if (value <= 0) return null;
                const bucket = getBucket(h, bucketKey);
                if (!sourceBucketSet.has(bucket)) return null;
                return {
                    symbol: h.Symbol as string,
                    name: (h['Name'] as string) ?? '',
                    bucket,
                    value,
                };
            })
            .filter((r): r is { symbol: string; name: string; bucket: string; value: number } => r != null);

        // Merge multiple lots of the same symbol so a stock held in two accounts
        // shows up once with combined weight.
        const bySymbol = new Map<string, typeof matched[number]>();
        for (const m of matched) {
            const prev = bySymbol.get(m.symbol);
            if (prev) {
                prev.value += m.value;
            } else {
                bySymbol.set(m.symbol, { ...m });
            }
        }

        const merged = Array.from(bySymbol.values()).sort((a, b) => b.value - a.value);
        const bucketTotal = merged.reduce((s, m) => s + m.value, 0);
        return merged.map(m => ({
            ...m,
            pctOfBucket: bucketTotal > 0 ? (m.value / bucketTotal) * 100 : 0,
            pctOfPortfolio: total > 0 ? (m.value / total) * 100 : 0,
        }));
    }, [selectedBucket, data, holdings, bucketKey, currency, total]);

    const visibleDrillRows = drillRows.slice(0, 12);

    return (
        <div className="metric-card p-5 flex flex-col gap-4">
            <div className="flex items-center justify-between shrink-0">
                <h3 className="section-label text-center flex-1">{title}</h3>
                <span className="flex items-center gap-1 text-[10px] text-muted-foreground/60 select-none">
                    <MousePointerClick className="w-2.5 h-2.5" />
                    <span className="hidden sm:inline">Click slice</span>
                </span>
            </div>

            <div className="relative w-full aspect-square">
                <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%"
                                cy="50%"
                                innerRadius="56%"
                                outerRadius="80%"
                                dataKey="value"
                                stroke="none"
                                paddingAngle={1.5}
                                {...({ activeIndex: activeIndex ?? undefined, activeShape: ActiveSlice } as object)}
                                onMouseEnter={(_, index) => setActiveIndex(index)}
                                onMouseLeave={() => setActiveIndex(null)}
                                onClick={(_, index) => {
                                    const slice = data[index];
                                    if (slice) handleSliceClick(slice.name);
                                }}
                            >
                                {data.map((entry, index) => {
                                    const isSelected = selectedBucket === entry.name;
                                    const isFadedByHover = activeIndex !== null && activeIndex !== index;
                                    const isFadedBySelection = selectedBucket !== null && !isSelected && activeIndex === null;
                                    const opacity = isSelected ? 1
                                        : isFadedByHover ? 0.35
                                        : isFadedBySelection ? 0.4
                                        : 1;
                                    return (
                                        <Cell
                                            key={index}
                                            fill={PALETTE[index % PALETTE.length]}
                                            opacity={opacity}
                                            stroke={isSelected ? PALETTE[index % PALETTE.length] : 'none'}
                                            strokeWidth={isSelected ? 2 : 0}
                                            className="transition-opacity duration-150 cursor-pointer"
                                        />
                                    );
                                })}
                            </Pie>
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* Center label */}
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none select-none">
                    {active ? (
                        <>
                            <p className="text-xs text-muted-foreground uppercase tracking-wide leading-none mb-1.5 max-w-[38%] truncate text-center">
                                {active.name}
                            </p>
                            <p className="text-xl font-bold text-foreground tabular-nums leading-none">
                                {formatCompactNumber(active.value, currency)}
                            </p>
                            <p className="text-sm text-muted-foreground tabular-nums leading-none mt-1.5">
                                {((active.value / total) * 100).toFixed(1)}%
                            </p>
                        </>
                    ) : (
                        <>
                            <p className="text-xs text-muted-foreground uppercase tracking-wide leading-none mb-1.5">
                                Total
                            </p>
                            <p className="text-xl font-bold text-foreground tabular-nums leading-none">
                                {formatCompactNumber(total, currency)}
                            </p>
                            <p className="text-sm text-muted-foreground leading-none mt-1.5">
                                {data.length} {data.length === 1 ? 'group' : 'groups'}
                            </p>
                        </>
                    )}
                </div>
            </div>

            {/* Legend */}
            <div className="space-y-0.5">
                {data.map((entry, index) => {
                    const pct = total > 0 ? (entry.value / total) * 100 : 0;
                    const isActive = activeIndex === index;
                    const isSelected = selectedBucket === entry.name;
                    return (
                        <button
                            key={entry.name}
                            type="button"
                            className={cn(
                                'w-full flex items-center gap-2 px-2 py-1.5 rounded-md cursor-pointer transition-colors text-xs text-left',
                                isSelected ? 'bg-primary/15 ring-1 ring-primary/30'
                                : isActive ? 'bg-muted'
                                : 'hover:bg-muted/50',
                            )}
                            onMouseEnter={() => setActiveIndex(index)}
                            onMouseLeave={() => setActiveIndex(null)}
                            onClick={() => handleSliceClick(entry.name)}
                        >
                            <span
                                className="w-2.5 h-2.5 rounded-sm shrink-0"
                                style={{ backgroundColor: PALETTE[index % PALETTE.length] }}
                            />
                            <span className={cn('flex-1 truncate', isSelected ? 'text-foreground font-semibold' : 'text-foreground')}>
                                {entry.name}
                            </span>
                            <span className="tabular-nums text-muted-foreground font-medium shrink-0">
                                {pct.toFixed(1)}%
                            </span>
                            <span className="tabular-nums text-muted-foreground/60 shrink-0 hidden sm:block min-w-[52px] text-right">
                                {formatCompactNumber(entry.value, currency)}
                            </span>
                        </button>
                    );
                })}
            </div>

            {/* Drill-down */}
            {selectedBucket && (
                <div className="border-t border-border/60 pt-3 mt-1 animate-in fade-in slide-in-from-top-1 duration-150">
                    <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                            <span className="text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold">
                                Holdings in
                            </span>
                            <span className="text-xs font-bold text-foreground truncate max-w-[180px]">
                                {selectedBucket}
                            </span>
                            <span className="text-[10px] text-muted-foreground tabular-nums">
                                · {drillRows.length} {drillRows.length === 1 ? 'stock' : 'stocks'}
                            </span>
                        </div>
                        <button
                            type="button"
                            onClick={() => setSelectedBucket(null)}
                            className="text-[10px] text-muted-foreground hover:text-foreground font-semibold uppercase tracking-wider"
                        >
                            Close
                        </button>
                    </div>

                    {drillRows.length === 0 ? (
                        <p className="text-xs text-muted-foreground/70 py-2">No matching holdings.</p>
                    ) : (
                        <div className="space-y-1">
                            {visibleDrillRows.map(row => (
                                <div key={`${row.symbol}-${row.bucket}`} className="grid grid-cols-[1fr_auto] gap-3 items-center text-xs">
                                    <div className="min-w-0">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="font-bold text-foreground truncate">{row.symbol}</span>
                                            {row.name && (
                                                <span className="text-muted-foreground/70 truncate text-[10px]">{row.name}</span>
                                            )}
                                        </div>
                                        <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
                                            <div
                                                className="absolute inset-y-0 left-0 bg-[#0097b2] rounded-full"
                                                style={{ width: `${Math.min(100, row.pctOfBucket)}%` }}
                                            />
                                        </div>
                                    </div>
                                    <div className="text-right shrink-0 tabular-nums">
                                        <div className="font-bold text-foreground">{row.pctOfBucket.toFixed(1)}%</div>
                                        <div className="text-[10px] text-muted-foreground/60">
                                            {row.pctOfPortfolio.toFixed(1)}% of total
                                        </div>
                                    </div>
                                </div>
                            ))}
                            {drillRows.length > visibleDrillRows.length && (
                                <p className="text-[10px] text-muted-foreground/60 text-center pt-2">
                                    + {drillRows.length - visibleDrillRows.length} more
                                </p>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

