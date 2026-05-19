'use client';
import React, { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Sector } from 'recharts';
import { Holding } from '../lib/api';
import { formatCompactNumber, cn } from '../lib/utils';
import AllocationDrift from './AllocationDrift';

interface AllocationProps {
    holdings: Holding[];
    currency: string;
}

const PALETTE = [
    '#6366f1', // indigo
    '#06b6d4', // cyan
    '#10b981', // emerald
    '#f59e0b', // amber
    '#ef4444', // red
    '#8b5cf6', // violet
    '#ec4899', // pink
    '#14b8a6', // teal
    '#f97316', // orange
    '#84cc16', // lime
];

interface AggregatedData {
    name: string;
    value: number;
    [key: string]: unknown;
}

interface AllocationPieChartProps {
    title: string;
    data: AggregatedData[];
    currency: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ActiveSlice = (props: any) => (
    <Sector {...props} outerRadius={props.outerRadius + 7} />
);

function AllocationPieChart({ title, data, currency }: AllocationPieChartProps) {
    const [activeIndex, setActiveIndex] = useState<number | null>(null);

    const total = data.reduce((s, d) => s + d.value, 0);
    const active = activeIndex !== null ? data[activeIndex] : null;

    return (
        <div className="metric-card p-5 flex flex-col gap-4">
            <h3 className="section-label text-center shrink-0">{title}</h3>

            {/* Full-width square container — ring fills ~80% of card width */}
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
                            >
                                {data.map((_, index) => (
                                    <Cell
                                        key={index}
                                        fill={PALETTE[index % PALETTE.length]}
                                        opacity={activeIndex === null || activeIndex === index ? 1 : 0.35}
                                        className="transition-opacity duration-150"
                                    />
                                ))}
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
                    return (
                        <div
                            key={entry.name}
                            className={cn(
                                'flex items-center gap-2 px-2 py-1.5 rounded-md cursor-default transition-colors text-xs',
                                isActive ? 'bg-muted' : 'hover:bg-muted/50',
                            )}
                            onMouseEnter={() => setActiveIndex(index)}
                            onMouseLeave={() => setActiveIndex(null)}
                        >
                            <span
                                className="w-2.5 h-2.5 rounded-sm shrink-0"
                                style={{ backgroundColor: PALETTE[index % PALETTE.length] }}
                            />
                            <span className="flex-1 text-foreground truncate">{entry.name}</span>
                            <span className="tabular-nums text-muted-foreground font-medium shrink-0">
                                {pct.toFixed(1)}%
                            </span>
                            <span className="tabular-nums text-muted-foreground/60 shrink-0 hidden sm:block min-w-[52px] text-right">
                                {formatCompactNumber(entry.value, currency)}
                            </span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

export default function Allocation({ holdings, currency }: AllocationProps) {
    if (!holdings || holdings.length === 0) {
        return <div className="p-4 text-center text-muted-foreground">No holdings data available.</div>;
    }

    const marketValueKey = `Market Value (${currency})`;

    const aggregateData = (key: keyof Holding | 'Sector' | 'Industry' | 'Country' | 'quoteType'): AggregatedData[] => {
        const aggregation: Record<string, number> = {};

        const isUnknown = (v: unknown) => {
            if (v == null) return true;
            const s = String(v).trim().toUpperCase();
            return s === ''
                || s === '-'
                || s === 'NONE'
                || s === 'NULL'
                || s === 'UNKNOWN'
                || s.startsWith('N/A')
                || s.startsWith('UNKNOWN');
        };

        holdings.forEach(h => {
            const value = (h[marketValueKey] as number) || 0;
            let raw: unknown;
            if (key === 'Country') {
                raw = (h['geography'] as string) || (h['Country'] as string);
            } else {
                raw = h[key];
            }
            const category = isUnknown(raw) ? 'Unknown' : (raw as string);
            aggregation[category] = (aggregation[category] || 0) + value;
        });

        const sorted = Object.entries(aggregation)
            .map(([name, value]) => ({ name, value }))
            .sort((a, b) => b.value - a.value);

        const totalVal = sorted.reduce((sum, item) => sum + item.value, 0);
        const top: AggregatedData[] = [];
        let otherVal = 0;

        sorted.forEach(item => {
            if (item.value / totalVal >= 0.02) {
                top.push(item);
            } else {
                otherVal += item.value;
            }
        });

        if (otherVal > 0) top.push({ name: 'Other', value: otherVal });

        return top;
    };

    const assetTypeData = aggregateData('quoteType');
    const sectorData    = aggregateData('Sector');
    const industryData  = aggregateData('Industry');
    const countryData   = aggregateData('Country');

    return (
        <div className="p-4 space-y-6">
            {/* Target vs actual drift */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <AllocationDrift
                    holdings={holdings}
                    currency={currency}
                    bucketKey="quoteType"
                    settingsBucket="quoteType"
                    title="Asset Type — drift vs target"
                    storageKey="allocation-target-quoteType"
                />
                <AllocationDrift
                    holdings={holdings}
                    currency={currency}
                    bucketKey="Sector"
                    settingsBucket="sector"
                    title="Sector — drift vs target"
                    storageKey="allocation-target-sector"
                />
            </div>

            {/* Donut charts */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <AllocationPieChart title="By Asset Type" data={assetTypeData} currency={currency} />
                <AllocationPieChart title="By Sector"     data={sectorData}    currency={currency} />
                <AllocationPieChart title="By Industry"   data={industryData}  currency={currency} />
                <AllocationPieChart title="By Country"    data={countryData}   currency={currency} />
            </div>
        </div>
    );
}
