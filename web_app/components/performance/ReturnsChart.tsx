'use client';
import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { AssetChangeData } from '../../lib/api';
import { formatCurrency, cn } from '../../lib/utils';

interface ReturnsChartProps {
    data: AssetChangeData | null;
    currency: string;
}

type PeriodKey = 'Y' | 'M' | 'W' | 'D';
type ViewMode = 'percent' | 'value';

const PERIODS: { key: PeriodKey; label: string; defaultCount: number }[] = [
    { key: 'Y', label: 'Annual',  defaultCount: 10 },
    { key: 'M', label: 'Monthly', defaultCount: 12 },
    { key: 'W', label: 'Weekly',  defaultCount: 12 },
    { key: 'D', label: 'Daily',   defaultCount: 30 },
];

const getBarColor = (name: string, index: number) => {
    const palette = ["#ef4444", "#0097b2", "#f59e0b", "#8b5cf6", "#e11d48", "#10b981"];
    const n = name.toLowerCase();
    if (n.includes('portfolio')) return "#ef4444";
    if (n.includes('s&p 500') || n.includes('500')) return "#0097b2";
    if (n.includes('dow')) return "#f59e0b";
    if (n.includes('nasdaq')) return "#8b5cf6";
    if (n.includes('russell')) return "#e11d48";
    return palette[index % palette.length];
};

export default function ReturnsChart({ data, currency }: ReturnsChartProps) {
    const [period, setPeriod] = useState<PeriodKey>('M');
    const [viewMode, setViewMode] = useState<ViewMode>('percent');
    const config = PERIODS.find(p => p.key === period)!;
    const [numPeriods, setNumPeriods] = useState<number>(config.defaultCount);

    // Reset numPeriods when switching period (keeps a sensible default per scale).
    const handlePeriodChange = (k: PeriodKey) => {
        setPeriod(k);
        const next = PERIODS.find(p => p.key === k)!;
        setNumPeriods(next.defaultCount);
    };

    const periodData = (data && data[period]) || [];
    const displayData = periodData.slice(-numPeriods);

    const dataKey = `${period}-Return`;
    const targetSuffix = viewMode === 'percent' ? dataKey : dataKey.replace('Return', 'Value');

    let keysToPlot: string[] = [];
    if (displayData.length > 0) {
        const sample = displayData[displayData.length - 1];
        Object.keys(sample).forEach(k => {
            if (k.endsWith(targetSuffix)) keysToPlot.push(k);
        });
    }
    if (viewMode === 'value') {
        keysToPlot = keysToPlot.filter(k => k.startsWith('Portfolio'));
    }
    keysToPlot.sort((a, b) => {
        if (a.startsWith('Portfolio')) return -1;
        if (b.startsWith('Portfolio')) return 1;
        return a.localeCompare(b);
    });

    const formatValue = (v: number) =>
        viewMode === 'percent' ? `${v.toFixed(2)}%` : formatCurrency(v, currency);

    return (
        <div className="metric-card p-5 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-violet-500 opacity-80" />

            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
                <div className="flex items-center gap-3">
                    <h3 className="section-label">Returns</h3>
                    <div className="inline-flex rounded-lg bg-secondary p-0.5">
                        {PERIODS.map(p => (
                            <button
                                key={p.key}
                                onClick={() => handlePeriodChange(p.key)}
                                className={cn(
                                    'px-2.5 py-1 rounded-md text-xs font-semibold transition-all',
                                    period === p.key
                                        ? 'bg-[#0097b2] text-white'
                                        : 'text-muted-foreground hover:text-foreground',
                                )}
                            >
                                {p.label}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2">
                        <label className="text-[11px] text-muted-foreground uppercase tracking-wider">Show</label>
                        <input
                            type="number"
                            min={1}
                            max={200}
                            value={numPeriods}
                            onChange={e => setNumPeriods(Math.max(1, parseInt(e.target.value) || 1))}
                            className="w-14 px-2 py-0.5 text-xs bg-secondary rounded text-foreground focus:outline-none focus:ring-1 focus:ring-violet-500 tabular-nums"
                        />
                    </div>
                    <div className="inline-flex rounded-lg bg-secondary p-0.5">
                        <button
                            onClick={() => setViewMode('percent')}
                            className={cn(
                                'px-2.5 py-1 rounded-md text-xs font-semibold transition-all',
                                viewMode === 'percent' ? 'bg-[#0097b2] text-white' : 'text-muted-foreground hover:text-foreground',
                            )}
                        >
                            %
                        </button>
                        <button
                            onClick={() => setViewMode('value')}
                            className={cn(
                                'px-2.5 py-1 rounded-md text-xs font-semibold transition-all',
                                viewMode === 'value' ? 'bg-[#0097b2] text-white' : 'text-muted-foreground hover:text-foreground',
                            )}
                        >
                            {currency}
                        </button>
                    </div>
                </div>
            </div>

            <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart key={`${viewMode}-${period}`} data={displayData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                        <XAxis
                            dataKey="Date"
                            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                            tickFormatter={(val) => {
                                if (!val || typeof val !== 'string') return '';
                                const datePart = val.split(' ')[0];
                                if (period === 'Y') return datePart.split('-')[0];
                                return datePart;
                            }}
                            axisLine={{ stroke: 'hsl(var(--border))' }}
                        />
                        <YAxis
                            tickFormatter={(val) =>
                                viewMode === 'percent'
                                    ? `${val.toFixed(0)}%`
                                    : new Intl.NumberFormat('en-US', { notation: 'compact', compactDisplay: 'short' }).format(val)
                            }
                            domain={['auto', 'auto']}
                            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
                            axisLine={{ stroke: 'hsl(var(--border))' }}
                            width={35}
                        />
                        {keysToPlot.map((key, index) => {
                            const name = key.replace(` ${targetSuffix}`, '');
                            return (
                                <Bar
                                    key={key}
                                    dataKey={key}
                                    name={name}
                                    fill={viewMode === 'percent' ? getBarColor(name, index) : undefined}
                                    radius={[4, 4, 0, 0]}
                                    // Near-zero months (e.g. +0.1%) would otherwise render as
                                    // invisible 1px bars and read as missing data.
                                    minPointSize={3}
                                >
                                    {viewMode === 'value' && displayData.map((entry: { [k: string]: unknown }, i: number) => (
                                        <Cell
                                            key={`cell-${i}`}
                                            fill={((entry[key] as number) || 0) >= 0 ? '#10b981' : '#ef4444'}
                                        />
                                    ))}
                                </Bar>
                            );
                        })}
                        <Tooltip
                            wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                            contentStyle={{ backgroundColor: 'transparent', border: 'none', boxShadow: 'none' }}
                            content={({ active, payload, label }) => {
                                if (!active || !payload || !payload.length) return null;
                                return (
                                    <div className="bg-background/98 backdrop-blur-2xl p-3 rounded-xl border border-border/60 shadow-2xl">
                                        <p className="font-medium text-foreground mb-1 text-sm">
                                            {typeof label === 'string' && !isNaN(Date.parse(label))
                                                ? new Date(label).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
                                                : label}
                                        </p>
                                        <div className="space-y-1">
                                            {payload.map((entry, idx) => (
                                                <div key={idx} className="flex items-center gap-2 text-xs">
                                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                                                    <span className="text-muted-foreground">{entry.name}:</span>
                                                    <span className={cn(
                                                        'font-medium',
                                                        Number(entry.value) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400',
                                                    )}>
                                                        {formatValue(Number(entry.value))}
                                                    </span>
                                                </div>
                                            ))}
                                            {viewMode === 'value' && payload[0]?.payload && (() => {
                                                const record = payload[0].payload as Record<string, number | undefined>;
                                                const netFlowKey = `Portfolio ${period}-NetFlow`;
                                                const netFlow = record[netFlowKey];
                                                if (netFlow !== undefined && Math.abs(netFlow) > 0.01) {
                                                    const portfolioEntry = payload.find(p => p.name === 'Portfolio');
                                                    const portfolioValue = portfolioEntry ? Number(portfolioEntry.value) : 0;
                                                    const totalChange = portfolioValue + netFlow;
                                                    return (
                                                        <>
                                                            <div className="flex items-center gap-2 text-xs mt-1 pt-2 border-t border-border/40">
                                                                <span className="w-2 h-2 rounded-full bg-cyan-500" />
                                                                <span className="text-muted-foreground">Net Flow:</span>
                                                                <span className={cn(
                                                                    'font-medium',
                                                                    netFlow >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400',
                                                                )}>
                                                                    {formatValue(netFlow)}
                                                                </span>
                                                            </div>
                                                            {!isNaN(totalChange) && (
                                                                <div className="flex items-center gap-2 text-xs">
                                                                    <span className="w-2 h-2 rounded-full bg-transparent" />
                                                                    <span className="text-muted-foreground">Total Change:</span>
                                                                    <span className={cn(
                                                                        'font-medium',
                                                                        totalChange >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400',
                                                                    )}>
                                                                        {formatValue(totalChange)}
                                                                    </span>
                                                                </div>
                                                            )}
                                                        </>
                                                    );
                                                }
                                                return null;
                                            })()}
                                        </div>
                                    </div>
                                );
                            }}
                            cursor={{ fill: 'var(--glass-hover)' }}
                        />
                        {viewMode === 'percent' && (
                            <Legend
                                verticalAlign="top"
                                wrapperStyle={{ fontSize: 10, paddingBottom: 10 }}
                                iconSize={8}
                                iconType="circle"
                            />
                        )}
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
