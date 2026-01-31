import React, { useState, useEffect } from 'react';
import { formatCurrency } from '../lib/utils';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { AssetChangeData } from '../lib/api';
import TabContentSkeleton from './skeletons/TabContentSkeleton';

interface AssetChangeProps {
    data: AssetChangeData | null;
    currency: string;
    isLoading?: boolean;
}

const PERIOD_CONFIGS = [
    { key: 'Y', title: 'Annual Returns', dataKey: 'Y-Return', defaultPeriods: 10 },
    { key: 'M', title: 'Monthly Returns', dataKey: 'M-Return', defaultPeriods: 12 },
    { key: 'W', title: 'Weekly Returns', dataKey: 'W-Return', defaultPeriods: 12 },
    { key: 'D', title: 'Daily Returns', dataKey: 'D-Return', defaultPeriods: 30 },
];

const COLORS = [
    "#ef4444", // Portfolio (Red)
    "#0097b2", // Investa Cyan (S&P 500)
    "#f59e0b", // Amber (Dow Jones)
    "#8b5cf6", // Violet (NASDAQ)
    "#e11d48", // Rose (Russell 2000)
    "#10b981", // Emerald (Fallback)
];

const getBarColor = (name: string, index: number) => {
    const normalized = name.toLowerCase();
    if (normalized.includes('portfolio')) return "#ef4444";
    if (normalized.includes('s&p 500') || normalized.includes('500')) return "#0097b2";
    if (normalized.includes('dow jones') || normalized.includes('dow')) return "#f59e0b";
    if (normalized.includes('nasdaq')) return "#8b5cf6";
    if (normalized.includes('russell 2000') || normalized.includes('2000')) return "#e11d48";
    return COLORS[index % COLORS.length];
};

interface AssetSectionProps {
    config: { key: string; title: string; dataKey: string; defaultPeriods: number };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data: any;
    currency: string;
    viewMode: 'percent' | 'value';
    formatValue: (val: number) => string;
}

const AssetSection = ({ config, data, currency, viewMode, formatValue }: AssetSectionProps) => {
    const periodData = (data && data[config.key]) || [];
    const [numPeriods, setNumPeriods] = useState(config.defaultPeriods);

    // Filter data based on numPeriods
    const displayData = periodData.slice(-numPeriods);

    // Determine target suffix based on viewMode
    const targetSuffix = viewMode === 'percent' ? config.dataKey : config.dataKey.replace('Return', 'Value');

    // Identify keys to plot
    let keysToPlot: string[] = [];
    if (displayData.length > 0) {
        const sampleRecord = displayData[displayData.length - 1];
        if (sampleRecord) {
            Object.keys(sampleRecord).forEach(k => {
                if (k.endsWith(targetSuffix)) {
                    keysToPlot.push(k);
                }
            });
        }
    }

    // In 'value' mode, only show Portfolio
    if (viewMode === 'value') {
        keysToPlot = keysToPlot.filter(key => key.startsWith('Portfolio'));
    }

    // Sort keys to put Portfolio first
    keysToPlot.sort((a, b) => {
        if (a.startsWith('Portfolio')) return -1;
        if (b.startsWith('Portfolio')) return 1;
        return a.localeCompare(b);
    });

    return (
        <div className="bg-card p-4 rounded-xl shadow-sm border border-border mb-6">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-foreground">{config.title} ({viewMode === 'percent' ? '%' : currency})</h3>
                <div className="flex items-center space-x-2">
                    <label className="text-sm text-muted-foreground">Periods:</label>
                    <input
                        type="number"
                        min="1"
                        max="100"
                        value={numPeriods}
                        onChange={(e) => setNumPeriods(parseInt(e.target.value) || 1)}
                        className="w-16 px-2 py-1 text-sm bg-secondary border border-border rounded text-foreground focus:outline-none focus:ring-1 focus:ring-cyan-500"
                    />
                </div>
            </div>

            {/* Chart */}
            <div className="h-64 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart key={`${viewMode}-${config.key}`} data={displayData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                        <XAxis
                            dataKey="Date"
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            tickFormatter={(val) => {
                                if (!val || typeof val !== 'string') return '';
                                const datePart = val.split(' ')[0];
                                if (config.key === 'Y') {
                                    return datePart.split('-')[0];
                                }
                                return datePart;
                            }}
                            axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                        />
                        <YAxis
                            tickFormatter={(val) => viewMode === 'percent' ? `${val.toFixed(0)}%` : new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(val)}
                            domain={['auto', 'auto']}
                            tick={{ fill: '#9ca3af', fontSize: 10 }}
                            axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                            width={35}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'var(--menu-solid)',
                                borderRadius: '8px',
                                border: '1px solid var(--border)',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                            }}
                            content={({ active, payload, label }) => {
                                if (active && payload && payload.length) {
                                    return (
                                        <div className="border border-border p-3 rounded-lg shadow-xl" style={{ backgroundColor: 'var(--menu-solid)' }}>
                                            <p className="font-medium text-foreground mb-1 text-sm">
                                                {typeof label === 'string' && !isNaN(Date.parse(label))
                                                    ? new Date(label).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
                                                    : label}
                                            </p>
                                            <div className="space-y-1">
                                                {payload.map((entry, index) => (
                                                    <div key={index} className="flex items-center gap-2 text-xs">
                                                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                                                        <span className="text-muted-foreground">{entry.name}:</span>
                                                        <span className={`font-medium ${Number(entry.value) >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                                                            {formatValue(Number(entry.value))}
                                                        </span>
                                                    </div>
                                                ))}
                                                {/* Show Net Flow and Total Change if available in Value mode */}
                                                {viewMode === 'value' && payload[0]?.payload && (() => {
                                                    const record = payload[0].payload;
                                                    const netFlowKey = `Portfolio ${config.key}-NetFlow`;
                                                    const netFlow = record[netFlowKey];

                                                    // Only show if netFlow is defined and non-zero
                                                    if (netFlow !== undefined && Math.abs(netFlow) > 0.01) {
                                                        const portfolioEntry = payload.find(p => p.name === 'Portfolio');
                                                        const portfolioValue = portfolioEntry ? Number(portfolioEntry.value) : 0;
                                                        const totalChange = portfolioValue + netFlow;
                                                        return (
                                                            <>
                                                                <div className="flex items-center gap-2 text-xs mt-1 pt-1 border-t border-border/50">
                                                                    <span className="w-2 h-2 rounded-full bg-cyan-500" />
                                                                    <span className="text-muted-foreground">Net Flow:</span>
                                                                    <span className={`font-medium ${Number(netFlow) >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                                                                        {formatValue(netFlow)}
                                                                    </span>
                                                                </div>
                                                                {!isNaN(totalChange) && (
                                                                    <div className="flex items-center gap-2 text-xs">
                                                                        <span className="w-2 h-2 rounded-full bg-transparent" />
                                                                        <span className="text-muted-foreground">Total Change:</span>
                                                                        <span className={`font-medium ${Number(totalChange) >= 0 ? "text-emerald-500" : "text-red-500"}`}>
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
                                }
                                return null;
                            }}
                            cursor={{ fill: 'var(--glass-hover)' }}
                        />
                        {viewMode === 'percent' && (
                            <Legend
                                wrapperStyle={{ fontSize: 10, paddingTop: 10 }}
                                iconSize={8}
                                iconType="circle"
                            />
                        )}
                        {keysToPlot.map((key, index) => {
                            const name = key.replace(` ${targetSuffix}`, '');
                            return (
                                <Bar
                                    key={key}
                                    dataKey={key}
                                    name={name}
                                    fill={viewMode === 'percent' ? getBarColor(name, index) : undefined}
                                    radius={[4, 4, 0, 0]}
                                >
                                    {viewMode === 'value' && displayData.map((entry: any, i: number) => (
                                        <Cell
                                            key={`cell-${i}`}
                                            fill={(entry[key] || 0) >= 0 ? '#10b981' : '#ef4444'}
                                        />
                                    ))}
                                </Bar>
                            );
                        })}
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default function AssetChange({ data, currency, isLoading }: AssetChangeProps) {
    const [viewMode, setViewMode] = useState<'percent' | 'value'>('percent');
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (isLoading) {
        return <TabContentSkeleton type="chart-only" />;
    }

    if (!data || Object.keys(data).length === 0) {
        return (
            <div className="p-12 text-center text-muted-foreground bg-card rounded-2xl border border-border border-dashed">
                <p className="font-medium text-sm">No asset change data available.</p>
                <p className="text-xs mt-1">Please ensure your portfolio history is populated.</p>
            </div>
        );
    }

    if (!mounted) {
        return <TabContentSkeleton type="chart-only" />;
    }

    const formatValue = (val: number) => {
        if (viewMode === 'percent') {
            return `${val.toFixed(2)}%`;
        }
        return formatCurrency(val, currency);
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-end space-x-2 mb-4">
                <span className="text-sm font-medium text-muted-foreground self-center">View:</span>
                <div className="inline-flex rounded-lg shadow-sm bg-secondary p-1 border border-border">
                    <button
                        onClick={() => setViewMode('percent')}
                        className={`whitespace-nowrap py-1.5 px-4 rounded text-sm font-medium transition-all ${viewMode === 'percent'
                            ? 'bg-[#0097b2] text-white shadow-sm'
                            : 'text-muted-foreground hover:bg-zinc-100 dark:hover:bg-zinc-800'
                            }`}
                    >
                        Percentage (%)
                    </button>
                    <button
                        onClick={() => setViewMode('value')}
                        className={`whitespace-nowrap py-1.5 px-4 rounded text-sm font-medium transition-all ${viewMode === 'value'
                            ? 'bg-[#0097b2] text-white shadow-sm'
                            : 'text-muted-foreground hover:bg-zinc-100 dark:hover:bg-zinc-800'
                            }`}
                    >
                        Value ({currency})
                    </button>
                </div>
            </div>
            {PERIOD_CONFIGS.map(config => (
                <AssetSection
                    key={config.key}
                    config={config}
                    data={data}
                    currency={currency}
                    viewMode={viewMode}
                    formatValue={formatValue}
                />
            ))}
        </div>
    );
}
