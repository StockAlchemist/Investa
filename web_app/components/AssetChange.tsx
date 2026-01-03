import React, { useState } from 'react';
import { formatCurrency } from '../lib/utils';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { AssetChangeData } from '../lib/api';

interface AssetChangeProps {
    data: AssetChangeData | null;
    currency: string;
}

const PERIOD_CONFIGS = [
    { key: 'Y', title: 'Annual Returns', dataKey: 'Y-Return', defaultPeriods: 10 },
    { key: 'M', title: 'Monthly Returns', dataKey: 'M-Return', defaultPeriods: 12 },
    { key: 'W', title: 'Weekly Returns', dataKey: 'W-Return', defaultPeriods: 12 },
    { key: 'D', title: 'Daily Returns', dataKey: 'D-Return', defaultPeriods: 30 },
];

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F'];

interface AssetSectionProps {
    config: { key: string; title: string; dataKey: string; defaultPeriods: number };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data: any;
    currency: string;
    viewMode: 'percent' | 'value';
    formatValue: (val: number) => string;
}

const AssetSection = ({ config, data, currency, viewMode, formatValue }: AssetSectionProps) => {
    const periodData = data[config.key] || [];
    const [numPeriods, setNumPeriods] = useState(config.defaultPeriods);

    // Filter data based on numPeriods
    const displayData = periodData.slice(-numPeriods);

    // Determine target suffix based on viewMode
    const targetSuffix = viewMode === 'percent' ? config.dataKey : config.dataKey.replace('Return', 'Value');

    // Identify keys to plot
    let keysToPlot: string[] = [];
    if (displayData.length > 0) {
        const sampleRecord = displayData[displayData.length - 1];
        Object.keys(sampleRecord).forEach(k => {
            if (k.endsWith(targetSuffix)) {
                keysToPlot.push(k);
            }
        });
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
        <div className="bg-card backdrop-blur-md p-4 rounded-xl shadow-sm border border-border mb-6">
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
                    <BarChart key={`${viewMode}-${config.key}`} data={displayData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                        <XAxis
                            dataKey="Date"
                            tick={{ fontSize: 10, fill: '#9ca3af' }}
                            tickFormatter={(val) => val} // Dates are already strings
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
                            content={({ active, payload, label }) => {
                                if (active && payload && payload.length) {
                                    return (
                                        <div className="bg-popover/95 backdrop-blur-sm border border-border p-3 rounded-lg shadow-xl">
                                            <p className="font-medium text-foreground mb-1">
                                                {typeof label === 'string' && !isNaN(Date.parse(label))
                                                    ? new Date(label).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
                                                    : label}
                                            </p>
                                            {payload.map((entry, index) => (
                                                <div key={index} className="flex items-center gap-2 text-sm">
                                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                                                    <span className="text-muted-foreground">{entry.name}:</span>
                                                    <span className="font-medium text-foreground">
                                                        {formatValue(entry.value)}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    );
                                }
                                return null;
                            }}
                            cursor={{ fill: 'var(--glass-hover)' }}
                        />
                        {viewMode === 'percent' && <Legend wrapperStyle={{ color: '#9ca3af' }} />}
                        {keysToPlot.map((key, index) => (
                            <Bar
                                key={key}
                                dataKey={key}
                                name={key.replace(` ${targetSuffix}`, '')}
                                fill={viewMode === 'percent' ? COLORS[index % COLORS.length] : undefined}
                                radius={[4, 4, 0, 0]}
                            >
                                {viewMode === 'value' && displayData.map((entry: Record<string, number>, i: number) => (
                                    <Cell
                                        key={`cell-${i}`}
                                        fill={entry[key] >= 0 ? '#10b981' : '#e11d48'}
                                    />
                                ))}
                            </Bar>
                        ))}
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default function AssetChange({ data, currency }: AssetChangeProps) {
    const [viewMode, setViewMode] = useState<'percent' | 'value'>('percent');

    if (!data) {
        return <div className="p-4 text-center text-muted-foreground">Loading asset change data...</div>;
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
