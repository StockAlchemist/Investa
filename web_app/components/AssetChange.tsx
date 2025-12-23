import React, { useState, useEffect } from 'react';
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

const AssetSection = ({ config, data, currency, viewMode, formatValue }: any) => {
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
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700 mb-6">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{config.title} ({viewMode === 'percent' ? '%' : currency})</h3>
                <div className="flex items-center space-x-2">
                    <label className="text-sm text-gray-500 dark:text-gray-400">Periods:</label>
                    <input
                        type="number"
                        min="1"
                        max="100"
                        value={numPeriods}
                        onChange={(e) => setNumPeriods(parseInt(e.target.value) || 1)}
                        className="w-16 px-2 py-1 text-sm border border-gray-300 rounded dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    />
                </div>
            </div>

            {/* Chart */}
            <div className="h-64 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart key={`${viewMode}-${config.key}`} data={displayData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                        <XAxis
                            dataKey="Date"
                            tick={{ fontSize: 10 }}
                            tickFormatter={(val) => val} // Dates are already strings
                        />
                        <YAxis
                            tickFormatter={(val) => viewMode === 'percent' ? `${val.toFixed(1)}%` : new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(val)}
                            domain={['auto', 'auto']}
                        />
                        <Tooltip
                            formatter={(value: number) => [formatValue(value), '']}
                            labelStyle={{ color: '#374151' }}
                        />
                        {viewMode === 'percent' && <Legend />}
                        {keysToPlot.map((key, index) => (
                            <Bar
                                key={key}
                                dataKey={key}
                                name={key.replace(` ${targetSuffix}`, '')}
                                fill={viewMode === 'percent' ? COLORS[index % COLORS.length] : undefined}
                            >
                                {viewMode === 'value' && displayData.map((entry: any, i: number) => (
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
        return <div className="p-4 text-center text-gray-500">Loading asset change data...</div>;
    }

    const formatValue = (val: number) => {
        if (viewMode === 'percent') {
            return `${val.toFixed(2)}%`;
        }
        return formatCurrency(val, currency);
    };

    // Custom Tooltip component
    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className="p-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-md">
                    <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">{label}</p>
                    {payload.map((entry: any, index: number) => (
                        <p key={`item-${index}`} className="text-sm font-medium" style={{ color: entry.color }}>
                            {entry.name}: {viewMode === 'percent'
                                ? `${entry.value > 0 ? '+' : ''}${entry.value.toFixed(2)}%`
                                : formatCurrency(entry.value, currency)
                            }
                        </p>
                    ))}
                </div>
            );
        }
        return null;
    };


    return (
        <div className="space-y-6">
            <div className="flex justify-end space-x-2 mb-4">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300 self-center">View:</span>
                <div className="inline-flex rounded-md shadow-sm" role="group">
                    <button
                        type="button"
                        onClick={() => setViewMode('percent')}
                        className={`px-4 py-2 text-sm font-medium border rounded-l-lg ${viewMode === 'percent'
                            ? 'bg-blue-600 text-white border-blue-600'
                            : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-100 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:hover:bg-gray-600'
                            }`}
                    >
                        Percentage (%)
                    </button>
                    <button
                        type="button"
                        onClick={() => setViewMode('value')}
                        className={`px-4 py-2 text-sm font-medium border rounded-r-lg ${viewMode === 'value'
                            ? 'bg-blue-600 text-white border-blue-600'
                            : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-100 dark:bg-gray-700 dark:border-gray-600 dark:text-white dark:hover:bg-gray-600'
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
