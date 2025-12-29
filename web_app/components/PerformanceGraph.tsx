import React, { useState, useEffect } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area,
    Legend
} from 'recharts';
import PeriodSelector from './PeriodSelector';
import BenchmarkSelector from './BenchmarkSelector';
import { fetchHistory, PerformanceData } from '../lib/api';
import { formatCurrency } from '../lib/utils';

interface PerformanceGraphProps {
    currency: string;
    accounts?: string[];
    benchmarks: string[];
    onBenchmarksChange: (benchmarks: string[]) => void;
}

const COLORS = [
    "#2563eb", // Portfolio (Blue)
    "#dc2626", // Red
    "#16a34a", // Green
    "#d97706", // Amber
    "#9333ea", // Purple
    "#0891b2", // Cyan
    "#db2777", // Pink
];

export default function PerformanceGraph({ currency, accounts, benchmarks, onBenchmarksChange }: PerformanceGraphProps) {
    const [view, setView] = useState<'return' | 'value' | 'drawdown'>('return');
    const [period, setPeriod] = useState('1y');
    const [data, setData] = useState<PerformanceData[]>([]);
    const [loading, setLoading] = useState(false);

    // Fetch data when period or benchmarks change
    useEffect(() => {
        const loadData = async () => {
            setLoading(true);
            try {
                const newData = await fetchHistory(currency, accounts, period, benchmarks);
                setData(newData);
            } catch (error) {
                console.error("Failed to fetch history:", error);
            } finally {
                setLoading(false);
            }
        };

        // Skip initial load if using initialData and params haven't changed (simplified check)
        // Actually, initialData is likely 1y with no benchmarks. 
        // If period is 1y and benchmarks empty, we could skip, but safer to just reload or check.
        // For now, let's just reload to be safe and consistent.
        loadData();
    }, [period, benchmarks, currency, accounts]);

    if (!data || data.length === 0) {
        return (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-100 dark:border-gray-700 h-64 flex items-center justify-center text-gray-500">
                {loading ? 'Loading...' : 'No historical data available.'}
            </div>
        );
    }

    const formatXAxis = (tickItem: string) => {
        const date = new Date(tickItem);
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    };

    const formatYAxis = (tickItem: number) => {
        if (view === 'return' || view === 'drawdown') {
            return `${tickItem.toFixed(1)}%`;
        } else {
            return new Intl.NumberFormat('en-US', {
                notation: "compact",
                maximumFractionDigits: 1
            }).format(tickItem);
        }
    };

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-popover/95 backdrop-blur-sm p-3 border border-border shadow-xl rounded-lg">
                    <p className="text-sm text-foreground mb-1">{new Date(label).toLocaleDateString()}</p>
                    {payload.map((entry: any, index: number) => (
                        <p key={index} className="text-sm font-medium" style={{ color: entry.color }}>
                            {entry.name}: {view === 'return' || view === 'drawdown'
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

    // Determine which keys to plot
    // Always plot 'twr' or 'value' for portfolio
    // For benchmarks, they are usually only relevant for 'return' view
    // Benchmarks keys in data will be their tickers/names.
    // We need to know which keys correspond to benchmarks.
    // A simple way is to look at keys in the first data point that are not 'date', 'value', 'twr'.
    const benchmarkKeys = data.length > 0
        ? Object.keys(data[0]).filter(k => k !== 'date' && k !== 'value' && k !== 'twr' && k !== 'drawdown')
        : [];

    return (
        <div className="bg-black/5 dark:bg-white/5 backdrop-blur-md rounded-xl p-4 shadow-sm border border-black/10 dark:border-white/10 mb-6">
            <div className="flex flex-col space-y-4 mb-4">
                <div className="flex justify-between items-center">
                    <h3 className="text-lg font-semibold text-foreground">
                        {view === 'return' ? 'Time-Weighted Return' : view === 'value' ? 'Portfolio Value' : 'Drawdown'}
                    </h3>
                    <div className="flex space-x-2 bg-black/5 dark:bg-white/5 rounded-lg p-1 border border-black/5 dark:border-white/5">
                        <button
                            onClick={() => setView('return')}
                            className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${view === 'return'
                                ? 'bg-black/10 dark:bg-white/10 text-foreground shadow-sm'
                                : 'text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                }`}
                        >
                            Return %
                        </button>
                        <button
                            onClick={() => setView('value')}
                            className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${view === 'value'
                                ? 'bg-black/10 dark:bg-white/10 text-foreground shadow-sm'
                                : 'text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                }`}
                        >
                            Value
                        </button>
                        <button
                            onClick={() => setView('drawdown')}
                            className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${view === 'drawdown'
                                ? 'bg-black/10 dark:bg-white/10 text-foreground shadow-sm'
                                : 'text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                }`}
                        >
                            Drawdown
                        </button>
                    </div>
                </div>

                <div className="flex justify-between items-center flex-wrap gap-2">
                    <PeriodSelector selectedPeriod={period} onPeriodChange={setPeriod} />
                    {view === 'return' && (
                        <BenchmarkSelector selectedBenchmarks={benchmarks} onBenchmarkChange={onBenchmarksChange} />
                    )}
                </div>
            </div>

            <div className="h-64 w-full relative">
                {loading && (
                    <div className="absolute inset-0 bg-white/50 dark:bg-gray-800/50 flex items-center justify-center z-10">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    </div>
                )}
                <ResponsiveContainer width="100%" height="100%">
                    {view === 'return' ? (
                        <LineChart syncId="portfolio-sync" data={data} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                            <XAxis
                                dataKey="date"
                                tickFormatter={formatXAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                minTickGap={30}
                            />
                            <YAxis
                                tickFormatter={formatYAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                width={45}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend />
                            <Line
                                name="Portfolio"
                                type="monotone"
                                dataKey="twr"
                                stroke={COLORS[0]}
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 6 }}
                            />
                            {benchmarkKeys.map((key, index) => (
                                <Line
                                    key={key}
                                    name={key}
                                    type="monotone"
                                    dataKey={key}
                                    stroke={COLORS[(index + 1) % COLORS.length]}
                                    strokeWidth={2}
                                    dot={false}
                                />
                            ))}
                        </LineChart>
                    ) : view === 'value' ? (
                        <AreaChart syncId="portfolio-sync" data={data} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.1} />
                                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                            <XAxis
                                dataKey="date"
                                tickFormatter={formatXAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                minTickGap={30}
                            />
                            <YAxis
                                tickFormatter={formatYAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                width={45}
                                domain={['auto', 'auto']}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Area
                                name="Portfolio Value"
                                type="monotone"
                                dataKey="value"
                                stroke="#2563eb"
                                fillOpacity={1}
                                fill="url(#colorValue)"
                                strokeWidth={2}
                            />
                        </AreaChart>
                    ) : (
                        <AreaChart syncId="portfolio-sync" data={data} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.1} />
                                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                            <XAxis
                                dataKey="date"
                                tickFormatter={formatXAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                minTickGap={30}
                            />
                            <YAxis
                                tickFormatter={formatYAxis}
                                tick={{ fontSize: 12, fill: '#9ca3af' }}
                                axisLine={false}
                                tickLine={false}
                                width={45}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Area
                                name="Drawdown"
                                type="monotone"
                                dataKey="drawdown"
                                stroke="#ef4444"
                                fillOpacity={1}
                                fill="url(#colorDrawdown)"
                                strokeWidth={2}
                            />
                        </AreaChart>
                    )}
                </ResponsiveContainer>
            </div>
        </div>
    );
}
