import React, { useState, useMemo } from 'react';
import { CapitalGain } from '../lib/api';
import { formatCurrency } from '../lib/utils';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

interface CapitalGainsProps {
    data: CapitalGain[] | null;
    currency: string;
}

export default function CapitalGains({ data, currency }: CapitalGainsProps) {
    const [sortConfig, setSortConfig] = useState<{ key: keyof CapitalGain; direction: 'ascending' | 'descending' } | null>({ key: 'Date', direction: 'descending' });
    const [visibleRows, setVisibleRows] = useState(10);

    // Group by Year for Chart
    const gainsByYear = useMemo(() => {
        if (!data) return [];
        const groups: Record<string, number> = {};
        data.forEach(item => {
            const year = item.Date.substring(0, 4);
            groups[year] = (groups[year] || 0) + (item['Realized Gain (Display)'] || 0);
        });
        return Object.entries(groups)
            .map(([year, gain]) => ({ year, gain }))
            .sort((a, b) => a.year.localeCompare(b.year));
    }, [data]);

    // Sorting
    const sortedData = useMemo(() => {
        if (!data) return [];
        // eslint-disable-next-line
        let sortableItems = [...data];
        if (sortConfig !== null) {
            sortableItems.sort((a, b) => {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const aValue = a[sortConfig.key] as any;
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const bValue = b[sortConfig.key] as any;

                if (aValue < bValue) {
                    return sortConfig.direction === 'ascending' ? -1 : 1;
                }
                if (aValue > bValue) {
                    return sortConfig.direction === 'ascending' ? 1 : -1;
                }
                return 0;
            });
        }
        return sortableItems;
    }, [data, sortConfig]);

    if (!data) {
        return <div className="p-4 text-center text-muted-foreground">Loading capital gains data...</div>;
    }

    if (data.length === 0) {
        return <div className="p-4 text-center text-muted-foreground">No realized capital gains found for the selected criteria.</div>;
    }

    // --- Calculations ---
    const totalRealizedGain = data.reduce((sum, item) => sum + (item['Realized Gain (Display)'] || 0), 0);
    const totalProceeds = data.reduce((sum, item) => sum + (item['Total Proceeds (Display)'] || 0), 0);
    const totalCostBasis = data.reduce((sum, item) => sum + (item['Total Cost Basis (Display)'] || 0), 0);

    const requestSort = (key: keyof CapitalGain) => {
        let direction: 'ascending' | 'descending' = 'ascending';
        if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    const visibleData = sortedData.slice(0, visibleRows);

    const handleShowMore = () => {
        setVisibleRows(prev => prev + 20);
    };

    const handleShowAll = () => {
        setVisibleRows(sortedData.length);
    };

    return (
        <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white/5 backdrop-blur-md p-4 rounded-xl shadow-sm border border-white/10">
                    <h3 className="text-sm font-medium text-muted-foreground">Total Realized Gain</h3>
                    <p className={`text-2xl font-bold ${totalRealizedGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                        {formatCurrency(totalRealizedGain, currency)}
                    </p>
                </div>
                <div className="bg-white/5 backdrop-blur-md p-4 rounded-xl shadow-sm border border-white/10">
                    <h3 className="text-sm font-medium text-muted-foreground">Total Proceeds</h3>
                    <p className="text-2xl font-bold text-foreground">
                        {formatCurrency(totalProceeds, currency)}
                    </p>
                </div>
                <div className="bg-white/5 backdrop-blur-md p-4 rounded-xl shadow-sm border border-white/10">
                    <h3 className="text-sm font-medium text-muted-foreground">Total Cost Basis</h3>
                    <p className="text-2xl font-bold text-foreground">
                        {formatCurrency(totalCostBasis, currency)}
                    </p>
                </div>
            </div>

            {/* Annual Gains Chart */}
            <div className="bg-white/5 backdrop-blur-md p-4 rounded-xl shadow-sm border border-white/10">
                <h3 className="text-lg font-semibold text-foreground mb-4">Annual Realized Gains</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={gainsByYear} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="year" tick={{ fontSize: 12, fill: '#9ca3af' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                            <YAxis
                                tickFormatter={(val) => new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(val)}
                                tick={{ fill: '#9ca3af', fontSize: 10 }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                                width={35}
                            />
                            <Tooltip
                                content={({ active, payload, label }) => {
                                    if (active && payload && payload.length) {
                                        return (
                                            <div className="bg-popover/95 backdrop-blur-sm border border-border p-3 rounded-lg shadow-xl">
                                                <p className="font-medium text-foreground mb-1">{label}</p>
                                                <div className="flex items-center gap-2 text-sm">
                                                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                                                    <span className="text-muted-foreground">Realized Gain:</span>
                                                    <span className="font-medium text-emerald-500">
                                                        {formatCurrency(payload[0].value as number, currency)}
                                                    </span>
                                                </div>
                                            </div>
                                        );
                                    }
                                    return null;
                                }}
                                cursor={{ fill: 'var(--glass-hover)' }}
                            />
                            <Bar dataKey="gain" fill="#10B981" name="Realized Gain" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Transactions Table */}
            <div className="bg-black/5 dark:bg-white/5 backdrop-blur-md rounded-xl shadow-sm border border-black/5 dark:border-white/10 overflow-hidden">
                <div className="p-4 border-b border-black/5 dark:border-white/5 flex justify-between items-center">
                    <h3 className="text-lg font-semibold text-foreground">Realized Gain Transactions</h3>
                    <div className="text-sm text-muted-foreground">
                        Showing {visibleData.length} of {sortedData.length} transactions
                    </div>
                </div>
                {/* Mobile Card View */}
                <div className="md:hidden space-y-4 p-4">
                    {visibleData.map((item, index) => (
                        <div key={`mobile-${index}`} className="bg-black/5 dark:bg-white/5 p-4 rounded-lg border border-black/5 dark:border-white/10 shadow-sm">
                            <div className="flex justify-between items-start mb-2">
                                <div>
                                    <h3 className="text-lg font-bold text-foreground">{item.Symbol}</h3>
                                    <div className="text-xs text-muted-foreground">
                                        {item.Date} • {item.Account}
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className={`text-lg font-bold ${(item['Realized Gain (Display)'] || 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                                        {formatCurrency(item['Realized Gain (Display)'] || 0, currency)}
                                    </div>
                                    <div className="text-xs text-muted-foreground uppercase tracking-wider">
                                        {item.Type}
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm mt-3 pt-3 border-t border-black/5 dark:border-white/10">
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Qty:</span>
                                    <span className="text-foreground font-medium">{item.Quantity}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Proceeds:</span>
                                    <span className="text-foreground font-medium whitespace-nowrap">
                                        {formatCurrency(item["Total Proceeds (Display)"] || 0, currency)}
                                    </span>
                                </div>
                                <div className="flex justify-between col-span-2">
                                    <span className="text-muted-foreground">Cost Basis:</span>
                                    <span className="text-foreground font-medium whitespace-nowrap">
                                        {formatCurrency(item["Total Cost Basis (Display)"] || 0, currency)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Desktop Table View */}
                <div className="hidden md:block overflow-x-auto">
                    <table className="min-w-full divide-y divide-black/5 dark:divide-white/10">
                        <thead className="bg-black/5 dark:bg-white/5">
                            <tr>
                                {['Date', 'Symbol', 'Account', 'Type', 'Quantity', 'Proceeds', 'Cost Basis', 'Realized Gain'].map((header) => (
                                    <th
                                        key={header}
                                        onClick={() => requestSort(
                                            header === 'Realized Gain' ? 'Realized Gain (Display)' :
                                                header === 'Proceeds' ? 'Total Proceeds (Display)' :
                                                    header === 'Cost Basis' ? 'Total Cost Basis (Display)' :
                                                        header as keyof CapitalGain
                                        )}
                                        className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                                    >
                                        {header}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-black/5 dark:divide-white/10">
                            {visibleData.map((item, index) => (
                                <tr key={index} className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-foreground">{item.Date}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-foreground">{item.Symbol}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-muted-foreground">{item.Account}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-muted-foreground">{item.Type}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-muted-foreground">{item.Quantity}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-muted-foreground">
                                        {formatCurrency(item["Total Proceeds (Display)"] || 0, currency)}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-muted-foreground">
                                        {formatCurrency(item["Total Cost Basis (Display)"] || 0, currency)}
                                    </td>
                                    <td className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${(item['Realized Gain (Display)'] || 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                                        }`}>
                                        {formatCurrency(item['Realized Gain (Display)'] || 0, currency)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                {visibleRows < sortedData.length && (
                    <div className="flex justify-center gap-4 p-4 border-t border-black/5 dark:border-white/10">
                        <button
                            onClick={handleShowMore}
                            className="px-4 py-2 bg-cyan-600/20 text-cyan-600 dark:text-cyan-400 border border-cyan-500/30 rounded-md hover:bg-cyan-600/30 transition-colors text-sm font-medium"
                        >
                            Show More
                        </button>
                        <button
                            onClick={handleShowAll}
                            className="px-4 py-2 bg-black/5 dark:bg-white/5 text-foreground border border-black/5 dark:border-white/10 rounded-md hover:bg-black/10 dark:hover:bg-white/10 transition-colors text-sm font-medium"
                        >
                            Show All
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
