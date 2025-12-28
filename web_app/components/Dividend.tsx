import React, { useState, useMemo } from 'react';
import type { Dividend } from '../lib/api';
import { formatCurrency } from '../lib/utils';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface DividendProps {
    data: Dividend[] | null;
    currency: string;
    expectedDividends?: number;
    children?: React.ReactNode;
}

export default function Dividend({ data, currency, expectedDividends, children }: DividendProps) {
    const [sortConfig, setSortConfig] = useState<{ key: keyof Dividend; direction: 'ascending' | 'descending' } | null>({ key: 'Date', direction: 'descending' });
    const [visibleRows, setVisibleRows] = useState(10);

    // Group by Year for Chart
    const dividendsByYear = useMemo(() => {
        if (!data) return [];
        const groups: Record<string, number> = {};
        data.forEach(item => {
            const year = item.Date.substring(0, 4);
            groups[year] = (groups[year] || 0) + (item['DividendAmountDisplayCurrency'] || 0);
        });
        return Object.entries(groups)
            .map(([year, amount]) => ({ year, amount }))
            .sort((a, b) => a.year.localeCompare(b.year));
    }, [data]);

    // Sorting
    const sortedData = useMemo(() => {
        if (!data) return [];
        let sortableItems = [...data];
        if (sortConfig !== null) {
            sortableItems.sort((a, b) => {
                const aValue = a[sortConfig.key];
                const bValue = b[sortConfig.key];

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
        return <div className="p-4 text-center text-muted-foreground">Loading dividend data...</div>;
    }

    if (data.length === 0) {
        return <div className="p-4 text-center text-muted-foreground">No dividend history found for the selected criteria.</div>;
    }

    // --- Calculations ---
    const totalDividends = data.reduce((sum, item) => sum + (item['DividendAmountDisplayCurrency'] || 0), 0);

    const requestSort = (key: keyof Dividend) => {
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
                    <h3 className="text-sm font-medium text-muted-foreground">Total Dividends</h3>
                    <p className="text-2xl font-bold text-emerald-500">
                        {formatCurrency(totalDividends, currency)}
                    </p>
                </div>
                {expectedDividends !== undefined && (
                    <div className="bg-white/5 backdrop-blur-md p-4 rounded-xl shadow-sm border border-white/10">
                        <h3 className="text-sm font-medium text-muted-foreground">Expected Dividends (Next 12M)</h3>
                        <p className="text-2xl font-bold text-cyan-400">
                            {formatCurrency(expectedDividends, currency)}
                        </p>
                    </div>
                )}
            </div>

            {/* Injected Content (e.g. Dividend Calendar) */}
            {children}

            {/* Annual Dividends Chart */}
            <div className="bg-white/5 backdrop-blur-md p-4 rounded-xl shadow-sm border border-white/10">
                <h3 className="text-lg font-semibold text-foreground mb-4">Annual Dividends</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={dividendsByYear} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="year" tick={{ fontSize: 12, fill: '#9ca3af' }} axisLine={{ stroke: 'rgba(255,255,255,0.1)' }} />
                            <YAxis
                                tickFormatter={(val) => new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(val)}
                                tick={{ fill: '#9ca3af' }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                            />
                            <Tooltip
                                formatter={(value: number | undefined) => [formatCurrency(value || 0, currency), 'Dividend Amount']}
                                contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: 'rgba(255,255,255,0.1)', color: '#fff' }}
                                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                            />
                            <Bar dataKey="amount" fill="#3B82F6" name="Dividend Amount" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Transactions Table */}
            <div className="bg-black/5 dark:bg-white/5 backdrop-blur-md rounded-xl shadow-sm border border-black/5 dark:border-white/10 overflow-hidden">
                <div className="p-4 border-b border-black/5 dark:border-white/5 flex justify-between items-center">
                    <h3 className="text-lg font-semibold text-foreground">Dividend Transactions</h3>
                    <div className="text-sm text-muted-foreground">
                        Showing {visibleData.length} of {sortedData.length} transactions
                    </div>
                </div>
                {/* Desktop Table View */}
                <div className="hidden md:block overflow-x-auto">
                    <table className="min-w-full divide-y divide-black/5 dark:divide-white/10">
                        <thead className="bg-black/5 dark:bg-white/5">
                            <tr>
                                {['Date', 'Symbol', 'Account', 'DividendAmountDisplayCurrency'].map((header) => (
                                    <th
                                        key={header}
                                        onClick={() => requestSort(header as keyof Dividend)}
                                        className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                                    >
                                        {header === 'DividendAmountDisplayCurrency' ? 'Amount' : header}
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
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-muted-foreground">
                                        {formatCurrency(item['DividendAmountDisplayCurrency'] || 0, currency)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Mobile Card View */}
                <div className="block md:hidden space-y-4 p-4">
                    {visibleData.map((item, index) => (
                        <div key={`mobile-div-${index}`} className="bg-black/5 dark:bg-white/5 rounded-lg border border-black/5 dark:border-white/10 shadow-sm p-4">
                            <div className="flex justify-between items-start mb-2">
                                <div>
                                    <h3 className="text-lg font-bold text-foreground">{item.Symbol}</h3>
                                    <div className="text-xs text-muted-foreground">{item.Date} • {item.Account}</div>
                                </div>
                                <div className="text-right">
                                    <div className="text-lg font-bold text-emerald-500">
                                        {formatCurrency(item['DividendAmountDisplayCurrency'] || 0, currency)}
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
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
