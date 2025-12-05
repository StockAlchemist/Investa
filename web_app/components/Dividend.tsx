import React, { useState, useMemo } from 'react';
import type { Dividend } from '../lib/api';
import { formatCurrency } from '../lib/utils';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface DividendProps {
    data: Dividend[] | null;
    currency: string;
    expectedDividends?: number;
}

export default function Dividend({ data, currency, expectedDividends }: DividendProps) {
    const [sortConfig, setSortConfig] = useState<{ key: keyof Dividend; direction: 'ascending' | 'descending' } | null>({ key: 'Date', direction: 'descending' });

    if (!data) {
        return <div className="p-4 text-center text-gray-500">Loading dividend data...</div>;
    }

    if (data.length === 0) {
        return <div className="p-4 text-center text-gray-500">No dividend history found for the selected criteria.</div>;
    }

    // --- Calculations ---
    const totalDividends = data.reduce((sum, item) => sum + (item['DividendAmountDisplayCurrency'] || 0), 0);

    // Group by Year for Chart
    const dividendsByYear = useMemo(() => {
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

    const requestSort = (key: keyof Dividend) => {
        let direction: 'ascending' | 'descending' = 'ascending';
        if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    const [visibleRows, setVisibleRows] = useState(10);

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
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Dividends</h3>
                    <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {formatCurrency(totalDividends, currency)}
                    </p>
                </div>
                {expectedDividends !== undefined && (
                    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
                        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Expected Dividends (Next 12M)</h3>
                        <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                            {formatCurrency(expectedDividends, currency)}
                        </p>
                    </div>
                )}
            </div>

            {/* Annual Dividends Chart */}
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Annual Dividends</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={dividendsByYear} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                            <XAxis dataKey="year" tick={{ fontSize: 12 }} />
                            <YAxis tickFormatter={(val) => new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(val)} />
                            <Tooltip
                                formatter={(value: number) => [formatCurrency(value, currency), 'Dividend Amount']}
                                labelStyle={{ color: '#374151' }}
                            />
                            <Bar dataKey="amount" fill="#3B82F6" name="Dividend Amount" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Transactions Table */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 overflow-hidden">
                <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Dividend Transactions</h3>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                        Showing {visibleData.length} of {sortedData.length} transactions
                    </div>
                </div>
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead className="bg-gray-50 dark:bg-gray-700">
                            <tr>
                                {['Date', 'Symbol', 'Account', 'DividendAmountDisplayCurrency'].map((header) => (
                                    <th
                                        key={header}
                                        onClick={() => requestSort(header as keyof Dividend)}
                                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                                    >
                                        {header === 'DividendAmountDisplayCurrency' ? 'Amount' : header}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                            {visibleData.map((item, index) => (
                                <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">{item.Date}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{item.Symbol}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{item.Account}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-500 dark:text-gray-300">
                                        {formatCurrency(item['DividendAmountDisplayCurrency'] || 0, currency)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                {visibleRows < sortedData.length && (
                    <div className="flex justify-center gap-4 p-4 border-t border-gray-200 dark:border-gray-700">
                        <button
                            onClick={handleShowMore}
                            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm font-medium"
                        >
                            Show More
                        </button>
                        <button
                            onClick={handleShowAll}
                            className="px-4 py-2 bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors text-sm font-medium"
                        >
                            Show All
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
