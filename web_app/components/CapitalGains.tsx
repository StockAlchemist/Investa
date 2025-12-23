import React, { useState, useMemo } from 'react';
import { CapitalGain } from '../lib/api';
import { formatCurrency } from '../lib/utils';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

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
        return <div className="p-4 text-center text-gray-500">Loading capital gains data...</div>;
    }

    if (data.length === 0) {
        return <div className="p-4 text-center text-gray-500">No realized capital gains found for the selected criteria.</div>;
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
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Realized Gain</h3>
                    <p className={`text-2xl font-bold ${totalRealizedGain >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {formatCurrency(totalRealizedGain, currency)}
                    </p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Proceeds</h3>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {formatCurrency(totalProceeds, currency)}
                    </p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Cost Basis</h3>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {formatCurrency(totalCostBasis, currency)}
                    </p>
                </div>
            </div>

            {/* Annual Gains Chart */}
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Annual Realized Gains</h3>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={gainsByYear} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                            <XAxis dataKey="year" tick={{ fontSize: 12 }} />
                            <YAxis tickFormatter={(val) => new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(val)} />
                            <Tooltip
                                formatter={(value: number) => [formatCurrency(value, currency), 'Realized Gain']}
                                labelStyle={{ color: '#374151' }}
                            />
                            <Bar dataKey="gain" fill="#10B981" name="Realized Gain" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Transactions Table */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 overflow-hidden">
                <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Realized Gain Transactions</h3>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                        Showing {visibleData.length} of {sortedData.length} transactions
                    </div>
                </div>
                {/* Mobile Card View */}
                <div className="md:hidden space-y-4 p-4">
                    {visibleData.map((item, index) => (
                        <div key={`mobile-${index}`} className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-100 dark:border-gray-700 shadow-sm">
                            <div className="flex justify-between items-start mb-2">
                                <div>
                                    <h3 className="text-lg font-bold text-gray-900 dark:text-white">{item.Symbol}</h3>
                                    <div className="text-xs text-gray-500 dark:text-gray-400">
                                        {item.Date} • {item.Account}
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className={`text-lg font-bold ${(item['Realized Gain (Display)'] || 0) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                                        {formatCurrency(item['Realized Gain (Display)'] || 0, currency)}
                                    </div>
                                    <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        {item.Type}
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
                                <div className="flex justify-between">
                                    <span className="text-gray-500 dark:text-gray-400">Qty:</span>
                                    <span className="text-gray-900 dark:text-gray-200 font-medium">{item.Quantity}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-500 dark:text-gray-400">Proceeds:</span>
                                    <span className="text-gray-900 dark:text-gray-200 font-medium whitespace-nowrap">
                                        {formatCurrency(item["Total Proceeds (Display)"] || 0, currency)}
                                    </span>
                                </div>
                                <div className="flex justify-between col-span-2">
                                    <span className="text-gray-500 dark:text-gray-400">Cost Basis:</span>
                                    <span className="text-gray-900 dark:text-gray-200 font-medium whitespace-nowrap">
                                        {formatCurrency(item["Total Cost Basis (Display)"] || 0, currency)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Desktop Table View */}
                <div className="hidden md:block overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead className="bg-gray-50 dark:bg-gray-700">
                            <tr>
                                {['Date', 'Symbol', 'Account', 'Type', 'Quantity', 'Proceeds', 'Cost Basis', 'Realized Gain'].map((header) => (
                                    <th
                                        key={header}
                                        onClick={() => requestSort(
                                            header === 'Realized Gain' ? 'Realized Gain (Display)' as any :
                                                header === 'Proceeds' ? 'Total Proceeds (Display)' as any :
                                                    header === 'Cost Basis' ? 'Total Cost Basis (Display)' as any :
                                                        header as any
                                        )}
                                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                                    >
                                        {header}
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
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{item.Type}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{item.Quantity}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-500 dark:text-gray-300">
                                        {formatCurrency(item["Total Proceeds (Display)"] || 0, currency)}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-500 dark:text-gray-300">
                                        {formatCurrency(item["Total Cost Basis (Display)"] || 0, currency)}
                                    </td>
                                    <td className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${(item['Realized Gain (Display)'] || 0) >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                                        }`}>
                                        {formatCurrency(item['Realized Gain (Display)'] || 0, currency)}
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
