import React, { useState } from 'react';
import { Transaction } from '../lib/api';

interface TransactionsTableProps {
    transactions: Transaction[];
}

export default function TransactionsTable({ transactions }: TransactionsTableProps) {
    const [symbolFilter, setSymbolFilter] = useState('');
    const [accountFilter, setAccountFilter] = useState('');
    const [visibleRows, setVisibleRows] = useState(10);

    if (!transactions || transactions.length === 0) {
        return <div className="p-4 text-center text-gray-500">No transactions found.</div>;
    }

    const filteredTransactions = transactions.filter(tx => {
        const symbolMatch = tx.Symbol.toLowerCase().includes(symbolFilter.toLowerCase());
        const accountMatch = tx.Account.toLowerCase().includes(accountFilter.toLowerCase());
        return symbolMatch && accountMatch;
    });

    const visibleTransactions = filteredTransactions.slice(0, visibleRows);

    const handleShowMore = () => {
        setVisibleRows(prev => prev + 20);
    };

    const handleShowAll = () => {
        setVisibleRows(filteredTransactions.length);
    };

    return (
        <div className="space-y-4">
            <div className="flex flex-col gap-3">
                <div className="flex flex-row gap-2">
                    <div className="relative flex-1 min-w-0">
                        <input
                            type="text"
                            placeholder="Filter Symbol..."
                            value={symbolFilter}
                            onChange={(e) => setSymbolFilter(e.target.value)}
                            className="w-full pl-3 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                        />
                        {symbolFilter && (
                            <button
                                onClick={() => setSymbolFilter('')}
                                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                            >
                                ✕
                            </button>
                        )}
                    </div>
                    <div className="relative flex-1 min-w-0">
                        <input
                            type="text"
                            placeholder="Filter Account..."
                            value={accountFilter}
                            onChange={(e) => setAccountFilter(e.target.value)}
                            className="w-full pl-3 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                        />
                        {accountFilter && (
                            <button
                                onClick={() => setAccountFilter('')}
                                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                            >
                                ✕
                            </button>
                        )}
                    </div>
                </div>
                <div className="flex justify-between items-center">
                    <button
                        onClick={() => { setSymbolFilter(''); setAccountFilter(''); }}
                        className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors text-sm font-medium"
                    >
                        Reset Filters
                    </button>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                        Showing {visibleTransactions.length} of {filteredTransactions.length} transactions
                    </div>
                </div>
            </div>

            <div className="overflow-x-auto">
                <table className="min-w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                        <tr>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Date</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Type</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Symbol</th>
                            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Qty</th>
                            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Price/Share</th>
                            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Total Amount</th>
                            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Commission</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Account</th>
                            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Split Ratio</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Note</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Currency</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                        {visibleTransactions.map((tx, index) => (
                            <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
                                <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-200 whitespace-nowrap">{tx.Date ? tx.Date.split('T')[0].split(' ')[0] : '-'}</td>
                                <td className="px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${tx.Type.toUpperCase() === 'BUY' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                                        tx.Type.toUpperCase() === 'SELL' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                                            'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                                        }`}>
                                        {tx.Type}
                                    </span>
                                </td>
                                <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">{tx.Symbol}</td>
                                <td className="px-4 py-3 text-sm text-right text-gray-900 dark:text-gray-200">{tx.Quantity}</td>
                                <td className="px-4 py-3 text-sm text-right text-gray-900 dark:text-gray-200">{tx["Price/Share"]?.toFixed(2)}</td>
                                <td className="px-4 py-3 text-sm text-right font-medium text-gray-900 dark:text-gray-200">
                                    {tx["Total Amount"] ? Math.abs(tx["Total Amount"]).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '-'}
                                </td>
                                <td className="px-4 py-3 text-sm text-right text-gray-900 dark:text-gray-200">
                                    {tx.Commission ? tx.Commission.toFixed(2) : '-'}
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-200 whitespace-nowrap">{tx.Account}</td>
                                <td className="px-4 py-3 text-sm text-right text-gray-900 dark:text-gray-200">{tx["Split Ratio"] || '-'}</td>
                                <td className="px-4 py-3 text-sm text-gray-500 dark:text-gray-400 truncate max-w-xs" title={tx.Note}>{tx.Note || '-'}</td>
                                <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-200">{tx["Local Currency"]}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {visibleRows < filteredTransactions.length && (
                <div className="flex justify-center gap-4 mt-4">
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
    );
}
