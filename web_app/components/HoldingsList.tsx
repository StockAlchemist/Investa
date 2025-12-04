import React from 'react';
import { Holding } from '../lib/api';

interface HoldingsListProps {
    holdings: Holding[];
}

export default function HoldingsList({ holdings }: HoldingsListProps) {
    if (!holdings || holdings.length === 0) {
        return <div className="p-4 text-center text-gray-500">No holdings found.</div>;
    }

    // Helper to find value by key prefix (e.g., "Market Value" -> "Market Value (USD)")
    const getValue = (holding: Holding, prefix: string) => {
        const key = Object.keys(holding).find(k => k.startsWith(prefix));
        const val = key ? holding[key] : 0;
        return (val === null || val === undefined) ? 0 : val;
    };

    // Sort by Market Value desc
    const sortedHoldings = [...holdings].sort((a, b) => getValue(b, 'Market Value') - getValue(a, 'Market Value'));

    return (
        <div className="bg-white dark:bg-gray-800 rounded-t-xl shadow-sm border-t border-gray-100 dark:border-gray-700 mt-4">
            <div className="p-4 border-b border-gray-100 dark:border-gray-700">
                <h2 className="text-lg font-bold text-gray-900 dark:text-white">Holdings</h2>
            </div>
            <div className="divide-y divide-gray-100 dark:divide-gray-700">
                {sortedHoldings.map((h, idx) => {
                    const dayChange = getValue(h, 'Day Change'); // Matches "Day Change (USD)"
                    const dayChangePct = getValue(h, 'Day Change %');
                    const marketValue = getValue(h, 'Market Value');
                    const price = getValue(h, 'Price');
                    const isPositive = dayChange >= 0;

                    return (
                        <div key={idx} className="p-4 flex justify-between items-center hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
                            <div>
                                <div className="font-bold text-gray-900 dark:text-white">{h.Symbol}</div>
                                <div className="text-sm text-gray-500 dark:text-gray-400">
                                    {h.Quantity?.toLocaleString()} shares @ ${price?.toFixed(2)}
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="font-bold text-gray-900 dark:text-white">
                                    ${marketValue?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </div>
                                <div className={`text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                                    {isPositive ? '+' : ''}{dayChange.toFixed(2)} ({dayChangePct.toFixed(2)}%)
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
