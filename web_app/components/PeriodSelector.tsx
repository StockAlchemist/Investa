import React from 'react';

interface PeriodSelectorProps {
    selectedPeriod: string;
    onPeriodChange: (period: string) => void;
}

const PERIODS = [
    { label: '1M', value: '1m' },
    { label: '3M', value: '3m' },
    { label: '6M', value: '6m' },
    { label: 'YTD', value: 'ytd' },
    { label: '1Y', value: '1y' },
    { label: '3Y', value: '3y' },
    { label: '5Y', value: '5y' },
    { label: 'All', value: 'all' },
];

export default function PeriodSelector({ selectedPeriod, onPeriodChange }: PeriodSelectorProps) {
    return (
        <div className="flex space-x-1 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            {PERIODS.map((period) => (
                <button
                    key={period.value}
                    onClick={() => onPeriodChange(period.value)}
                    className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${selectedPeriod === period.value
                            ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                            : 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                        }`}
                >
                    {period.label}
                </button>
            ))}
        </div>
    );
}
