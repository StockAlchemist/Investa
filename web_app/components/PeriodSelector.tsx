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
        <div className="flex space-x-1 bg-black/5 dark:bg-white/5 backdrop-blur-md rounded-lg p-1 border border-black/10 dark:border-white/10">
            {PERIODS.map((period) => (
                <button
                    key={period.value}
                    onClick={() => onPeriodChange(period.value)}
                    className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${selectedPeriod === period.value
                        ? 'bg-black/10 dark:bg-white/10 text-foreground shadow-sm border border-black/5 dark:border-white/5'
                        : 'text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5'
                        }`}
                >
                    {period.label}
                </button>
            ))}
        </div>
    );
}
