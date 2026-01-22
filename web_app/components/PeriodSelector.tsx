import React from 'react';

interface PeriodSelectorProps {
    selectedPeriod: string;
    onPeriodChange: (period: string) => void;
}

const PERIODS = [
    { label: '1D', value: '1d' },
    { label: '5D', value: '5d' },
    { label: '1M', value: '1m' },
    { label: '3M', value: '3m' },
    { label: '6M', value: '6m' },
    { label: 'YTD', value: 'ytd' },
    { label: '1Y', value: '1y' },
    { label: '3Y', value: '3y' },
    { label: '5Y', value: '5y' },
    { label: '10Y', value: '10y' },
    { label: 'All', value: 'all' },
    { label: 'Custom', value: 'custom' },
];

export default function PeriodSelector({ selectedPeriod, onPeriodChange }: PeriodSelectorProps) {
    return (
        <div className="flex shrink-0 space-x-1 bg-secondary rounded-lg p-1 border border-border">
            {PERIODS.map((period) => (
                <button
                    key={period.value}
                    onClick={() => onPeriodChange(period.value)}
                    className={`px-2.5 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-all ${selectedPeriod === period.value
                        ? 'bg-[#0097b2] text-white shadow-sm'
                        : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                        }`}
                >
                    {period.label}
                </button>
            ))}
        </div>
    );
}
