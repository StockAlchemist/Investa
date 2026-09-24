import React from 'react';

interface PeriodSelectorProps {
    selectedPeriod: string;
    onPeriodChange: (period: string) => void;
}

const PERIODS = [
    { label: '1D',     value: '1d' },
    { label: '5D',     value: '5d' },
    { label: '1M',     value: '1m' },
    { label: '3M',     value: '3m' },
    { label: '6M',     value: '6m' },
    { label: 'YTD',    value: 'ytd' },
    { label: '1Y',     value: '1y' },
    { label: '3Y',     value: '3y' },
    { label: '5Y',     value: '5y' },
    { label: '10Y',    value: '10y' },
    { label: 'All',    value: 'all' },
    { label: 'Custom', value: 'custom' },
];

export default function PeriodSelector({ selectedPeriod, onPeriodChange }: PeriodSelectorProps) {
    return (
        <div role="radiogroup" aria-label="Period" className="segmented h-8">
            {PERIODS.map((period) => {
                const active = selectedPeriod === period.value;
                return (
                    <button
                        key={period.value}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        onClick={() => onPeriodChange(period.value)}
                        className="text-xs whitespace-nowrap"
                    >
                        {period.label}
                    </button>
                );
            })}
        </div>
    );
}
