import React from 'react';
import { cn } from '@/lib/utils';

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
        <div className="inline-flex items-center gap-0.5 bg-muted/60 border border-border/60 rounded-lg p-0.5">
            {PERIODS.map((period) => {
                const active = selectedPeriod === period.value;
                return (
                    <button
                        key={period.value}
                        onClick={() => onPeriodChange(period.value)}
                        className={cn(
                            'relative px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all duration-150 whitespace-nowrap',
                            active
                                ? 'bg-indigo-600 text-white font-bold shadow'
                                : 'text-muted-foreground hover:text-foreground hover:bg-background/50',
                        )}
                    >
                        {period.label}
                    </button>
                );
            })}
        </div>
    );
}
