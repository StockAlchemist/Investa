'use client';
import React from 'react';
import { cn } from '../../lib/utils';

/** Trailing-12-month vs. since-inception — the window both income cards offer. */
export type IncomeWindow = '12m' | 'all';

interface WindowToggleProps {
    value: IncomeWindow;
    onChange: (value: IncomeWindow) => void;
}

export default function WindowToggle({ value, onChange }: WindowToggleProps) {
    return (
        <div className="inline-flex rounded-lg bg-secondary p-0.5">
            {(['12m', 'all'] as const).map(w => (
                <button
                    key={w}
                    onClick={() => onChange(w)}
                    className={cn(
                        'px-2.5 py-1 rounded-md text-xs font-semibold transition-all whitespace-nowrap',
                        value === w ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground',
                    )}
                >
                    {w === '12m' ? '12M' : 'All time'}
                </button>
            ))}
        </div>
    );
}
