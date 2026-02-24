import React from 'react';
import { cn } from '@/lib/utils';

interface InlineProgressBarProps {
    value: number; // usually percentage, 0 to 100
    max?: number;
    colorClass?: string;
    className?: string;
    children?: React.ReactNode;
}

export function InlineProgressBar({
    value,
    max = 100,
    colorClass = 'bg-cyan-500/10 dark:bg-cyan-500/20',
    className,
    children,
}: InlineProgressBarProps) {
    const percentage = Math.max(0, Math.min((value / max) * 100, 100));

    return (
        <div className={cn("relative w-full h-full min-h-[1.75rem] flex items-center rounded overflow-hidden group/progress", className)}>
            <div
                className={cn("absolute left-0 top-0 bottom-0 transition-all duration-500 ease-out", colorClass)}
                style={{ width: `${percentage}%` }}
            />
            <div className="relative z-10 w-full flex items-center justify-between px-2">
                {children}
            </div>
        </div>
    );
}
