import React from 'react';
import { cn } from '../../../lib/utils';

interface StatCardProps {
    icon: React.ElementType;
    label: React.ReactNode;
    value: React.ReactNode;
    subValue?: React.ReactNode;
    color?: string;
    valueColor?: string;
    subValueColor?: string;
    extra?: React.ReactNode;
    rangeMin?: number | string;
    rangeMax?: number | string;
    rotate?: string;
    className?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
    icon: Icon,
    label,
    value,
    subValue,
    color,
    valueColor,
    subValueColor,
    extra,
    rangeMin,
    rangeMax,
    rotate,
}) => {
    void Icon; void color; void rotate;
    return (
        // Ledger inset tile: muted fill, a quiet label, the figure. No glow and
        // no icon chip — the label names the figure; colour is kept for sign.
        <div className="bg-muted py-3 px-3.5 rounded-inset flex items-center gap-3 relative h-full">
            <div className="flex-1 min-w-0 relative">
                <p className="text-xs leading-4 text-muted-foreground whitespace-nowrap">{label}</p>
                <div className="flex items-baseline gap-1.5">
                    <p className={cn("text-lg leading-6 font-semibold tabular-nums whitespace-nowrap", valueColor || "text-foreground")}>{value}</p>
                    {subValue && (
                        <span className={cn("text-xs font-semibold tabular-nums whitespace-nowrap", subValueColor)}>
                            {subValue}
                        </span>
                    )}
                </div>
                {(rangeMin && rangeMax) ? (
                    <p className="text-[10px] text-muted-foreground font-medium grayscale opacity-70">
                        Range: {rangeMin} - {rangeMax}
                    </p>
                ) : extra ? (
                    <div>
                        {extra}
                    </div>
                ) : null}
            </div>
        </div>
    );
};
