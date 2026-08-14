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
    return (
        <div className="bg-muted py-1.5 px-3 rounded-xl flex items-center gap-3 transition-all hover:bg-muted/50 group relative overflow-hidden">
            {/* Soft background glow */}
            <div className={cn(
                "absolute -top-8 -right-8 w-20 h-20 blur-[25px] opacity-10 transition-opacity duration-500 group-hover:opacity-20 pointer-events-none rounded-full",
                color?.includes('emerald') ? 'bg-emerald-500' :
                    color?.includes('rose') ? 'bg-rose-500' :
                        color?.includes('indigo') ? 'bg-indigo-500' :
                            color?.includes('amber') ? 'bg-amber-500' :
                                color?.includes('purple') ? 'bg-purple-500' :
                                    'bg-slate-500'
            )} />

            <div className={cn("p-2 rounded-lg bg-card relative z-10", color, rotate)}>
                <Icon className="w-4 h-4" />
            </div>
            <div className="flex-1 overflow-hidden relative z-10">
                <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider truncate">{label}</p>
                <div className="flex items-baseline gap-1.5">
                    <p className={cn("text-base font-bold tracking-tight whitespace-nowrap", valueColor || "text-foreground")}>{value}</p>
                    {subValue && (
                        <span className={cn("text-xs whitespace-nowrap", subValueColor)}>
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
