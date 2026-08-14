import React from 'react';
import { Activity as LucideActivity } from 'lucide-react';

interface FiftyTwoWeekCardProps {
    low?: number | null;
    high?: number | null;
    price?: number | null;
    format: (v: number) => string;
}

export const FiftyTwoWeekCard: React.FC<FiftyTwoWeekCardProps> = ({ low, high, price, format }) => {
    const usable = low != null && high != null && high > low;
    const position = usable && price != null
        ? Math.max(0, Math.min(1, (price - low) / (high - low)))
        : null;

    return (
        <div className="bg-muted py-1.5 px-3 rounded-xl flex items-center gap-3 relative overflow-hidden">
            <div className="p-2 rounded-lg bg-card text-blue-400 relative z-10">
                <LucideActivity className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0 relative z-10">
                <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">52-Week Range</p>
                {usable ? (
                    <>
                        <div className="flex items-baseline justify-between gap-2 text-[13px] font-bold tabular-nums">
                            <span>{format(low)}</span>
                            <span>{format(high)}</span>
                        </div>
                        <div className="relative h-1 rounded-full bg-gradient-to-r from-rose-500/40 via-amber-400/40 to-emerald-500/50 mt-1 mb-1">
                            {position !== null && (
                                <div
                                    className="absolute top-1/2 w-1.5 h-3 -translate-y-1/2 -translate-x-1/2 rounded-full bg-foreground shadow"
                                    style={{ left: `${position * 100}%` }}
                                    title={price != null ? format(price) : undefined}
                                />
                            )}
                        </div>
                    </>
                ) : (
                    <p className="text-base font-bold tracking-tight">-</p>
                )}
            </div>
        </div>
    );
};
