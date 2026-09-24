import React from 'react';
import { Skeleton } from "@/components/ui/skeleton";
import { cn, formatCurrency } from '@/lib/utils';
import { LucideIcon, Loader2 } from 'lucide-react';

export interface MetricCardProps {
    title: string;
    value: string | number | null;
    subValue?: number | string | null;
    isCurrency?: boolean;
    colorClass?: string;
    valueClassName?: string;
    containerClassName?: string;
    subValueClassName?: string;
    currency?: string;
    isHero?: boolean;
    trend?: number | string | null;
    icon?: LucideIcon;
    isLoading?: boolean;
    isRefreshing?: boolean;
    /** Retired — a watermark chart behind a figure informs no one. */
    sparklineData?: { value: number }[];
    accentColor?: string;
    variant?: 'card' | 'seamless';
    onClick?: () => void;
}

// The eleven-entry accent map is gone: a KPI tile no longer paints a 128px
// blurred colour glow in its corner, nor a 10%-opacity sparkline watermark
// across its bottom. Neither was readable enough to inform and both were
// visible enough to distract, with the figure sitting on top of them. Tiles
// carry one accent; if a trend matters it gets a real chart with an axis.
//
// `accentColor` is still accepted so the ~20 call sites keep compiling.

export function MetricCard({
    title,
    value,
    subValue,
    isCurrency = true,
    colorClass = '',
    valueClassName = 'text-[22px]',
    containerClassName = '',
    subValueClassName = '',
    currency = 'USD',
    isLoading = false,
    isRefreshing = false,
    icon: Icon,
    variant = 'card',
    onClick,
}: MetricCardProps) {
    // Label → figure → delta, with the delta row reserved even when empty, so a
    // grid row of tiles shares one baseline and one height.
    const display = value !== null && value !== undefined
        ? (isCurrency && typeof value === 'number' ? formatCurrency(value, currency) : String(value))
        : '\u2014';

    // The full number is always shown — never abbreviated, never ellipsized.
    // The figure steps down as the string grows so long amounts still fit.
    const fitClass =
        display.length > 13 ? 'text-sm sm:text-base' :
        display.length > 11 ? 'text-base sm:text-lg' :
        display.length > 9 ? 'text-lg sm:text-xl' : '';

    const seamless = variant === 'seamless';
    void Icon;

    // Ledger KPI tile: a quiet label, the figure, and one line under it. A
    // numeric sub-value is a signed percentage in the gain/loss colour; a
    // string sub-value is context ("p.a.", "on cost") in the second ink.
    const subLine = subValue !== undefined && subValue !== null ? (
        typeof subValue === 'number' ? (
            <span className={cn('font-semibold', subValue >= 0 ? 'text-up' : 'text-down')}>
                {subValue === Infinity
                    ? '\u221e'
                    : `${subValue >= 0 ? '+' : '\u2212'}${Math.abs(subValue).toFixed(2)}%`}
            </span>
        ) : (
            <span className={cn('text-ink-2', subValueClassName && 'font-medium')}>{subValue}</span>
        )
    ) : null;

    return (
        <div
            className={cn(
                'card-standard relative h-full p-4 flex flex-col gap-1',
                seamless && 'min-h-[104px]',
                onClick ? 'cursor-pointer' : 'cursor-default',
                containerClassName,
            )}
            onClick={onClick}
        >
            {/* Label */}
            <div className="flex items-start justify-between gap-2">
                <p className="text-xs leading-4 text-muted-foreground min-w-0 line-clamp-2">{title}</p>
                {isRefreshing && (
                    <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin text-muted-foreground" />
                )}
            </div>

            {/* Figure */}
            <div className="min-w-0">
                {isLoading ? (
                    <Skeleton className="h-7 w-28 rounded-md" />
                ) : (
                    <span
                        title={value !== null && value !== undefined && isCurrency && typeof value === 'number'
                            ? formatCurrency(value, currency)
                            : undefined}
                        className={cn(
                            'block font-semibold leading-7 tabular-nums text-foreground whitespace-nowrap',
                            colorClass,
                            valueClassName,
                            fitClass,
                        )}
                    >
                        {display}
                    </span>
                )}
            </div>

            {/* Sub-line — the row keeps its height so tiles in a row line up. */}
            <div className="mt-auto min-h-4 flex items-center text-xs leading-4 tabular-nums whitespace-nowrap">
                {isLoading ? <Skeleton className="h-3.5 w-14 rounded" /> : subLine}
            </div>
        </div>
    );
}
