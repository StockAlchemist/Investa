import React from 'react';
import { Badge } from "@/components/ui/badge";
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
    valueClassName = 'text-xl sm:text-2xl',
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

    const deltaBadge = subValue !== undefined && subValue !== null ? (
        <Badge
            variant={typeof subValue === 'number' ? (subValue >= 0 ? 'success' : 'destructive') : 'outline'}
            className={cn('shrink-0', subValueClassName)}
        >
            {typeof subValue === 'number'
                ? (subValue === Infinity ? '\u221e' : `${subValue >= 0 ? '+' : ''}${subValue.toFixed(2)}%`)
                : subValue}
        </Badge>
    ) : null;

    return (
        <div
            className={cn(
                'card-standard relative h-full p-4 flex flex-col gap-2',
                seamless && 'min-h-[112px]',
                onClick ? 'cursor-pointer' : 'cursor-default',
                containerClassName,
            )}
            onClick={onClick}
        >
            {/* Label + icon */}
            <div className="flex items-start justify-between gap-2">
                <p className="section-label pr-1 leading-tight min-w-0 line-clamp-2">{title}</p>
                <div className="flex items-center gap-1.5 shrink-0">
                    {isRefreshing && (
                        <Loader2 className="w-3.5 h-3.5 animate-spin text-primary/60" />
                    )}
                    {Icon && (
                        <div className="p-1.5 rounded-md bg-primary/12 text-primary-ink">
                            <Icon className="w-3.5 h-3.5" />
                        </div>
                    )}
                </div>
            </div>

            {/* Figure */}
            <div className="min-w-0">
                {isLoading ? (
                    <Skeleton className="h-7 w-28 opacity-50 rounded-md" />
                ) : (
                    <span
                        title={value !== null && value !== undefined && isCurrency && typeof value === 'number'
                            ? formatCurrency(value, currency)
                            : undefined}
                        className={cn(
                            'block font-bold tracking-tight leading-none tabular-nums text-foreground truncate',
                            colorClass,
                            valueClassName,
                            fitClass,
                        )}
                    >
                        {display}
                    </span>
                )}
            </div>

            {/* Delta — the row keeps its height so card footers line up. */}
            <div className="mt-auto pt-1 min-h-[24px] flex items-center">
                {isLoading ? <Skeleton className="h-4 w-14 rounded-full opacity-50" /> : deltaBadge}
            </div>
        </div>
    );
}
