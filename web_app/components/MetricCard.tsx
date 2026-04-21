import React from 'react';
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, formatCurrency } from '@/lib/utils';
import { ResponsiveContainer, LineChart, Line } from 'recharts';
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
    isPercent?: boolean;
    trend?: number | string | null;
    icon?: LucideIcon;
    isLoading?: boolean;
    isRefreshing?: boolean;
    sparklineData?: { value: number }[];
    accentColor?: string;
    variant?: 'card' | 'seamless';
    onClick?: () => void;
}

// Maps accent color name → CSS values for icon bg, icon text, top-border glow
const ACCENT_MAP: Record<string, {
    iconBg: string;
    iconText: string;
    topBar: string;
    glowShadow: string;
    sparkColor: string;
}> = {
    'indigo-500': {
        iconBg: 'bg-indigo-500/15 dark:bg-indigo-500/20',
        iconText: 'text-indigo-500',
        topBar: 'bg-indigo-500',
        glowShadow: 'hover:shadow-indigo-500/10',
        sparkColor: '#6366f1',
    },
    'sky-500': {
        iconBg: 'bg-sky-500/15 dark:bg-sky-500/20',
        iconText: 'text-sky-500',
        topBar: 'bg-sky-500',
        glowShadow: 'hover:shadow-sky-500/10',
        sparkColor: '#0ea5e9',
    },
    'teal-500': {
        iconBg: 'bg-teal-500/15 dark:bg-teal-500/20',
        iconText: 'text-teal-500',
        topBar: 'bg-teal-500',
        glowShadow: 'hover:shadow-teal-500/10',
        sparkColor: '#14b8a6',
    },
    'slate-500': {
        iconBg: 'bg-slate-500/15 dark:bg-slate-500/20',
        iconText: 'text-slate-400',
        topBar: 'bg-slate-400',
        glowShadow: 'hover:shadow-slate-500/10',
        sparkColor: '#94a3b8',
    },
    'purple-500': {
        iconBg: 'bg-purple-500/15 dark:bg-purple-500/20',
        iconText: 'text-purple-500',
        topBar: 'bg-purple-500',
        glowShadow: 'hover:shadow-purple-500/10',
        sparkColor: '#a855f7',
    },
    'violet-500': {
        iconBg: 'bg-violet-500/15 dark:bg-violet-500/20',
        iconText: 'text-violet-500',
        topBar: 'bg-violet-500',
        glowShadow: 'hover:shadow-violet-500/10',
        sparkColor: '#8b5cf6',
    },
    'amber-500': {
        iconBg: 'bg-amber-500/15 dark:bg-amber-500/20',
        iconText: 'text-amber-500',
        topBar: 'bg-amber-500',
        glowShadow: 'hover:shadow-amber-500/10',
        sparkColor: '#f59e0b',
    },
    'emerald-500': {
        iconBg: 'bg-emerald-500/15 dark:bg-emerald-500/20',
        iconText: 'text-emerald-500',
        topBar: 'bg-emerald-500',
        glowShadow: 'hover:shadow-emerald-500/10',
        sparkColor: '#10b981',
    },
    'rose-500': {
        iconBg: 'bg-rose-500/15 dark:bg-rose-500/20',
        iconText: 'text-rose-500',
        topBar: 'bg-rose-500',
        glowShadow: 'hover:shadow-rose-500/10',
        sparkColor: '#f43f5e',
    },
    'cyan-500': {
        iconBg: 'bg-cyan-500/15 dark:bg-cyan-500/20',
        iconText: 'text-cyan-500',
        topBar: 'bg-cyan-500',
        glowShadow: 'hover:shadow-cyan-500/10',
        sparkColor: '#06b6d4',
    },
    'zinc-500': {
        iconBg: 'bg-zinc-500/15 dark:bg-zinc-500/20',
        iconText: 'text-zinc-400',
        topBar: 'bg-zinc-400',
        glowShadow: 'hover:shadow-zinc-500/10',
        sparkColor: '#71717a',
    },
};

export function MetricCard({
    title,
    value,
    subValue,
    isCurrency = true,
    isPercent = false,
    colorClass = '',
    valueClassName = 'text-xl sm:text-2xl',
    containerClassName = '',
    subValueClassName = '',
    currency = 'USD',
    isLoading = false,
    isRefreshing = false,
    icon: Icon,
    accentColor = 'indigo-500',
    variant = 'card',
    onClick,
    sparklineData
}: MetricCardProps) {
    const accent = ACCENT_MAP[accentColor] || ACCENT_MAP['indigo-500'];

    if (variant === 'seamless') {
        // Seamless variant: flat, no card chrome, used in the scalar metrics grid
        return (
            <div
                className={cn(
                    'metric-card card-shine relative overflow-hidden h-full min-h-[90px] p-4 cursor-default group',
                    onClick ? 'cursor-pointer' : '',
                    containerClassName
                )}
                onClick={onClick}
            >
                {/* Soft background glow */}
                <div className="absolute -top-12 -right-12 w-32 h-32 blur-[40px] opacity-10 transition-opacity duration-500 group-hover:opacity-20 pointer-events-none"
                    style={{ backgroundColor: accent.sparkColor }} />

                {/* Header row */}
                <div className="flex items-start justify-between mb-3 relative z-10">
                    <p className="section-label pr-2 leading-tight">{title}</p>
                    <div className="flex items-center gap-1.5 shrink-0">
                        {isRefreshing && !isLoading && (
                            <Loader2 className="w-2.5 h-2.5 animate-spin text-muted-foreground/40" />
                        )}
                        {Icon && (
                            <div className={cn(
                                'p-1.5 rounded-lg transition-all duration-300 group-hover:scale-110 backdrop-blur-sm',
                                accent.iconBg,
                                accent.iconText
                            )}>
                                <Icon className="w-3.5 h-3.5" />
                            </div>
                        )}
                    </div>
                </div>

                {/* Value row */}
                <div className="flex items-end gap-2 flex-wrap relative z-10">
                    <div className="flex flex-col min-w-0">
                        {isLoading ? (
                            <Skeleton className="h-8 w-28 mb-1 opacity-50 rounded-lg" />
                        ) : (
                            <span className={cn(
                                'font-bold tracking-tight leading-none tabular-nums text-foreground',
                                colorClass,
                                valueClassName
                            )}>
                                {value !== null && value !== undefined
                                    ? (isCurrency && typeof value === 'number' ? formatCurrency(value, currency) : value)
                                    : '—'}
                            </span>
                        )}
                    </div>

                    {isLoading ? (
                        <Skeleton className="h-5 w-12 rounded-full opacity-50" />
                    ) : subValue !== undefined && subValue !== null && (
                        <Badge
                            variant={(typeof subValue === 'number' ? subValue >= 0 : true) ? 'success' : 'destructive'}
                            className={cn(
                                'text-[10px] sm:text-xs font-bold px-2 py-0.5 rounded-full border-none shrink-0',
                                subValueClassName
                            )}
                        >
                            {typeof subValue === 'number'
                                ? (subValue === Infinity ? '∞' : `${subValue.toFixed(2)}%`)
                                : subValue}
                        </Badge>
                    )}
                </div>

                {/* Sparkline */}
                {!isLoading && sparklineData && sparklineData.length > 1 && (
                    <div className="absolute inset-x-0 bottom-0 h-12 z-0 pointer-events-none opacity-10 group-hover:opacity-20 transition-opacity duration-500">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={sparklineData}>
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke={accent.sparkColor}
                                    strokeWidth={2}
                                    dot={false}
                                    isAnimationActive={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </div>
        );
    }

    // Card variant (used for complex/tall metrics)
    return (
        <div
            className={cn(
                'metric-card card-shine relative overflow-hidden h-full p-5 cursor-default group',
                onClick ? 'cursor-pointer' : '',
                containerClassName
            )}
            onClick={onClick}
        >
            {/* Soft background glow */}
            <div className="absolute -top-16 -right-16 w-48 h-48 blur-[60px] opacity-8 transition-opacity duration-500 group-hover:opacity-15 pointer-events-none"
                style={{ backgroundColor: accent.sparkColor }} />

            {/* Header */}
            <div className="flex items-start justify-between mb-4 relative z-10">
                <div className="flex items-center gap-1.5">
                    <p className="section-label">{title}</p>
                    {isRefreshing && !isLoading && (
                        <Loader2 className="w-2.5 h-2.5 animate-spin text-muted-foreground/40" />
                    )}
                </div>
                {Icon && (
                    <div className={cn(
                        'p-2 rounded-xl transition-all duration-300 group-hover:scale-110 group-hover:rotate-3 backdrop-blur-sm',
                        accent.iconBg,
                        accent.iconText
                    )}>
                        <Icon className="w-4 h-4" />
                    </div>
                )}
            </div>

            {/* Value */}
            <div className="flex items-end justify-between gap-3 relative z-10">
                <div className="flex flex-col min-w-0">
                    {isLoading ? (
                        <Skeleton className="h-9 w-32 mb-1 opacity-50 rounded-lg" />
                    ) : (
                        <span className={cn(
                            'font-bold tracking-tight leading-none tabular-nums text-foreground',
                            colorClass,
                            valueClassName
                        )}>
                            {value !== null && value !== undefined
                                ? (isCurrency && typeof value === 'number' ? formatCurrency(value, currency) : value)
                                : '—'}
                        </span>
                    )}
                </div>

                {isLoading ? (
                    <Skeleton className="h-5 w-12 rounded-full opacity-50" />
                ) : subValue !== undefined && subValue !== null && (
                    <Badge
                        variant={(typeof subValue === 'number' ? subValue >= 0 : true) ? 'success' : 'destructive'}
                        className={cn(
                            'text-[10px] sm:text-xs font-bold px-2 py-0.5 rounded-full border-none shrink-0',
                            subValueClassName
                        )}
                    >
                        {typeof subValue === 'number'
                            ? (subValue === Infinity ? '∞' : `${subValue.toFixed(2)}%`)
                            : subValue}
                    </Badge>
                )}
            </div>

            {/* Sparkline */}
            {!isLoading && sparklineData && sparklineData.length > 1 && (
                <div className="absolute inset-x-0 bottom-0 h-16 z-0 pointer-events-none opacity-8 group-hover:opacity-15 transition-opacity duration-500">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={sparklineData}>
                            <Line
                                type="monotone"
                                dataKey="value"
                                stroke={accent.sparkColor}
                                strokeWidth={3}
                                dot={false}
                                isAnimationActive={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
}
