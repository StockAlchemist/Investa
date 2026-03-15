import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
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
    accentColor = 'cyan-500',
    variant = 'card',
    onClick,
    sparklineData
}: MetricCardProps) {
    const colorMap: Record<string, { textHover: string; bgHover: string; accent: string }> = {
        'indigo-500': { textHover: 'group-hover:text-indigo-500', bgHover: 'group-hover:bg-indigo-500/5', accent: 'bg-indigo-500/10' },
        'sky-500': { textHover: 'group-hover:text-sky-500', bgHover: 'group-hover:bg-sky-500/5', accent: 'bg-sky-500/10' },
        'teal-500': { textHover: 'group-hover:text-teal-500', bgHover: 'group-hover:bg-teal-500/5', accent: 'bg-teal-500/10' },
        'slate-500': { textHover: 'group-hover:text-slate-500', bgHover: 'group-hover:bg-slate-500/5', accent: 'bg-slate-500/10' },
        'purple-500': { textHover: 'group-hover:text-purple-500', bgHover: 'group-hover:bg-purple-500/5', accent: 'bg-purple-500/10' },
        'violet-500': { textHover: 'group-hover:text-violet-500', bgHover: 'group-hover:bg-violet-500/5', accent: 'bg-violet-500/10' },
        'amber-500': { textHover: 'group-hover:text-amber-500', bgHover: 'group-hover:bg-amber-500/5', accent: 'bg-amber-500/10' },
        'emerald-500': { textHover: 'group-hover:text-emerald-500', bgHover: 'group-hover:bg-emerald-500/5', accent: 'bg-emerald-500/10' },
        'rose-500': { textHover: 'group-hover:text-rose-500', bgHover: 'group-hover:bg-rose-500/5', accent: 'bg-rose-500/10' },
        'cyan-500': { textHover: 'group-hover:text-cyan-500', bgHover: 'group-hover:bg-cyan-500/5', accent: 'bg-cyan-500/10' },
        'zinc-500': { textHover: 'group-hover:text-zinc-500', bgHover: 'group-hover:bg-zinc-500/5', accent: 'bg-zinc-500/10' }
    };

    const activeClasses = colorMap[accentColor] || colorMap['cyan-500'];

    return (
        <Card
            className={cn(
                "h-full transition-all duration-300 relative overflow-hidden group rounded-2xl border-none",
                variant === 'card' && "bg-card hover:shadow-sm",
                variant === 'seamless' && "bg-transparent shadow-none hover:bg-muted/30",
                onClick ? "cursor-pointer active:scale-[0.98]" : "",
                containerClassName
            )}
            onClick={onClick}
        >
            <CardContent className={cn(
                "h-full flex flex-col justify-start gap-1 p-3 relative",
                variant === 'seamless' && "p-1"
            )}>
                <div className="flex justify-between items-start z-10">
                    <div className="flex items-center gap-2">
                        <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest opacity-80">{title}</p>
                        {isRefreshing && !isLoading && (
                            <Loader2 className="w-3 h-3 animate-spin text-cyan-500 opacity-70" />
                        )}
                    </div>
                    {Icon && (
                        <div className={cn(
                            "p-2 rounded-xl transition-all duration-300",
                            "group-hover:scale-105",
                            activeClasses.accent,
                            activeClasses.textHover
                        )}>
                            <Icon className="w-4 h-4" />
                        </div>
                    )}
                </div>

                <div className="flex items-end justify-between gap-2 relative z-10">
                    <div className="flex flex-col">
                        {isLoading ? (
                            <Skeleton className="h-9 w-32 mb-1 opacity-50" />
                        ) : (
                            <h3 className={cn("font-bold tracking-tight leading-none text-foreground", colorClass, valueClassName)}>
                                {value !== null && value !== undefined ? (isCurrency && typeof value === 'number' ? formatCurrency(value, currency) : value) : '-'}
                            </h3>
                        )}
                    </div>

                    {isLoading ? (
                        <Skeleton className="h-5 w-12 rounded-full opacity-50" />
                    ) : subValue !== undefined && subValue !== null && (
                        <Badge
                            variant={(typeof subValue === 'number' ? subValue >= 0 : true) ? "success" : "destructive"}
                            className={cn(
                                "text-[10px] sm:text-xs font-bold px-2 py-0.5 rounded-full border-none",
                                subValueClassName
                            )}
                        >
                            {typeof subValue === 'number' ? (
                                `${subValue > 0 ? '+' : ''}${subValue.toFixed(2)}%`
                            ) : subValue}
                        </Badge>
                    )}
                </div>

                {!isLoading && sparklineData && sparklineData.length > 1 && (
                    <div className="absolute inset-x-0 bottom-0 h-16 z-0 pointer-events-none opacity-5 group-hover:opacity-10 transition-all duration-500">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={sparklineData}>
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke="currentColor"
                                    strokeWidth={3}
                                    dot={false}
                                    isAnimationActive={true}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

// Add a helper for mask-linear-fade if not in global css, but usually it's fine without or added in globals.css
// .mask-linear-fade { mask-image: linear-gradient(to top, black, transparent); }
