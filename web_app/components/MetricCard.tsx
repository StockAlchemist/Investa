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
    vertical?: boolean;
    sparklineData?: { value: number }[];
    isLoading?: boolean;
    isRefreshing?: boolean;
    icon?: LucideIcon;
    accentColor?: string;
}

export function MetricCard({
    title,
    value,
    subValue,
    isCurrency = true,
    colorClass = '',
    valueClassName = 'text-xl sm:text-2xl',
    containerClassName = '',
    subValueClassName = '',
    vertical = false,
    sparklineData,
    currency = 'USD',
    isLoading = false,
    isRefreshing = false,
    icon: Icon,
    accentColor = 'cyan-500',
    onClick
}: MetricCardProps & { onClick?: () => void }) {
    const colorMap: Record<string, { borderHover: string; bgGlow: string; textHover: string; bgHover: string }> = {
        'indigo-500': { borderHover: 'hover:border-r-indigo-500', bgGlow: 'bg-gradient-to-br from-indigo-500/5 via-transparent to-transparent dark:from-indigo-500/10', textHover: 'group-hover:text-indigo-500', bgHover: 'group-hover:bg-indigo-500/10' },
        'sky-500': { borderHover: 'hover:border-r-sky-500', bgGlow: 'bg-gradient-to-br from-sky-500/5 via-transparent to-transparent dark:from-sky-500/10', textHover: 'group-hover:text-sky-500', bgHover: 'group-hover:bg-sky-500/10' },
        'teal-500': { borderHover: 'hover:border-r-teal-500', bgGlow: 'bg-gradient-to-br from-teal-500/5 via-transparent to-transparent dark:from-teal-500/10', textHover: 'group-hover:text-teal-500', bgHover: 'group-hover:bg-teal-500/10' },
        'slate-500': { borderHover: 'hover:border-r-slate-500', bgGlow: 'bg-gradient-to-br from-slate-500/5 via-transparent to-transparent dark:from-slate-500/10', textHover: 'group-hover:text-slate-500', bgHover: 'group-hover:bg-slate-500/10' },
        'purple-500': { borderHover: 'hover:border-r-purple-500', bgGlow: 'bg-gradient-to-br from-purple-500/5 via-transparent to-transparent dark:from-purple-500/10', textHover: 'group-hover:text-purple-500', bgHover: 'group-hover:bg-purple-500/10' },
        'violet-500': { borderHover: 'hover:border-r-violet-500', bgGlow: 'bg-gradient-to-br from-violet-500/5 via-transparent to-transparent dark:from-violet-500/10', textHover: 'group-hover:text-violet-500', bgHover: 'group-hover:bg-violet-500/10' },
        'amber-500': { borderHover: 'hover:border-r-amber-500', bgGlow: 'bg-gradient-to-br from-amber-500/5 via-transparent to-transparent dark:from-amber-500/10', textHover: 'group-hover:text-amber-500', bgHover: 'group-hover:bg-amber-500/10' },
        'emerald-500': { borderHover: 'hover:border-r-emerald-500', bgGlow: 'bg-gradient-to-br from-emerald-500/5 via-transparent to-transparent dark:from-emerald-500/10', textHover: 'group-hover:text-emerald-500', bgHover: 'group-hover:bg-emerald-500/10' },
        'rose-500': { borderHover: 'hover:border-r-rose-500', bgGlow: 'bg-gradient-to-br from-rose-500/5 via-transparent to-transparent dark:from-rose-500/10', textHover: 'group-hover:text-rose-500', bgHover: 'group-hover:bg-rose-500/10' },
        'cyan-500': { borderHover: 'hover:border-r-cyan-500', bgGlow: 'bg-gradient-to-br from-cyan-500/5 via-transparent to-transparent dark:from-cyan-500/10', textHover: 'group-hover:text-cyan-500', bgHover: 'group-hover:bg-cyan-500/10' },
        'zinc-500': { borderHover: 'hover:border-r-zinc-500', bgGlow: 'bg-gradient-to-br from-zinc-500/5 via-transparent to-transparent dark:from-zinc-500/10', textHover: 'group-hover:text-zinc-500', bgHover: 'group-hover:bg-zinc-500/10' }
    };

    const activeClasses = colorMap[accentColor] || colorMap['cyan-500'];

    return (
        <Card
            className={cn(
                "h-full transition-all duration-300 relative overflow-hidden group border-r-2",
                onClick ? "cursor-pointer active:scale-[0.98]" : "",
                containerClassName,
                activeClasses.borderHover,
                `border-r-transparent`,
                activeClasses.bgGlow
            )}
            onClick={onClick}
        >
            <CardContent className="h-full flex flex-col justify-between p-4 relative">
                <div className="flex justify-between items-start z-10">
                    <div className="flex items-center gap-2">
                        <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest">{title}</p>
                        {isRefreshing && !isLoading && (
                            <Loader2 className="w-3 h-3 animate-spin text-cyan-500 opacity-70" />
                        )}
                    </div>
                    {Icon && (
                        <div className={cn(
                            "absolute top-3 right-3 p-1.5 rounded-lg bg-secondary/50 text-muted-foreground transition-all duration-300",
                            activeClasses.textHover,
                            activeClasses.bgHover
                        )}>
                            <Icon className="w-4 h-4" />
                        </div>
                    )}
                </div>

                <div className="mt-2 flex items-end justify-between gap-2 relative z-10">
                    <div className="flex flex-col">
                        {isLoading ? (
                            <Skeleton className="h-8 w-32 mb-1" />
                        ) : (
                            <h3 className={cn("font-bold tracking-tight leading-none", colorClass || "text-foreground", valueClassName)}>
                                {value !== null && value !== undefined ? (isCurrency && typeof value === 'number' ? formatCurrency(value, currency) : value) : '-'}
                            </h3>
                        )}
                    </div>

                    {isLoading ? (
                        <Skeleton className="h-5 w-12 rounded-full" />
                    ) : subValue !== undefined && subValue !== null && (
                        <Badge
                            variant={(typeof subValue === 'number' ? subValue >= 0 : true) ? "success" : "destructive"}
                            className={cn("text-[10px] sm:text-xs font-bold px-1.5 py-0.5", subValueClassName)}
                        >
                            {typeof subValue === 'number' ? (
                                `${subValue > 0 ? '+' : ''}${subValue.toFixed(2)}%`
                            ) : subValue}
                        </Badge>
                    )}
                </div>

                {!isLoading && sparklineData && sparklineData.length > 1 && (
                    <div className="absolute inset-x-0 bottom-0 h-16 z-0 pointer-events-none opacity-20 group-hover:opacity-40 transition-opacity mask-linear-fade">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={sparklineData}>
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke={(typeof subValue === 'number' ? subValue >= 0 : true) ? "#10b981" : "#ef4444"}
                                    strokeWidth={3}
                                    dot={false}
                                    isAnimationActive={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {isLoading && (
                    <Skeleton className="absolute inset-0 opacity-10" />
                )}
            </CardContent>
        </Card>
    );
}

// Add a helper for mask-linear-fade if not in global css, but usually it's fine without or added in globals.css
// .mask-linear-fade { mask-image: linear-gradient(to top, black, transparent); }
