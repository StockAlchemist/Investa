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
    onClick
}: MetricCardProps & { onClick?: () => void }) {
    return (
        <Card
            className={cn(
                "h-full transition-all duration-300 relative overflow-hidden group",
                "hover:bg-accent/5 hover:shadow-md",
                onClick ? "cursor-pointer active:scale-[0.98]" : "",
                containerClassName
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
                            "group-hover:text-cyan-500 group-hover:bg-cyan-500/10"
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
