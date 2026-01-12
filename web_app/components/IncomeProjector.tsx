'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { ProjectedIncome } from '../lib/api';
import { Skeleton } from "@/components/ui/skeleton";
import { formatCurrency } from '../lib/utils';

interface IncomeProjectorProps {
    data: ProjectedIncome[] | null;
    isLoading: boolean;
    currency: string;
}

export function IncomeProjector({ data, isLoading, currency }: IncomeProjectorProps) {
    // Generate unique colors for each symbol dynamically
    const colors = [
        "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899", "#f43f5e",
        "#f59e0b", "#10b981", "#6366f1", "#14b8a6", "#f97316"
    ];

    // Memoize keys to prevent re-renders and ensure consistent order
    const stackedKeys = React.useMemo(() => {
        if (!data) return [];
        const keys = new Set<string>();
        data.forEach(item => {
            Object.keys(item).forEach(k => {
                if (k !== 'month' && k !== 'value' && k !== 'year_month') {
                    keys.add(k);
                }
            });
        });
        return Array.from(keys).sort(); // Sort for consistent color assignment
    }, [data]);

    if (isLoading) {
        return <Skeleton className="h-[300px] w-full rounded-xl mb-6 bg-muted/10" />;
    }

    if (!data || data.length === 0) {
        return null;
    }

    const totalProjected = data.reduce((sum, item) => sum + item.value, 0);

    return (
        <Card className="bg-card border border-border overflow-hidden mb-6 shadow-sm">
            <CardHeader className="flex flex-row items-center justify-between pb-2 px-6 pt-6 space-y-0">
                <div>
                    <CardTitle className="text-lg font-semibold text-foreground">
                        Projected 12M Income
                    </CardTitle>
                    <p className="text-sm text-muted-foreground mt-1">
                        Est. Total: <span className="text-emerald-500 font-bold">{formatCurrency(totalProjected, currency)}</span>
                    </p>
                </div>
            </CardHeader>
            <CardContent className="h-[300px] w-full pl-0 pb-2">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(128,128,128,0.1)" />
                        <XAxis
                            dataKey="month"
                            tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                            axisLine={false}
                            tickLine={false}
                            dy={10}
                        />
                        <YAxis
                            tickFormatter={(val) => new Intl.NumberFormat('en-US', { notation: "compact", compactDisplay: "short" }).format(val)}
                            tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
                            axisLine={false}
                            tickLine={false}
                            width={40}
                        />
                        <Tooltip
                            cursor={{ fill: 'var(--muted)', opacity: 0.1 }}
                            contentStyle={{
                                backgroundColor: 'var(--menu-solid)',
                                borderRadius: '12px',
                                border: '1px solid var(--border)',
                                color: 'var(--foreground)'
                            }}
                            content={({ active, payload, label }) => {
                                if (active && payload && payload.length) {
                                    // Sort payload by value desc for better reading in tooltip
                                    const sortedPayload = [...payload].sort((a, b) => (b.value as number) - (a.value as number));
                                    return (
                                        <div className="border border-border p-3 rounded-lg shadow-xl text-xs min-w-[180px]" style={{ backgroundColor: 'var(--menu-solid)' }}>
                                            <p className="font-semibold mb-2 text-foreground">{label}</p>
                                            {sortedPayload.map((entry, idx) => (
                                                <div key={`tooltip-${idx}`} className="flex items-center gap-3 mb-1 last:mb-0">
                                                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }}></div>
                                                    <span className="text-muted-foreground">{entry.name}:</span>
                                                    <span className="font-mono text-emerald-500 font-bold ml-auto">
                                                        {formatCurrency(entry.value as number, currency)}
                                                    </span>
                                                </div>
                                            ))}
                                            <div className="border-t border-border mt-2 pt-2 flex justify-between font-bold">
                                                <span>Total</span>
                                                <span>{formatCurrency(payload.reduce((s, p) => s + (p.value as number), 0), currency)}</span>
                                            </div>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        {stackedKeys.map((key, index) => (
                            <Bar
                                key={key}
                                dataKey={key}
                                stackId="a"
                                fill={colors[index % colors.length]}
                                stroke="none"
                                radius={[index === stackedKeys.length - 1 ? 6 : 0, index === stackedKeys.length - 1 ? 6 : 0, 0, 0]}
                                isAnimationActive={false} // Disable animation to prevent initial black flicker
                                name={key}
                            />
                        ))}
                    </BarChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    );
}
