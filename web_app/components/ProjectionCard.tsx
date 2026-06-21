import React, { useMemo } from 'react';
import { Loader2, TrendingUp } from 'lucide-react';
import {
    ComposedChart,
    Area,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from 'recharts';

import { Projection } from '../lib/api';
import { formatCurrency, formatPercent } from '../lib/utils';

interface ProjectionCardProps {
    data?: Projection;
    isLoading?: boolean;
    isRefreshing?: boolean;
    currency: string;
}

function compactCurrency(value: number, currency: string): string {
    const abs = Math.abs(value);
    if (abs >= 1e9) return `${formatCurrency(value / 1e9, currency)}B`.replace(/(\.\d*?)0+B$/, '$1B').replace(/\.B$/, 'B');
    if (abs >= 1e6) return `${formatCurrency(value / 1e6, currency)}M`.replace(/(\.\d*?)0+M$/, '$1M').replace(/\.M$/, 'M');
    if (abs >= 1e3) return `${formatCurrency(value / 1e3, currency)}K`.replace(/(\.\d*?)0+K$/, '$1K').replace(/\.K$/, 'K');
    return formatCurrency(value, currency);
}

export default function ProjectionCard({ data, isLoading, isRefreshing, currency }: ProjectionCardProps) {
    const cur = data?.currency || currency;

    const chartData = useMemo(() => {
        if (!data?.available || !data.horizons) return [];
        return data.horizons.map(h => ({
            label: `${h.years}Y`,
            median: h.median_value,
            band90: [h.p10, h.p90] as [number, number],
            band50: [h.p25, h.p75] as [number, number],
        }));
    }, [data]);

    return (
        <div className="metric-card card-shine p-4 sm:p-6 mb-6 overflow-visible relative">
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-indigo-500 opacity-80" />

            <div className="flex flex-col gap-1 md:flex-row md:justify-between md:items-center mb-4">
                <h3 className="text-base font-bold text-foreground/90 flex items-center gap-2 tracking-tight">
                    <TrendingUp className="w-4 h-4 text-indigo-500" />
                    Projected Value
                    {isRefreshing && <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-500 opacity-70" />}
                </h3>
                {data?.available && (
                    <div className="flex items-baseline gap-4 text-sm">
                        <span className="text-muted-foreground">
                            Assumed return{' '}
                            <span className="font-bold text-foreground">{formatPercent(data.annual_return_pct ?? 0)}</span>/yr
                        </span>
                        <span className="text-muted-foreground">
                            Volatility{' '}
                            <span className="font-bold text-foreground">{formatPercent(data.annual_volatility_pct ?? 0)}</span>
                        </span>
                    </div>
                )}
            </div>

            {isLoading && !data ? (
                <div className="h-[280px] flex items-center justify-center text-muted-foreground">
                    <Loader2 className="w-5 h-5 animate-spin" />
                </div>
            ) : !data?.available ? (
                <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground text-center px-6">
                    Not enough history yet to project a return. Projections appear once the
                    portfolio has a longer track record.
                </div>
            ) : (
                <>
                    <div className="h-[280px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={chartData} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" vertical={false} />
                                <XAxis dataKey="label" tickLine={false} axisLine={false} className="text-xs" />
                                <YAxis
                                    tickFormatter={(v) => compactCurrency(v as number, cur)}
                                    tickLine={false}
                                    axisLine={false}
                                    width={64}
                                    className="text-xs"
                                />
                                <Tooltip
                                    formatter={(value, name) => {
                                        if (Array.isArray(value)) {
                                            return [`${formatCurrency(value[0], cur)} – ${formatCurrency(value[1], cur)}`, name === 'band90' ? '10–90%' : '25–75%'];
                                        }
                                        return [formatCurrency(value as number, cur), 'Median'];
                                    }}
                                    labelFormatter={(l) => `In ${l}`}
                                    contentStyle={{ background: 'var(--background)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                                />
                                <Area dataKey="band90" stroke="none" fill="#6366f1" fillOpacity={0.12} isAnimationActive={false} />
                                <Area dataKey="band50" stroke="none" fill="#6366f1" fillOpacity={0.22} isAnimationActive={false} />
                                <Line dataKey="median" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="mt-4 overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-[11px] uppercase tracking-wider text-muted-foreground text-right">
                                    <th className="text-left font-semibold py-1.5">Horizon</th>
                                    <th className="font-semibold py-1.5">Median value</th>
                                    <th className="font-semibold py-1.5">Return</th>
                                    <th className="font-semibold py-1.5">Range (10–90%)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.horizons!.map(h => (
                                    <tr key={h.years} className="border-t border-border/40 text-right">
                                        <td className="text-left font-semibold py-2">{h.years} {h.years === 1 ? 'year' : 'years'}</td>
                                        <td className="py-2 font-bold tabular-nums">{formatCurrency(h.median_value, cur)}</td>
                                        <td className={`py-2 font-semibold tabular-nums ${h.median_return_pct >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}`}>
                                            {h.median_return_pct >= 0 ? '+' : ''}{formatPercent(h.median_return_pct)}
                                        </td>
                                        <td className="py-2 text-muted-foreground tabular-nums">
                                            {compactCurrency(h.p10, cur)} – {compactCurrency(h.p90, cur)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <p className="mt-3 text-[11px] text-muted-foreground leading-relaxed">
                        Projections compound the portfolio&apos;s historical annualized return and
                        volatility forward (lognormal model). The median is the central estimate;
                        the shaded band shows the 10th–90th percentile range. Past performance does
                        not guarantee future results — longer horizons are far more uncertain.
                    </p>
                </>
            )}
        </div>
    );
}
