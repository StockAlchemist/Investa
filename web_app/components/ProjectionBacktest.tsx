'use client';
import React, { useMemo } from 'react';
import { Loader2 } from 'lucide-react';
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

import { ProjectionBacktest as ProjectionBacktestData } from '../lib/api';
import { formatCurrency, formatPercent } from '../lib/utils';
import { formatCalendarDate } from '../lib/market_time';

interface ProjectionBacktestProps {
    data?: ProjectionBacktestData;
    isLoading?: boolean;
    error?: boolean;
    currency: string;
    /** Compact currency formatter shared with the forecast view. */
    compact: (value: number, currency: string) => string;
}

const ACTUAL_COLOR = '#059669';   // emerald — the path that actually happened
const MODEL_COLOR = '#6366f1';    // indigo — the cone the model drew back then

const VERDICTS: Record<string, { label: string; className: string }> = {
    calibrated: { label: 'Well calibrated', className: 'text-up' },
    narrow: { label: 'Bands too narrow', className: 'text-down' },
    wide: { label: 'Bands conservative', className: 'text-muted-foreground' },
};

/** `MMM yyyy` — a monthly replay point has no day of its own worth showing. */
function monthLabel(iso: string): string {
    return formatCalendarDate(iso, { month: 'short', year: 'numeric' });
}

function years(n: number): string {
    return `${n} ${n === 1 ? 'year' : 'years'}`;
}

export default function ProjectionBacktest({ data, isLoading, error, currency, compact }: ProjectionBacktestProps) {
    const cur = data?.currency || currency;
    const replay = data?.replay ?? null;
    const money = (v: number) => (replay?.indexed ? v.toFixed(1) : formatCurrency(v, cur));

    const chartData = useMemo(() => {
        if (!replay) return [];
        return replay.points.map(p => ({
            date: p.date,
            actual: p.actual,
            median: p.median,
            band90: [p.p10, p.p90] as [number, number],
            band50: [p.p25, p.p75] as [number, number],
        }));
    }, [replay]);

    // Ticks roughly yearly, always including the anchor and the last point.
    const ticks = useMemo(() => {
        if (chartData.length < 2) return undefined;
        const step = Math.max(1, Math.round((chartData.length - 1) / 5));
        const out: string[] = [];
        for (let i = 0; i < chartData.length - 1; i += step) out.push(chartData[i].date);
        out.push(chartData[chartData.length - 1].date);
        return out;
    }, [chartData]);

    if (isLoading) {
        return (
            <div className="h-[280px] flex items-center justify-center text-muted-foreground">
                <Loader2 className="w-5 h-5 animate-spin" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground text-center px-6">
                Couldn&apos;t run the backtest just now. Try again in a moment.
            </div>
        );
    }

    if (!data?.available) {
        const need = data?.required_years;
        const has = data?.history_years;
        return (
            <div className="h-[200px] flex items-center justify-center text-sm text-muted-foreground text-center px-6">
                {need
                    ? `Backtesting needs about ${need.toFixed(0)} years of history — enough to fit the model on the past and still have years left to check it against. This portfolio has ${has?.toFixed(1)}.`
                    : 'Not enough history yet to backtest the projection.'}
            </div>
        );
    }

    const outcomeLine = replay && (
        replay.outcome === 'inside'
            ? 'The actual path finished inside the 10–90% band the model drew back then.'
            : replay.outcome === 'below'
                ? 'The actual path finished below the 10th percentile — the model was too optimistic over this stretch.'
                : 'The actual path finished above the 90th percentile — the model was too cautious over this stretch.'
    );

    return (
        <>
            {replay && (
                <>
                    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-sm mb-1">
                        <span className="font-semibold text-foreground">
                            {years(Math.round(replay.years))} from {formatCalendarDate(replay.anchor_date)}
                        </span>
                        <span className="text-muted-foreground">
                            Projected median{' '}
                            <span className="font-bold text-foreground tabular-nums">{money(replay.final_median)}</span>
                        </span>
                        <span className="text-muted-foreground">
                            Actual{' '}
                            <span className="font-bold tabular-nums" style={{ color: ACTUAL_COLOR }}>{money(replay.final_actual)}</span>
                        </span>
                        <span className="text-muted-foreground">
                            Band{' '}
                            <span className="tabular-nums">{compact(replay.final_p10, cur)} – {compact(replay.final_p90, cur)}</span>
                        </span>
                    </div>
                    <p className="text-[11px] text-muted-foreground mb-3">{outcomeLine}</p>

                    <div className="h-[260px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={chartData} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" className="stroke-border/40" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    ticks={ticks}
                                    tickFormatter={monthLabel}
                                    tickLine={false}
                                    axisLine={false}
                                    className="text-xs"
                                />
                                <YAxis
                                    tickFormatter={(v) => (replay.indexed ? String(v) : compact(v as number, cur))}
                                    tickLine={false}
                                    axisLine={false}
                                    width={64}
                                    className="text-xs"
                                />
                                <Tooltip
                                    formatter={(value, name) => {
                                        if (Array.isArray(value)) {
                                            return [`${money(value[0])} – ${money(value[1])}`, name === 'band90' ? '10–90%' : '25–75%'];
                                        }
                                        return [money(value as number), name === 'actual' ? 'Actual' : 'Projected median'];
                                    }}
                                    labelFormatter={(iso) => formatCalendarDate(iso as string)}
                                    contentStyle={{ background: 'var(--background)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                                />
                                <Area dataKey="band90" stroke="none" fill={MODEL_COLOR} fillOpacity={0.12} isAnimationActive={false} />
                                <Area dataKey="band50" stroke="none" fill={MODEL_COLOR} fillOpacity={0.22} isAnimationActive={false} />
                                <Line dataKey="median" stroke={MODEL_COLOR} strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                                <Line dataKey="actual" stroke={ACTUAL_COLOR} strokeWidth={2.5} dot={false} isAnimationActive={false} connectNulls />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="flex flex-wrap items-center gap-4 mt-2 text-[11px] text-muted-foreground">
                        <span className="flex items-center gap-1.5">
                            <span className="inline-block w-4 h-0.5" style={{ background: ACTUAL_COLOR }} />
                            Actual (time-weighted, no later deposits)
                        </span>
                        <span className="flex items-center gap-1.5">
                            <span className="inline-block w-4 h-0.5 border-t-2 border-dashed" style={{ borderColor: MODEL_COLOR }} />
                            Projected median
                        </span>
                        <span className="flex items-center gap-1.5">
                            <span className="inline-block w-4 h-2 rounded-sm" style={{ background: MODEL_COLOR, opacity: 0.18 }} />
                            10–90% band
                        </span>
                    </div>
                </>
            )}

            <div className="mt-4 overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="text-[11px] uppercase tracking-wider text-muted-foreground text-right">
                            <th className="text-left font-semibold py-1.5">Horizon</th>
                            <th className="font-semibold py-1.5">Checks</th>
                            <th className="font-semibold py-1.5">Inside 10–90%</th>
                            <th className="font-semibold py-1.5 hidden sm:table-cell">Below 10%</th>
                            <th className="font-semibold py-1.5 hidden md:table-cell">Median projected</th>
                            <th className="font-semibold py-1.5 hidden md:table-cell">Median actual</th>
                            <th className="font-semibold py-1.5">Verdict</th>
                        </tr>
                    </thead>
                    <tbody>
                        {(data.horizons ?? []).map(h => {
                            const verdict = VERDICTS[h.verdict] ?? VERDICTS.calibrated;
                            return (
                                <tr key={h.years} className="border-t border-border/40 text-right">
                                    <td className="text-left font-semibold py-2">{years(h.years)}</td>
                                    <td className="py-2 tabular-nums text-muted-foreground">{h.samples}</td>
                                    <td className="py-2 font-bold tabular-nums">{h.in_band_pct.toFixed(0)}%</td>
                                    <td className="py-2 tabular-nums text-muted-foreground hidden sm:table-cell">{h.below_p10_pct.toFixed(0)}%</td>
                                    <td className="py-2 tabular-nums text-muted-foreground hidden md:table-cell">
                                        {formatPercent(h.median_projected_return_pct / 100)}
                                    </td>
                                    <td className="py-2 tabular-nums font-semibold hidden md:table-cell" style={{ color: ACTUAL_COLOR }}>
                                        {formatPercent(h.median_actual_return_pct / 100)}
                                    </td>
                                    <td className={`py-2 font-semibold ${verdict.className}`}>{verdict.label}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            <p className="mt-3 text-[11px] text-muted-foreground leading-relaxed">
                Walk-forward test on this portfolio&apos;s own history
                {data.history_start && data.history_end
                    ? ` (${formatCalendarDate(data.history_start)} – ${formatCalendarDate(data.history_end)})`
                    : ''}: at each month in the past the model is refitted on the data that existed
                <em> then</em> — never later — and its cone is scored against what followed. &ldquo;Inside 10–90%&rdquo;
                should come out near 80%: much less and the bands are too narrow to trust, much more and
                they are wider than they need to be. Returns are time-weighted, so deposits and withdrawals
                after each start date don&apos;t flatter the result.
            </p>
        </>
    );
}
