import React from 'react';
import {
    ComposedChart,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    ReferenceLine,
    Bar,
    ResponsiveContainer
} from 'recharts';
import { ChartPoint, ChartSeries } from '../types';
import {
    StatementPeriod,
    formatStatementValue,
    isFiniteNumber
} from '../../../lib/statement_chart';
import { formatCalendarDate } from '../../../lib/market_time';
import { cn } from '../../../lib/utils';

export function RoundedBar(props: {
    x?: number; y?: number; width?: number; height?: number; fill?: string; value?: number;
}) {
    const { x = 0, y = 0, width = 0, height = 0, fill } = props;
    if (!width || !height) return null;
    const negative = (props.value ?? 0) < 0;
    const r = Math.max(0, Math.min(4, width / 2, height));
    const path = negative
        ? `M${x},${y} L${x + width},${y} L${x + width},${y + height - r} Q${x + width},${y + height} ${x + width - r},${y + height} L${x + r},${y + height} Q${x},${y + height} ${x},${y + height - r} Z`
        : `M${x},${y + height} L${x},${y + r} Q${x},${y} ${x + r},${y} L${x + width - r},${y} Q${x + width},${y} ${x + width},${y + r} L${x + width},${y + height} Z`;
    return <path d={path} fill={fill} />;
}

export function StatementTooltip({ active, payload, series }: {
    active?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Recharts' tooltip payload shape
    payload?: any[];
    series: ChartSeries[];
}) {
    if (!active || !payload || !payload.length) return null;
    const period = payload[0]?.payload?.period as string | undefined;
    return (
        <div className="bg-white/95 dark:bg-zinc-950/95 backdrop-blur-md rounded-xl px-3 py-2 shadow-lg border border-black/5 dark:border-white/10">
            <p className="text-[11px] font-medium text-muted-foreground mb-1.5">
                {period ? formatCalendarDate(period) : ''}
            </p>
            <div className="space-y-1">
                {series.map(s => {
                    const entry = payload.find(p => p.dataKey === s.key);
                    const value = entry?.value;
                    return (
                        <div key={s.key} className="flex items-center justify-between gap-6">
                            <span className="flex items-center gap-2 text-xs text-muted-foreground">
                                <span className="w-3 h-0.5 rounded-full flex-shrink-0" style={{ backgroundColor: s.color }} />
                                {s.label}
                            </span>
                            <span className="text-sm font-bold tabular-nums text-foreground">
                                {isFiniteNumber(value) ? formatStatementValue(value) : '—'}
                            </span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

export function StatementChart({ points, series, periodType }: {
    points: ChartPoint[];
    series: ChartSeries[];
    periodType: StatementPeriod;
}) {
    const hasNegative = points.some(p => series.some(s => isFiniteNumber(p[s.key]) && (p[s.key] as number) < 0));

    return (
        <div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 mb-2">
                {series.map(s => (
                    <span key={s.key} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                        <span className="w-2.5 h-2.5 rounded-[2px] flex-shrink-0" style={{ backgroundColor: s.color }} />
                        {s.label}
                    </span>
                ))}
            </div>
            <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={points} margin={{ top: 8, right: 4, left: 0, bottom: 0 }} barGap={2} barCategoryGap="10%">
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" opacity={0.1} vertical={false} />
                        <XAxis
                            dataKey="label"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fontSize: 10 }}
                            className="fill-muted-foreground"
                            interval="preserveStartEnd"
                            minTickGap={periodType === 'quarterly' ? 4 : 12}
                        />
                        <YAxis
                            axisLine={false}
                            tickLine={false}
                            tick={{ fontSize: 10 }}
                            width={56}
                            className="fill-muted-foreground"
                            tickFormatter={(val) => formatStatementValue(Number(val))}
                        />
                        {hasNegative && <ReferenceLine y={0} stroke="currentColor" className="text-border" strokeWidth={1} />}
                        <Tooltip
                            wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                            cursor={{ fill: 'currentColor', className: 'text-muted-foreground', opacity: 0.08 }}
                            content={<StatementTooltip series={series} />}
                        />
                        {series.map(s => (
                            <Bar
                                key={s.key}
                                dataKey={s.key}
                                name={s.label}
                                fill={s.color}
                                shape={<RoundedBar />}
                                animationDuration={600}
                            />
                        ))}
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

export function MetricChangeStrip({ label, color, points, seriesKey, periodType }: {
    label: string;
    color: string;
    points: ChartPoint[];
    seriesKey: string;
    periodType: StatementPeriod;
}) {
    const values = points.map(p => (isFiniteNumber(p[seriesKey]) ? (p[seriesKey] as number) : null));
    const lastIdx = values.map((v, i) => (v === null ? -1 : i)).filter(i => i >= 0).pop();
    if (lastIdx === undefined) return null;

    const latest = values[lastIdx] as number;
    const back = (n: number) => (lastIdx - n >= 0 ? values[lastIdx - n] : null);
    const yearBack = periodType === 'quarterly' ? 4 : 1;

    const change = (prior: number | null) =>
        prior === null || prior <= 0 ? null : ((latest - prior) / prior) * 100;
    const formatChange = (pct: number) => `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;

    const priorChange = change(back(1));
    const yoyChange = periodType === 'quarterly' ? change(back(yearBack)) : change(back(1));

    const cells: { title: string; value: string; tone?: number | null }[] = [
        {
            title: `${label} · ${String(points[lastIdx].label ?? '')}`,
            value: formatStatementValue(latest)
        },
        ...(periodType === 'quarterly'
            ? [{
                title: 'vs prior quarter',
                value: priorChange === null ? '—' : formatChange(priorChange),
                tone: priorChange
            }]
            : []),
        {
            title: periodType === 'quarterly' ? 'vs year ago' : 'vs prior year',
            value: yoyChange === null ? '—' : formatChange(yoyChange),
            tone: yoyChange
        }
    ];

    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {cells.map((cell, i) => (
                <div key={cell.title} className="bg-background/60 rounded-xl px-3 py-2">
                    <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground truncate">
                        {i === 0 && <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />}
                        <span className="truncate" title={cell.title}>{cell.title}</span>
                    </div>
                    <div className={cn(
                        "text-lg font-bold tabular-nums mt-0.5",
                        cell.tone == null ? "text-foreground" : cell.tone >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"
                    )}>
                        {cell.value}
                    </div>
                </div>
            ))}
        </div>
    );
}
