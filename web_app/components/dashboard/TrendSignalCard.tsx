'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { ShieldCheck, ShieldAlert, Loader2, AlertTriangle } from 'lucide-react';
import { fetchTrendSignal, type TrendSignal } from '@/lib/api';
import { cn } from '@/lib/utils';

/**
 * The market-trend indicator, as a dashboard card.
 *
 * **Advisory only.** No strategy acts on this signal — gating the stock book
 * with it was measured and rejected (13.0%/yr against 16.3% for staying
 * invested, with a deeper drawdown). The card therefore describes the market's
 * state and never tells the reader to do anything; the wording is deliberate,
 * because a card that reads like an instruction is one users will follow.
 *
 * It still keeps two facts apart. `state` is the active reading, decided at the
 * last completed month-end. `provisional_state` is what the comparison would
 * say if the month ended now — a preview of next month's reading, shown only
 * when it diverges, so a mid-month price is never mistaken for the current one.
 */

const fmtMoney = (value: number | null | undefined): string =>
    value === null || value === undefined ? '—' : `$${value.toFixed(2)}`;

const monthLabel = (yyyymm: string): string => {
    const [year, month] = yyyymm.split('-').map(Number);
    if (!year || !month) return yyyymm;
    return new Date(year, month - 1, 1).toLocaleString(undefined, { month: 'long', year: 'numeric' });
};

const dayLabel = (iso: string): string => {
    const parsed = new Date(`${iso}T00:00:00`);
    return Number.isNaN(parsed.getTime())
        ? iso
        : parsed.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
};

/** Month-end closes against their moving average — enough to read the trend, no axes. */
function SignalSparkline({ history }: { history: TrendSignal['history'] }) {
    const points = history.filter(p => p.sma !== null);
    if (points.length < 2) return null;

    const values = points.flatMap(p => [p.close, p.sma as number]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const width = 100;
    const height = 28;

    const path = (pick: (p: typeof points[number]) => number) =>
        points
            .map((p, i) => {
                const x = (i / (points.length - 1)) * width;
                const y = height - ((pick(p) - min) / span) * height;
                return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
            })
            .join(' ');

    return (
        <svg
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="none"
            className="w-full h-7"
            role="img"
            aria-label="Month-end closes against the moving average"
        >
            <path d={path(p => p.sma as number)} fill="none" stroke="currentColor"
                  className="text-muted-foreground/40" strokeWidth={1.5} strokeDasharray="3 2"
                  vectorEffect="non-scaling-stroke" />
            <path d={path(p => p.close)} fill="none" stroke="currentColor"
                  className="text-foreground/70" strokeWidth={1.5}
                  vectorEffect="non-scaling-stroke" />
        </svg>
    );
}

interface TrendSignalCardProps {
    symbol?: string;
    smaMonths?: number;
    className?: string;
}

export default function TrendSignalCard({
    symbol = 'QQQ', smaMonths = 10, className,
}: TrendSignalCardProps) {
    const { data, isLoading, isError } = useQuery({
        queryKey: ['trendSignal', symbol, smaMonths],
        queryFn: ({ signal }) => fetchTrendSignal(symbol, smaMonths, signal),
        // The signal is built from daily closes and can only change once a day.
        staleTime: 15 * 60 * 1000,
        retry: 1,
    });

    if (isLoading) {
        return (
            <div className={cn('metric-card p-5 h-full flex items-center gap-2 text-sm text-muted-foreground', className)}>
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading market trend…
            </div>
        );
    }

    if (isError || !data) {
        return (
            <div className={cn('metric-card p-5 h-full text-sm text-muted-foreground', className)}>
                <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" />
                    Market trend unavailable
                </div>
            </div>
        );
    }

    const isIn = data.state === 'in';
    const Icon = isIn ? ShieldCheck : ShieldAlert;

    return (
        <div className={cn('metric-card p-5 h-full flex flex-col gap-3', className)}>
            <div className="flex items-start justify-between gap-3">
                <div className="flex flex-col gap-0.5 min-w-0">
                    <span className="section-label">
                        Market trend · {data.signal_symbol} {data.sma_months}-month
                    </span>
                    <div className="flex items-center gap-2">
                        <Icon className={cn('w-5 h-5 shrink-0',
                            isIn ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400')} />
                        <span className={cn('text-xl font-bold leading-none',
                            isIn ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400')}>
                            {isIn ? 'Uptrend' : 'Downtrend'}
                        </span>
                    </div>
                    <span className="text-xs text-muted-foreground mt-0.5">
                        {isIn
                            ? 'Trading above its moving average.'
                            : 'Trading below its moving average.'}
                        {' '}Context only — no strategy acts on this.
                    </span>
                </div>
                <div className="w-24 shrink-0 hidden sm:block">
                    <SignalSparkline history={data.history} />
                </div>
            </div>

            <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
                <div className="flex flex-col">
                    <dt className="text-muted-foreground">Set at</dt>
                    <dd className="tabular-nums font-medium">
                        {dayLabel(data.decision_date)} · {fmtMoney(data.decision_close)}
                    </dd>
                </div>
                <div className="flex flex-col">
                    <dt className="text-muted-foreground">{data.sma_months}-month average</dt>
                    <dd className="tabular-nums font-medium">{fmtMoney(data.sma)}</dd>
                </div>
                <div className="flex flex-col">
                    <dt className="text-muted-foreground">Governs</dt>
                    <dd className="font-medium">{monthLabel(data.governs_month)}</dd>
                </div>
                <div className="flex flex-col">
                    <dt className="text-muted-foreground">Next check</dt>
                    <dd className="font-medium">{dayLabel(data.next_decision_date)}</dd>
                </div>
            </dl>

            {/*
              The provisional reading. Rendered as a forward-looking note rather
              than a state, and only when it disagrees with the active signal —
              a matching provisional value carries no information and would just
              add a number to misread.
            */}
            {data.would_flip ? (
                <div className="rounded-lg bg-amber-500/10 border border-amber-500/20 px-3 py-2 text-xs">
                    <span className="font-semibold text-amber-700 dark:text-amber-400">
                        On track to turn {data.provisional_state === 'in' ? 'up' : 'down'}
                    </span>
                    <span className="text-muted-foreground">
                        {' '}at the {dayLabel(data.next_decision_date)} close — the reading only
                        changes on month-end closes.
                    </span>
                </div>
            ) : (
                <div className="text-xs text-muted-foreground">
                    Unchanged unless {data.signal_symbol} closes{' '}
                    {data.state === 'in' ? 'below' : 'above'}{' '}
                    <span className="tabular-nums font-medium text-foreground">{fmtMoney(data.flip_close)}</span>
                    {data.distance_pct !== null && (
                        <> on {dayLabel(data.next_decision_date)} — {Math.abs(data.distance_pct).toFixed(1)}% away.</>
                    )}
                </div>
            )}
        </div>
    );
}
