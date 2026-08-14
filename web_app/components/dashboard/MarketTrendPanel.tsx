'use client';

import React from 'react';
import { useQueries } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';
import { fetchTrendSignal, MARKET_TREND_INDICES, type TrendSignal } from '@/lib/api';
import { formatCalendarDate } from '@/lib/market_time';
import { cn } from '@/lib/utils';

/**
 * The market-trend panel: one moving-average reading per market, stacked.
 *
 * **Advisory only.** No strategy acts on these signals — gating the stock book
 * with the NASDAQ reading was measured and rejected (13.0%/yr against 16.3% for
 * staying invested, with a deeper drawdown). The panel therefore describes the
 * markets' state and never tells the reader to do anything; the wording is
 * deliberate, because a panel that reads like an instruction is one users will
 * follow.
 *
 * **Why more than one market.** A single index reads as a verdict on "the
 * market". Two that disagree as often as the S&P 500 and the NASDAQ 100 show
 * how much of the answer depends on which index was picked — which is the
 * honest way to present an indicator nothing acts on. The set comes from the
 * backend (`MARKET_SIGNAL_INDICES` in `src/strategies.py`) and each reading is
 * fetched independently, so one market's price feed failing still leaves the
 * other readable.
 *
 * Each row keeps two facts apart. The state is the *active* reading, decided at
 * the last completed month-end. `provisional_state` is what the comparison
 * would say if the month ended now — a preview of the next reading, surfaced
 * only when it diverges, so a mid-month price is never mistaken for the current
 * one. The timing those readings share (which month they govern, when they are
 * next checked) is stated once in the footer rather than repeated per row.
 */

/*
 * A note on the muting. Secondary text pairs `text-muted-foreground` with an
 * `opacity-*` utility rather than relying on the token alone: this app imports
 * Tailwind v4 without bridging its shadcn colour tokens, so `muted-foreground`
 * generates no rule and muted text silently renders at full foreground colour.
 * Opacity mutes the inherited colour in both themes, and keeps working — rather
 * than doubling up — if the tokens are ever wired in.
 */

const fmtMoney = (value: number | null | undefined): string =>
    value === null || value === undefined ? '—' : `$${value.toFixed(2)}`;

/** `2026-07` -> `July 2026`. Parsed by hand: a month is not an instant. */
const monthLabel = (yyyymm: string): string => {
    const [year, month] = yyyymm.split('-').map(Number);
    if (!year || !month) return yyyymm;
    return new Date(Date.UTC(year, month - 1, 1)).toLocaleDateString(undefined, {
        month: 'long',
        year: 'numeric',
        timeZone: 'UTC',
    });
};

/** `2026-06-30` -> `30 Jun`, pinned to UTC — these are calendar days, not instants. */
const dayLabel = (iso: string): string =>
    formatCalendarDate(iso, { day: 'numeric', month: 'short' });

/**
 * Whether a reading has everything the panel states about it.
 *
 * A half-populated payload is treated as an unavailable market, not rendered
 * with gaps: this panel shares the dashboard's top section, so a missing field
 * that threw during render would take the whole page down with it.
 */
function isRenderable(signal: TrendSignal | undefined): signal is TrendSignal {
    return !!signal
        && (signal.state === 'in' || signal.state === 'out')
        && typeof signal.sma === 'number'
        && typeof signal.decision_close === 'number'
        && typeof signal.decision_date === 'string'
        && typeof signal.governs_month === 'string'
        && typeof signal.next_decision_date === 'string'
        && Array.isArray(signal.history);
}

/** Month-end closes against their moving average — enough to read the shape, no axes. */
function SignalSparkline({ history }: { history: TrendSignal['history'] }) {
    const points = history.filter(p => p.sma !== null);
    if (points.length < 2) return null;

    // One scale for both series, so a crossing is drawn where it happened.
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
                  className="opacity-40" strokeWidth={1.5} strokeDasharray="3 2"
                  vectorEffect="non-scaling-stroke" />
            <path d={path(p => p.close)} fill="none" stroke="currentColor"
                  className="opacity-90" strokeWidth={1.5}
                  vectorEffect="non-scaling-stroke" />
        </svg>
    );
}

/** One market's reading. */
function TrendRow({ signal, label }: { signal: TrendSignal; label: string }) {
    const isUp = signal.state === 'in';
    const Icon = isUp ? TrendingUp : TrendingDown;
    const tone = isUp
        ? 'text-emerald-700 dark:text-emerald-400'
        : 'text-amber-700 dark:text-amber-400';

    // The margin of the *active* reading: the month-end close that set it,
    // against the average it was compared with. Same comparison as the state
    // beside it, so the word and the number can never disagree.
    const marginPct = signal.sma ? (signal.decision_close / signal.sma - 1) * 100 : null;

    return (
        <div className="py-2.5 flex flex-col gap-1.5">
            <div className="flex items-center justify-between gap-3">
                <div className="flex items-baseline gap-2 min-w-0">
                    <span className="text-sm font-semibold truncate">
                        {signal.signal_name || label}
                    </span>
                    <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                        {signal.signal_symbol}
                    </span>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                    <Icon className={cn('w-3.5 h-3.5', tone)} />
                    <span className={cn('text-sm font-bold leading-none', tone)}>
                        {isUp ? 'Uptrend' : 'Downtrend'}
                    </span>
                    {marginPct !== null && (
                        <span
                            className={cn('text-xs font-medium tabular-nums leading-none', tone)}
                            title={`${dayLabel(signal.decision_date)} close ${fmtMoney(signal.decision_close)} against its ${signal.sma_months}-month average of ${fmtMoney(signal.sma)}`}
                        >
                            {marginPct >= 0 ? '+' : ''}{marginPct.toFixed(1)}%
                        </span>
                    )}
                </div>
            </div>

            <div className="flex items-center gap-2.5">
                <div className="w-16 shrink-0">
                    <SignalSparkline history={signal.history} />
                </div>
                {/*
                  The provisional reading, phrased as a forward-looking note
                  rather than a state, and only when it disagrees with the
                  active one — a matching provisional value carries no
                  information and would just add a number to misread.
                */}
                {signal.would_flip ? (
                    <span className="text-xs leading-snug">
                        <span className="font-semibold text-amber-700 dark:text-amber-400">
                            On track to turn {signal.provisional_state === 'in' ? 'up' : 'down'}
                        </span>
                        <span className="text-slate-600 dark:text-slate-400">
                            {' '}at the next month-end close.
                        </span>
                    </span>
                ) : (
                    <span className="text-xs leading-snug text-slate-600 dark:text-slate-400">
                        Turns {isUp ? 'down below' : 'up above'}{' '}
                        <span className="tabular-nums font-medium text-foreground">
                            {fmtMoney(signal.flip_close)}
                        </span>
                        {signal.distance_pct !== null && (
                            <> — now {Math.abs(signal.distance_pct).toFixed(1)}% away.</>
                        )}
                    </span>
                )}
            </div>
        </div>
    );
}

/** A market whose prices could not be read. Named, so its absence is visible. */
function UnavailableRow({ label, symbol }: { label: string; symbol: string }) {
    return (
        <div className="py-2.5 flex items-center gap-2 text-xs text-muted-foreground opacity-70">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
            <span>
                <span className="font-medium text-foreground">{label}</span> unavailable —
                not enough price history for {symbol}.
            </span>
        </div>
    );
}

/**
 * A row's placeholder while its market is being read.
 *
 * Deliberately not `bg-muted`, which resolves to nothing here (see the note on
 * muting above) and would animate an invisible box.
 */
function SkeletonRow() {
    const bar = 'rounded bg-slate-200 dark:bg-slate-700 animate-pulse';
    return (
        <div className="py-2.5 flex flex-col gap-2">
            <div className="flex items-center justify-between gap-3">
                <div className={cn('h-3.5 w-24', bar)} />
                <div className={cn('h-3.5 w-20', bar)} />
            </div>
            <div className={cn('h-3 w-2/3', bar)} />
        </div>
    );
}

interface MarketTrendPanelProps {
    /** Markets to read, in display order. Defaults to the backend's set. */
    indices?: readonly { symbol: string; label: string }[];
    smaMonths?: number;
    className?: string;
}

export default function MarketTrendPanel({
    indices = MARKET_TREND_INDICES, smaMonths = 10, className,
}: MarketTrendPanelProps) {
    const results = useQueries({
        queries: indices.map(({ symbol }) => ({
            queryKey: ['trendSignal', symbol, smaMonths],
            queryFn: ({ signal }: { signal?: AbortSignal }) =>
                fetchTrendSignal(symbol, smaMonths, signal),
            // Built from daily closes: it can only change once a day.
            staleTime: 15 * 60 * 1000,
            retry: 1,
        })),
    });

    const isLoading = results.some(r => r.isLoading);
    const signals = results.map(r => r.data).filter(isRenderable);

    const header = (
        <div className="flex items-baseline justify-between gap-3">
            <span className="section-label">Market trend</span>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                {smaMonths}-month average
            </span>
        </div>
    );

    if (isLoading) {
        return (
            <div className={cn('metric-card p-5 h-full flex flex-col gap-2', className)}>
                {header}
                <div className="divide-y divide-border/60">
                    {indices.map(idx => <SkeletonRow key={idx.symbol} />)}
                </div>
            </div>
        );
    }

    if (signals.length === 0) {
        return (
            <div className={cn('metric-card p-5 h-full flex flex-col gap-2', className)}>
                {header}
                <div className="flex items-center gap-2 text-sm text-muted-foreground opacity-70">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    Market trend unavailable
                </div>
            </div>
        );
    }

    // The timing every reading shares. Taken from the first available signal:
    // the month a reading governs and its next check are calendar facts of the
    // same market clock, so they agree across rows by construction. The
    // set-at date is only claimed as shared when the rows actually agree.
    const first = signals[0];
    const decisionDates = new Set(signals.map(s => s.decision_date));
    const setAt = decisionDates.size === 1
        ? `the ${dayLabel(first.decision_date)} close`
        : 'each market’s last month-end close';

    return (
        <div className={cn('metric-card p-5 h-full flex flex-col gap-2', className)}>
            {header}

            <div className="divide-y divide-border/60">
                {indices.map(({ symbol, label }) => {
                    const signal = signals.find(s => s.signal_symbol === symbol);
                    return signal
                        ? <TrendRow key={symbol} signal={signal} label={label} />
                        : <UnavailableRow key={symbol} label={label} symbol={symbol} />;
                })}
            </div>

            <p className="text-xs text-muted-foreground opacity-70 leading-snug mt-auto pt-1">
                Set at {setAt}, governing {monthLabel(first.governs_month)}; next checked{' '}
                {dayLabel(first.next_decision_date)} — readings only change on month-end
                closes. Context only — no strategy acts on these.
            </p>
        </div>
    );
}
