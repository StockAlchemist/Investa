'use client';

import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    Loader2, AlertTriangle, Info, TrendingUp, ShieldCheck, ShieldAlert, Check,
} from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useStockModal } from '@/context/StockModalContext';
import StockIcon from '@/components/StockIcon';
import { cn, formatCurrency, CURRENCY_SYMBOLS } from '@/lib/utils';
import { formatCalendarDate } from '@/lib/market_time';
import {
    fetchStrategies,
    fetchStrategyAllocation,
    fetchBuffettRankRun,
    type StrategyDefinition,
    type StrategySleeve,
} from '@/lib/api';

/**
 * The strategy catalogue and today's sized allocation.
 *
 * These are *rules*, not suggestions the app generates: each was measured in
 * `scripts/buffett_strategy_search.py` and written up in
 * `docs/quality_strategy_report.pdf`. The view therefore shows each strategy's
 * historical record and its risks together — a return figure without the
 * drawdown beside it is the number that gets people hurt.
 *
 * Nothing here places an order. It answers "what does the rule say today";
 * acting on it stays a deliberate, manual step.
 */

const pct = (value: number | null | undefined, digits = 1): string =>
    value === null || value === undefined ? '—' : `${value.toFixed(digits)}%`;

/**
 * Thousands separators for the capital field, applied as the user types.
 *
 * A seven-digit portfolio value is unreadable as a bare digit run, and this is
 * the one number on the screen the user enters rather than reads. Separators
 * are cosmetic only — `applyCapital` strips everything but digits and the
 * decimal point before parsing, so what is typed and what is sent stay the same
 * number.
 */
const groupDigits = (raw: string): string => {
    const cleaned = raw.replace(/[^0-9.]/g, '');
    const [whole = '', ...fraction] = cleaned.split('.');
    const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return fraction.length > 0 ? `${grouped}.${fraction.join('')}` : grouped;
};

function StatCell({ label, value, tone }: { label: string; value: string; tone?: 'good' | 'bad' }) {
    return (
        <div className="flex flex-col gap-0.5">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold">
                {label}
            </span>
            <span className={cn(
                'text-base font-bold tabular-nums leading-none',
                tone === 'good' && 'text-up',
                tone === 'bad' && 'text-down',
            )}>
                {value}
            </span>
        </div>
    );
}

function StrategyCard({
    strategy, selected, onSelect,
}: {
    strategy: StrategyDefinition;
    selected: boolean;
    onSelect: () => void;
}) {
    const b = strategy.backtest;
    return (
        <button
            type="button"
            onClick={onSelect}
            aria-pressed={selected}
            className="group metric-card relative text-left p-4 w-full h-full transition-all"
        >
            {/*
             * Selection and focus are both drawn as child elements rather than
             * with `outline` or `ring`/`box-shadow`, neither of which survives
             * here: globals.css carries a global
             * `*:focus { outline: none !important; box-shadow: none !important }`
             * that erased the indicator the instant a card was clicked (clicking
             * focuses it), and `.metric-card` sets `box-shadow` as unlayered CSS
             * that outranks Tailwind's layered `ring-*`. A child's own border is
             * subject to neither.
             */}
            <span
                aria-hidden
                className="pointer-events-none absolute -inset-[3px] rounded-[1.4rem] border-2 border-blue-600 opacity-0 transition-opacity group-focus-visible:opacity-100"
            />
            {selected && (
                <span
                    aria-hidden
                    className="pointer-events-none absolute inset-0 rounded-[1.25rem] border-2 border-blue-500 bg-blue-500/[0.05]"
                />
            )}
            <div className="flex items-start justify-between gap-2 mb-2">
                <span className={cn(
                    'font-semibold leading-tight flex items-center gap-1.5',
                    selected && 'text-blue-600 dark:text-blue-400',
                )}>
                    {/* Selection is carried by the ring, the title colour and this
                        mark together — never by colour alone. */}
                    {selected && <Check className="w-4 h-4 shrink-0" aria-hidden />}
                    {strategy.name}
                </span>
                {strategy.is_default && (
                    <Badge variant="secondary" className="shrink-0 text-[10px]">Recommended</Badge>
                )}
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed mb-3">{strategy.summary}</p>
            <div className="grid grid-cols-3 gap-3">
                <StatCell label={`CAGR ${b.window ?? ''}`} value={pct(b.cagr)} tone="good" />
                <StatCell label="Max drawdown" value={pct(b.max_drawdown)} tone="bad" />
                <StatCell label="Sharpe" value={b.sharpe?.toFixed(2) ?? '—'} />
            </div>
        </button>
    );
}

function SleeveTable({ sleeve, currency }: { sleeve: StrategySleeve; currency: string }) {
    const { openStockDetail } = useStockModal();
    if (sleeve.positions.length === 0) {
        return (
            <p className="text-sm text-muted-foreground px-1 py-3">
                No positions — the sleeve could not be built.
            </p>
        );
    }

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-sm">
                <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-muted-foreground/80 border-b">
                        <th className="text-left font-semibold py-2 px-2 whitespace-nowrap">Symbol</th>
                        <th className="text-left font-semibold py-2 px-2 hidden md:table-cell">Industry</th>
                        <th className="text-right font-semibold py-2 px-2 whitespace-nowrap">Weight</th>
                        <th className="text-right font-semibold py-2 px-2 whitespace-nowrap">Amount</th>
                        <th className="text-right font-semibold py-2 px-2 whitespace-nowrap">
                            {sleeve.price_source === 'snapshot' ? 'Price (stored)' : 'Price'}
                        </th>
                        <th className="text-right font-semibold py-2 px-2 whitespace-nowrap">Shares</th>
                    </tr>
                </thead>
                <tbody>
                    {sleeve.positions.map(position => (
                        <tr key={position.symbol} className="border-b last:border-0 hover:bg-muted/40">
                            <td className="py-2 px-2 whitespace-nowrap">
                                <div className="flex items-center gap-2">
                                    <StockIcon symbol={position.symbol} size={24} className="shrink-0" />
                                    <div className="min-w-0">
                                        <button
                                            type="button"
                                            onClick={() => openStockDetail(position.symbol, currency)}
                                            className="font-semibold hover:underline"
                                        >
                                            {position.symbol}
                                        </button>
                                        {position.name && (
                                            <div className="text-xs text-muted-foreground truncate max-w-[16rem]">
                                                {position.name}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </td>
                            <td className="py-2 px-2 text-muted-foreground hidden md:table-cell">
                                {position.industry ?? '—'}
                            </td>
                            <td className="py-2 px-2 text-right tabular-nums whitespace-nowrap">
                                {(position.weight * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 px-2 text-right tabular-nums whitespace-nowrap">
                                {formatCurrency(position.amount, currency)}
                            </td>
                            <td className="py-2 px-2 text-right tabular-nums text-muted-foreground whitespace-nowrap">
                                {position.price != null ? position.price.toFixed(2) : '—'}
                            </td>
                            <td className="py-2 px-2 text-right tabular-nums whitespace-nowrap">
                                {position.shares != null ? position.shares.toLocaleString() : '—'}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

interface StrategiesViewProps {
    currency?: string;
    /** Portfolio value, used as the default amount to size the allocation against. */
    defaultCapital?: number;
}

export default function StrategiesView({ currency = 'USD', defaultCapital }: StrategiesViewProps) {
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [capitalInput, setCapitalInput] = useState<string>(
        groupDigits(String(Math.round(defaultCapital || 100000)))
    );
    const [appliedCapital, setAppliedCapital] = useState<number>(
        defaultCapital && defaultCapital > 0 ? Math.round(defaultCapital) : 100000
    );
    // An unknown code falls back to the code itself, which needs the wider gutter.
    const symbol = CURRENCY_SYMBOLS[currency.toUpperCase()] ?? currency;

    const { data: catalogue, isLoading, isError } = useQuery({
        queryKey: ['strategies'],
        queryFn: ({ signal }) => fetchStrategies(signal),
        staleTime: 60 * 60 * 1000,   // static definitions
    });

    const activeId = selectedId ?? catalogue?.default ?? null;
    const active = useMemo(
        () => catalogue?.strategies.find(s => s.id === activeId) ?? null,
        [catalogue, activeId]
    );

    // The holdings are an output of the ranking, so the cached allocation is
    // keyed on which ranking run produced it. Without this the list is pinned
    // for a full staleTime after the nightly worker publishes a new run — the
    // one moment the book actually changes. This request is metadata only.
    const { data: rankRun } = useQuery({
        queryKey: ['buffett-rank-run'],
        queryFn: ({ signal }) => fetchBuffettRankRun(signal),
        staleTime: 5 * 60 * 1000,
    });

    const { data: allocation, isLoading: allocationLoading, isError: allocationError } = useQuery({
        queryKey: ['strategyAllocation', activeId, appliedCapital, rankRun?.run_id ?? null],
        queryFn: ({ signal }) => fetchStrategyAllocation(activeId!, appliedCapital, signal),
        enabled: !!activeId && appliedCapital > 0,
        staleTime: 15 * 60 * 1000,
    });

    if (isLoading) {
        return (
            <div className="flex items-center gap-2 text-muted-foreground p-6">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading strategies…
            </div>
        );
    }

    if (isError || !catalogue) {
        return (
            <div className="flex items-center gap-2 text-muted-foreground p-6">
                <AlertTriangle className="w-4 h-4" /> Could not load strategies.
            </div>
        );
    }

    const applyCapital = () => {
        const parsed = Number(capitalInput.replace(/[^0-9.]/g, ''));
        if (Number.isFinite(parsed) && parsed > 0) setAppliedCapital(Math.round(parsed));
    };

    return (
        <div className="space-y-6">
            <header className="space-y-1">
                <p className="text-sm text-muted-foreground max-w-3xl">
                    Fixed rules, not live suggestions. Each was backtested point-in-time; the
                    returns below assume the rule is followed mechanically. Nothing here places
                    an order.
                </p>
                <p className="text-xs text-muted-foreground max-w-3xl">
                    <b className="text-foreground">Individual stocks only — no leverage, no funds.</b>{' '}
                    Each strategy holds the ranked companies directly and stays fully invested;
                    there is no defensive mode and no index or cash ETF. None of them
                    out-returns the NASDAQ-100 over 2013&ndash;2025.
                </p>
            </header>

            <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {catalogue.strategies.map(strategy => (
                    <StrategyCard
                        key={strategy.id}
                        strategy={strategy}
                        selected={strategy.id === activeId}
                        onSelect={() => setSelectedId(strategy.id)}
                    />
                ))}
            </section>

            {active && (
                <section className="metric-card p-5 space-y-4">
                    {/* Stacked until the title has room for its own line: sharing
                        the row on a narrow screen wraps the strategy name through
                        the middle and squeezes the Apply button. */}
                    <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-end sm:justify-between">
                        <div className="min-w-0">
                            <h2 className="text-lg font-semibold text-balance">{active.name} — today&apos;s allocation</h2>
                            <p className="text-xs text-muted-foreground">
                                Weights are the rule; share counts are indicative and move with price.
                            </p>
                        </div>
                        <div className="flex items-end gap-2">
                            <label className="flex flex-col gap-1 flex-1 sm:flex-none">
                                <span className="text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold whitespace-nowrap">
                                    Amount to allocate
                                </span>
                                {/* The symbol is chrome, not content: it sits in the
                                    field's padding so the user never types or deletes
                                    it, and it follows the display currency. */}
                                <span className="relative flex items-center">
                                    <span
                                        aria-hidden
                                        className="pointer-events-none absolute left-3 text-sm text-muted-foreground"
                                    >
                                        {symbol}
                                    </span>
                                    <Input
                                        value={capitalInput}
                                        onChange={e => setCapitalInput(groupDigits(e.target.value))}
                                        onKeyDown={e => { if (e.key === 'Enter') applyCapital(); }}
                                        inputMode="decimal"
                                        aria-label={`Amount to allocate, in ${currency}`}
                                        className={cn('w-full sm:w-44 tabular-nums',
                                                      symbol.length > 1 ? 'pl-11' : 'pl-7')}
                                    />
                                </span>
                            </label>
                            <Button onClick={applyCapital} variant="secondary" className="shrink-0">Apply</Button>
                        </div>
                    </div>

                    {/* Rules and their record, side by side with what they cost. */}
                    <div className="grid gap-4 md:grid-cols-2">
                        <div className="space-y-2">
                            <h3 className="text-xs uppercase tracking-wider text-muted-foreground/80 font-semibold">
                                The rule
                            </h3>
                            <ul className="text-sm space-y-1.5">
                                {active.ranking && (
                                    <li className="flex gap-2">
                                        <TrendingUp className="w-4 h-4 mt-0.5 shrink-0 text-muted-foreground" />
                                        <span>
                                            Top {active.ranking.top_n} of
                                            the quality ranking at {(active.ranking.quality_weight * 100).toFixed(0)}% quality,
                                            equally weighted
                                            {active.ranking.max_per_sector != null && (
                                                <>, max {active.ranking.max_per_sector} per industry</>
                                            )}
                                            {active.ranking.min_market_cap != null && active.ranking.min_market_cap > 0 && (
                                                <>, min ${(active.ranking.min_market_cap / 1e9).toFixed(0)}B market cap</>
                                            )}. {active.ranking.rebalance}.
                                        </span>
                                    </li>
                                )}
                                <li className="flex gap-2">
                                    <ShieldCheck className="w-4 h-4 mt-0.5 shrink-0 text-muted-foreground" />
                                    <span>
                                        Fully invested at all times — no market timing, no cash
                                        position, no funds.
                                    </span>
                                </li>
                            </ul>
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                                <StatCell label={`CAGR ${active.backtest.window ?? ''}`} value={pct(active.backtest.cagr)} tone="good" />
                                <StatCell label="Volatility" value={pct(active.backtest.volatility)} />
                                <StatCell label="Max drawdown" value={pct(active.backtest.max_drawdown)} tone="bad" />
                                <StatCell label="Sharpe" value={active.backtest.sharpe?.toFixed(2) ?? '—'} />
                            </div>
                            {active.backtest.long_cagr != null && (
                                <p className="text-xs text-muted-foreground">
                                    Over {active.backtest.long_window}: {pct(active.backtest.long_cagr)} a year.
                                </p>
                            )}
                        </div>

                        <div className="space-y-2">
                            <h3 className="text-xs uppercase tracking-wider text-muted-foreground/80 font-semibold">
                                What it costs you
                            </h3>
                            <ul className="text-sm space-y-1.5">
                                {active.risks.map(risk => (
                                    <li key={risk} className="flex gap-2 text-muted-foreground">
                                        <ShieldAlert className="w-4 h-4 mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
                                        <span>{risk}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>

                    {allocationLoading && (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Loader2 className="w-4 h-4 animate-spin" /> Building allocation…
                        </div>
                    )}
                    {allocationError && (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <AlertTriangle className="w-4 h-4" /> Could not build the allocation.
                        </div>
                    )}

                    {allocation?.warnings.map(warning => (
                        <div key={warning} className="flex items-start gap-2 rounded-lg bg-amber-500/10 border border-amber-500/20 px-3 py-2 text-xs">
                            <Info className="w-4 h-4 shrink-0 text-amber-600 dark:text-amber-400" />
                            <span>{warning}</span>
                        </div>
                    ))}

                    {allocation?.sleeves.map(sleeve => (
                        <div key={sleeve.key} className="space-y-1">
                            <div className="flex items-baseline justify-between gap-2 border-b pb-1">
                                <h3 className="font-semibold text-sm">{sleeve.label}</h3>
                                <span className="text-xs text-muted-foreground tabular-nums">
                                    {(sleeve.weight * 100).toFixed(0)}% · {formatCurrency(sleeve.amount, currency)}
                                    {sleeve.ranked_at && <> · ranked {formatCalendarDate(sleeve.ranked_at)}</>}
                                    {allocation.ranking_age_days != null && (
                                        <span className={cn(
                                            allocation.ranking_is_stale && 'text-amber-600 dark:text-amber-400 font-semibold',
                                        )}>
                                            {' '}({allocation.ranking_age_days === 0
                                                ? 'today'
                                                : `${allocation.ranking_age_days}d ago`})
                                        </span>
                                    )}
                                </span>
                            </div>
                            <SleeveTable sleeve={sleeve} currency={currency} />
                        </div>
                    ))}
                </section>
            )}
        </div>
    );
}
