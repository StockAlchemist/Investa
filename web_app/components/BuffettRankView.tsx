"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ChevronDown, ChevronRight, Info, Loader2, ShieldAlert, TrendingUp, TrendingDown } from 'lucide-react';
import WatchlistStar from './WatchlistStar';
import StockIcon from './StockIcon';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { formatPercent, cn } from "@/lib/utils";
import { formatCalendarDate } from '@/lib/market_time';
import { useStockModal } from '@/context/StockModalContext';
import {
    fetchBuffettRankings,
    fetchBuffettExclusions,
    fetchBuffettRankRun,
    type BuffettModel,
    type BuffettRankRow,
} from '@/lib/api';

const MODEL_LABELS: Record<BuffettModel, string> = {
    generic: 'Operating business',
    bank: 'Bank',
    insurer: 'Insurer',
    reit: 'REIT',
};

// Pillars are shown in the order they are weighted, so the columns read as the
// argument for a company's position rather than an arbitrary metric dump.
const PILLARS: Array<{ key: keyof BuffettRankRow; label: string; weight: string }> = [
    { key: 'returns_on_capital', label: 'Returns', weight: '30%' },
    { key: 'financial_strength', label: 'Strength', weight: '20%' },
    { key: 'predictability', label: 'Predictable', weight: '20%' },
    { key: 'growth', label: 'Growth', weight: '15%' },
    { key: 'capital_allocation', label: 'Capital', weight: '15%' },
];

// The weights, spelled out. They are the whole argument for a company's
// position, and a column headed "Capital" means nothing without them.
const QUALITY_WEIGHTS: Array<[string, string]> = [
    ['Returns on capital', '30'],
    ['Financial strength', '20'],
    ['Predictability', '20'],
    ['Growth', '15'],
    ['Capital allocation', '15'],
];

const VALUE_WEIGHTS: Array<[string, string]> = [
    ['Earnings yield', '60'],
    ['Free-cash-flow yield', '40'],
];

const PAGE_SIZE = 100;

/** Percentile scores share one scale, so one colour ramp serves all of them. */
const scoreClass = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return 'text-muted-foreground';
    if (value >= 70) return 'text-emerald-600 dark:text-emerald-400';
    if (value >= 50) return 'text-cyan-600 dark:text-cyan-400';
    if (value >= 30) return 'text-amber-600 dark:text-amber-400';
    return 'text-red-600 dark:text-red-500';
};

const fmtScore = (value: number | null | undefined): string =>
    value === null || value === undefined ? '—' : value.toFixed(0);

const fmtMoney = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return '—';
    if (Math.abs(value) >= 1e12) return `$${(value / 1e12).toFixed(1)}T`;
    if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
    if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(0)}M`;
    return `$${value.toFixed(2)}`;
};

/** One run count, carrying the swatch that ties it to the split bar. */
const Statistic: React.FC<{
    label: string;
    value?: number | null;
    text?: string;
    dotClass?: string;
}> = ({ label, value, text, dotClass }) => (
    <div>
        <dt className="flex items-center gap-1.5 text-xs text-muted-foreground">
            {dotClass && <span className={cn('h-1.5 w-1.5 shrink-0 rounded-full', dotClass)} />}
            {label}
        </dt>
        <dd className="text-lg font-semibold tabular-nums">
            {text ?? (value === null || value === undefined ? '—' : value.toLocaleString())}
        </dd>
    </div>
);

const WeightGroup: React.FC<{
    title: string;
    items: Array<[string, string]>;
    chipClass: string;
    numberClass: string;
}> = ({ title, items, chipClass, numberClass }) => (
    <div>
        <h3 className="text-[10px] font-extrabold uppercase tracking-[0.12em] text-muted-foreground">
            {title}
        </h3>
        <div className="mt-2 flex flex-wrap gap-2">
            {items.map(([label, weight]) => (
                <span
                    key={label}
                    className={cn(
                        'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs text-muted-foreground',
                        chipClass
                    )}
                >
                    {label}
                    <b className={cn('font-bold tabular-nums', numberClass)}>{weight}</b>
                </span>
            ))}
        </div>
    </div>
);

/**
 * How a company's position was arrived at.
 *
 * In place rather than in a doc nobody opens, and collapsed by default so the
 * ranking itself stays the first thing on the screen.
 */
const MethodNote: React.FC = () => (
    <div className="mt-3 space-y-3 rounded-lg bg-secondary/60 p-4">
        <WeightGroup
            title="Quality — 60% of the composite"
            items={QUALITY_WEIGHTS}
            chipClass="bg-cyan-500/10"
            numberClass="text-cyan-700 dark:text-cyan-400"
        />
        <WeightGroup
            title="Value — 40%"
            items={VALUE_WEIGHTS}
            chipClass="bg-indigo-500/10"
            numberClass="text-indigo-600 dark:text-indigo-400"
        />
        <p className="text-xs text-muted-foreground">
            Every figure is a percentile against the companies scored under the same model, so a
            bank&apos;s leverage is judged against other banks and never against an industrial.
            Fundamentals come from SEC EDGAR filings; missing ones lower a company&apos;s
            confidence — and with it its score — rather than failing it outright. Banks and
            insurers have no owner-earnings figure to derive a free-cash-flow yield from, so their
            value score is the earnings yield alone.
        </p>
    </div>
);

interface BuffettRankViewProps {
    currency?: string;
}

const BuffettRankView: React.FC<BuffettRankViewProps> = ({ currency = 'USD' }) => {
    const [model, setModel] = useState<BuffettModel | 'all'>('all');
    const [page, setPage] = useState(0);
    const [search, setSearch] = useState('');
    const [debouncedSearch, setDebouncedSearch] = useState('');
    const [showExcluded, setShowExcluded] = useState(false);
    const [showMethod, setShowMethod] = useState(false);
    const { openStockDetail } = useStockModal();

    // Debounced so typing does not fire a query per keystroke against a
    // 1,100-row table. Any change resets to the first page — staying on page 4
    // of a result set that now has one match would show nothing.
    useEffect(() => {
        const timer = setTimeout(() => {
            setDebouncedSearch(search.trim());
            setPage(0);
        }, 250);
        return () => clearTimeout(timer);
    }, [search]);

    const { data: run } = useQuery({
        queryKey: ['buffett-rank-run'],
        queryFn: ({ signal }) => fetchBuffettRankRun(signal),
        staleTime: 5 * 60 * 1000,
    });

    const { data: rankPage, isFetching, isError } = useQuery({
        queryKey: ['buffett-rank', model, page, debouncedSearch],
        queryFn: ({ signal }) =>
            fetchBuffettRankings(
                PAGE_SIZE,
                page * PAGE_SIZE,
                model === 'all' ? undefined : model,
                debouncedSearch || undefined,
                signal
            ),
        staleTime: 5 * 60 * 1000,
        enabled: !showExcluded,
        // No placeholderData here on purpose. Carrying the previous page over
        // while a new query runs means that mid-search the user sees the full
        // unfiltered list alongside its stale match count — which reads as
        // "search returned everything" rather than "still loading".
    });

    // Treat the whole window from keystroke to resolved response as loading:
    // between the two, `debouncedSearch` still lags `search`, so the query has
    // not even been issued for what is currently typed.
    const searchPending = search.trim() !== debouncedSearch;
    const isBusy = isFetching || searchPending;

    const { data: exclusionPage, isFetching: exclusionsFetching } = useQuery({
        queryKey: ['buffett-exclusions', page, debouncedSearch],
        queryFn: ({ signal }) =>
            fetchBuffettExclusions(PAGE_SIZE, page * PAGE_SIZE, debouncedSearch || undefined, signal),
        staleTime: 5 * 60 * 1000,
        enabled: showExcluded,
    });

    const visibleRows = useMemo(() => rankPage?.rows ?? [], [rankPage]);
    const totalMatches = rankPage?.total ?? 0;
    const exclusionTotal = exclusionPage?.total ?? 0;
    const activeTotal = showExcluded ? exclusionTotal : totalMatches;
    // Of what the run actually scored, not of the universe: the universe count
    // includes listings dropped before scoring, and a bar that did not add up
    // to its own total would be the wrong picture.
    const rankedShare = (run?.ranked_count ?? 0) / Math.max((run?.ranked_count ?? 0) + (run?.excluded_count ?? 0), 1);
    const isLastPage = (page + 1) * PAGE_SIZE >= activeTotal;

    const changeModel = (next: BuffettModel | 'all') => {
        setModel(next);
        setPage(0);
    };

    if (!run) {
        return (
            <div className="rounded-xl border border-border bg-card p-8 text-center">
                <ShieldAlert className="mx-auto h-8 w-8 text-muted-foreground" />
                <h2 className="mt-3 text-lg font-semibold">No ranking run yet</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                    Run <code className="rounded bg-secondary px-1.5 py-0.5">python src/buffett_rank_worker.py</code>{' '}
                    to build the first snapshot.
                </p>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <header className="rounded-xl border border-border bg-card p-5">
                <h1 className="text-xl font-semibold">Buffett &amp; Value Ranking</h1>
                <p className="mt-1 max-w-4xl text-sm text-muted-foreground">
                    Every US-listed common stock, scored 60% on business quality and 40% on
                    value. Quality gates run first — a company that fails one is excluded
                    rather than ranked low, because cheapness never rescues a broken business.
                </p>

                {/* The counts as the split they describe. Four fifths of the listed
                    market fails a gate, and that proportion is the shape of the whole
                    screen — three bare numbers never said so. */}
                <div className="mt-4 flex h-2 gap-0.5" aria-hidden="true">
                    <div
                        className="rounded-full bg-gradient-to-r from-cyan-600 to-cyan-400"
                        style={{ width: `${(rankedShare * 100).toFixed(2)}%` }}
                    />
                    <div className="flex-1 rounded-full bg-muted-foreground/20" />
                </div>

                <dl className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
                    <Statistic label="Ranked" value={run.ranked_count} dotClass="bg-cyan-500" />
                    <Statistic label="Excluded" value={run.excluded_count} dotClass="bg-muted-foreground/40" />
                    <Statistic label="Universe" value={run.universe_size} />
                    {run.finished_at && <Statistic label="Run" text={formatCalendarDate(run.finished_at)} />}
                </dl>

                <button
                    type="button"
                    onClick={() => setShowMethod((open) => !open)}
                    className="mt-3 flex items-center gap-1 text-xs font-semibold text-cyan-700 hover:underline dark:text-cyan-400"
                    aria-expanded={showMethod}
                >
                    {showMethod ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                    How it&apos;s scored
                </button>
                {showMethod && <MethodNote />}
            </header>

            <div className="flex flex-wrap items-center gap-2">
                <Button
                    variant={showExcluded ? 'outline' : 'default'}
                    size="sm"
                    onClick={() => { setShowExcluded(false); setPage(0); }}
                >
                    Ranked
                </Button>
                <Button
                    variant={showExcluded ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => { setShowExcluded(true); setPage(0); }}
                >
                    Excluded ({run.excluded_count ?? 0})
                </Button>

                {!showExcluded && (
                    <>
                        <div className="mx-2 h-6 w-px bg-border" />
                        {(['all', 'generic', 'bank', 'insurer', 'reit'] as const).map((option) => (
                            <Button
                                key={option}
                                variant={model === option ? 'default' : 'outline'}
                                size="sm"
                                onClick={() => changeModel(option)}
                            >
                                {option === 'all' ? 'All models' : MODEL_LABELS[option]}
                            </Button>
                        ))}
                    </>
                )}

                {/* Search serves both tabs: when a company is missing from the
                    ranking, looking it up in the excluded list is the very next
                    thing you want to do. */}
                <div className="ml-auto flex items-center gap-2">
                    {debouncedSearch && !isBusy && (
                        <span className="text-xs text-muted-foreground">
                            {activeTotal} {activeTotal === 1 ? 'match' : 'matches'}
                        </span>
                    )}
                    <Input
                        value={search}
                        onChange={(event) => setSearch(event.target.value)}
                        placeholder={showExcluded ? 'Search excluded stocks…' : 'Search all ranked stocks…'}
                        className="h-9 w-64"
                    />
                </div>
            </div>

            {showExcluded ? (
                <ExclusionTable
                    rows={exclusionPage?.rows ?? []}
                    loading={exclusionsFetching || searchPending}
                    searchTerm={debouncedSearch}
                />
            ) : (
                <RankTable
                    rows={visibleRows}
                    loading={isBusy}
                    error={isError}
                    currency={currency}
                    searchTerm={debouncedSearch}
                    onOpen={openStockDetail}
                />
            )}

            <div className="flex items-center justify-between">
                <Button
                    variant="outline"
                    size="sm"
                    disabled={page === 0}
                    onClick={() => setPage((current) => Math.max(0, current - 1))}
                >
                    Previous
                </Button>
                <span className="text-sm text-muted-foreground">
                    {isBusy
                        ? '…'
                        : `${activeTotal === 0 ? 0 : page * PAGE_SIZE + 1}–${Math.min((page + 1) * PAGE_SIZE, activeTotal)} of ${activeTotal}`}
                </span>
                <Button
                    variant="outline"
                    size="sm"
                    disabled={isLastPage}
                    onClick={() => setPage((current) => current + 1)}
                >
                    Next
                </Button>
            </div>
        </div>
    );
};

const RankTable: React.FC<{
    rows: BuffettRankRow[];
    loading: boolean;
    error: boolean;
    currency: string;
    searchTerm: string;
    onOpen: (symbol: string, currency: string) => void;
}> = ({ rows, loading, error, currency, searchTerm, onOpen }) => {
    if (loading) {
        return (
            <div className="flex items-center justify-center rounded-xl border border-border bg-card py-16">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
        );
    }
    if (error) {
        return (
            <div className="rounded-xl border border-border bg-card p-8 text-center text-sm text-red-500">
                Could not load the ranking.
            </div>
        );
    }
    if (!rows.length) {
        return (
            <div className="rounded-xl border border-border bg-card p-8 text-center text-sm text-muted-foreground">
                {searchTerm ? (
                    <>
                        No ranked company matches <span className="font-medium">“{searchTerm}”</span>.
                        <div className="mt-1 text-xs">
                            It may have been excluded by a quality gate — check the Excluded tab.
                        </div>
                    </>
                ) : (
                    'No companies on this page.'
                )}
            </div>
        );
    }

    return (
        <div className="overflow-x-auto rounded-xl border border-border bg-card">
            <table className="min-w-full divide-y divide-border">
                <thead className="bg-secondary/30">
                    <tr>
                        <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">#</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">Company</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Score</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Quality</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Value</th>
                        {PILLARS.map((pillar) => (
                            <th
                                key={String(pillar.key)}
                                className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                                title={`Pillar weight ${pillar.weight}`}
                            >
                                {pillar.label}
                            </th>
                        ))}
                        <th
                            className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                            title="Earnings yield — net income over market cap, and the heaviest input to the value score"
                        >
                            E/P
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Mkt Cap</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-border">
                    {rows.map((row) => (
                        <tr key={row.symbol} className="transition-colors hover:bg-secondary/20">
                            <td className="px-4 py-3 text-sm tabular-nums text-muted-foreground">{row.rank ?? '—'}</td>
                            <td className="px-4 py-3">
                                <div className="flex items-center gap-3">
                                    {/* Carries the logo itself — the star is a badge
                                        on its corner — so this row needs no separate
                                        StockIcon. */}
                                    <WatchlistStar symbol={row.symbol} size={32} />
                                    <button
                                        onClick={() => onOpen(row.symbol, currency)}
                                        className="flex flex-col items-start text-left transition-colors hover:text-cyan-500"
                                    >
                                        <span className="font-semibold">{row.symbol}</span>
                                        <span className="max-w-[16rem] truncate text-xs text-muted-foreground">
                                            {row.name}
                                        </span>
                                    </button>
                                    {row.model !== 'generic' && (
                                        <Badge variant="outline" className="text-[10px]">
                                            {MODEL_LABELS[row.model]}
                                        </Badge>
                                    )}
                                </div>
                            </td>
                            <td className={cn('px-4 py-3 text-right text-sm font-semibold tabular-nums', scoreClass(row.composite_score))}>
                                {fmtScore(row.composite_score)}
                                {/* Confidence below 1 means the score was cut for thin data,
                                    so it belongs next to the number it modified. */}
                                {row.confidence !== null && row.confidence < 0.999 && (
                                    <span
                                        className="ml-1 text-[10px] font-normal text-amber-500"
                                        title={`Reduced to ${(row.confidence * 100).toFixed(0)}% for incomplete data`}
                                    >
                                        ▾
                                    </span>
                                )}
                            </td>
                            <td className={cn('px-4 py-3 text-right text-sm tabular-nums', scoreClass(row.quality_score))}>
                                {fmtScore(row.quality_score)}
                            </td>
                            <td className={cn('px-4 py-3 text-right text-sm tabular-nums', scoreClass(row.value_score))}>
                                {fmtScore(row.value_score)}
                            </td>
                            {PILLARS.map((pillar) => (
                                <td
                                    key={String(pillar.key)}
                                    className={cn('px-3 py-3 text-right text-sm tabular-nums', scoreClass(row[pillar.key] as number | null))}
                                >
                                    {fmtScore(row[pillar.key] as number | null)}
                                </td>
                            ))}
                            <td className="px-4 py-3 text-right text-sm tabular-nums">
                                {row.earnings_yield === null ? (
                                    <span className="text-muted-foreground" title="No reported earnings for this period">—</span>
                                ) : (
                                    <span
                                        className={cn(
                                            'inline-flex items-center',
                                            row.earnings_yield > 0
                                                ? 'text-emerald-600 dark:text-emerald-400'
                                                : 'text-red-600 dark:text-red-500'
                                        )}
                                    >
                                        {row.earnings_yield > 0 ? (
                                            <TrendingUp className="mr-1 h-3 w-3" />
                                        ) : (
                                            <TrendingDown className="mr-1 h-3 w-3" />
                                        )}
                                        {formatPercent(row.earnings_yield / 100)}
                                    </span>
                                )}
                            </td>
                            <td className="px-4 py-3 text-right text-sm tabular-nums text-muted-foreground">
                                {fmtMoney(row.market_cap)}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

const ExclusionTable: React.FC<{
    rows: Array<{ symbol: string; name: string | null; model: string; reasons: string; period_count: number | null }>;
    loading: boolean;
    searchTerm: string;
}> = ({ rows, loading, searchTerm }) => {
    if (loading) {
        return (
            <div className="flex items-center justify-center rounded-xl border border-border bg-card py-16">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (!rows.length) {
        return (
            <div className="rounded-xl border border-border bg-card p-8 text-center text-sm text-muted-foreground">
                {searchTerm
                    ? `No excluded company matches “${searchTerm}”.`
                    : 'No exclusions on this page.'}
            </div>
        );
    }

    return (
        <div className="space-y-3">
            <div className="flex items-start gap-2 rounded-lg border border-border bg-secondary/20 p-3 text-sm text-muted-foreground">
                <Info className="mt-0.5 h-4 w-4 shrink-0" />
                <p>
                    Most of the listed market is excluded, which is expected when ranking every
                    listing rather than an index. A gate only fires on something the filings
                    actually show — missing data never fails a company, it reduces its confidence
                    score instead.
                </p>
            </div>
            <div className="overflow-x-auto rounded-xl border border-border bg-card">
                <table className="min-w-full divide-y divide-border">
                    <thead className="bg-secondary/30">
                        <tr>
                            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">Symbol</th>
                            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">Name</th>
                            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">Model</th>
                            <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">Years</th>
                            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">Reasons</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                        {rows.map((row) => (
                            <tr key={row.symbol} className="hover:bg-secondary/20">
                                <td className="px-4 py-3 text-sm font-semibold">
                                    {/* Same identity treatment as the ranked
                                        table, so the two tabs read as one list
                                        seen two ways. */}
                                    <span className="flex items-center gap-2">
                                        <StockIcon symbol={row.symbol} size={24} className="shrink-0" />
                                        {row.symbol}
                                    </span>
                                </td>
                                <td className="max-w-[20rem] truncate px-4 py-3 text-sm text-muted-foreground">{row.name}</td>
                                <td className="px-4 py-3 text-sm text-muted-foreground">{row.model}</td>
                                <td className="px-4 py-3 text-right text-sm tabular-nums text-muted-foreground">
                                    {row.period_count ?? '—'}
                                </td>
                                <td className="px-4 py-3">
                                    <div className="flex flex-wrap gap-1">
                                        {row.reasons.split(',').map((reason) => (
                                            <Badge key={reason} variant="outline" className="text-[10px]">
                                                {reason.trim().replace(/_/g, ' ')}
                                            </Badge>
                                        ))}
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default BuffettRankView;
