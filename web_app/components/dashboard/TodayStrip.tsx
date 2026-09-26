'use client';
import React, { useMemo } from 'react';
import { TrendingUp, TrendingDown, Globe, BarChart3 } from 'lucide-react';
import { Holding } from '../../lib/api';
import { formatCompactNumber, cn } from '../../lib/utils';
import { useStockModal } from '@/context/StockModalContext';

interface MarketIndex {
    name: string;
    price: number;
    change: number;
    changesPercentage: number;
}

interface TodayStripProps {
    holdings: Holding[];
    currency: string;
    portfolioDayChangePct: number | null;
    indices?: Record<string, MarketIndex>;
}

interface MoverRow {
    symbol: string;
    pct: number;
    contribution: number;
}

function isCashSymbol(s: string): boolean {
    const u = (s || '').toUpperCase();
    return u === '$CASH' || u === 'CASH' || u.startsWith('CASH (');
}

/** The caps label every column opens with, so the four read as one strip. */
function ColumnLabel({ icon: Icon, children, tone }: {
    icon: React.ComponentType<{ className?: string }>;
    children: React.ReactNode;
    tone?: string;
}) {
    return (
        <div className={cn(
            'flex items-center gap-1.5 text-[11px] uppercase tracking-[0.06em] font-semibold mb-2',
            tone ?? 'text-slate-600 dark:text-slate-400',
        )}>
            <Icon className="w-3 h-3" />
            <span>{children}</span>
        </div>
    );
}

/**
 * One row of a column: a label, then its figures right beside it.
 *
 * The label column is capped rather than `1fr`: stretched across a third of a
 * wide dashboard it pushed each figure ~400px from the name it belongs to, and
 * the strip read as two unrelated lists with a gap down the middle.
 */
const ROW = 'grid grid-cols-[minmax(0,4.5rem)_4rem_4.25rem] items-baseline gap-x-3 text-xs';

/** Your own day, as the figure the rest of the strip is read against. */
function YouColumn({ portfolioDayChangePct, benchmark }: {
    portfolioDayChangePct: number | null;
    benchmark: MarketIndex | null;
}) {
    // The column stays when the figure is unknown: it carries the strip's
    // "Today" label, and a dash says the day is not in yet.
    if (portfolioDayChangePct == null) {
        return (
            <div className="min-w-0">
                <ColumnLabel icon={BarChart3}>Today</ColumnLabel>
                <div className="text-2xl font-bold leading-none text-muted-foreground">—</div>
            </div>
        );
    }
    const up = portfolioDayChangePct >= 0;
    const delta = benchmark ? portfolioDayChangePct - benchmark.changesPercentage : null;
    return (
        <div className="min-w-0">
            <ColumnLabel icon={BarChart3}>Today</ColumnLabel>
            <div className={cn('text-2xl font-bold tabular-nums leading-none', up ? 'text-up' : 'text-down')}>
                {up ? '+' : ''}{portfolioDayChangePct.toFixed(2)}%
            </div>
            <div className="mt-1.5 text-xs text-muted-foreground whitespace-nowrap">
                Your portfolio
                {delta != null && benchmark && (
                    <>
                        {' · '}
                        <span className={cn('tabular-nums font-semibold', delta >= 0 ? 'text-up' : 'text-down')}>
                            {delta >= 0 ? '+' : ''}{delta.toFixed(2)}
                        </span>
                        {' vs '}{benchmark.name}
                    </>
                )}
            </div>
        </div>
    );
}

/** The day's index moves, each with how far you ran ahead of or behind it. */
function MarketsColumn({ indices, portfolioDayChangePct }: {
    indices: MarketIndex[];
    portfolioDayChangePct: number | null;
}) {
    if (indices.length === 0) return null;
    return (
        <div className="min-w-0">
            <ColumnLabel icon={Globe}>Markets</ColumnLabel>
            <div className="space-y-1">
                {indices.map(idx => {
                    const positive = idx.changesPercentage >= 0;
                    const delta = portfolioDayChangePct != null ? portfolioDayChangePct - idx.changesPercentage : null;
                    return (
                        <div key={idx.name} className={ROW} title={delta != null ? `You vs ${idx.name}` : undefined}>
                            <span className="text-foreground truncate">{idx.name}</span>
                            <span className={cn('tabular-nums font-bold text-right', positive ? 'text-up' : 'text-down')}>
                                {positive ? '+' : ''}{idx.changesPercentage.toFixed(2)}%
                            </span>
                            {delta != null ? (
                                <span className={cn('text-[10px] tabular-nums text-right', delta >= 0 ? 'text-up' : 'text-down')}>
                                    ({delta >= 0 ? '+' : ''}{delta.toFixed(2)})
                                </span>
                            ) : <span />}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

function MoversColumn({ rows, currency, positive, onPick }: {
    rows: MoverRow[];
    currency: string;
    positive: boolean;
    onPick: (sym: string) => void;
}) {
    const tone = positive ? 'text-up' : 'text-down';
    return (
        <div className="min-w-0">
            <ColumnLabel icon={positive ? TrendingUp : TrendingDown} tone={tone}>
                {positive ? 'Top gainers' : 'Top losers'}
            </ColumnLabel>
            {rows.length === 0 ? (
                <p className="text-xs text-slate-600 dark:text-slate-400">No movers.</p>
            ) : (
                <div className="space-y-1">
                    {rows.map(r => (
                        <button
                            key={r.symbol}
                            type="button"
                            onClick={() => onPick(r.symbol)}
                            className={cn(ROW, 'hover:bg-muted/40 -mx-1.5 px-1.5 rounded transition-colors text-left')}
                        >
                            <span className="text-foreground font-bold truncate">{r.symbol}</span>
                            <span className={cn('tabular-nums font-bold text-right', tone)}>
                                {r.pct >= 0 ? '+' : ''}{r.pct.toFixed(2)}%
                            </span>
                            <span className={cn('text-[10px] tabular-nums text-right', tone)}>
                                {r.contribution >= 0 ? '+' : ''}{formatCompactNumber(r.contribution, currency)}
                            </span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}

export default function TodayStrip({ holdings, currency, portfolioDayChangePct, indices }: TodayStripProps) {
    const { openStockDetail } = useStockModal();

    const { gainers, losers } = useMemo(() => {
        const mvKey = `Market Value (${currency})`;
        // Aggregate by symbol — a stock held in two accounts shouldn't appear
        // twice (and would otherwise produce duplicate React keys).
        const bySymbol = new Map<string, { mv: number; pct: number }>();
        for (const h of holdings) {
            if (isCashSymbol(h.Symbol)) continue;
            const pct = h['Day Change %'];
            if (typeof pct !== 'number') continue;
            const mv = (h[mvKey] as number) || 0;
            const cur = bySymbol.get(h.Symbol);
            if (cur) {
                cur.mv += mv; // Day Change % is per-symbol, keep first observation
            } else {
                bySymbol.set(h.Symbol, { mv, pct });
            }
        }
        const rows: MoverRow[] = Array.from(bySymbol.entries()).map(([symbol, v]) => ({
            symbol,
            pct: v.pct,
            // Today's $ change ≈ MV × pct / 100; close enough for ranking.
            contribution: v.mv * (v.pct / 100),
        }));
        const sorted = [...rows].sort((a, b) => b.contribution - a.contribution);
        return {
            gainers: sorted.filter(r => r.contribution > 0).slice(0, 3),
            losers: sorted.filter(r => r.contribution < 0).slice(-3).reverse(),
        };
    }, [holdings, currency]);

    const hasMovers = gainers.length > 0 || losers.length > 0;
    const hasMarket = (indices && Object.keys(indices).length > 0) || portfolioDayChangePct != null;
    if (!hasMovers && !hasMarket) return null;

    // Sorted by the size of the move, as before, and three at most.
    const indexList = indices
        ? [...Object.values(indices)]
            .sort((a, b) => Math.abs(b.changesPercentage) - Math.abs(a.changesPercentage))
            .slice(0, 3)
        : [];
    // "vs the S&P" is the comparison people make; fall back to whatever leads.
    const benchmark = indexList.find(i => /s&p/i.test(i.name)) ?? indexList[0] ?? null;

    // One row of four on a wide screen, with no title row above it: the first
    // column carries the "Today" label and your own figure, so the strip spends
    // its height on data. Two by two at tablet width, stacked on a phone.
    return (
        <div className="metric-card px-5 py-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-x-8 gap-y-5">
                <YouColumn portfolioDayChangePct={portfolioDayChangePct} benchmark={benchmark} />
                <MarketsColumn indices={indexList} portfolioDayChangePct={portfolioDayChangePct} />
                <MoversColumn rows={gainers} currency={currency} positive onPick={s => openStockDetail(s, currency)} />
                <MoversColumn rows={losers} currency={currency} positive={false} onPick={s => openStockDetail(s, currency)} />
            </div>
        </div>
    );
}
