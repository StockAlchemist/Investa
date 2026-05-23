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

function MarketContextColumn({ indices, portfolioDayChangePct }: { indices?: Record<string, MarketIndex>; portfolioDayChangePct: number | null }) {
    const list = indices ? Object.values(indices) : [];
    if (list.length === 0 && portfolioDayChangePct == null) return null;

    // Sort indices by absolute change desc and show the top 3.
    const top = [...list].sort((a, b) => Math.abs(b.changesPercentage) - Math.abs(a.changesPercentage)).slice(0, 3);

    return (
        <div className="flex flex-col gap-2 min-w-0">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold">
                <Globe className="w-3 h-3" />
                <span>Market Today</span>
            </div>
            {portfolioDayChangePct != null && (
                <div className="flex items-baseline gap-2 mb-1">
                    <span className={cn(
                        'text-2xl font-bold tabular-nums leading-none',
                        portfolioDayChangePct >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400',
                    )}>
                        {portfolioDayChangePct >= 0 ? '+' : ''}{portfolioDayChangePct.toFixed(2)}%
                    </span>
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">you</span>
                </div>
            )}
            <div className="space-y-1">
                {top.map(idx => {
                    const positive = idx.changesPercentage >= 0;
                    const delta = portfolioDayChangePct != null ? portfolioDayChangePct - idx.changesPercentage : null;
                    return (
                        <div key={idx.name} className="grid grid-cols-[1fr_auto_auto] items-baseline gap-3 text-xs">
                            <span className="text-foreground truncate">{idx.name}</span>
                            <span className={cn(
                                'tabular-nums font-bold',
                                positive ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400',
                            )}>
                                {positive ? '+' : ''}{idx.changesPercentage.toFixed(2)}%
                            </span>
                            {delta != null && (
                                <span className={cn(
                                    'text-[10px] tabular-nums w-12 text-right',
                                    delta >= 0 ? 'text-emerald-600/70 dark:text-emerald-400/70' : 'text-red-600/70 dark:text-red-400/70',
                                )}>
                                    ({delta >= 0 ? '+' : ''}{delta.toFixed(2)})
                                </span>
                            )}
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
    const Icon = positive ? TrendingUp : TrendingDown;
    const tone = positive ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400';
    return (
        <div className="flex flex-col gap-2 min-w-0">
            <div className={cn('flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-semibold', tone)}>
                <Icon className="w-3 h-3" />
                <span>{positive ? 'Top Gainers' : 'Top Losers'}</span>
            </div>
            {rows.length === 0 ? (
                <p className="text-xs text-muted-foreground/60 py-1">No movers.</p>
            ) : (
                <div className="space-y-1">
                    {rows.map(r => (
                        <button
                            key={r.symbol}
                            type="button"
                            onClick={() => onPick(r.symbol)}
                            className="w-full grid grid-cols-[1fr_auto_auto] items-baseline gap-3 text-xs hover:bg-muted/40 -mx-2 px-2 py-0.5 rounded transition-colors text-left"
                        >
                            <span className="text-foreground font-bold truncate">{r.symbol}</span>
                            <span className={cn('tabular-nums font-bold', tone)}>
                                {r.pct >= 0 ? '+' : ''}{r.pct.toFixed(2)}%
                            </span>
                            <span className={cn('text-[10px] tabular-nums w-16 text-right', tone)}>
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

    return (
        <div className="metric-card p-5">
            <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="w-3.5 h-3.5 text-muted-foreground" />
                <h3 className="section-label">Today</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-x-8 gap-y-4">
                <MarketContextColumn indices={indices} portfolioDayChangePct={portfolioDayChangePct} />
                <MoversColumn rows={gainers} currency={currency} positive onPick={s => openStockDetail(s, currency)} />
                <MoversColumn rows={losers} currency={currency} positive={false} onPick={s => openStockDetail(s, currency)} />
            </div>
        </div>
    );
}
