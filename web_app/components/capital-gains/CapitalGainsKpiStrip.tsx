'use client';
import React, { useMemo } from 'react';
import { CircleDollarSign, Target, TrendingUp, TrendingDown, Hash, ArrowUpRight, Scale } from 'lucide-react';
import { CapitalGain } from '../../lib/api';
import { formatCompactNumber, formatCurrency, cn } from '../../lib/utils';

interface CapitalGainsKpiStripProps {
    data: CapitalGain[];
    currency: string;
}

interface KpiTileProps {
    label: string;
    value: string;
    sub?: React.ReactNode;
    tone?: 'neutral' | 'pos' | 'neg' | 'warn';
    icon?: React.ComponentType<{ className?: string }>;
}

function KpiTile({ label, value, sub, tone = 'neutral', icon: Icon }: KpiTileProps) {
    const toneClass =
        tone === 'pos'  ? 'text-emerald-600 dark:text-emerald-400'
        : tone === 'neg'  ? 'text-red-600 dark:text-red-400'
        : tone === 'warn' ? 'text-amber-600 dark:text-amber-400'
        : 'text-foreground';
    // Mobile → tablet: fills its responsive grid cell. xl+: becomes a
    // single-row flex strip item with vertical dividers (handled on the parent).
    return (
        <div className="min-w-0 px-1 py-1.5 xl:flex-1 xl:min-w-[120px] xl:px-4 xl:py-3 xl:first:pl-0 xl:last:pr-0">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold mb-1.5">
                {Icon && <Icon className="w-3 h-3 shrink-0" />}
                <span className="truncate">{label}</span>
            </div>
            <div className={cn('text-lg sm:text-2xl font-bold tabular-nums leading-none truncate', toneClass)}>
                {value}
            </div>
            {sub && (
                <div className="text-[11px] text-muted-foreground/80 mt-1.5 leading-tight truncate">
                    {sub}
                </div>
            )}
        </div>
    );
}

export default function CapitalGainsKpiStrip({ data, currency }: CapitalGainsKpiStripProps) {
    const m = useMemo(() => {
        let totalGain = 0, totalProceeds = 0, totalCost = 0;
        let wins = 0, losses = 0, breakeven = 0;
        let winSum = 0, lossSum = 0;
        let biggestWin: { symbol: string; date: string; gain: number } | null = null;
        let biggestLoss: { symbol: string; date: string; gain: number } | null = null;

        for (const item of data) {
            const gain = item['Realized Gain (Display)'] || 0;
            totalGain += gain;
            totalProceeds += item['Total Proceeds (Display)'] || 0;
            totalCost += item['Total Cost Basis (Display)'] || 0;

            if (gain > 0) {
                wins += 1;
                winSum += gain;
                if (!biggestWin || gain > biggestWin.gain) biggestWin = { symbol: item.Symbol, date: item.Date, gain };
            } else if (gain < 0) {
                losses += 1;
                lossSum += gain;
                if (!biggestLoss || gain < biggestLoss.gain) biggestLoss = { symbol: item.Symbol, date: item.Date, gain };
            } else {
                breakeven += 1;
            }
        }

        const decided = wins + losses;
        const winRate = decided > 0 ? (wins / decided) * 100 : null;
        const avgWin = wins > 0 ? winSum / wins : null;
        const avgLoss = losses > 0 ? lossSum / losses : null;
        const returnPct = totalCost !== 0 ? (totalGain / totalCost) * 100 : null;

        return {
            totalGain, totalProceeds, totalCost, returnPct,
            wins, losses, breakeven, winRate, avgWin, avgLoss,
            sales: data.length, biggestWin, biggestLoss,
        };
    }, [data]);

    const fmt = (v: number) => formatCompactNumber(v, currency);

    return (
        <div className="space-y-3">
            <div className="metric-card p-4">
                <div className="grid grid-cols-2 gap-x-3 gap-y-4 sm:grid-cols-3 lg:grid-cols-4 xl:flex xl:gap-0 xl:divide-x xl:divide-border/60">
                    <KpiTile
                        label="Total Realized"
                        value={fmt(m.totalGain)}
                        sub={m.returnPct != null
                            ? <span className={m.returnPct >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}>
                                {m.returnPct >= 0 ? '+' : ''}{m.returnPct.toFixed(1)}% on cost
                              </span>
                            : 'on cost basis'}
                        tone={m.totalGain >= 0 ? 'pos' : 'neg'}
                        icon={CircleDollarSign}
                    />
                    <KpiTile
                        label="Win Rate"
                        value={m.winRate != null ? `${m.winRate.toFixed(0)}%` : '–'}
                        sub={`${m.wins} W · ${m.losses} L${m.breakeven ? ` · ${m.breakeven} flat` : ''}`}
                        tone={(m.winRate ?? 0) >= 50 ? 'pos' : 'warn'}
                        icon={Target}
                    />
                    <KpiTile
                        label="Avg Win"
                        value={m.avgWin != null ? fmt(m.avgWin) : '–'}
                        sub="per winning sale"
                        tone="pos"
                        icon={TrendingUp}
                    />
                    <KpiTile
                        label="Avg Loss"
                        value={m.avgLoss != null ? fmt(m.avgLoss) : '–'}
                        sub="per losing sale"
                        tone="neg"
                        icon={TrendingDown}
                    />
                    <KpiTile
                        label="Sales"
                        value={m.sales.toLocaleString()}
                        sub="closing lots"
                        icon={Hash}
                    />
                    <KpiTile
                        label="Proceeds"
                        value={fmt(m.totalProceeds)}
                        sub="gross sold"
                        icon={ArrowUpRight}
                    />
                    <KpiTile
                        label="Cost Basis"
                        value={fmt(m.totalCost)}
                        sub="of sold lots"
                        icon={Scale}
                    />
                </div>
            </div>

            {/* Biggest win / biggest loss callouts */}
            {(m.biggestWin || m.biggestLoss) && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {m.biggestWin && (
                        <div className="metric-card p-4 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <TrendingUp className="w-4 h-4 text-emerald-500" />
                                <div>
                                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">Biggest Win</div>
                                    <div className="text-sm font-bold text-foreground">{m.biggestWin.symbol}</div>
                                    <div className="text-[10px] text-muted-foreground/60 tabular-nums">{m.biggestWin.date}</div>
                                </div>
                            </div>
                            <div className="text-lg font-bold tabular-nums text-emerald-600 dark:text-emerald-400">
                                +{formatCurrency(m.biggestWin.gain, currency)}
                            </div>
                        </div>
                    )}
                    {m.biggestLoss && (
                        <div className="metric-card p-4 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <TrendingDown className="w-4 h-4 text-red-500" />
                                <div>
                                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">Biggest Loss</div>
                                    <div className="text-sm font-bold text-foreground">{m.biggestLoss.symbol}</div>
                                    <div className="text-[10px] text-muted-foreground/60 tabular-nums">{m.biggestLoss.date}</div>
                                </div>
                            </div>
                            <div className="text-lg font-bold tabular-nums text-red-600 dark:text-red-400">
                                {formatCurrency(m.biggestLoss.gain, currency)}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
