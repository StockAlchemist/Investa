'use client';
import React, { useMemo, useState } from 'react';
import { TrendingUp, TrendingDown, AlertCircle, Info } from 'lucide-react';
import { Holding, Lot } from '../lib/api';
import { formatCurrency, cn } from '../lib/utils';

interface Props {
    holdings: Holding[];
    currency: string;
}

const ONE_YEAR_MS = 365 * 24 * 60 * 60 * 1000;
const MIN_HARVEST_LOSS = 100; // ignore lots with tiny losses below this absolute threshold

type LotClass = 'ST' | 'LT';

interface ClassifiedLot {
    symbol: string;
    account?: string;
    lot: Lot;
    cls: LotClass;
    holdingPeriodDays: number;
    /** Days until this short-term lot graduates to long-term (negative if already LT) */
    daysToLongTerm: number;
}

function classifyLots(holdings: Holding[]): ClassifiedLot[] {
    const now = Date.now();
    const out: ClassifiedLot[] = [];
    for (const h of holdings) {
        if (!h.lots) continue;
        for (const lot of h.lots) {
            if (!lot.Date) continue;
            const lotMs = new Date(lot.Date).getTime();
            if (isNaN(lotMs)) continue;
            const held = now - lotMs;
            const heldDays = Math.floor(held / (24 * 60 * 60 * 1000));
            const cls: LotClass = held >= ONE_YEAR_MS ? 'LT' : 'ST';
            out.push({
                symbol: h.Symbol,
                account: h.Account,
                lot,
                cls,
                holdingPeriodDays: heldDays,
                daysToLongTerm: Math.max(0, 365 - heldDays),
            });
        }
    }
    return out;
}

export default function UnrealizedTaxView({ holdings, currency }: Props) {
    const [maxCandidates, setMaxCandidates] = useState(10);

    const { stTotal, ltTotal, harvest, ripening } = useMemo(() => {
        const classified = classifyLots(holdings);
        let st = 0;
        let lt = 0;
        const losses: ClassifiedLot[] = [];
        const ripeningSoon: ClassifiedLot[] = [];

        for (const c of classified) {
            const gain = (c.lot['Unreal. Gain'] as number) || 0;
            if (c.cls === 'ST') st += gain;
            else lt += gain;

            // Harvesting candidates: any lot with gain < -MIN_HARVEST_LOSS
            if (gain < -MIN_HARVEST_LOSS) losses.push(c);

            // Ripening: short-term lots within 30 days of becoming long-term, with positive gain
            if (c.cls === 'ST' && c.daysToLongTerm > 0 && c.daysToLongTerm <= 30 && gain > 0) {
                ripeningSoon.push(c);
            }
        }

        // Sort losses by deepest first (most absolute loss)
        losses.sort((a, b) => (a.lot['Unreal. Gain'] as number) - (b.lot['Unreal. Gain'] as number));
        // Sort ripening by soonest first
        ripeningSoon.sort((a, b) => a.daysToLongTerm - b.daysToLongTerm);

        return { stTotal: st, ltTotal: lt, harvest: losses, ripening: ripeningSoon };
    }, [holdings]);

    const totalUnrealized = stTotal + ltTotal;
    const visibleHarvest = harvest.slice(0, maxCandidates);

    return (
        <div className="space-y-4">
            {/* Summary cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <SummaryTile
                    label="Short-term"
                    value={stTotal}
                    currency={currency}
                    sublabel="Taxed as ordinary income if sold today"
                    accent="amber"
                />
                <SummaryTile
                    label="Long-term"
                    value={ltTotal}
                    currency={currency}
                    sublabel="Taxed at LTCG rate if sold today"
                    accent="emerald"
                />
                <SummaryTile
                    label="Total unrealized"
                    value={totalUnrealized}
                    currency={currency}
                    sublabel={`${holdings.reduce((s, h) => s + (h.lots?.length || 0), 0)} tax lots`}
                    accent="cyan"
                />
            </div>

            {/* Tax-loss harvesting candidates */}
            <div className="metric-card card-shine p-5">
                <div className="flex items-center justify-between mb-3">
                    <div>
                        <h3 className="section-label">Tax-loss harvesting candidates</h3>
                        <p className="text-[10px] text-muted-foreground mt-0.5">
                            Lots with unrealized loss &gt; {formatCurrency(MIN_HARVEST_LOSS, currency)}. Sorted by deepest loss.
                        </p>
                    </div>
                    {harvest.length > maxCandidates && (
                        <button
                            onClick={() => setMaxCandidates(maxCandidates + 10)}
                            className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded hover:bg-muted transition-colors"
                        >
                            Show more ({harvest.length - maxCandidates})
                        </button>
                    )}
                </div>

                {harvest.length === 0 ? (
                    <p className="text-sm text-muted-foreground py-4 text-center">
                        No lots with significant unrealized losses — nothing to harvest right now.
                    </p>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                            <thead>
                                <tr className="text-left text-[10px] uppercase tracking-wider text-muted-foreground/70 border-b border-border/50">
                                    <th className="py-1.5 pr-3 font-semibold">Symbol</th>
                                    <th className="py-1.5 px-3 font-semibold">Acquired</th>
                                    <th className="py-1.5 px-3 font-semibold text-right">Qty</th>
                                    <th className="py-1.5 px-3 font-semibold text-right">Cost</th>
                                    <th className="py-1.5 px-3 font-semibold text-right">Value</th>
                                    <th className="py-1.5 px-3 font-semibold text-right">Loss</th>
                                    <th className="py-1.5 pl-3 font-semibold">Term</th>
                                </tr>
                            </thead>
                            <tbody>
                                {visibleHarvest.map((c, i) => {
                                    const loss = (c.lot['Unreal. Gain'] as number) || 0;
                                    const lossPct = (c.lot['Unreal. Gain %'] as number) || 0;
                                    return (
                                        <tr key={`${c.symbol}-${i}`} className="border-b border-border/30 hover:bg-muted/30">
                                            <td className="py-1.5 pr-3 font-bold text-foreground">
                                                {c.symbol}
                                                {c.account && (
                                                    <span className="text-muted-foreground/60 font-normal ml-1">{c.account}</span>
                                                )}
                                            </td>
                                            <td className="py-1.5 px-3 text-muted-foreground tabular-nums">
                                                {c.lot.Date}
                                            </td>
                                            <td className="py-1.5 px-3 text-right tabular-nums">
                                                {c.lot.Quantity.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                                            </td>
                                            <td className="py-1.5 px-3 text-right tabular-nums text-muted-foreground">
                                                {formatCurrency(c.lot['Cost Basis'], currency)}
                                            </td>
                                            <td className="py-1.5 px-3 text-right tabular-nums">
                                                {formatCurrency(c.lot['Market Value'], currency)}
                                            </td>
                                            <td className="py-1.5 px-3 text-right tabular-nums font-bold text-red-500">
                                                {formatCurrency(loss, currency)} <span className="text-muted-foreground/60 font-normal">({lossPct.toFixed(1)}%)</span>
                                            </td>
                                            <td className="py-1.5 pl-3">
                                                <span className={cn(
                                                    'text-[10px] font-bold uppercase px-1.5 py-0.5 rounded',
                                                    c.cls === 'LT'
                                                        ? 'bg-emerald-500/15 text-emerald-600'
                                                        : 'bg-amber-500/15 text-amber-600',
                                                )}>
                                                    {c.cls}
                                                </span>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                )}

                <p className="text-[10px] text-muted-foreground/70 mt-3 flex items-start gap-1.5">
                    <Info className="w-3 h-3 mt-0.5 shrink-0" />
                    <span>
                        Watch out for the wash-sale rule: if you sell at a loss and buy substantially the same
                        security within 30 days (before or after), the IRS disallows the deduction.
                    </span>
                </p>
            </div>

            {/* Ripening lots: short-term graduating to long-term within 30 days */}
            {ripening.length > 0 && (
                <div className="metric-card card-shine p-5">
                    <h3 className="section-label mb-3">Ripening to long-term within 30 days</h3>
                    <div className="space-y-1.5">
                        {ripening.slice(0, 8).map((c, i) => {
                            const gain = (c.lot['Unreal. Gain'] as number) || 0;
                            return (
                                <div key={`r-${i}`} className="flex items-center justify-between text-xs">
                                    <div className="flex items-center gap-2">
                                        <AlertCircle className="w-3 h-3 text-amber-500" />
                                        <span className="font-bold text-foreground">{c.symbol}</span>
                                        <span className="text-muted-foreground">acquired {c.lot.Date}</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className="tabular-nums text-emerald-600 font-medium">
                                            +{formatCurrency(gain, currency)}
                                        </span>
                                        <span className="tabular-nums text-amber-600 font-bold">
                                            {c.daysToLongTerm}d
                                        </span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                    <p className="text-[10px] text-muted-foreground/70 mt-3">
                        Holding ≥30 more days converts these gains to LTCG treatment (typically lower tax).
                    </p>
                </div>
            )}
        </div>
    );
}

function SummaryTile({
    label, value, currency, sublabel, accent,
}: {
    label: string;
    value: number;
    currency: string;
    sublabel: string;
    accent: 'amber' | 'emerald' | 'cyan';
}) {
    const positive = value >= 0;
    const Icon = positive ? TrendingUp : TrendingDown;
    const colorClass = positive ? 'text-emerald-500' : 'text-red-500';
    const accentBar = {
        amber: 'bg-amber-500',
        emerald: 'bg-emerald-500',
        cyan: 'bg-[#0097b2]',
    }[accent];

    return (
        <div className="metric-card card-shine p-4 relative overflow-hidden">
            <div className={cn('absolute top-0 left-0 right-0 h-[2px] opacity-80', accentBar)} />
            <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70 mb-1">{label}</p>
            <div className="flex items-baseline gap-1.5">
                <span className={cn('text-xl font-bold tabular-nums leading-none', colorClass)}>
                    {positive ? '+' : ''}{formatCurrency(value, currency)}
                </span>
                <Icon className={cn('w-4 h-4', colorClass)} />
            </div>
            <p className="text-[10px] text-muted-foreground/70 mt-2">{sublabel}</p>
        </div>
    );
}
