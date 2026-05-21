'use client';
import React, { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Scale, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { Holding, fetchSettings } from '../../lib/api';
import { formatCurrency, cn } from '../../lib/utils';

interface RebalanceHelperProps {
    holdings: Holding[];
    currency: string;
}

type BucketDim = 'quoteType' | 'sector' | 'country';

const DIMS: { key: BucketDim; label: string; field: keyof Holding | 'Sector' | 'Country' | 'quoteType' }[] = [
    { key: 'quoteType', label: 'Asset Type', field: 'quoteType' },
    { key: 'sector', label: 'Sector', field: 'Sector' },
    { key: 'country', label: 'Country', field: 'Country' },
];

function isUnknown(v: unknown): boolean {
    if (v == null) return true;
    const s = String(v).trim().toUpperCase();
    return s === '' || s === '-' || s === 'NONE' || s === 'NULL' || s === 'UNKNOWN'
        || s.startsWith('N/A') || s.startsWith('UNKNOWN');
}

export default function RebalanceHelper({ holdings, currency }: RebalanceHelperProps) {
    const [dim, setDim] = useState<BucketDim>('quoteType');
    const settingsQuery = useQuery({ queryKey: ['settings'], queryFn: fetchSettings, staleTime: 5 * 60 * 1000 });

    const { rows, total, hasTargets } = useMemo(() => {
        const mvKey = `Market Value (${currency})`;
        const dimDef = DIMS.find(d => d.key === dim)!;
        const targets = (settingsQuery.data?.target_allocation as Record<string, Record<string, number>> | undefined)?.[dim] ?? {};

        const agg: Record<string, number> = {};
        for (const h of holdings) {
            const v = Math.max(0, (h[mvKey] as number) || 0);
            const raw = dimDef.field === 'Country'
                ? ((h['geography'] as string) || (h['Country'] as string))
                : (h[dimDef.field] as unknown);
            const cat = isUnknown(raw) ? 'Unknown' : (raw as string);
            agg[cat] = (agg[cat] || 0) + v;
        }
        const tot = Object.values(agg).reduce((s, v) => s + v, 0);

        const allBuckets = new Set([...Object.keys(agg), ...Object.keys(targets)]);
        const out = Array.from(allBuckets).map(bucket => {
            const currentVal = agg[bucket] || 0;
            const currentPct = tot > 0 ? (currentVal / tot) * 100 : 0;
            const targetPct = targets[bucket] ?? 0;
            const targetVal = (targetPct / 100) * tot;
            const delta = targetVal - currentVal; // >0 buy, <0 sell
            return { bucket, currentPct, targetPct, currentVal, targetVal, delta };
        })
            // Only show rows that have a target or a current position; sort by trade size.
            .filter(r => r.targetPct > 0 || r.currentVal > 0)
            .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

        const targetSum = Object.values(targets).reduce((s, v) => s + v, 0);
        return { rows: out, total: tot, hasTargets: targetSum > 0 };
    }, [holdings, currency, dim, settingsQuery.data]);

    return (
        <div className="metric-card p-5">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Scale className="w-3.5 h-3.5 text-emerald-500" />
                    <h3 className="section-label">Rebalance Helper</h3>
                </div>
                <div className="inline-flex rounded-lg bg-secondary p-0.5">
                    {DIMS.map(d => (
                        <button
                            key={d.key}
                            onClick={() => setDim(d.key)}
                            className={cn(
                                'px-2.5 py-1 rounded-md text-xs font-semibold transition-all',
                                dim === d.key ? 'bg-[#0097b2] text-white' : 'text-muted-foreground hover:text-foreground',
                            )}
                        >
                            {d.label}
                        </button>
                    ))}
                </div>
            </div>

            {!hasTargets ? (
                <p className="text-sm text-muted-foreground text-center py-8">
                    No targets set for {DIMS.find(d => d.key === dim)!.label.toLowerCase()}.
                    Set them in the drift card above to see suggested rebalancing trades.
                </p>
            ) : (
                <>
                    <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                            <thead>
                                <tr className="text-[10px] uppercase tracking-wider text-muted-foreground/70 border-b border-border/50">
                                    <th className="py-1.5 pr-3 text-left font-semibold">Bucket</th>
                                    <th className="py-1.5 px-2 text-right font-semibold">Current</th>
                                    <th className="py-1.5 px-2 text-right font-semibold">Target</th>
                                    <th className="py-1.5 pl-2 text-right font-semibold">Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows.map(r => {
                                    const isBuy = r.delta > 0;
                                    const negligible = Math.abs(r.delta) < total * 0.005; // < 0.5% of portfolio
                                    return (
                                        <tr key={r.bucket} className="border-b border-border/30 last:border-0 hover:bg-muted/30">
                                            <td className="py-2 pr-3 font-bold text-foreground truncate max-w-[140px]">{r.bucket}</td>
                                            <td className="py-2 px-2 text-right tabular-nums text-muted-foreground">{r.currentPct.toFixed(1)}%</td>
                                            <td className="py-2 px-2 text-right tabular-nums text-foreground">{r.targetPct.toFixed(1)}%</td>
                                            <td className="py-2 pl-2 text-right tabular-nums font-semibold">
                                                {negligible ? (
                                                    <span className="text-muted-foreground/50">On target</span>
                                                ) : (
                                                    <span className={cn(
                                                        'inline-flex items-center gap-0.5 justify-end',
                                                        isBuy ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400',
                                                    )}>
                                                        {isBuy ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                                                        {isBuy ? 'Buy' : 'Sell'} {formatCurrency(Math.abs(r.delta), currency)}
                                                    </span>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                    <p className="text-[10px] text-muted-foreground/50 mt-3 leading-relaxed">
                        Trades to align each bucket with its target weight at the current portfolio value
                        ({formatCurrency(total, currency)}). Buckets within 0.5% are treated as on target.
                    </p>
                </>
            )}
        </div>
    );
}
