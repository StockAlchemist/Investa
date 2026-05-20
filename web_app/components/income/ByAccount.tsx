'use client';
import React, { useMemo } from 'react';
import { Building2 } from 'lucide-react';
import { Dividend } from '../../lib/api';
import { formatCurrency } from '../../lib/utils';

interface ByAccountProps {
    dividends: Dividend[];
    currency: string;
}

export default function ByAccount({ dividends, currency }: ByAccountProps) {
    const accounts = useMemo(() => {
        const now = new Date();
        const cutoff = new Date(now);
        cutoff.setFullYear(now.getFullYear() - 1);

        const byAccount = new Map<string, { gross12m: number; grossAll: number; count12m: number }>();
        for (const div of dividends) {
            const d = new Date(div.Date);
            const acc = div.Account || '—';
            const gross = div.DividendAmountDisplayCurrency || 0;
            const cur = byAccount.get(acc) ?? { gross12m: 0, grossAll: 0, count12m: 0 };
            cur.grossAll += gross;
            if (!isNaN(d.getTime()) && d >= cutoff) {
                cur.gross12m += gross;
                cur.count12m += 1;
            }
            byAccount.set(acc, cur);
        }
        return Array.from(byAccount.entries())
            .map(([account, v]) => ({ account, ...v }))
            .sort((a, b) => b.gross12m - a.gross12m || b.grossAll - a.grossAll);
    }, [dividends]);

    if (accounts.length === 0) return null;

    const total12m = accounts.reduce((s, a) => s + a.gross12m, 0);

    return (
        <div className="metric-card p-5">
            <div className="flex items-center gap-2 mb-4">
                <Building2 className="w-3.5 h-3.5 text-cyan-500" />
                <h3 className="section-label">By Account</h3>
                <span className="ml-auto text-[10px] text-muted-foreground/60 uppercase tracking-wider font-semibold">
                    Trailing 12M
                </span>
            </div>

            <div className="space-y-3">
                {accounts.map(acc => {
                    const pct12m = total12m > 0 ? (acc.gross12m / total12m) * 100 : 0;
                    return (
                        <div key={acc.account} className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 items-center">
                            <div className="min-w-0">
                                <div className="flex items-center justify-between gap-2 mb-1">
                                    <span className="text-xs font-bold text-foreground truncate">{acc.account}</span>
                                    <span className="text-[10px] tabular-nums text-muted-foreground shrink-0">
                                        {pct12m.toFixed(1)}%
                                    </span>
                                </div>
                                <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
                                    <div
                                        className="absolute inset-y-0 left-0 bg-cyan-500 rounded-full"
                                        style={{ width: `${Math.min(100, pct12m)}%` }}
                                    />
                                </div>
                            </div>
                            <div className="text-right shrink-0 tabular-nums">
                                <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400">
                                    {formatCurrency(acc.gross12m, currency)}
                                </div>
                                <div className="text-[10px] text-muted-foreground/60">
                                    {acc.count12m} {acc.count12m === 1 ? 'event' : 'events'}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
