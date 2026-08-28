'use client';
import React, { useMemo, useState } from 'react';
import { Building2 } from 'lucide-react';
import { Dividend } from '../../lib/api';
import { formatCurrency } from '../../lib/utils';
import WindowToggle, { IncomeWindow } from './WindowToggle';

interface ByAccountProps {
    dividends: Dividend[];
    currency: string;
}

export default function ByAccount({ dividends, currency }: ByAccountProps) {
    const [window, setWindow] = useState<IncomeWindow>('12m');

    const accounts = useMemo(() => {
        const now = new Date();
        const cutoff = new Date(now);
        cutoff.setFullYear(now.getFullYear() - 1);

        const byAccount = new Map<string, { gross: number; count: number }>();
        for (const div of dividends) {
            if (window === '12m') {
                const d = new Date(div.Date);
                if (isNaN(d.getTime()) || d < cutoff) continue;
            }
            const acc = div.Account || '—';
            const cur = byAccount.get(acc) ?? { gross: 0, count: 0 };
            cur.gross += div.DividendAmountDisplayCurrency || 0;
            cur.count += 1;
            byAccount.set(acc, cur);
        }
        return Array.from(byAccount.entries())
            .map(([account, v]) => ({ account, ...v }))
            .sort((a, b) => b.gross - a.gross);
    }, [dividends, window]);

    const total = accounts.reduce((s, a) => s + a.gross, 0);

    return (
        <div className="metric-card p-5">
            <div className="flex items-center justify-between gap-2 mb-4">
                <div className="flex items-center gap-2">
                    <Building2 className="w-3.5 h-3.5 text-cyan-500" />
                    <h3 className="section-label">By Account</h3>
                </div>
                <WindowToggle value={window} onChange={setWindow} />
            </div>

            {accounts.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                    {window === '12m' ? 'No dividends in the last 12 months.' : 'No dividends recorded.'}
                </p>
            ) : (
                <div className="space-y-3">
                    {accounts.map(acc => {
                        const pct = total > 0 ? (acc.gross / total) * 100 : 0;
                        return (
                            <div key={acc.account} className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 items-center">
                                <div className="min-w-0">
                                    <div className="flex items-center justify-between gap-2 mb-1">
                                        <span className="text-xs font-bold text-foreground truncate">{acc.account}</span>
                                        <span className="text-[10px] tabular-nums text-muted-foreground shrink-0">
                                            {pct.toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
                                        <div
                                            className="absolute inset-y-0 left-0 bg-cyan-500 rounded-full"
                                            style={{ width: `${Math.min(100, pct)}%` }}
                                        />
                                    </div>
                                </div>
                                <div className="text-right shrink-0 tabular-nums">
                                    <div className="text-xs font-bold text-up">
                                        {formatCurrency(acc.gross, currency)}
                                    </div>
                                    <div className="text-[10px] text-muted-foreground/60">
                                        {acc.count} {acc.count === 1 ? 'event' : 'events'}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
