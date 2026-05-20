'use client';
import React, { useMemo, useState } from 'react';
import { Trophy } from 'lucide-react';
import { Dividend } from '../../lib/api';
import { formatCurrency, cn } from '../../lib/utils';
import StockIcon from '../StockIcon';
import { useStockModal } from '@/context/StockModalContext';

interface TopPayersProps {
    dividends: Dividend[];
    currency: string;
    limit?: number;
}

type Window = '12m' | 'all';

export default function TopPayers({ dividends, currency, limit = 10 }: TopPayersProps) {
    const [window, setWindow] = useState<Window>('12m');
    const { openStockDetail } = useStockModal();

    const rows = useMemo(() => {
        const now = new Date();
        const cutoff = new Date(now);
        cutoff.setFullYear(now.getFullYear() - 1);

        const bySymbol = new Map<string, { gross: number; tax: number; count: number }>();
        for (const div of dividends) {
            if (window === '12m') {
                const d = new Date(div.Date);
                if (isNaN(d.getTime()) || d < cutoff) continue;
            }
            const sym = div.Symbol;
            const cur = bySymbol.get(sym) ?? { gross: 0, tax: 0, count: 0 };
            cur.gross += div.DividendAmountDisplayCurrency || 0;
            cur.tax += div.TaxAmountDisplayCurrency || 0;
            cur.count += 1;
            bySymbol.set(sym, cur);
        }

        const arr = Array.from(bySymbol.entries())
            .map(([symbol, v]) => ({ symbol, ...v, net: v.gross - v.tax }))
            .sort((a, b) => b.gross - a.gross);

        const grandTotal = arr.reduce((s, x) => s + x.gross, 0);
        return arr.slice(0, limit).map(p => ({
            ...p,
            pct: grandTotal > 0 ? (p.gross / grandTotal) * 100 : 0,
        }));
    }, [dividends, window, limit]);

    if (rows.length === 0) return null;

    return (
        <div className="metric-card p-5">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Trophy className="w-3.5 h-3.5 text-amber-500" />
                    <h3 className="section-label">Top Dividend Payers</h3>
                </div>
                <div className="inline-flex rounded-lg bg-secondary p-0.5">
                    <button
                        onClick={() => setWindow('12m')}
                        className={cn(
                            'px-2.5 py-1 rounded-md text-xs font-semibold transition-all',
                            window === '12m' ? 'bg-[#0097b2] text-white' : 'text-muted-foreground hover:text-foreground',
                        )}
                    >
                        12M
                    </button>
                    <button
                        onClick={() => setWindow('all')}
                        className={cn(
                            'px-2.5 py-1 rounded-md text-xs font-semibold transition-all',
                            window === 'all' ? 'bg-[#0097b2] text-white' : 'text-muted-foreground hover:text-foreground',
                        )}
                    >
                        All time
                    </button>
                </div>
            </div>

            <div className="space-y-2">
                {rows.map((row, idx) => (
                    <div key={row.symbol} className="grid grid-cols-[20px_minmax(0,1fr)_auto] gap-3 items-center">
                        <span className="text-[11px] tabular-nums text-muted-foreground/60 font-bold text-right">
                            {idx + 1}
                        </span>
                        <div className="min-w-0">
                            <button
                                type="button"
                                onClick={() => openStockDetail(row.symbol, currency)}
                                className="flex items-center gap-1.5 mb-1 group"
                            >
                                <StockIcon symbol={row.symbol} size={16} />
                                <span className="text-xs font-bold text-foreground group-hover:text-cyan-500 transition-colors truncate">
                                    {row.symbol}
                                </span>
                                <span className="text-[10px] text-muted-foreground/70 tabular-nums">
                                    · {row.count} {row.count === 1 ? 'pay' : 'pays'}
                                </span>
                            </button>
                            <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
                                <div
                                    className="absolute inset-y-0 left-0 bg-emerald-500 rounded-full"
                                    style={{ width: `${Math.min(100, row.pct)}%` }}
                                />
                            </div>
                        </div>
                        <div className="text-right shrink-0 tabular-nums">
                            <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400">
                                {formatCurrency(row.gross, currency)}
                            </div>
                            <div className="text-[10px] text-muted-foreground/60">
                                {row.pct.toFixed(1)}% of top
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
