'use client';
import React, { useMemo } from 'react';
import { CalendarClock, CheckCircle2, Clock } from 'lucide-react';
import { DividendEvent } from '../../lib/api';
import { formatCurrency, cn } from '../../lib/utils';
import { useStockModal } from '@/context/StockModalContext';

interface DashboardEventsProps {
    events: DividendEvent[];
    currency: string;
    windowDays?: number;
}

function relativeDay(iso: string): string {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const target = new Date(d);
    target.setHours(0, 0, 0, 0);
    const diffDays = Math.round((target.getTime() - today.getTime()) / 86400000);
    if (diffDays === 0) return 'today';
    if (diffDays === 1) return 'tomorrow';
    if (diffDays < 0) return `${-diffDays}d ago`;
    if (diffDays < 7) return `${diffDays}d`;
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export default function DashboardEvents({ events, currency, windowDays = 14 }: DashboardEventsProps) {
    const { openStockDetail } = useStockModal();

    const upcoming = useMemo(() => {
        const now = new Date();
        const cutoff = new Date(now);
        cutoff.setDate(now.getDate() + windowDays);
        return (events || [])
            .filter(e => {
                const d = new Date(e.dividend_date);
                return !isNaN(d.getTime()) && d >= new Date(now.toDateString()) && d <= cutoff;
            })
            .sort((a, b) => new Date(a.dividend_date).getTime() - new Date(b.dividend_date).getTime())
            .slice(0, 8);
    }, [events, windowDays]);

    return (
        <div className="metric-card p-5 h-full">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <CalendarClock className="w-3.5 h-3.5 text-cyan-500" />
                    <h3 className="section-label">Upcoming Dividends</h3>
                </div>
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-semibold">
                    Next {windowDays}d
                </span>
            </div>

            {upcoming.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-6">
                    No dividend events in the next {windowDays} days.
                </p>
            ) : (
                <div className="space-y-1.5">
                    {upcoming.map((e, i) => (
                        <button
                            key={`${e.symbol}-${e.dividend_date}-${i}`}
                            type="button"
                            onClick={() => openStockDetail(e.symbol, currency)}
                            className="w-full grid grid-cols-[1fr_auto_auto] items-baseline gap-3 px-2 py-1.5 -mx-2 rounded-md hover:bg-muted/40 transition-colors text-left"
                        >
                            <div className="flex items-center gap-1.5 min-w-0">
                                <span className="text-xs font-bold text-foreground truncate">{e.symbol}</span>
                                {e.status === 'estimated' && (
                                    <span className="inline-flex items-center gap-0.5 text-[9px] text-amber-600 dark:text-amber-400">
                                        <Clock className="w-2.5 h-2.5" />
                                        est.
                                    </span>
                                )}
                                {e.status === 'confirmed' && (
                                    <CheckCircle2 className="w-2.5 h-2.5 text-emerald-500 shrink-0" />
                                )}
                            </div>
                            <span className={cn(
                                'text-[10px] uppercase tracking-wider tabular-nums font-semibold',
                                relativeDay(e.dividend_date) === 'today' || relativeDay(e.dividend_date) === 'tomorrow'
                                    ? 'text-emerald-600 dark:text-emerald-400'
                                    : 'text-muted-foreground',
                            )}>
                                {relativeDay(e.dividend_date)}
                            </span>
                            <span className="text-xs font-bold tabular-nums text-emerald-600 dark:text-emerald-400 w-20 text-right">
                                {formatCurrency(e.amount, currency)}
                            </span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
