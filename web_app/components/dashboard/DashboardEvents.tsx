'use client';
import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { CalendarClock, CheckCircle2, Clock, X } from 'lucide-react';
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

function fullDate(iso: string): string {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

/** Modal listing every confirmed dividend, grouped by month. */
function ConfirmedDividendsModal({
    events,
    currency,
    onClose,
    onSelectSymbol,
}: {
    events: DividendEvent[];
    currency: string;
    onClose: () => void;
    onSelectSymbol: (symbol: string) => void;
}) {
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [onClose]);

    const confirmed = useMemo(() => {
        return (events || [])
            .filter(e => e.status === 'confirmed')
            .sort((a, b) => new Date(a.dividend_date).getTime() - new Date(b.dividend_date).getTime());
    }, [events]);

    const total = useMemo(
        () => confirmed.reduce((sum, e) => sum + (e.amount || 0), 0),
        [confirmed],
    );

    if (typeof document === 'undefined') return null;

    return createPortal(
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 isolate">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in" onClick={onClose} />

            <div
                style={{ backgroundColor: 'var(--menu-solid)' }}
                className="relative w-full max-w-lg max-h-[85vh] rounded-[2rem] flex flex-col overflow-hidden animate-in zoom-in-95 slide-in-from-bottom-10 duration-300"
            >
                {/* Header */}
                <div className="sticky top-0 z-10 bg-card flex-shrink-0 px-6 pt-6 pb-4 flex items-start justify-between">
                    <div>
                        <div className="flex items-center gap-2 mb-1">
                            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                            <h2 className="text-lg font-black tracking-tight text-foreground">Confirmed Dividends</h2>
                        </div>
                        <p className="text-xs text-muted-foreground font-semibold">
                            {confirmed.length} payment{confirmed.length === 1 ? '' : 's'} · {formatCurrency(total, currency)} total
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-all duration-200 text-muted-foreground hover:text-foreground group"
                    >
                        <X className="w-5 h-5 group-hover:rotate-90 transition-transform duration-300" />
                    </button>
                </div>

                {/* List */}
                <div className="flex-1 overflow-y-auto px-4 pb-6 custom-scrollbar">
                    {confirmed.length === 0 ? (
                        <p className="text-sm text-muted-foreground text-center py-12">
                            No confirmed dividends.
                        </p>
                    ) : (
                        <div className="space-y-1">
                            {confirmed.map((e, i) => (
                                <button
                                    key={`${e.symbol}-${e.dividend_date}-${i}`}
                                    type="button"
                                    onClick={() => { onSelectSymbol(e.symbol); onClose(); }}
                                    className="w-full grid grid-cols-[1fr_auto_auto] items-baseline gap-3 px-3 py-2 rounded-lg hover:bg-muted/40 transition-colors text-left"
                                >
                                    <div className="flex items-center gap-1.5 min-w-0">
                                        <span className="text-sm font-bold text-foreground truncate">{e.symbol}</span>
                                        <CheckCircle2 className="w-2.5 h-2.5 text-emerald-500 shrink-0" />
                                    </div>
                                    <span className="text-[11px] tabular-nums text-muted-foreground whitespace-nowrap">
                                        {fullDate(e.dividend_date)}
                                    </span>
                                    <span className="text-sm font-bold tabular-nums text-emerald-600 dark:text-emerald-400 w-24 text-right">
                                        {formatCurrency(e.amount, currency)}
                                    </span>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>,
        document.body,
    );
}

export default function DashboardEvents({ events, currency, windowDays = 14 }: DashboardEventsProps) {
    const { openStockDetail } = useStockModal();
    const [showAll, setShowAll] = useState(false);

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

    const confirmedCount = useMemo(
        () => (events || []).filter(e => e.status === 'confirmed').length,
        [events],
    );

    return (
        <div className="metric-card p-5 h-full">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <CalendarClock className="w-3.5 h-3.5 text-cyan-500" />
                    <h3 className="section-label">Upcoming Dividends</h3>
                </div>
                <button
                    type="button"
                    onClick={() => setShowAll(true)}
                    disabled={confirmedCount === 0}
                    className="text-[10px] uppercase tracking-wider font-semibold text-cyan-600 dark:text-cyan-400 hover:underline disabled:opacity-40 disabled:no-underline disabled:cursor-default"
                    title="View all confirmed dividends"
                >
                    Confirmed ({confirmedCount}) →
                </button>
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

            {showAll && (
                <ConfirmedDividendsModal
                    events={events}
                    currency={currency}
                    onClose={() => setShowAll(false)}
                    onSelectSymbol={(symbol) => openStockDetail(symbol, currency)}
                />
            )}
        </div>
    );
}
