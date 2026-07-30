'use client';
import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { BarChart3, CalendarClock, CheckCircle2, Clock, X } from 'lucide-react';
import { DividendEvent, EarningsEvent } from '../../lib/api';
import { formatCurrency, cn } from '../../lib/utils';
import { formatCalendarDate, marketDayDiff } from '../../lib/market_time';
import { useStockModal } from '@/context/StockModalContext';
import StockIcon from '../StockIcon';

interface DashboardEventsProps {
    events: DividendEvent[];
    /** Upcoming earnings reports, interleaved with dividends by date. */
    earnings?: EarningsEvent[];
    currency: string;
    windowDays?: number;
}

/** A dividend payment or an earnings report, normalized onto one timeline. */
type TimelineRow =
    | { kind: 'dividend'; date: string; symbol: string; event: DividendEvent }
    | { kind: 'earnings'; date: string; symbol: string; event: EarningsEvent };

/**
 * "today" / "3d" / "Aug 12" for an event date, counted in the *market's* local
 * time. Reckoning against the browser clock puts a US report a day out for a
 * viewer in Asia, where the device has already rolled into tomorrow.
 */
function relativeDay(iso: string, timeZone?: string | null): string {
    const diffDays = marketDayDiff(iso, timeZone);
    if (diffDays === null) return '';
    if (diffDays === 0) return 'today';
    if (diffDays === 1) return 'tomorrow';
    if (diffDays < 0) return `${-diffDays}d ago`;
    if (diffDays < 7) return `${diffDays}d`;
    return formatCalendarDate(iso, { month: 'short', day: 'numeric' });
}

function fullDate(iso: string): string {
    return formatCalendarDate(iso);
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
                                    className="w-full grid grid-cols-[1fr_auto_auto] items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted/40 transition-colors text-left"
                                >
                                    <div className="flex items-center gap-2 min-w-0">
                                        <StockIcon symbol={e.symbol} size={24} />
                                        <div className="min-w-0">
                                            <div className="flex items-center gap-1.5 min-w-0">
                                                <span className="text-sm font-bold text-foreground truncate">{e.symbol}</span>
                                                <CheckCircle2 className="w-2.5 h-2.5 text-emerald-500 shrink-0" />
                                            </div>
                                            {e.name && (
                                                <span className="block text-[10px] text-muted-foreground/70 truncate leading-tight">
                                                    {e.name}
                                                </span>
                                            )}
                                        </div>
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

export default function DashboardEvents({ events, earnings = [], currency, windowDays = 14 }: DashboardEventsProps) {
    const { openStockDetail } = useStockModal();
    const [showAll, setShowAll] = useState(false);

    const upcoming = useMemo<TimelineRow[]>(() => {
        // The window is measured on each event's own exchange clock, so the row
        // shown as "today" is the one the market calls today.
        const inWindow = (iso: string, timeZone?: string | null) => {
            const diff = marketDayDiff(iso, timeZone);
            return diff !== null && diff >= 0 && diff <= windowDays;
        };

        const rows: TimelineRow[] = [];
        for (const e of events || []) {
            if (inWindow(e.dividend_date, e.market_timezone)) {
                rows.push({ kind: 'dividend', date: e.dividend_date, symbol: e.symbol, event: e });
            }
        }
        for (const e of earnings || []) {
            if (inWindow(e.earnings_date, e.market_timezone)) {
                rows.push({ kind: 'earnings', date: e.earnings_date, symbol: e.symbol, event: e });
            }
        }
        return rows
            .sort((a, b) => {
                const diff = new Date(a.date).getTime() - new Date(b.date).getTime();
                return diff !== 0 ? diff : a.symbol.localeCompare(b.symbol);
            })
            .slice(0, 8);
    }, [events, earnings, windowDays]);

    const confirmedCount = useMemo(
        () => (events || []).filter(e => e.status === 'confirmed').length,
        [events],
    );

    return (
        <div className="metric-card p-5 h-full">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <CalendarClock className="w-3.5 h-3.5 text-cyan-500" />
                    <h3 className="section-label">Upcoming Events</h3>
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
                    No dividend or earnings events in the next {windowDays} days.
                </p>
            ) : (
                <div className="space-y-1.5">
                    {upcoming.map((row, i) => {
                        const rel = relativeDay(row.date, row.event.market_timezone);
                        const isSoon = rel === 'today' || rel === 'tomorrow';
                        return (
                            <button
                                key={`${row.kind}-${row.symbol}-${row.date}-${i}`}
                                type="button"
                                onClick={() => openStockDetail(row.symbol, currency)}
                                className="w-full grid grid-cols-[1fr_auto_auto] items-center gap-3 px-2 py-1.5 -mx-2 rounded-md hover:bg-muted/40 transition-colors text-left"
                            >
                                {/* Logo, then ticker + type marker on top with the company
                                    name beneath, so a narrow row never has to wrap the
                                    marker's text. */}
                                <div className="flex items-center gap-2 min-w-0">
                                    <StockIcon symbol={row.symbol} size={22} />
                                    <div className="min-w-0">
                                        <div className="flex items-center gap-1.5 min-w-0">
                                            <span className="text-xs font-bold text-foreground truncate">{row.symbol}</span>
                                            {row.kind === 'earnings' ? (
                                                <span
                                                    className="inline-flex items-center gap-0.5 whitespace-nowrap text-[9px] text-violet-600 dark:text-violet-400 shrink-0"
                                                    title={row.event.status === 'estimated'
                                                        ? 'Earnings date projected from past reporting cadence'
                                                        : 'Confirmed earnings date'}
                                                >
                                                    <BarChart3 className="w-2.5 h-2.5" />
                                                    earnings{row.event.status === 'estimated' ? ' est.' : ''}
                                                </span>
                                            ) : row.event.status === 'estimated' ? (
                                                <span className="inline-flex items-center gap-0.5 whitespace-nowrap text-[9px] text-amber-600 dark:text-amber-400 shrink-0">
                                                    <Clock className="w-2.5 h-2.5" />
                                                    est.
                                                </span>
                                            ) : (
                                                <CheckCircle2 className="w-2.5 h-2.5 text-emerald-500 shrink-0" />
                                            )}
                                        </div>
                                        {row.event.name && (
                                            <span className="block text-[10px] text-muted-foreground/70 truncate leading-tight">
                                                {row.event.name}
                                            </span>
                                        )}
                                    </div>
                                </div>
                                <span
                                    title={row.kind === 'earnings' && row.event.earnings_date_end
                                        ? `${fullDate(row.date)} – ${fullDate(row.event.earnings_date_end)}`
                                        : fullDate(row.date)}
                                    className={cn(
                                        'text-[10px] uppercase tracking-wider tabular-nums font-semibold',
                                        isSoon ? 'text-emerald-600 dark:text-emerald-400' : 'text-muted-foreground',
                                    )}
                                >
                                    {rel}
                                </span>
                                {row.kind === 'earnings' ? (
                                    <span
                                        className="text-xs font-bold tabular-nums text-violet-600 dark:text-violet-400 w-20 text-right"
                                        title={row.event.eps_year_ago != null
                                            ? `Consensus EPS estimate — ${row.event.eps_year_ago.toFixed(2)} a year ago`
                                            : 'Consensus EPS estimate for the quarter'}
                                    >
                                        {row.event.eps_estimate != null
                                            ? `${row.event.eps_estimate.toFixed(2)} EPS`
                                            : 'Report'}
                                    </span>
                                ) : (
                                    <span className="text-xs font-bold tabular-nums text-emerald-600 dark:text-emerald-400 w-20 text-right">
                                        {formatCurrency(row.event.amount, currency)}
                                    </span>
                                )}
                            </button>
                        );
                    })}
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
