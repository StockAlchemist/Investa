import StockDetailModal from './StockDetailModal';
import StockIcon from './StockIcon';
import { useState, useMemo } from 'react';
import { CheckCircle2, Clock } from 'lucide-react';
import TableSkeleton from './skeletons/TableSkeleton';

interface DividendEvent {
    symbol: string;
    dividend_date: string;
    ex_dividend_date: string;
    amount: number;
    status: 'confirmed' | 'estimated';
}

interface DividendCalendarProps {
    events: DividendEvent[];
    isLoading: boolean;
    currency: string;
}

export default function DividendCalendar({ events, isLoading, currency }: DividendCalendarProps) {
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
    const [viewDuration, setViewDuration] = useState<'3m' | '1y'>('3m');

    const filteredEvents = useMemo(() => {
        if (!events || !Array.isArray(events)) return [];
        const now = new Date();
        const cutoff = new Date();
        if (viewDuration === '3m') {
            cutoff.setMonth(now.getMonth() + 3);
        } else {
            cutoff.setFullYear(now.getFullYear() + 1);
        }

        return events.filter(e => {
            const d = new Date(e.dividend_date);
            return d <= cutoff;
        }).sort((a, b) => new Date(a.dividend_date).getTime() - new Date(b.dividend_date).getTime());
    }, [events, viewDuration]);

    if (isLoading) {
        return <TableSkeleton />;
    }

    if (!events || !Array.isArray(events) || events.length === 0) {
        return (
            <div className="bg-card rounded-xl p-12 shadow-sm border border-border text-center">
                <p className="text-zinc-500 dark:text-zinc-400">No upcoming dividend events found.</p>
            </div>
        );
    }

    return (
        <div className="bg-card rounded-xl shadow-sm border border-border overflow-hidden">
            <div className="p-4 sm:p-6 border-b border-border flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div className="flex items-center justify-between w-full sm:w-auto gap-3">
                    <h3 className="text-base font-semibold text-foreground">Dividend Calendar</h3>
                    <span className="px-2.5 py-1 rounded-md bg-secondary text-xs font-medium text-muted-foreground border border-border">
                        {filteredEvents.length} events
                    </span>
                </div>

                <div className="flex items-center gap-1 bg-secondary/50 p-1 rounded-lg w-full sm:w-auto border border-border/50">
                    <button
                        onClick={() => setViewDuration('3m')}
                        className={`text-xs font-medium px-4 py-2 rounded-md transition-all flex-1 sm:flex-none text-center ${viewDuration === '3m'
                            ? 'bg-white dark:bg-zinc-800 text-foreground shadow-sm ring-1 ring-black/5 dark:ring-white/10'
                            : 'text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5'
                            }`}
                    >
                        3 Months
                    </button>
                    <button
                        onClick={() => setViewDuration('1y')}
                        className={`text-xs font-medium px-4 py-2 rounded-md transition-all flex-1 sm:flex-none text-center ${viewDuration === '1y'
                            ? 'bg-white dark:bg-zinc-800 text-foreground shadow-sm ring-1 ring-black/5 dark:ring-white/10'
                            : 'text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5'
                            }`}
                    >
                        1 Year
                    </button>
                </div>
            </div>

            <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
                <table className="w-full text-sm text-left">
                    <thead className="bg-secondary/50 font-semibold border-b border-border sticky top-0 z-10 backdrop-blur-md">
                        <tr>
                            <th className="px-6 py-3 text-xs font-semibold text-muted-foreground">Symbol</th>
                            <th className="px-6 py-3 text-xs font-semibold text-muted-foreground">Ex-Dividend</th>
                            <th className="px-6 py-3 text-xs font-semibold text-muted-foreground">Pay Date</th>
                            <th className="px-6 py-3 text-xs font-semibold text-muted-foreground">Status</th>
                            <th className="px-6 py-3 text-xs font-semibold text-muted-foreground text-right">Amount</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border/50">
                        {filteredEvents.map((event, idx) => (
                            <tr key={`${event.symbol}-${event.dividend_date}-${idx}`} className="hover:bg-accent/5 transition-colors group">
                                <td
                                    className="px-6 py-3 font-medium text-foreground cursor-pointer hover:text-cyan-500 transition-colors"
                                    onClick={() => setSelectedSymbol(event.symbol)}
                                >
                                    <div className="flex items-center gap-2">
                                        <StockIcon symbol={event.symbol} size={20} />
                                        {event.symbol}
                                    </div>
                                </td>
                                <td className="px-6 py-3 text-muted-foreground">
                                    {event.ex_dividend_date ? new Date(event.ex_dividend_date).toLocaleDateString() : '-'}
                                </td>
                                <td className="px-6 py-3 text-foreground">
                                    {new Date(event.dividend_date).toLocaleDateString()}
                                </td>
                                <td className="px-6 py-3">
                                    {event.status === 'confirmed' ? (
                                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                                            <CheckCircle2 className="w-3 h-3" />
                                            Confirmed
                                        </span>
                                    ) : (
                                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                                            <Clock className="w-3 h-3" />
                                            Estimated
                                        </span>
                                    )}
                                </td>
                                <td className="px-6 py-3 text-right font-medium text-emerald-500 dark:text-emerald-400 tabular-nums">
                                    ${event.amount.toFixed(2)}
                                </td>
                            </tr>
                        ))}
                        {filteredEvents.length === 0 && (
                            <tr>
                                <td colSpan={5} className="text-center py-8 text-muted-foreground">
                                    No events in this period.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>

            {selectedSymbol && (
                <StockDetailModal
                    symbol={selectedSymbol}
                    isOpen={!!selectedSymbol}
                    onClose={() => setSelectedSymbol(null)}
                    currency={currency}
                />
            )}
        </div>
    );
}
