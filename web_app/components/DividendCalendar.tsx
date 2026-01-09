import StockDetailModal from './StockDetailModal';
import { useState } from 'react';

interface DividendEvent {
    symbol: string;
    dividend_date: string;
    ex_dividend_date: string;
    amount: number;
}

interface DividendCalendarProps {
    events: DividendEvent[];
    isLoading: boolean;
    currency: string;  // Added currency prop
}

export default function DividendCalendar({ events, isLoading, currency }: DividendCalendarProps) {
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

    if (isLoading) {
        return (
            <div className="bg-card rounded-xl p-6 shadow-sm border border-border animate-pulse h-64">
            </div>
        );
    }

    // DEBUG: Log data


    if (!events || !Array.isArray(events) || events.length === 0) {
        return (
            <div className="bg-card rounded-xl p-12 shadow-sm border border-border text-center">
                <p className="text-zinc-500 dark:text-zinc-400">No upcoming dividend events found for your current holdings.</p>
            </div>
        );
    }

    // Sort events by date
    const sortedEvents = [...events].sort((a, b) =>
        new Date(a.dividend_date).getTime() - new Date(b.dividend_date).getTime()
    );

    return (
        <div className="bg-card rounded-xl shadow-sm border border-border overflow-hidden">
            <div className="p-6 border-b border-border">
                <h3 className="text-sm font-semibold text-muted-foreground">Dividend Calendar</h3>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                    <thead className="bg-secondary/50 font-semibold border-b border-border">
                        <tr>
                            <th className="px-6 py-3 text-xs font-semibold text-muted-foreground">Symbol</th>
                            <th className="px-6 py-3 text-xs font-semibold text-muted-foreground">Ex-Dividend Date</th>
                            <th className="px-6 py-3 text-xs font-semibold text-muted-foreground">Dividend Date</th>
                            <th className="px-6 py-3 text-xs font-semibold text-muted-foreground text-right">Amount</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border/50">
                        {sortedEvents.map((event, idx) => (
                            <tr key={`${event.symbol}-${idx}`} className="hover:bg-accent/5 transition-colors">
                                <td
                                    className="px-6 py-3 font-medium text-foreground cursor-pointer hover:text-cyan-500 transition-colors"
                                    onClick={() => setSelectedSymbol(event.symbol)}
                                >
                                    {event.symbol}
                                </td>
                                <td className="px-6 py-3 text-muted-foreground">
                                    {new Date(event.ex_dividend_date).toLocaleDateString()}
                                </td>
                                <td className="px-6 py-3 text-muted-foreground italic">
                                    {new Date(event.dividend_date).toLocaleDateString()}
                                </td>
                                <td className="px-6 py-3 text-right font-medium text-emerald-500 dark:text-emerald-400 tabular-nums">
                                    ${event.amount.toFixed(2)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Stock Detail Modal */}
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
