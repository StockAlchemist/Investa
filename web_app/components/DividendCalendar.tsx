'use client';

import React from 'react';

interface DividendEvent {
    symbol: string;
    dividend_date: string;
    ex_dividend_date: string;
    amount: number;
}

interface DividendCalendarProps {
    events: DividendEvent[];
    isLoading: boolean;
}

export default function DividendCalendar({ events, isLoading }: DividendCalendarProps) {
    if (isLoading) {
        return (
            <div className="bg-white dark:bg-zinc-900 rounded-xl p-6 shadow-sm border border-zinc-200 dark:border-zinc-800 animate-pulse h-64">
            </div>
        );
    }

    // DEBUG: Log data
    console.log('[DividendCalendar] Received events:', events);

    if (!events || !Array.isArray(events) || events.length === 0) {
        return (
            <div className="bg-white dark:bg-zinc-900 rounded-xl p-12 shadow-sm border border-zinc-200 dark:border-zinc-800 text-center">
                <p className="text-zinc-500 dark:text-zinc-400">No upcoming dividend events found for your current holdings.</p>
            </div>
        );
    }

    // Sort events by date
    const sortedEvents = [...events].sort((a, b) =>
        new Date(a.dividend_date).getTime() - new Date(b.dividend_date).getTime()
    );

    return (
        <div className="bg-white/5 backdrop-blur-md rounded-xl shadow-sm border border-white/10 overflow-hidden">
            <div className="p-6 border-b border-white/10">
                <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Dividend Calendar</h3>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                    <thead className="text-xs text-muted-foreground uppercase bg-white/5">
                        <tr>
                            <th className="px-6 py-3 font-medium">Symbol</th>
                            <th className="px-6 py-3 font-medium">Ex-Dividend Date</th>
                            <th className="px-6 py-3 font-medium">Dividend Date</th>
                            <th className="px-6 py-3 font-medium text-right">Amount</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/10">
                        {sortedEvents.map((event, idx) => (
                            <tr key={`${event.symbol}-${idx}`} className="hover:bg-white/5 transition-colors">
                                <td className="px-6 py-4 font-bold text-foreground">{event.symbol}</td>
                                <td className="px-6 py-4 text-muted-foreground">
                                    {new Date(event.ex_dividend_date).toLocaleDateString()}
                                </td>
                                <td className="px-6 py-4 text-muted-foreground italic">
                                    {new Date(event.dividend_date).toLocaleDateString()}
                                </td>
                                <td className="px-6 py-4 text-right font-medium text-emerald-500">
                                    ${event.amount.toFixed(2)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
