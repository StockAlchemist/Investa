'use client';

import { useQuery } from '@tanstack/react-query';
import { AlertTriangle } from 'lucide-react';

import { fetchDataQuality } from '@/lib/api';
import { formatCalendarDate } from '@/lib/market_time';

/**
 * Warns that a symbol's stored price history is known to be unreliable.
 *
 * The archive has always known this and only ever said so in a terminal. A
 * chart that steps 30x in the middle because a reverse split was never applied
 * looks exactly like a chart of a stock that fell 97%, and nothing on screen
 * distinguished them.
 *
 * Two severities, because the two are not equally certain. `high` means a split
 * is on record that the prices do not reflect — the series is definitely wrong
 * somewhere. `medium` means a jump nothing explains, which is worth knowing and
 * is not proof; plenty of thin stocks really do move like that.
 */
export function DataQualityBanner({ symbol }: { symbol: string }) {
    const { data } = useQuery({
        queryKey: ['data-quality', symbol],
        queryFn: () => fetchDataQuality([symbol]),
        enabled: !!symbol,
        // The flags are rebuilt by a nightly scan, so re-asking within a session
        // cannot produce a different answer.
        staleTime: 60 * 60 * 1000,
    });

    const flag = data?.symbols?.[symbol.toUpperCase()] ?? data?.symbols?.[symbol];
    if (!flag) return null;

    const high = flag.severity === 'high';
    const when = flag.occurred_on ? formatCalendarDate(flag.occurred_on) : null;

    return (
        <div
            role="status"
            className={`mb-4 flex items-start gap-3 rounded-xl border px-4 py-3 ${
                high
                    ? 'border-destructive/30 bg-destructive/10 text-destructive'
                    : 'border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400'
            }`}
        >
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <div className="min-w-0 text-sm">
                <p className="font-semibold">
                    {high
                        ? 'This price history is known to be wrong'
                        : 'This price history has an unexplained jump'}
                </p>
                {/* Prose, so it is allowed to wrap — unlike the figures in a row
                    or a table cell, which must stay on one line. */}
                <p className="mt-0.5 text-foreground/80">
                    {flag.detail}
                    {when ? ` Around ${when}.` : ''}
                    {flag.findings > 1 ? ` ${flag.findings} findings in total.` : ''}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                    Charts, returns and any figure derived from this stock&apos;s history may be
                    affected. Your recorded transactions are not.
                </p>
            </div>
        </div>
    );
}
