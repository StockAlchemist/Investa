import React from 'react';
import { cn } from '../../../lib/utils';
import { formatCalendarDate, marketDayDiff } from '../../../lib/market_time';

function formatEventDate(iso: string): string {
    return formatCalendarDate(iso);
}

function relativeEventDay(iso: string, timeZone?: string | null): string | null {
    const days = marketDayDiff(iso, timeZone);
    if (days === null) return null;
    if (days === 0) return 'today';
    if (days === 1) return 'tomorrow';
    if (days < 0) return `${-days} day${days === -1 ? '' : 's'} ago`;
    return `in ${days} days`;
}

interface UpcomingEventRowProps {
    icon: React.ElementType;
    color: string;
    label: string;
    status: 'confirmed' | 'estimated' | 'reported';
    date: string;
    dateEnd?: string | null;
    detail?: string;
    detailColor?: string;
    timeZone?: string | null;
}

export const UpcomingEventRow: React.FC<UpcomingEventRowProps> = ({
    icon: Icon,
    color,
    label,
    status,
    date,
    dateEnd,
    detail,
    detailColor,
    timeZone
}) => {
    const relative = relativeEventDay(date, timeZone);
    const badge = {
        confirmed: { text: 'confirmed', tone: 'text-up bg-up/12', title: 'Announced by the company' },
        estimated: { text: 'est.', tone: 'text-amber-600 dark:text-amber-400 bg-amber-500/10', title: 'Projected from the past reporting/payment cadence' },
        reported: { text: 'reported', tone: 'text-violet-600 dark:text-violet-400 bg-violet-500/10', title: 'Already reported by the company' },
    }[status];

    return (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 px-3 py-2 transition-colors hover:bg-muted/60">
            <div className={cn("p-1.5 rounded-lg bg-card shrink-0", color)}>
                <Icon className="w-3.5 h-3.5" />
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
                <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">{label}</p>
                <span
                    className={cn("text-[9px] font-bold uppercase tracking-wider px-1 py-px rounded", badge.tone)}
                    title={badge.title}
                >
                    {badge.text}
                </span>
            </div>
            <p className="text-sm font-bold">
                {formatEventDate(date)}{dateEnd ? ` – ${formatEventDate(dateEnd)}` : ''}
                {relative && <span className="text-muted-foreground font-medium"> · {relative}</span>}
            </p>
            {detail && (
                <p className={cn(
                    "text-[11px] w-full sm:w-auto sm:ml-auto sm:text-right truncate",
                    detailColor || "text-muted-foreground",
                )}>
                    {detail}
                </p>
            )}
        </div>
    );
};
