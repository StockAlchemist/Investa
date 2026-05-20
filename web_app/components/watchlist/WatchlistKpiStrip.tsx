'use client';
import React, { useMemo } from 'react';
import { Hash, TrendingUp, TrendingDown, Activity, Target } from 'lucide-react';
import { WatchlistItem } from '../../lib/api';
import { cn } from '../../lib/utils';

interface WatchlistKpiStripProps {
    items: WatchlistItem[];
}

interface KpiTileProps {
    label: string;
    value: string;
    sub?: string;
    tone?: 'neutral' | 'pos' | 'neg' | 'warn';
    icon?: React.ComponentType<{ className?: string }>;
}

function KpiTile({ label, value, sub, tone = 'neutral', icon: Icon }: KpiTileProps) {
    const toneClass =
        tone === 'pos'  ? 'text-emerald-600 dark:text-emerald-400'
        : tone === 'neg'  ? 'text-red-600 dark:text-red-400'
        : tone === 'warn' ? 'text-amber-600 dark:text-amber-400'
        : 'text-foreground';
    return (
        <div className="flex-1 min-w-[120px] px-4 py-2.5 first:pl-0 last:pr-0">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold mb-1.5">
                {Icon && <Icon className="w-3 h-3" />}
                <span>{label}</span>
            </div>
            <div className={cn('text-lg sm:text-xl font-bold tabular-nums leading-none truncate', toneClass)}>
                {value}
            </div>
            {sub && (
                <div className="text-[10px] text-muted-foreground/70 tabular-nums mt-1 leading-none truncate">
                    {sub}
                </div>
            )}
        </div>
    );
}

export default function WatchlistKpiStrip({ items }: WatchlistKpiStripProps) {
    const m = useMemo(() => {
        const withChange = items.filter(i => typeof i['Day Change %'] === 'number');
        const changes = withChange.map(i => i['Day Change %'] as number);
        const avg = changes.length > 0 ? changes.reduce((s, v) => s + v, 0) / changes.length : null;

        let best: { symbol: string; pct: number } | null = null;
        let worst: { symbol: string; pct: number } | null = null;
        for (const i of withChange) {
            const pct = i['Day Change %'] as number;
            if (!best || pct > best.pct) best = { symbol: i.Symbol, pct };
            if (!worst || pct < worst.pct) worst = { symbol: i.Symbol, pct };
        }

        // "Opportunities": items trading below intrinsic value (positive margin of safety).
        const opportunities = items.filter(i => typeof i.margin_of_safety === 'number' && (i.margin_of_safety as number) > 0).length;
        const advancers = changes.filter(v => v > 0).length;
        const decliners = changes.filter(v => v < 0).length;

        return { count: items.length, avg, best, worst, opportunities, advancers, decliners };
    }, [items]);

    const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;

    return (
        <div className="metric-card p-3 sm:p-4">
            <div className="flex flex-wrap divide-x divide-border/60">
                <KpiTile
                    label="Symbols"
                    value={m.count.toLocaleString()}
                    sub={m.count === 1 ? 'tracked' : 'tracked'}
                    icon={Hash}
                />
                <KpiTile
                    label="Avg Day Change"
                    value={m.avg != null ? fmtPct(m.avg) : '–'}
                    sub={`${m.advancers} up · ${m.decliners} down`}
                    tone={(m.avg ?? 0) >= 0 ? 'pos' : 'neg'}
                    icon={Activity}
                />
                <KpiTile
                    label="Best Today"
                    value={m.best ? fmtPct(m.best.pct) : '–'}
                    sub={m.best?.symbol}
                    tone="pos"
                    icon={TrendingUp}
                />
                <KpiTile
                    label="Worst Today"
                    value={m.worst ? fmtPct(m.worst.pct) : '–'}
                    sub={m.worst?.symbol}
                    tone="neg"
                    icon={TrendingDown}
                />
                <KpiTile
                    label="Opportunities"
                    value={m.opportunities.toLocaleString()}
                    sub="below fair value"
                    tone={m.opportunities > 0 ? 'pos' : 'neutral'}
                    icon={Target}
                />
            </div>
        </div>
    );
}
