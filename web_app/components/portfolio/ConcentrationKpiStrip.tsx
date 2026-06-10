'use client';
import React, { useMemo } from 'react';
import { Hash, Crown, Layers, Sigma, AlertTriangle } from 'lucide-react';
import { Holding } from '../../lib/api';
import { cn } from '../../lib/utils';

interface ConcentrationKpiStripProps {
    holdings: Holding[];
    currency: string;
}

interface KpiTileProps {
    label: string;
    value: string;
    sub?: string;
    tone?: 'neutral' | 'pos' | 'warn' | 'neg';
    icon?: React.ComponentType<{ className?: string }>;
}

function KpiTile({ label, value, sub, tone = 'neutral', icon: Icon }: KpiTileProps) {
    const toneClass =
        tone === 'pos'  ? 'text-emerald-600 dark:text-emerald-400'
        : tone === 'warn' ? 'text-amber-600 dark:text-amber-400'
        : tone === 'neg'  ? 'text-red-600 dark:text-red-400'
        : 'text-foreground';

    // Mobile → tablet: fills its responsive grid cell. xl+: becomes a
    // single-row flex strip item with vertical dividers (handled on the parent).
    return (
        <div className="min-w-0 px-1 py-1.5 xl:flex-1 xl:min-w-[120px] xl:px-3 xl:py-2.5 xl:first:pl-0 xl:last:pr-0">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold mb-1">
                {Icon && <Icon className="w-3 h-3 shrink-0" />}
                <span className="truncate">{label}</span>
            </div>
            <div className={cn('text-base sm:text-lg font-bold tabular-nums leading-none truncate', toneClass)}>
                {value}
            </div>
            {sub && (
                <div className="text-[10px] text-muted-foreground/70 tabular-nums mt-1 leading-tight truncate">
                    {sub}
                </div>
            )}
        </div>
    );
}

// Treat $CASH / Cash (...) / CASH symbols as cash positions — they're real money
// but conceptually separate from "stock holdings" for diversification metrics.
function isCashSymbol(symbol: string | undefined): boolean {
    if (!symbol) return false;
    const s = symbol.toUpperCase();
    return s === '$CASH' || s === 'CASH' || s.startsWith('CASH (');
}

export default function ConcentrationKpiStrip({ holdings, currency }: ConcentrationKpiStripProps) {
    const metrics = useMemo(() => {
        const mvKey = `Market Value (${currency})`;
        const positions = (holdings ?? [])
            .map(h => ({
                symbol: h.Symbol as string,
                isCash: isCashSymbol(h.Symbol as string),
                value: Math.max(0, (h[mvKey] as number) || 0),
            }))
            .filter(p => p.value > 0);

        const total = positions.reduce((s, p) => s + p.value, 0);
        const stockPositions = positions.filter(p => !p.isCash);
        const cashPositions  = positions.filter(p => p.isCash);

        // Sort stock positions by weight desc for top-N math.
        const sorted = [...stockPositions].sort((a, b) => b.value - a.value);
        const weights = sorted.map(p => (total > 0 ? p.value / total : 0));

        const largest = sorted[0] ?? null;
        const largestPct = total > 0 && largest ? (largest.value / total) * 100 : null;

        const sumTopN = (n: number) =>
            weights.slice(0, n).reduce((s, w) => s + w, 0) * 100;

        const top5 = sorted.length >= 1 ? sumTopN(5) : null;
        const top10 = sorted.length >= 1 ? sumTopN(10) : null;

        // Herfindahl-Hirschman Index over stock positions (cash excluded — it has
        // no concentration risk).
        const hhi = weights.reduce((s, w) => s + w * w, 0);
        // Effective number of stocks: 1 / HHI. Caps the "diversification equivalent".
        const effectiveN = hhi > 0 ? 1 / hhi : null;

        // Cash drag — purely informational, not a concentration metric.
        const cashTotal = cashPositions.reduce((s, p) => s + p.value, 0);
        const cashPct = total > 0 ? (cashTotal / total) * 100 : 0;

        return {
            stockCount: stockPositions.length,
            cashCount: cashPositions.length,
            largest: largest ? { symbol: largest.symbol, pct: largestPct } : null,
            top5,
            top10,
            effectiveN,
            cashPct,
        };
    }, [holdings, currency]);

    // Concentration thresholds: HHI > 0.25 ≈ "concentrated" (i.e. effective N < 4).
    // Use effective N tone instead of raw HHI for readability.
    const effNTone = (() => {
        if (metrics.effectiveN == null) return 'neutral' as const;
        if (metrics.effectiveN >= 10) return 'pos' as const;
        if (metrics.effectiveN >= 5)  return 'neutral' as const;
        return 'warn' as const;
    })();

    const largestTone = (() => {
        if (metrics.largest?.pct == null) return 'neutral' as const;
        if (metrics.largest.pct >= 25) return 'warn' as const;
        if (metrics.largest.pct >= 15) return 'neutral' as const;
        return 'pos' as const;
    })();

    return (
        <div className="metric-card p-3 sm:p-4">
            <div className="grid grid-cols-2 gap-x-3 gap-y-3 sm:grid-cols-3 lg:grid-cols-4 xl:flex xl:gap-0 xl:divide-x xl:divide-border/60">
                <KpiTile
                    label="Holdings"
                    value={metrics.stockCount.toString()}
                    sub={metrics.cashCount > 0 ? `+ ${metrics.cashCount} cash` : 'stocks & funds'}
                    icon={Hash}
                />
                <KpiTile
                    label="Largest"
                    value={metrics.largest ? metrics.largest.symbol : '–'}
                    sub={metrics.largest?.pct != null ? `${metrics.largest.pct.toFixed(1)}%` : undefined}
                    tone={largestTone}
                    icon={Crown}
                />
                <KpiTile
                    label="Top 5"
                    value={metrics.top5 != null ? `${metrics.top5.toFixed(1)}%` : '–'}
                    sub="of portfolio"
                    icon={Layers}
                />
                <KpiTile
                    label="Top 10"
                    value={metrics.top10 != null ? `${metrics.top10.toFixed(1)}%` : '–'}
                    sub="of portfolio"
                    icon={Layers}
                />
                <KpiTile
                    label="Effective N"
                    value={metrics.effectiveN != null ? metrics.effectiveN.toFixed(1) : '–'}
                    sub="equal-weight equiv."
                    tone={effNTone}
                    icon={Sigma}
                />
                <KpiTile
                    label="Cash"
                    value={`${metrics.cashPct.toFixed(1)}%`}
                    sub={metrics.cashPct > 20 ? 'heavy cash drag' : 'of portfolio'}
                    tone={metrics.cashPct > 20 ? 'warn' : 'neutral'}
                    icon={AlertTriangle}
                />
            </div>
        </div>
    );
}
