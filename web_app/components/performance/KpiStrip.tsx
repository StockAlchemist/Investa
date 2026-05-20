'use client';
import React, { useMemo } from 'react';
import { ArrowUpRight, ArrowDownRight, TrendingUp, Calendar, Target, AlertTriangle, Trophy, Skull } from 'lucide-react';
import { AssetChangeData, PortfolioSummary, RiskMetrics } from '../../lib/api';
import { cn } from '../../lib/utils';

interface KpiStripProps {
    data: AssetChangeData | null;
    summary: PortfolioSummary | null;
    riskMetrics?: RiskMetrics | null;
    benchmarks: string[];
}

interface KpiTileProps {
    label: string;
    value: string;
    sub?: string;
    tone?: 'pos' | 'neg' | 'neutral' | 'warn';
    icon?: React.ComponentType<{ className?: string }>;
}

function KpiTile({ label, value, sub, tone = 'neutral', icon: Icon }: KpiTileProps) {
    const toneClass =
        tone === 'pos' ? 'text-emerald-600 dark:text-emerald-400'
        : tone === 'neg' ? 'text-red-600 dark:text-red-400'
        : tone === 'warn' ? 'text-amber-600 dark:text-amber-400'
        : 'text-foreground';

    return (
        <div className="flex-1 min-w-[110px] px-3 py-2.5 first:pl-0 last:pr-0">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold mb-1">
                {Icon && <Icon className="w-3 h-3" />}
                <span>{label}</span>
            </div>
            <div className={cn('text-base sm:text-lg font-bold tabular-nums leading-none', toneClass)}>
                {value}
            </div>
            {sub && (
                <div className="text-[10px] text-muted-foreground/70 tabular-nums mt-1 leading-none">
                    {sub}
                </div>
            )}
        </div>
    );
}

// Pull a monthly return series for a named series (e.g. "Portfolio", "S&P 500").
function getMonthlyReturns(data: AssetChangeData | null, seriesName: string): { date: string; value: number }[] {
    const monthly = data?.M;
    if (!monthly) return [];
    const key = `${seriesName} M-Return`;
    return monthly
        .map(r => ({ date: r.Date as string, value: r[key] as number | undefined }))
        .filter((r): r is { date: string; value: number } => typeof r.value === 'number' && !Number.isNaN(r.value));
}

function formatPct(v: number | null | undefined, signed = true): string {
    if (v == null || Number.isNaN(v)) return '–';
    const sign = signed && v > 0 ? '+' : '';
    return `${sign}${v.toFixed(2)}%`;
}

export default function KpiStrip({ data, summary, riskMetrics = null, benchmarks }: KpiStripProps) {
    const metrics = useMemo(() => {
        const portfolioMonthly = getMonthlyReturns(data, 'Portfolio');
        const sumMetrics = summary?.metrics;
        const primaryBenchmark = benchmarks[0];
        const benchMonthly = primaryBenchmark ? getMonthlyReturns(data, primaryBenchmark) : [];

        // YTD: prefer backend metric, fallback to summing current-year monthly returns.
        let ytd: number | null = sumMetrics?.ytd_return ?? null;
        if (ytd == null && portfolioMonthly.length > 0) {
            const currentYear = new Date().getFullYear();
            const ytdMonths = portfolioMonthly.filter(m => new Date(m.date).getFullYear() === currentYear);
            ytd = ytdMonths.reduce((s, m) => s + m.value, 0);
        }

        // 1Y: sum of last 12 monthly returns (compounded would be more correct, but additive matches the existing
        // monthly bar chart semantics).
        const last12 = portfolioMonthly.slice(-12);
        const oneYear = last12.length > 0
            ? last12.reduce((acc, m) => acc * (1 + m.value / 100), 1) * 100 - 100
            : null;

        // Win rate over all months we have.
        const winRate = portfolioMonthly.length > 0
            ? (portfolioMonthly.filter(m => m.value > 0).length / portfolioMonthly.length) * 100
            : null;

        // Best / worst month.
        const best = portfolioMonthly.length > 0
            ? portfolioMonthly.reduce((a, b) => (b.value > a.value ? b : a))
            : null;
        const worst = portfolioMonthly.length > 0
            ? portfolioMonthly.reduce((a, b) => (b.value < a.value ? b : a))
            : null;

        // Max drawdown: prefer the risk_metrics endpoint (fraction, e.g. -0.142)
        // and convert to percent; fall back to summary metric which is already in percent.
        const riskDD = riskMetrics?.['Max Drawdown'];
        const maxDD = riskDD != null
            ? riskDD * 100
            : (sumMetrics?.max_drawdown ?? null);

        // vs Primary benchmark over last 12 months: compounded portfolio − compounded benchmark.
        let vsBench: number | null = null;
        if (primaryBenchmark && last12.length > 0 && benchMonthly.length > 0) {
            const benchLast12 = benchMonthly.slice(-12);
            const portRet = last12.reduce((acc, m) => acc * (1 + m.value / 100), 1) * 100 - 100;
            const benchRet = benchLast12.reduce((acc, m) => acc * (1 + m.value / 100), 1) * 100 - 100;
            vsBench = portRet - benchRet;
        }

        return { ytd, oneYear, winRate, best, worst, maxDD, vsBench, primaryBenchmark };
    }, [data, summary, riskMetrics, benchmarks]);

    const fmtMonth = (date: string) => {
        try {
            return new Date(date).toLocaleDateString(undefined, { month: 'short', year: '2-digit' });
        } catch {
            return date;
        }
    };

    return (
        <div className="metric-card p-3 sm:p-4">
            <div className="flex flex-wrap divide-x divide-border/60">
                <KpiTile
                    label="YTD"
                    value={formatPct(metrics.ytd)}
                    tone={(metrics.ytd ?? 0) >= 0 ? 'pos' : 'neg'}
                    icon={Calendar}
                />
                <KpiTile
                    label="1Y"
                    value={formatPct(metrics.oneYear)}
                    tone={(metrics.oneYear ?? 0) >= 0 ? 'pos' : 'neg'}
                    icon={TrendingUp}
                />
                <KpiTile
                    label="Win Rate"
                    value={metrics.winRate != null ? `${metrics.winRate.toFixed(0)}%` : '–'}
                    sub="of months"
                    tone={(metrics.winRate ?? 0) >= 50 ? 'pos' : 'warn'}
                    icon={Target}
                />
                <KpiTile
                    label="Best Month"
                    value={metrics.best ? formatPct(metrics.best.value) : '–'}
                    sub={metrics.best ? fmtMonth(metrics.best.date) : undefined}
                    tone="pos"
                    icon={Trophy}
                />
                <KpiTile
                    label="Worst Month"
                    value={metrics.worst ? formatPct(metrics.worst.value) : '–'}
                    sub={metrics.worst ? fmtMonth(metrics.worst.date) : undefined}
                    tone="neg"
                    icon={Skull}
                />
                <KpiTile
                    label="Max DD"
                    value={metrics.maxDD != null ? `${metrics.maxDD.toFixed(2)}%` : '–'}
                    tone="warn"
                    icon={AlertTriangle}
                />
                {metrics.primaryBenchmark && (
                    <KpiTile
                        label={`vs ${metrics.primaryBenchmark}`}
                        value={metrics.vsBench != null ? formatPct(metrics.vsBench) : '–'}
                        sub="last 12M"
                        tone={(metrics.vsBench ?? 0) >= 0 ? 'pos' : 'neg'}
                        icon={(metrics.vsBench ?? 0) >= 0 ? ArrowUpRight : ArrowDownRight}
                    />
                )}
            </div>
        </div>
    );
}
