'use client';
import React, { useMemo } from 'react';
import { Calendar, TrendingUp, Wallet, CalendarClock, Percent, Receipt, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { Dividend } from '../../lib/api';
import { formatCompactNumber, cn } from '../../lib/utils';

interface IncomeKpiStripProps {
    dividends: Dividend[];
    currency: string;
    expectedDividends?: number;
    dividendYield?: number;
}

interface KpiTileProps {
    label: string;
    value: string;
    sub?: React.ReactNode;
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
        <div className="flex-1 min-w-[150px] px-4 py-3 first:pl-0 last:pr-0">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold mb-1.5">
                {Icon && <Icon className="w-3 h-3" />}
                <span>{label}</span>
            </div>
            <div className={cn('text-xl sm:text-2xl font-bold tabular-nums leading-none truncate', toneClass)}>
                {value}
            </div>
            {sub && (
                <div className="text-[11px] text-muted-foreground/80 mt-1.5 leading-tight">
                    {sub}
                </div>
            )}
        </div>
    );
}

export default function IncomeKpiStrip({
    dividends,
    currency,
    expectedDividends,
    dividendYield,
}: IncomeKpiStripProps) {
    const metrics = useMemo(() => {
        const now = new Date();
        const currentYear = now.getFullYear();

        // "Same point in the prior year" cutoff — used so YoY YTD isn't biased
        // by the partial year-to-date window comparing against a full prior year.
        const priorCutoff = new Date(now);
        priorCutoff.setFullYear(currentYear - 1);

        const oneYearAgo = new Date(now);
        oneYearAgo.setFullYear(now.getFullYear() - 1);

        let ytd = 0;
        let priorYtd = 0;
        let trailing12m = 0;
        let trailing12mTax = 0;
        let totalGross = 0;
        let totalTax = 0;

        for (const div of dividends) {
            const d = new Date(div.Date);
            if (isNaN(d.getTime())) continue;
            const gross = div.DividendAmountDisplayCurrency || 0;
            const tax = div.TaxAmountDisplayCurrency || 0;

            totalGross += gross;
            totalTax += tax;

            if (d.getFullYear() === currentYear) {
                ytd += gross;
            } else if (d.getFullYear() === currentYear - 1 && d <= priorCutoff) {
                priorYtd += gross;
            }

            if (d >= oneYearAgo) {
                trailing12m += gross;
                trailing12mTax += tax;
            }
        }

        const yoyPct = priorYtd > 0 ? ((ytd - priorYtd) / priorYtd) * 100 : null;
        const yoyAbs = ytd - priorYtd;
        const avgMonthly = trailing12m / 12;
        // Tax efficiency: % of GROSS that stays with the investor. 100% means no
        // tax was withheld; lower means more was lost to tax.
        const taxEfficiency = trailing12m > 0 ? ((trailing12m - trailing12mTax) / trailing12m) * 100 : null;

        return {
            ytd, priorYtd, yoyPct, yoyAbs,
            trailing12m, avgMonthly,
            taxEfficiency,
            totalGross, totalTax,
        };
    }, [dividends]);

    const fmt = (v: number) => formatCompactNumber(v, currency);

    return (
        <div className="metric-card p-4">
            <div className="flex flex-wrap divide-x divide-border/60">
                <KpiTile
                    label="YTD Received"
                    value={fmt(metrics.ytd)}
                    sub={
                        metrics.yoyPct != null ? (
                            <span className={cn(
                                'inline-flex items-center gap-0.5 font-semibold tabular-nums',
                                metrics.yoyPct >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400',
                            )}>
                                {metrics.yoyPct >= 0 ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                                {metrics.yoyPct >= 0 ? '+' : ''}{metrics.yoyPct.toFixed(1)}% YoY
                                <span className="text-muted-foreground/60 ml-1 font-normal">
                                    ({metrics.yoyAbs >= 0 ? '+' : ''}{fmt(Math.abs(metrics.yoyAbs))})
                                </span>
                            </span>
                        ) : (
                            <span className="text-muted-foreground/60">vs prior YTD</span>
                        )
                    }
                    tone="pos"
                    icon={Calendar}
                />
                <KpiTile
                    label="Trailing 12M"
                    value={fmt(metrics.trailing12m)}
                    sub="received in last year"
                    tone="neutral"
                    icon={TrendingUp}
                />
                <KpiTile
                    label="Avg Monthly"
                    value={fmt(metrics.avgMonthly)}
                    sub="trailing 12M ÷ 12"
                    tone="neutral"
                    icon={Wallet}
                />
                {expectedDividends != null && (
                    <KpiTile
                        label="Expected 12M"
                        value={fmt(expectedDividends)}
                        sub="forward indicated rate"
                        tone="pos"
                        icon={CalendarClock}
                    />
                )}
                {dividendYield != null && (
                    <KpiTile
                        label="Annual Yield"
                        value={`${dividendYield.toFixed(2)}%`}
                        sub="on current portfolio"
                        tone="neutral"
                        icon={Percent}
                    />
                )}
                {metrics.taxEfficiency != null && (
                    <KpiTile
                        label="Tax Efficiency"
                        value={`${metrics.taxEfficiency.toFixed(0)}%`}
                        sub={`${fmt(metrics.totalTax)} paid · trailing 12M`}
                        tone={metrics.taxEfficiency >= 85 ? 'pos' : metrics.taxEfficiency >= 70 ? 'warn' : 'neg'}
                        icon={Receipt}
                    />
                )}
            </div>
        </div>
    );
}
