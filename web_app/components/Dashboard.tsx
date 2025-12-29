import React from 'react';
import { PortfolioSummary } from '../lib/api';
import { formatCurrency, cn } from '../lib/utils';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
interface DashboardProps {
    summary: PortfolioSummary;
    currency: string;
}

interface MetricCardProps {
    title: string;
    value: string | number;
    subValue?: number;
    isCurrency?: boolean;
    colorClass?: string;
    valueClassName?: string;
    containerClassName?: string;
    subValueClassName?: string;
    currency?: string;
    isHero?: boolean;
    isPercent?: boolean;
    vertical?: boolean;
}

const MetricCard = ({
    title,
    value,
    subValue,
    isCurrency = true,
    colorClass = '',
    valueClassName = 'text-xl sm:text-2xl',
    containerClassName = '',
    subValueClassName = '',
    currency = 'USD'
}: MetricCardProps) => (
    <Card className={cn(
        "h-full transition-all duration-300 relative overflow-hidden group",
        "hover:bg-accent/5 transition-colors",
        containerClassName
    )}>

        <CardContent className="h-full flex flex-col justify-center p-4 sm:p-6">
            <p className="text-sm font-medium text-muted-foreground relative z-10 uppercase tracking-wider text-[10px]">{title}</p>

            <div className="mt-2 flex items-center gap-2 sm:gap-3 flex-wrap relative z-10">
                <h3 className={cn("font-bold tracking-tight", colorClass || "text-foreground", valueClassName)}>
                    {value !== null && value !== undefined ? (isCurrency && typeof value === 'number' ? formatCurrency(value, currency) : value) : '-'}
                </h3>
                {subValue !== undefined && subValue !== null && (
                    <Badge variant={subValue >= 0 ? "success" : "destructive"} className={cn("text-[9px] sm:text-[11px] font-bold px-1.5 sm:px-2 py-0.5", subValueClassName)}>
                        {subValue > 0 ? '+' : ''}{subValue.toFixed(2)}%
                    </Badge>
                )}
            </div>
        </CardContent>
    </Card>
);

const DEFAULT_ITEMS = [
    { id: 'portfolioValue', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'dayGL', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'totalReturn', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'unrealizedGL', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'fxGL', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'realizedGain', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'annualTWR', colSpan: '' },
    { id: 'cashBalance', colSpan: '' },
    { id: 'ytdDividends', colSpan: '' },
    { id: 'fees', colSpan: '' },
];

export default function Dashboard({ summary, currency }: DashboardProps) {
    const m = summary?.metrics;
    const am = summary?.account_metrics;

    if (!m) {
        return (
            <div className="flex items-center justify-center p-12">
                <div className="animate-pulse flex flex-col items-center">
                    <div className="h-4 w-32 bg-slate-200 dark:bg-slate-700 rounded mb-4"></div>
                    <div className="h-8 w-48 bg-slate-200 dark:bg-slate-700 rounded"></div>
                </div>
            </div>
        );
    }

    // Prepare data helpers
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const cashBalance = (am?.['Cash'] as any)?.['total_market_value_display'] || 0;
    const dayGL = (m.day_change_display as number) || 0;
    const dayGLPct = (m.day_change_percent as number) || 0;
    const unrealizedGL = (m.unrealized_gain as number) || 0;
    const unrealizedGLPct = ((m.unrealized_gain as number) / ((m.cost_basis_held as number) || 1)) * 100;
    const fxGL = (m.fx_gain_loss_display as number) || 0;
    const fxGLPct = (m.fx_gain_loss_pct as number) || 0;

    const dayGLColor = dayGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const unrealizedGLColor = unrealizedGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const fxGLColor = fxGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    const totalGain = (m.total_gain as number) || 0;
    const realizedGain = (m.realized_gain as number) || 0;

    const totalReturnColor = totalGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const realizedGainColor = realizedGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    // Render helper same as before
    const renderContent = (id: string) => {
        switch (id) {
            case 'portfolioValue':
                return <MetricCard
                    title="Total Portfolio Value"
                    value={m.market_value}
                    valueClassName="text-3xl sm:text-5xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                />;
            case 'dayGL':
                return <MetricCard
                    title="Day's Gain/Loss"
                    value={dayGL}
                    subValue={dayGLPct}
                    colorClass={dayGLColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", dayGLPct >= 0 ? "bg-emerald-600 text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                />;
            case 'totalReturn':
                return <MetricCard
                    title="Total Return"
                    value={totalGain}
                    subValue={m.total_return_pct}
                    colorClass={totalReturnColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", (m.total_return_pct || 0) >= 0 ? "bg-emerald-600 text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                />;
            case 'annualTWR':
                return <MetricCard
                    title="Annual TWR"
                    value={m.annualized_twr !== undefined && m.annualized_twr !== null ? `${m.annualized_twr.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass={m.annualized_twr && m.annualized_twr >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}
                />;
            case 'unrealizedGL':
                return <MetricCard
                    title="Unrealized G/L"
                    value={unrealizedGL}
                    subValue={unrealizedGLPct}
                    colorClass={unrealizedGLColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", unrealizedGLPct >= 0 ? "bg-emerald-600 text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                />;
            case 'fxGL':
                return <MetricCard
                    title="FX Gain/Loss"
                    value={fxGL}
                    subValue={fxGLPct}
                    colorClass={fxGLColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", fxGLPct >= 0 ? "bg-emerald-600 text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                />;
            case 'realizedGain':
                return <MetricCard
                    title="Realized Gain"
                    value={realizedGain}
                    colorClass={realizedGainColor}
                    valueClassName="text-2xl sm:text-3xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                />;
            case 'cashBalance':
                return <MetricCard
                    title="Cash Balance"
                    value={cashBalance}
                    currency={currency}
                />;
            case 'ytdDividends':
                return <MetricCard
                    title="YTD Dividends"
                    value={m.dividends}
                    currency={currency}
                />;
            case 'fees':
                return <MetricCard
                    title="Fees"
                    value={m.commissions}
                    colorClass="text-rose-600 dark:text-rose-400"
                    currency={currency}
                />;
            default:
                return null;
        }
    };

    return (
        <div className="mb-10">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {DEFAULT_ITEMS.map((item) => (
                    <div key={item.id} className={item.colSpan}>
                        {renderContent(item.id)}
                    </div>
                ))}
            </div>
        </div>
    );
}
