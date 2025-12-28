import React from 'react';
import { PortfolioSummary } from '../lib/api';
import { formatCurrency, cn } from '../lib/utils';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
interface DashboardProps {
    summary: PortfolioSummary;
    currency: string;
}

const MetricCard = ({
    title,
    value,
    subValue,
    isCurrency = true,
    isPercent = false,
    colorClass = '',
    vertical = false,
    valueClassName = 'text-2xl',
    containerClassName = '',
    isHero = false,
    subValueClassName = '',
    currency = 'USD'
}: any) => (
    <Card className={cn(
        "h-full transition-all duration-300 relative overflow-hidden group",
        "bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 border-black/5 dark:border-white/10",
        containerClassName
    )}>

        <CardContent className="h-full flex flex-col justify-center p-6">
            <p className="text-sm font-medium text-muted-foreground relative z-10 uppercase tracking-wider text-[10px]">{title}</p>

            <div className="mt-2 flex items-center gap-3 flex-wrap relative z-10">
                <h3 className={cn("font-bold tracking-tight", colorClass || "text-foreground", valueClassName)}>
                    {value !== null && value !== undefined ? (isCurrency ? formatCurrency(value, currency) : value) : '-'}
                </h3>
                {subValue && (
                    <Badge variant={subValue >= 0 ? "success" : "destructive"} className={cn("text-xs font-semibold px-2 py-0.5", subValueClassName)}>
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
    { id: 'totalReturn', colSpan: '' },
    { id: 'annualTWR', colSpan: '' },
    { id: 'unrealizedGL', colSpan: '' },
    { id: 'unrealizedGLPct', colSpan: '' },
    { id: 'realizedGain', colSpan: '' },
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
    const cashBalance = am?.['Cash']?.['total_market_value_display'] || 0;
    const dayGL = m.day_change_display || 0;
    const dayGLPct = m.day_change_percent || 0;
    const unrealizedGL = m.unrealized_gain || 0;
    const unrealizedGLPct = (m.unrealized_gain / (m.cost_basis_held || 1)) * 100;

    const dayGLColor = dayGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const unrealizedGLColor = unrealizedGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    const totalGain = m.total_gain || 0;
    const realizedGain = m.realized_gain || 0;

    const totalReturnColor = totalGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const realizedGainColor = realizedGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    // Render helper same as before
    const renderContent = (id: string) => {
        switch (id) {
            case 'portfolioValue':
                return <MetricCard
                    title="Total Portfolio Value"
                    value={m.market_value}
                    valueClassName="text-4xl"
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
                    valueClassName="text-4xl"
                    subValueClassName="text-2xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                />;
            case 'totalReturn':
                return <MetricCard
                    title="Total Return"
                    value={totalGain}
                    colorClass={totalReturnColor}
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
                    colorClass={unrealizedGLColor}
                    currency={currency}
                />;
            case 'unrealizedGLPct':
                return <MetricCard
                    title="Unrealized G/L %"
                    value={unrealizedGLPct !== undefined && unrealizedGLPct !== null ? `${unrealizedGLPct.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass={unrealizedGLColor}
                />;
            case 'realizedGain':
                return <MetricCard
                    title="Realized Gain"
                    value={realizedGain}
                    colorClass={realizedGainColor}
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
