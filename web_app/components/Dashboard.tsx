import React from 'react';
import { PortfolioSummary } from '../lib/api';
import { formatCurrency } from '../lib/utils';

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
    <div className={`
        relative overflow-hidden rounded-2xl p-5 transition-all duration-300 h-full
        ${isHero
            ? 'bg-gradient-to-br from-white to-slate-50 dark:from-slate-800 dark:to-slate-900 shadow-md hover:shadow-lg border border-slate-200 dark:border-slate-700'
            : 'bg-white dark:bg-slate-800 shadow-sm hover:shadow border border-slate-100 dark:border-slate-700'
        }
        ${containerClassName}
    `}>
        {isHero && (
            <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-gradient-to-br from-slate-100 to-transparent dark:from-slate-700 rounded-full opacity-50 blur-2xl pointer-events-none"></div>
        )}

        <p className="text-sm font-medium text-slate-500 dark:text-slate-400 relative z-10">{title}</p>

        <div className={`mt-2 flex items-baseline gap-2 flex-wrap relative z-10`}>
            <h3 className={`font-extrabold tracking-tight ${valueClassName} ${colorClass || 'text-slate-900 dark:text-white'}`}>
                {value !== null && value !== undefined ? (isCurrency ? formatCurrency(value, currency) : value) : '-'}
            </h3>
            {subValue && (
                <span className={`
                    ${subValueClassName || valueClassName} font-semibold
                    ${subValue >= 0
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : 'text-rose-600 dark:text-rose-400'
                    }
                `}>
                    ({subValue > 0 ? '+' : ''}{subValue.toFixed(2)}%)
                </span>
            )}
        </div>
    </div>
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
