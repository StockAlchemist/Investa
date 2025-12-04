import React from 'react';
import { PortfolioSummary } from '../lib/api';

interface DashboardProps {
    summary: PortfolioSummary;
}

const MetricCard = ({ title, value, subValue, isCurrency = true, isPercent = false, colorClass = '', vertical = false, valueClassName = 'text-2xl', containerClassName = '' }: any) => (
    <div className={`bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-100 dark:border-gray-700 ${containerClassName}`}>
        <p className="text-sm text-gray-500 dark:text-gray-400 font-medium">{title}</p>
        <div className={`mt-1 flex ${vertical ? 'flex-col' : 'items-baseline justify-between'}`}>
            <h3 className={`font-bold ${colorClass} ${valueClassName}`}>
                {value !== null && value !== undefined ? (isCurrency ? `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : value) : '-'}
            </h3>
            {subValue && (
                <span className={`text-sm font-semibold ${subValue >= 0 ? 'text-green-600' : 'text-red-600'} ${vertical ? 'mt-0.5' : ''}`}>
                    {subValue > 0 ? '+' : ''}{subValue.toFixed(2)}%
                </span>
            )}
        </div>
    </div>
);

export default function Dashboard({ summary }: DashboardProps) {
    const m = summary?.metrics;
    const am = summary?.account_metrics;

    if (!m) {
        return <div className="p-4 text-center text-gray-500">Loading metrics...</div>;
    }

    // Calculate Cash Balance from Account Metrics if available, or look for it in metrics
    // Based on logs, Cash is an account in account_metrics
    const cashBalance = am?.['Cash']?.['total_market_value_display'] || 0;

    const dayGL = m.day_change_display || 0;
    const dayGLPct = m.day_change_percent || 0;
    const unrealizedGL = m.unrealized_gain || 0;
    // Calculate unrealized GL % if not provided directly
    const unrealizedGLPct = (m.unrealized_gain / (m.cost_basis_held || 1)) * 100;

    const dayGLColor = dayGL >= 0 ? 'text-green-600' : 'text-red-600';
    const unrealizedGLColor = unrealizedGL >= 0 ? 'text-green-600' : 'text-red-600';

    const totalReturnPct = m.total_return_pct || 0;
    const totalGain = m.total_gain || 0;
    const realizedGain = m.realized_gain || 0;

    const totalReturnColor = totalGain >= 0 ? 'text-green-600' : 'text-red-600';
    const realizedGainColor = realizedGain >= 0 ? 'text-green-600' : 'text-red-600';

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div className="col-span-1 md:col-span-2 lg:col-span-2">
                <MetricCard
                    title="Total Portfolio Value"
                    value={m.market_value}
                    valueClassName="text-4xl"
                    containerClassName="h-full flex flex-col justify-center"
                />
            </div>

            <div className="col-span-1 md:col-span-2 lg:col-span-2">
                <MetricCard
                    title="Day's Gain/Loss"
                    value={dayGL}
                    subValue={dayGLPct}
                    colorClass={dayGLColor}
                    vertical={true}
                    valueClassName="text-4xl"
                    containerClassName="h-full flex flex-col justify-center"
                />
            </div>

            <MetricCard
                title="Total Return"
                value={totalGain}
                colorClass={totalReturnColor}
                vertical={true}
            />

            <MetricCard
                title="Total Return %"
                value={totalReturnPct !== undefined && totalReturnPct !== null ? `${totalReturnPct.toFixed(2)}%` : '-'}
                isCurrency={false}
                colorClass={totalReturnColor}
                vertical={true}
            />

            <MetricCard
                title="Annual TWR"
                value={m.annualized_twr !== undefined && m.annualized_twr !== null ? `${m.annualized_twr.toFixed(2)}%` : '-'}
                isCurrency={false}
                colorClass={m.annualized_twr && m.annualized_twr >= 0 ? 'text-green-600' : 'text-red-600'}
                vertical={true}
            />

            <MetricCard
                title="Unrealized G/L"
                value={unrealizedGL}
                subValue={unrealizedGLPct}
                colorClass={unrealizedGLColor}
                vertical={true}
            />

            <MetricCard
                title="Realized Gain"
                value={realizedGain}
                colorClass={realizedGainColor}
            />

            <MetricCard
                title="Cash Balance"
                value={cashBalance}
            />

            <MetricCard
                title="YTD Dividends"
                value={m.dividends}
            />

            <MetricCard
                title="Fees"
                value={m.commissions}
            />

        </div>
    );
}

