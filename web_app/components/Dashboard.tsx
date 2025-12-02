import React from 'react';
import { PortfolioSummary } from '../lib/api';

interface DashboardProps {
    summary: PortfolioSummary;
}

const MetricCard = ({ title, value, subValue, isCurrency = true, isPercent = false, colorClass = '' }: any) => (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-100 dark:border-gray-700">
        <p className="text-sm text-gray-500 dark:text-gray-400 font-medium">{title}</p>
        <div className="mt-1 flex items-baseline justify-between">
            <h3 className={`text-2xl font-bold ${colorClass}`}>
                {value !== null && value !== undefined ? (isCurrency ? `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : value) : '-'}
            </h3>
            {subValue && (
                <span className={`text-sm font-semibold ${subValue >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {subValue > 0 ? '+' : ''}{subValue.toFixed(2)}%
                </span>
            )}
        </div>
    </div>
);

export default function Dashboard({ summary }: DashboardProps) {
    const m = summary?.metrics;

    if (!m) {
        return <div className="p-4 text-center text-gray-500">Loading metrics...</div>;
    }

    const dayGLColor = (m['Day\'s G/L'] || 0) >= 0 ? 'text-green-600' : 'text-red-600';
    const unrealizedGLColor = (m['Unrealized G/L'] || 0) >= 0 ? 'text-green-600' : 'text-red-600';

    return (
        <div className="grid grid-cols-1 gap-4 p-4">
            <MetricCard
                title="Total Portfolio Value"
                value={m['Total Value']}
                subValue={m['Day\'s G/L %']}
            />

            <div className="grid grid-cols-2 gap-4">
                <MetricCard
                    title="Day's Gain/Loss"
                    value={m['Day\'s G/L']}
                    colorClass={dayGLColor}
                />
                <MetricCard
                    title="Unrealized G/L"
                    value={m['Unrealized G/L']}
                    subValue={m['Unrealized G/L %']}
                    colorClass={unrealizedGLColor}
                />
            </div>

            <div className="grid grid-cols-2 gap-4">
                <MetricCard
                    title="Cash Balance"
                    value={m['Cash Balance']}
                />
                <MetricCard
                    title="YTD Dividends"
                    value={m['Dividends']}
                />
            </div>
        </div>
    );
}
