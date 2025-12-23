'use client';

import React from 'react';

interface RiskMetricsProps {
    metrics: {
        'Max Drawdown'?: number;
        'Volatility (Ann.)'?: number;
        'Sharpe Ratio'?: number;
        'Sortino Ratio'?: number;
    };
    isLoading: boolean;
}

export default function RiskMetrics({ metrics, isLoading }: RiskMetricsProps) {
    if (isLoading) {
        return (
            <div className="bg-white dark:bg-zinc-900 rounded-xl p-6 shadow-sm border border-zinc-200 dark:border-zinc-800 animate-pulse">
                <div className="h-6 w-32 bg-zinc-200 dark:bg-zinc-800 rounded mb-6"></div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[1, 2, 3, 4].map((i) => (
                        <div key={i} className="space-y-2">
                            <div className="h-4 w-20 bg-zinc-100 dark:bg-zinc-800 rounded"></div>
                            <div className="h-8 w-16 bg-zinc-200 dark:bg-zinc-800 rounded"></div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    const formatPercent = (val: number | undefined) => {
        if (val === undefined) return 'N/A';
        return `${(val * 100).toFixed(2)}%`;
    };

    const formatNumber = (val: number | undefined) => {
        if (val === undefined) return 'N/A';
        return val.toFixed(2);
    };

    const items = [
        {
            label: 'Sharpe Ratio',
            value: formatNumber(metrics['Sharpe Ratio']),
            description: 'Risk-adjusted return',
            color: metrics['Sharpe Ratio'] && metrics['Sharpe Ratio'] > 1 ? 'text-emerald-500' : 'text-zinc-500',
        },
        {
            label: 'Volatility',
            value: formatPercent(metrics['Volatility (Ann.)']),
            description: 'Annualized std dev',
            color: 'text-zinc-500',
        },
        {
            label: 'Max Drawdown',
            value: formatPercent(metrics['Max Drawdown']),
            description: 'Peak to trough decline',
            color: 'text-rose-500',
        },
        {
            label: 'Sortino Ratio',
            value: formatNumber(metrics['Sortino Ratio']),
            description: 'Downside risk-adjusted',
            color: metrics['Sortino Ratio'] && metrics['Sortino Ratio'] > 1 ? 'text-emerald-500' : 'text-zinc-500',
        },
    ];

    return (
        <div className="bg-white dark:bg-zinc-900 rounded-xl p-6 shadow-sm border border-zinc-200 dark:border-zinc-800">
            <h3 className="text-sm font-medium text-zinc-500 dark:text-zinc-400 mb-6 uppercase tracking-wider">Risk Analytics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                {items.map((item) => (
                    <div key={item.label} className="group">
                        <p className="text-xs text-zinc-400 dark:text-zinc-500 mb-1">{item.label}</p>
                        <p className={`text-2xl font-bold ${item.color} tracking-tight`}>
                            {item.value}
                        </p>
                        <p className="text-[10px] text-zinc-400 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            {item.description}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
}
