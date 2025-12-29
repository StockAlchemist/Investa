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
            <div className="bg-card backdrop-blur-md rounded-xl p-6 shadow-sm border border-border animate-pulse">
                <div className="h-6 w-32 bg-white/10 rounded mb-6"></div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[1, 2, 3, 4].map((i) => (
                        <div key={i} className="space-y-2">
                            <div className="h-4 w-20 bg-white/5 rounded"></div>
                            <div className="h-8 w-16 bg-white/10 rounded"></div>
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
            color: metrics['Sharpe Ratio'] && metrics['Sharpe Ratio'] > 1 ? 'text-emerald-500' : 'text-muted-foreground',
        },
        {
            label: 'Volatility',
            value: formatPercent(metrics['Volatility (Ann.)']),
            description: 'Annualized std dev',
            color: 'text-muted-foreground',
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
            color: metrics['Sortino Ratio'] && metrics['Sortino Ratio'] > 1 ? 'text-emerald-500' : 'text-muted-foreground',
        },
    ];

    return (
        <div className="bg-card backdrop-blur-md rounded-xl p-6 shadow-sm border border-border">
            <h3 className="text-sm font-medium text-muted-foreground mb-6 uppercase tracking-wider">Risk Analytics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                {items.map((item) => (
                    <div key={item.label} className="group">
                        <p className="text-xs text-muted-foreground mb-1">{item.label}</p>
                        <p className={`text-2xl font-bold ${item.color} tracking-tight`}>
                            {item.value}
                        </p>
                        <p className="text-[10px] text-muted-foreground mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            {item.description}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
}
