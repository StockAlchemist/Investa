'use client';

import React from 'react';

interface AttributionData {
    sectors: {
        sector: string;
        gain: number;
        value: number;
        contribution: number;
    }[];
    stocks: {
        symbol: string;
        name: string;
        gain: number;
        value: number;
        sector: string;
    }[];
    total_gain: number;
}

interface AttributionChartProps {
    data: AttributionData;
    isLoading: boolean;
    currency: string;
}

export default function AttributionChart({ data, isLoading, currency }: AttributionChartProps) {
    if (isLoading) {
        return (
            <div className="bg-white dark:bg-zinc-900 rounded-xl p-6 shadow-sm border border-zinc-200 dark:border-zinc-800 animate-pulse h-80">
            </div>
        );
    }

    // DEBUG: Log data to see what we are receiving
    console.log('[AttributionChart] Received data:', data);

    const formatCurrency = (val: number) => {
        const symbol = currency === 'THB' ? '฿' : (new Intl.NumberFormat('en-US', { style: 'currency', currency: currency }).formatToParts(0).find(part => part.type === 'currency')?.value || currency);
        return `${symbol}${val.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
    };

    const formatPercent = (val: number) => {
        return `${(val * 100).toFixed(1)}%`;
    };

    const hasSectors = data?.sectors && data.sectors.length > 0;
    const hasStocks = data?.stocks && data.stocks.length > 0;

    if (!hasSectors && !hasStocks) {
        return (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-zinc-900 rounded-xl p-6 shadow-sm border border-zinc-200 dark:border-zinc-800 flex items-center justify-center h-40">
                    <p className="text-zinc-500">No sector attribution data available</p>
                </div>
                <div className="bg-white dark:bg-zinc-900 rounded-xl p-6 shadow-sm border border-zinc-200 dark:border-zinc-800 flex items-center justify-center h-40">
                    <p className="text-zinc-500">No top contributor data available</p>
                </div>
            </div>
        )
    }

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Sector Attribution */}
            <div className="bg-white dark:bg-zinc-900 rounded-xl p-6 shadow-sm border border-zinc-200 dark:border-zinc-800">
                <h3 className="text-sm font-medium text-zinc-500 dark:text-zinc-400 mb-6 uppercase tracking-wider">Sector Contribution</h3>
                <div className="space-y-4">
                    {hasSectors ? data.sectors.map((s) => (
                        <div key={s.sector}>
                            <div className="flex justify-between text-xs mb-1.5">
                                <span className="font-medium text-zinc-700 dark:text-zinc-300">{s.sector}</span>
                                <span className={s.gain >= 0 ? 'text-emerald-500' : 'text-rose-500'}>
                                    {formatCurrency(s.gain)} ({formatPercent(s.contribution)})
                                </span>
                            </div>
                            <div className="w-full bg-zinc-100 dark:bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                                <div
                                    className={`h-full ${s.gain >= 0 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                                    style={{ width: `${Math.min(100, Math.abs(s.contribution * 100))}%` }}
                                ></div>
                            </div>
                        </div>
                    )) : (
                        <p className="text-sm text-zinc-400 italic">No data</p>
                    )}
                </div>
            </div>

            {/* Top Contributors */}
            <div className="bg-white dark:bg-zinc-900 rounded-xl p-6 shadow-sm border border-zinc-200 dark:border-zinc-800">
                <h3 className="text-sm font-medium text-zinc-500 dark:text-zinc-400 mb-6 uppercase tracking-wider">Top Contributors</h3>
                <div className="space-y-3">
                    {hasStocks ? data.stocks.map((stock, idx) => (
                        <div key={`${stock.symbol}-${idx}`} className="flex items-center justify-between p-2 hover:bg-zinc-50 dark:hover:bg-zinc-800/50 rounded-lg transition-colors">
                            <div className="flex flex-col">
                                <span className="text-sm font-bold text-zinc-800 dark:text-zinc-200">{stock.symbol}</span>
                                <span className="text-[10px] text-zinc-400 truncate max-w-[120px]">{stock.name}</span>
                            </div>
                            <div className="text-right">
                                <p className={`text-sm font-medium ${stock.gain >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                                    {stock.gain >= 0 ? '+' : ''}{formatCurrency(stock.gain)}
                                </p>
                                <p className="text-[10px] text-zinc-400 uppercase">{stock.sector}</p>
                            </div>
                        </div>
                    )) : (
                        <p className="text-sm text-zinc-400 italic">No data</p>
                    )}
                </div>
            </div>
        </div>
    );
}
