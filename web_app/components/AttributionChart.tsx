'use client';

import StockDetailModal from './StockDetailModal';
import { useState } from 'react';

export interface AttributionData {
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
        contribution: number;
    }[];
    total_gain: number;
}

interface CommonProps {
    data: AttributionData;
    isLoading: boolean;
    currency: string;
}

const formatCurrencyHelper = (val: number, currency: string) => {
    const symbol = currency === 'THB' ? '฿' : (new Intl.NumberFormat('en-US', { style: 'currency', currency: currency }).formatToParts(0).find(part => part.type === 'currency')?.value || currency);
    return `${symbol}${val.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
};

const formatPercentHelper = (val: number) => {
    return `${(val * 100).toFixed(1)}%`;
};

export function SectorAttribution({ data, isLoading, currency }: CommonProps) {
    if (isLoading) {
        return (
            <div className="bg-card rounded-xl p-6 shadow-sm border border-border animate-pulse h-80"></div>
        );
    }

    const hasSectors = data?.sectors && data.sectors.length > 0;

    return (
        <div className="bg-card rounded-xl p-6 shadow-sm border border-border h-full">
            <h3 className="text-sm font-medium text-muted-foreground mb-6 uppercase tracking-wider">Sector Contribution</h3>
            <div className="space-y-4">
                {hasSectors ? data.sectors.map((s) => (
                    <div key={s.sector}>
                        <div className="flex justify-between text-xs mb-1.5">
                            <span className="font-medium text-foreground">{s.sector}</span>
                            <span className={s.gain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}>
                                {formatCurrencyHelper(s.gain, currency)} ({formatPercentHelper(s.contribution)})
                            </span>
                        </div>
                        <div className="w-full bg-secondary rounded-full h-1.5 overflow-hidden">
                            <div
                                className={`h-full ${s.gain >= 0 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                                style={{ width: `${Math.min(100, Math.abs(s.contribution * 100))}%` }}
                            ></div>
                        </div>
                    </div>
                )) : (
                    <p className="text-sm text-muted-foreground italic">No sector data available</p>
                )}
            </div>
        </div>
    );
}

export function TopContributors({ data, isLoading, currency }: CommonProps) {
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

    if (isLoading) {
        return (
            <div className="bg-card rounded-xl p-6 shadow-sm border border-border animate-pulse h-80"></div>
        );
    }

    const hasStocks = data?.stocks && data.stocks.length > 0;

    return (
        <>
            <div className="bg-card rounded-xl p-6 shadow-sm border border-border h-full">
                <h3 className="text-sm font-medium text-muted-foreground mb-6 uppercase tracking-wider">Top Contributors</h3>
                <div className="space-y-3">
                    {hasStocks ? data.stocks.map((stock, idx) => (
                        <div
                            key={`${stock.symbol}-${idx}`}
                            className="flex items-center justify-between p-2 hover:bg-accent/10 rounded-lg transition-colors cursor-pointer group"
                            onClick={() => {
                                const symbols = stock.symbol.split(',').map(s => s.trim());
                                if (symbols.length === 1) {
                                    setSelectedSymbol(symbols[0]);
                                }
                            }}
                        >
                            <div className="flex flex-col">
                                <div className="flex flex-wrap gap-1">
                                    {stock.symbol.split(',').map(s => s.trim()).map((sym, i, arr) => (
                                        <span
                                            key={sym}
                                            className="text-sm font-bold text-foreground hover:text-cyan-500 transition-colors cursor-pointer z-10"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setSelectedSymbol(sym);
                                            }}
                                        >
                                            {sym}{i < arr.length - 1 ? ',' : ''}
                                        </span>
                                    ))}
                                </div>
                                <span className="text-[10px] text-muted-foreground truncate max-w-[120px]">{stock.name}</span>
                            </div>
                            <div className="text-right">
                                <p className={`text-sm font-medium ${stock.gain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                                    {stock.gain >= 0 ? '+' : ''}{formatCurrencyHelper(stock.gain, currency)}
                                    <span className="text-xs ml-1">({formatPercentHelper(stock.contribution)})</span>
                                </p>
                                <p className="text-[10px] text-muted-foreground uppercase">{stock.sector}</p>
                            </div>
                        </div>
                    )) : (
                        <p className="text-sm text-muted-foreground italic">No contributor data available</p>
                    )}
                </div>
            </div>

            {selectedSymbol && (
                <StockDetailModal
                    symbol={selectedSymbol}
                    isOpen={!!selectedSymbol}
                    onClose={() => setSelectedSymbol(null)}
                    currency={currency}
                />
            )}
        </>
    );
}

export default function AttributionChart({ data, isLoading, currency }: CommonProps) {
    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SectorAttribution data={data} isLoading={isLoading} currency={currency} />
            <TopContributors data={data} isLoading={isLoading} currency={currency} />
        </div>
    );
}
