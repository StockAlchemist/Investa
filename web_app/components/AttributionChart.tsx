'use client';

import StockDetailModal from './StockDetailModal';
import { useState, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { Loader2, TrendingUp, TrendingDown, X, Info, LayoutDashboard } from 'lucide-react';
import StockIcon from './StockIcon';
import { cn } from '@/lib/utils';
import { fetchAttribution } from '../lib/api';

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
    isRefreshing?: boolean;
    currency: string;
}

const formatCurrencyHelper = (val: number, currency: string) => {
    const symbol = currency === 'THB' ? '฿' : (new Intl.NumberFormat('en-US', { style: 'currency', currency: currency }).formatToParts(0).find(part => part.type === 'currency')?.value || currency);
    return `${symbol}${val.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
};

const formatPercentHelper = (val: number) => {
    return `${(val * 100).toFixed(1)}%`;
};

export function SectorAttribution({ data, isLoading, isRefreshing = false, currency }: CommonProps) {
    if (isLoading) {
        return (
            <div className="bg-card rounded-xl p-6 shadow-sm animate-pulse h-80"></div>
        );
    }

    const hasSectors = data?.sectors && data.sectors.length > 0;

    return (
        <div className="bg-card rounded-xl p-6 shadow-sm h-full relative overflow-hidden">
            <div className="flex items-center gap-2 mb-6">
                <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Sector Contribution</h3>
                {isRefreshing && !isLoading && (
                    <Loader2 className="w-3 h-3 animate-spin text-cyan-500 opacity-70" />
                )}
            </div>
            <div className="space-y-4">
                {hasSectors ? [...data.sectors].sort((a, b) => b.gain - a.gain).map((s) => (
                    <div key={s.sector}>
                        <div className="flex justify-between text-xs mb-1.5">
                            <span className="font-medium text-foreground">{s.sector}</span>
                            <span className={s.gain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}>
                                {formatCurrencyHelper(s.gain, currency)} ({formatPercentHelper(s.contribution)})
                            </span>
                        </div>
                        <div className="w-full bg-secondary rounded-full h-1.5 overflow-hidden">
                            <div
                                className={`h-full ${s.gain >= 0 ? 'bg-emerald-500' : 'bg-red-500'}`}
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

interface FullContributorsModalProps {
    isOpen: boolean;
    onClose: () => void;
    initialData: AttributionData['stocks'];
    currency: string;
    accounts?: string[];
}

function FullContributorsModal({ isOpen, onClose, initialData, currency, accounts }: FullContributorsModalProps) {
    const [fullData, setFullData] = useState<AttributionData['stocks']>(initialData);
    const [loading, setLoading] = useState(false);
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState('');

    const loadFullData = async () => {
        setLoading(true);
        try {
            const data = await fetchAttribution(currency, accounts, true);
            setFullData(data.stocks);
        } catch (err) {
            console.error("Failed to fetch full contributor list:", err);
        } finally {
            setLoading(false);
        }
    };

    useMemo(() => {
        if (isOpen && fullData.length <= 10) {
            loadFullData();
        }
    }, [isOpen]);

    if (!isOpen) return null;

    const filteredData = fullData
        .filter(s =>
            s.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
            s.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            s.sector.toLowerCase().includes(searchTerm.toLowerCase())
        )
        .sort((a, b) => b.gain - a.gain);

    return createPortal(
        <div className="fixed inset-0 z-[100] flex flex-col justify-end sm:justify-center items-center p-0 sm:p-4 isolate">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

            <div
                style={{ backgroundColor: 'var(--menu-solid)' }}
                className="relative w-full max-w-5xl h-[94vh] sm:h-auto sm:max-h-[90vh] rounded-t-[2.5rem] sm:rounded-[2rem] flex flex-col shadow-2xl overflow-hidden animate-in slide-in-from-bottom sm:zoom-in-95 duration-300 pointer-events-auto"
            >
                {/* Mobile Drag Handle */}
                <div className="sm:hidden w-full flex justify-center pt-3 pb-1 flex-shrink-0">
                    <div className="w-12 h-1.5 bg-border/50 rounded-full" />
                </div>

                {/* Header */}
                <div className="sticky top-0 z-50 bg-card/50 backdrop-blur-md border-b border-border flex-shrink-0 shadow-sm overflow-hidden">
                    <div className="hidden sm:block absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-[100px] -mr-32 -mt-32" />

                    <div className="p-5 sm:p-8 pb-4 sm:pb-6 flex justify-between items-start relative z-10">
                        <div className="flex items-center gap-4 sm:gap-6 flex-1">
                            <div className="w-10 h-10 sm:w-16 sm:h-16 rounded-xl sm:rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 flex-shrink-0">
                                <LayoutDashboard className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                            </div>
                            <div className="flex-1 min-w-0">
                                <h2 className="text-lg sm:text-3xl font-bold tracking-tight text-foreground">All Contributors</h2>
                                <p className="text-muted-foreground text-[10px] sm:text-sm font-medium mt-0.5 uppercase tracking-wider opacity-70">
                                    Impact of individual holdings on performance
                                </p>
                            </div>
                        </div>

                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-muted rounded-full transition-colors text-muted-foreground hover:text-foreground"
                            aria-label="Close modal"
                        >
                            <X className="w-5 h-5 sm:w-6 sm:h-6" />
                        </button>
                    </div>

                    <div className="px-5 sm:px-8 pb-4 sm:pb-6 relative z-10">
                        <div className="relative">
                            <input
                                type="search"
                                placeholder="Search symbols, names, or sectors..."
                                className="w-full bg-background/50 border border-border rounded-xl sm:rounded-2xl px-5 py-3 sm:py-4 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/20 focus:border-cyan-500 transition-all backdrop-blur-sm"
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                autoFocus
                            />
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-4 sm:p-8 custom-scrollbar bg-card/20">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-24 gap-4">
                            <div className="relative">
                                <div className="w-12 h-12 rounded-full border-4 border-cyan-500/20 border-t-cyan-500 animate-spin" />
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <TrendingUp className="w-5 h-5 text-cyan-500" />
                                </div>
                            </div>
                            <p className="text-sm font-bold text-muted-foreground animate-pulse tracking-widest uppercase">Calculating Full Impact...</p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {filteredData.map((stock, idx) => (
                                <div
                                    key={`${stock.symbol}-${idx}`}
                                    className="flex items-center justify-between p-4 bg-card/50 hover:bg-accent/10 rounded-2xl border border-border/50 hover:border-cyan-500/30 transition-all cursor-pointer group hover:scale-[1.01] active:scale-[0.99] duration-200"
                                    onClick={() => {
                                        const symbols = stock.symbol.split(',').map(s => s.trim());
                                        if (symbols.length === 1) {
                                            setSelectedSymbol(symbols[0]);
                                        }
                                    }}
                                >
                                    <div className="flex items-center gap-4">
                                        <div className="bg-background w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform shadow-sm border border-border/50 p-2">
                                            <StockIcon symbol={stock.symbol.split(',')[0]} size="100%" />
                                        </div>
                                        <div className="flex flex-col min-w-0">
                                            <div className="flex flex-wrap gap-1.5 items-center">
                                                {stock.symbol.split(',').map(s => s.trim()).map((sym, i, arr) => (
                                                    <span
                                                        key={sym}
                                                        className="text-base font-bold text-foreground hover:text-cyan-500 transition-colors z-10"
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            setSelectedSymbol(sym);
                                                        }}
                                                    >
                                                        {sym}{i < arr.length - 1 ? ',' : ''}
                                                    </span>
                                                ))}
                                                {stock.value > 0 && (
                                                    <span className="ml-1 px-1.5 py-0.5 rounded-md bg-emerald-500/10 text-emerald-500 text-[9px] font-bold uppercase tracking-wider border border-emerald-500/20">
                                                        Held
                                                    </span>
                                                )}
                                            </div>
                                            <span className="text-xs text-muted-foreground truncate max-w-[180px] font-medium">{stock.name}</span>
                                        </div>
                                    </div>
                                    <div className="text-right flex flex-col items-end gap-1">
                                        <div className="flex items-center justify-end gap-2">
                                            {stock.gain >= 0 ?
                                                <div className="bg-emerald-500/10 p-1 rounded-full"><TrendingUp className="w-3.5 h-3.5 text-emerald-500" /></div> :
                                                <div className="bg-red-500/10 p-1 rounded-full"><TrendingDown className="w-3.5 h-3.5 text-red-500" /></div>
                                            }
                                            <span className={cn("text-base font-medium tracking-tight", stock.gain >= 0 ? 'text-emerald-500' : 'text-red-500')}>
                                                {stock.gain >= 0 ? '+' : ''}{formatCurrencyHelper(stock.gain, currency)}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest bg-secondary/50 px-2 py-0.5 rounded-md">
                                                {formatPercentHelper(stock.contribution)}
                                            </span>
                                            <span className="text-[10px] font-medium text-cyan-500 uppercase tracking-widest">{stock.sector}</span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                            {filteredData.length === 0 && (
                                <div className="col-span-full py-20 text-center bg-card/30 rounded-3xl border border-dashed border-border/50">
                                    <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4 opacity-30">
                                        <Info className="w-8 h-8" />
                                    </div>
                                    <p className="text-lg font-bold text-muted-foreground">No matching holdings</p>
                                    <p className="text-sm text-muted-foreground/60 mt-1">Try searching for a different symbol or industry</p>
                                </div>
                            )}
                        </div>
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
        </div>,
        document.body
    );
}

export function TopContributors({ data, isLoading, isRefreshing = false, currency, accounts }: CommonProps & { accounts?: string[] }) {
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
    const [isAllModalOpen, setIsAllModalOpen] = useState(false);

    if (isLoading) {
        return (
            <div className="bg-card rounded-xl p-6 shadow-sm animate-pulse h-80"></div>
        );
    }

    const hasStocks = data?.stocks && data.stocks.length > 0;
    // Strictly limit to 10 for dashboard as requested
    const dashboardStocks = data?.stocks ? data.stocks.slice(0, 10) : [];

    return (
        <>
            <div className="bg-card rounded-xl p-6 shadow-sm h-full relative overflow-hidden">
                <div className="flex items-center gap-2 mb-6">
                    <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Top Contributors</h3>
                    {isRefreshing && !isLoading && (
                        <Loader2 className="w-3 h-3 animate-spin text-cyan-500 opacity-70" />
                    )}
                </div>
                <div className="space-y-3">
                    {hasStocks ? dashboardStocks.map((stock, idx) => (
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
                            <div className="flex items-center gap-2.5">
                                <StockIcon symbol={stock.symbol.split(',')[0]} size={20} className="flex-shrink-0" />
                                <div className="flex flex-col min-w-0">
                                    <div className="flex flex-wrap gap-1 items-center">
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
                                        {stock.value > 0 && (
                                            <span className="ml-1 px-1 py-0.5 rounded bg-emerald-500/10 text-emerald-500 text-[8px] font-bold uppercase tracking-wider border border-emerald-500/20 leading-none">
                                                Held
                                            </span>
                                        )}
                                    </div>
                                    <span className="text-[10px] text-muted-foreground truncate max-w-[120px]">{stock.name}</span>
                                </div>
                            </div>
                            <div className="text-right">
                                <p className={`text-sm font-medium ${stock.gain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}`}>
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

                {hasStocks && (
                    <div className="mt-6 pt-4 border-t border-border/50">
                        <button
                            onClick={() => setIsAllModalOpen(true)}
                            className="w-full py-2.5 rounded-xl text-xs font-bold text-muted-foreground hover:text-cyan-500 hover:bg-cyan-500/5 transition-all border border-transparent hover:border-cyan-500/20 uppercase tracking-widest"
                        >
                            View All Contributors
                        </button>
                    </div>
                )}
            </div>

            <FullContributorsModal
                isOpen={isAllModalOpen}
                onClose={() => setIsAllModalOpen(false)}
                initialData={data?.stocks || []}
                currency={currency}
            />

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

export default function AttributionChart({ data, isLoading, currency, accounts }: CommonProps & { accounts?: string[] }) {
    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SectorAttribution data={data} isLoading={isLoading} currency={currency} />
            <TopContributors data={data} isLoading={isLoading} currency={currency} accounts={accounts} />
        </div>
    );
}
