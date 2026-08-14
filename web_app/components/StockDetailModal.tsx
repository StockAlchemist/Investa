'use client';

import React, { useState, useEffect, useMemo, Suspense } from 'react';
import dynamic from 'next/dynamic';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { createPortal } from 'react-dom';
import {
    fetchFundamentals,
    fetchFinancials,
    fetchRatios,
    fetchIntrinsicValue,
    fetchHoldings,
    Holding
} from '../lib/api';
import { Info } from 'lucide-react';
import { Skeleton } from './ui/skeleton';
import { TabType, StockDetailModalProps } from './stock-detail/types';
import { StockDetailHeader } from './stock-detail/components/StockDetailHeader';
import { StockDetailTabs } from './stock-detail/components/StockDetailTabs';
import { OverviewTab } from './stock-detail/tabs/OverviewTab';

// Dynamically import heavier tabs for optimal code splitting & Lighthouse performance
const ChartTab = dynamic(
    () => import('./stock-detail/tabs/ChartTab').then(mod => mod.ChartTab),
    { loading: () => <Skeleton className="h-96 w-full rounded-2xl" /> }
);
const FinancialsTab = dynamic(
    () => import('./stock-detail/tabs/FinancialsTab').then(mod => mod.FinancialsTab),
    { loading: () => <Skeleton className="h-96 w-full rounded-2xl" /> }
);
const RatiosTab = dynamic(
    () => import('./stock-detail/tabs/RatiosTab').then(mod => mod.RatiosTab),
    { loading: () => <Skeleton className="h-96 w-full rounded-2xl" /> }
);
const ValuationTab = dynamic(
    () => import('./stock-detail/tabs/ValuationTab').then(mod => mod.ValuationTab),
    { loading: () => <Skeleton className="h-96 w-full rounded-2xl" /> }
);
const HoldingsTab = dynamic(
    () => import('./stock-detail/tabs/HoldingsTab').then(mod => mod.HoldingsTab),
    { loading: () => <Skeleton className="h-96 w-full rounded-2xl" /> }
);
const AnalysisTab = dynamic(
    () => import('./stock-detail/tabs/AnalysisTab').then(mod => mod.AnalysisTab),
    { loading: () => <Skeleton className="h-96 w-full rounded-2xl" /> }
);
const NewsTab = dynamic(
    () => import('./stock-detail/tabs/NewsTab').then(mod => mod.NewsTab),
    { loading: () => <Skeleton className="h-96 w-full rounded-2xl" /> }
);

function getDomain(url: string | undefined): string | undefined {
    if (!url) return undefined;
    try {
        const fullUrl = url.startsWith('http') ? url : `https://${url}`;
        const hostname = new URL(fullUrl).hostname;
        return hostname.replace(/^www\./, '');
    } catch {
        return undefined;
    }
}

export default function StockDetailModal({ symbol, isOpen, onClose, currency }: StockDetailModalProps) {
    const [activeTab, setActiveTab] = useState<TabType>('overview');
    const [mounted, setMounted] = useState(false);
    const queryClient = useQueryClient();

    useEffect(() => {
        setMounted(true);
    }, []);

    const fundamentalsQuery = useQuery({
        queryKey: ['stock-fundamentals', symbol],
        queryFn: () => fetchFundamentals(symbol),
        enabled: isOpen && !!symbol,
        staleTime: 5 * 60 * 1000,
    });
    const fundamentals = fundamentalsQuery.data ?? null;
    const loading = fundamentalsQuery.isLoading;
    const error = fundamentalsQuery.error ? (fundamentalsQuery.error as Error).message : null;

    const intrinsicValueQuery = useQuery({
        queryKey: ['stock-intrinsic-value', symbol],
        queryFn: async () => {
            const data = await fetchIntrinsicValue(symbol);
            if (data) {
                window.dispatchEvent(new CustomEvent('stock-intrinsic-value-updated', {
                    detail: { symbol, data }
                }));
            }
            return data;
        },
        enabled: isOpen && !!symbol,
        staleTime: 30 * 60 * 1000,
    });
    const intrinsicValue = intrinsicValueQuery.data ?? null;

    const filters = useMemo(() => {
        if (typeof window === 'undefined') return { accounts: [], showClosed: false };
        try {
            const savedAccounts = localStorage.getItem('investa_selected_accounts');
            const savedShowClosed = localStorage.getItem('investa_show_closed');
            return {
                accounts: savedAccounts ? JSON.parse(savedAccounts) : [],
                showClosed: savedShowClosed === 'true'
            };
        } catch {
            return { accounts: [], showClosed: false };
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen]);

    const { data: holdings = [] } = useQuery({
        queryKey: ['holdings', symbol, currency, filters.accounts, filters.showClosed],
        queryFn: () => fetchHoldings(currency, filters.accounts, filters.showClosed),
        enabled: isOpen && !!symbol,
        staleTime: 5 * 60 * 1000,
    });

    const userPosition = useMemo(() => {
        if (!holdings.length) return null;

        const matchingHoldings = holdings.filter(h => h.Symbol === symbol);
        if (!matchingHoldings.length) return null;

        const isCash = symbol === '$CASH' || symbol === 'CASH' || symbol.toUpperCase().includes('CASH (');

        const aggregate = matchingHoldings.reduce((acc, curr) => {
            const getVal = (h: Holding, prefix: string) => {
                const exact = h[prefix];
                if (typeof exact === 'number') return exact;
                const withCurr = h[`${prefix} (${currency})`];
                if (typeof withCurr === 'number') return withCurr;
                const foundKey = Object.keys(h).find(k => k.startsWith(prefix));
                if (foundKey && typeof h[foundKey] === 'number') return h[foundKey] as number;
                return 0;
            };

            const qty = curr.Quantity || 0;
            const mktVal = getVal(curr, "Market Value");
            const costBasis = getVal(curr, "Cost Basis");
            const totalBuyCost = getVal(curr, "Total Buy Cost");
            const unrealizedGain = getVal(curr, "Unreal. Gain");
            const totalGain = getVal(curr, "Total Gain") || unrealizedGain;
            const dividends = getVal(curr, "Dividends") || 0;

            return {
                Quantity: acc.Quantity + qty,
                "Market Value": acc["Market Value"] + mktVal,
                "Cost Basis": acc["Cost Basis"] + costBasis,
                "Total Buy Cost": acc["Total Buy Cost"] + totalBuyCost,
                "Unreal. Gain": acc["Unreal. Gain"] + unrealizedGain,
                "Total Gain": acc["Total Gain"] + totalGain,
                "Dividends": acc["Dividends"] + dividends,
                "Weighted IRR": (acc["Weighted IRR"] || 0) + ((curr["IRR (%)"] || 0) * mktVal),
                "fx_rate": (typeof curr.fx_rate === 'number' ? curr.fx_rate : 0) || acc.fx_rate || 1,
            };
        }, {
            Quantity: 0,
            "Market Value": 0,
            "Cost Basis": 0,
            "Total Buy Cost": 0,
            "Unreal. Gain": 0,
            "Total Gain": 0,
            "Dividends": 0,
            "Weighted IRR": 0,
            "fx_rate": 1
        });

        const costBasis = aggregate["Cost Basis"];
        const totalBuyCost = aggregate["Total Buy Cost"];
        const EPSILON = 0.0001;
        const denominator = (Math.abs(totalBuyCost) > EPSILON) ? totalBuyCost : costBasis;
        const hasDenominator = Math.abs(denominator) > EPSILON;

        const totalReturnPct = hasDenominator
            ? (aggregate["Total Gain"] / denominator) * 100
            : (aggregate["Total Gain"] > EPSILON ? Infinity : 0);

        const unrealizedGainPct = hasDenominator
            ? (aggregate["Unreal. Gain"] / denominator) * 100
            : (aggregate["Unreal. Gain"] > EPSILON ? Infinity : 0);

        const aggregateIrr = aggregate["Market Value"] > EPSILON ? aggregate["Weighted IRR"] / aggregate["Market Value"] : 0;
        const avgCost = isCash ? 1.0 : (aggregate.Quantity > 0 ? costBasis / aggregate.Quantity : 0);

        return {
            ...aggregate,
            "Avg Cost": avgCost,
            "Total Return %": totalReturnPct,
            "Unreal. Gain %": unrealizedGainPct,
            "IRR %": aggregateIrr
        };
    }, [holdings, symbol, currency]);

    const fxRate = useMemo(() => userPosition?.fx_rate ?? 1, [userPosition]);
    const domain = useMemo(() => getDomain(fundamentals?.website), [fundamentals?.website]);
    const isEtf = !!fundamentals?.etf_data;

    const loadData = async (force: boolean = false) => {
        if (force) {
            await Promise.allSettled([
                queryClient.fetchQuery({
                    queryKey: ['stock-fundamentals', symbol],
                    queryFn: () => fetchFundamentals(symbol, true),
                    staleTime: 5 * 60 * 1000,
                }),
                queryClient.fetchQuery({
                    queryKey: ['stock-financials', symbol, 'quarterly'],
                    queryFn: () => fetchFinancials(symbol, 'quarterly', true),
                    staleTime: 30 * 60 * 1000,
                }),
                queryClient.fetchQuery({
                    queryKey: ['stock-financials', symbol, 'annual'],
                    queryFn: () => fetchFinancials(symbol, 'annual', true),
                    staleTime: 30 * 60 * 1000,
                }),
                queryClient.fetchQuery({
                    queryKey: ['stock-ratios', symbol, 'quarterly'],
                    queryFn: () => fetchRatios(symbol, 'quarterly', true),
                    staleTime: 30 * 60 * 1000,
                }),
                queryClient.fetchQuery({
                    queryKey: ['stock-ratios', symbol, 'annual'],
                    queryFn: () => fetchRatios(symbol, 'annual', true),
                    staleTime: 30 * 60 * 1000,
                }),
                queryClient.fetchQuery({
                    queryKey: ['stock-intrinsic-value', symbol],
                    queryFn: async () => {
                        const data = await fetchIntrinsicValue(symbol, true);
                        if (data) {
                            window.dispatchEvent(new CustomEvent('stock-intrinsic-value-updated', {
                                detail: { symbol, data }
                            }));
                        }
                        return data;
                    },
                    staleTime: 30 * 60 * 1000,
                }),
            ]);
        } else {
            await Promise.allSettled([
                fundamentalsQuery.refetch(),
                intrinsicValueQuery.refetch(),
            ]);
        }
    };

    if (!mounted || !isOpen) return null;

    return createPortal(
        <div className="fixed inset-0 z-[100] flex flex-col justify-end sm:justify-center items-center p-0 sm:p-4 isolate">
            <div className="absolute inset-0 bg-black/60 cursor-pointer" onClick={onClose} />

            <div className="relative w-full max-w-5xl h-[94vh] sm:h-auto sm:max-h-[90vh] rounded-t-[2.5rem] sm:rounded-[2rem] flex flex-col overflow-hidden animate-in slide-in-from-bottom sm:zoom-in-95 duration-300 bg-white dark:bg-zinc-950">
                {/* Mobile Drag Handle */}
                <div className="sm:hidden w-full flex justify-center pt-3 pb-1 flex-shrink-0">
                    <div className="w-12 h-1.5 bg-secondary rounded-full" />
                </div>

                {/* Sticky Header & Tabs Container */}
                <div className="sticky top-0 z-50 bg-white/95 dark:bg-zinc-950/95 backdrop-blur-md flex-shrink-0 border-b border-border/40">
                    <StockDetailHeader
                        symbol={symbol}
                        fundamentals={fundamentals}
                        currency={currency}
                        fxRate={fxRate}
                        domain={domain}
                        onClose={onClose}
                    />
                    <StockDetailTabs
                        activeTab={activeTab}
                        setActiveTab={setActiveTab}
                        isEtf={isEtf}
                    />
                </div>

                {/* Content Area */}
                <div className="flex-1 overflow-y-auto p-4 sm:p-6 pt-4 custom-scrollbar">
                    {loading ? (
                        <div className="space-y-4">
                            <Skeleton className="h-40 w-full rounded-2xl" />
                            <div className="grid grid-cols-3 gap-4">
                                <Skeleton className="h-24 rounded-2xl" />
                                <Skeleton className="h-24 rounded-2xl" />
                                <Skeleton className="h-24 rounded-2xl" />
                            </div>
                            <Skeleton className="h-60 w-full rounded-2xl" />
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center py-20 text-center">
                            <div className="w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center mb-4">
                                <Info className="w-8 h-8 text-destructive" />
                            </div>
                            <h3 className="text-xl font-bold mb-2">Something went wrong</h3>
                            <p className="text-muted-foreground max-w-md">{error}</p>
                            <button
                                onClick={() => loadData()}
                                className="mt-6 px-6 py-2 bg-secondary hover:bg-muted rounded-full transition-colors cursor-pointer"
                            >
                                Try Again
                            </button>
                        </div>
                    ) : (
                        <Suspense fallback={<Skeleton className="h-96 w-full rounded-2xl" />}>
                            {activeTab === 'overview' && (
                                <OverviewTab
                                    fundamentals={fundamentals}
                                    intrinsicValue={intrinsicValue}
                                    userPosition={userPosition}
                                    currency={currency}
                                    fxRate={fxRate}
                                    loading={loading}
                                    onRefreshData={() => loadData(true)}
                                />
                            )}
                            {activeTab === 'chart' && (
                                <ChartTab
                                    symbol={symbol}
                                    currency={currency}
                                    avgCost={userPosition?.["Avg Cost"]}
                                    fxRate={userPosition?.["fx_rate"]}
                                    accounts={filters.accounts}
                                    exchange={fundamentals?.exchange}
                                />
                            )}
                            {activeTab === 'financials' && (
                                <FinancialsTab
                                    symbol={symbol}
                                    fundamentals={fundamentals}
                                    isOpen={isOpen}
                                />
                            )}
                            {activeTab === 'ratios' && (
                                <RatiosTab
                                    symbol={symbol}
                                    isOpen={isOpen}
                                />
                            )}
                            {activeTab === 'valuation' && (
                                <ValuationTab
                                    symbol={symbol}
                                    intrinsicValue={intrinsicValue}
                                    fundamentals={fundamentals}
                                    currency={currency}
                                    fxRate={fxRate}
                                />
                            )}
                            {activeTab === 'holdings' && (
                                <HoldingsTab fundamentals={fundamentals} />
                            )}
                            {activeTab === 'analysis' && (
                                <AnalysisTab
                                    symbol={symbol}
                                    isOpen={isOpen}
                                />
                            )}
                            {activeTab === 'news' && (
                                <NewsTab
                                    symbol={symbol}
                                    isOpen={isOpen}
                                />
                            )}
                        </Suspense>
                    )}
                </div>
            </div>
        </div>,
        document.body
    );
}
