'use client';

import React, { useMemo, useState } from 'react';
import { Sparkles, Layers } from 'lucide-react';
import { cn, formatCurrency } from '@/lib/utils';
import type { IntrinsicValueResponse, RecommendedValuationMethod } from '@/lib/api';

interface ValuationComparisonChartProps {
    symbol: string;
    intrinsicValue: IntrinsicValueResponse;
    currency: string;
    fxRate: number;
    recommendedMethod?: RecommendedValuationMethod | null;
    customModelValues?: Record<string, number | null>;
    customBlendedValue?: number | null;
    onSelectMethod?: (methodKey: string) => void;
}

interface ValuationItem {
    key: string;
    name: string;
    category: 'cash_earnings' | 'multiples' | 'relative';
    categoryLabel: string;
    categoryColor: string;
    value: number;
    defaultValue?: number;
    isCustom?: boolean;
    bear?: number;
    bull?: number;
    upsidePct: number;
    isRecommended: boolean;
    isBlended?: boolean;
    bestSuitedFor?: string;
    keyCaveats?: string;
}

const CATEGORY_STYLES = {
    cash_earnings: {
        label: 'Cash Flow & Earnings',
        badge: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20',
        dot: 'bg-emerald-500',
    },
    multiples: {
        label: 'Multiples & Growth',
        badge: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20',
        dot: 'bg-blue-500',
    },
    relative: {
        label: 'Heuristics & Relative',
        badge: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20',
        dot: 'bg-purple-500',
    },
};

export default function ValuationComparisonChart({
    symbol,
    intrinsicValue,
    currency,
    fxRate,
    recommendedMethod,
    customModelValues,
    customBlendedValue,
    onSelectMethod,
}: ValuationComparisonChartProps) {
    const [sortMode, setSortMode] = useState<'default' | 'value_desc' | 'upside_desc'>('default');
    const [hoveredKey, setHoveredKey] = useState<string | null>(null);

    const currentPrice = intrinsicValue.current_price ?? 0;
    const models = intrinsicValue.models;

    // Collect all valid valuation points
    const items = useMemo(() => {
        if (!models) return [];
        const list: ValuationItem[] = [];

        const addItem = (
            key: string,
            name: string,
            category: 'cash_earnings' | 'multiples' | 'relative',
            defaultVal?: number | null,
            mc?: { bear?: number | null; bull?: number | null } | null,
            bestSuitedFor?: string,
            keyCaveats?: string,
        ) => {
            const customVal = customModelValues?.[key];
            const val = customVal !== undefined ? customVal : defaultVal;
            if (val == null || val <= 0 || isNaN(val)) return;
            const isCustom = customVal != null && defaultVal != null && Math.abs(customVal - defaultVal) > 0.001;
            const upsidePct = currentPrice > 0 ? ((val - currentPrice) / currentPrice) * 100 : 0;
            const isRecommended = recommendedMethod?.method_key === key;

            list.push({
                key,
                name,
                category,
                categoryLabel: CATEGORY_STYLES[category].label,
                categoryColor: CATEGORY_STYLES[category].badge,
                value: val,
                defaultValue: defaultVal != null ? defaultVal : undefined,
                isCustom,
                bear: mc?.bear && mc.bear > 0 ? mc.bear : undefined,
                bull: mc?.bull && mc.bull > 0 ? mc.bull : undefined,
                upsidePct,
                isRecommended,
                bestSuitedFor,
                keyCaveats,
            });
        };

        // 1. DCF
        addItem(
            'dcf',
            'Discounted Free Cash Flow (DCF)',
            'cash_earnings',
            models.dcf?.intrinsic_value,
            models.dcf?.mc,
            'Cash-generative companies with steady, predictable Free Cash Flow.',
            'Highly sensitive to growth and discount rate (WACC) inputs.',
        );

        // 2. D-CFO
        addItem(
            'dcfo',
            'Discounted Cash from Operations (D-CFO)',
            'cash_earnings',
            models.dcfo?.intrinsic_value,
            models.dcfo?.mc,
            'Companies with consistent operating cash flow but lumpy multi-year CapEx.',
            'Excludes ongoing reinvestment needs (CapEx).',
        );

        // 3. D-NI
        addItem(
            'dni',
            'Discounted Net Income (D-NI)',
            'cash_earnings',
            models.dni?.intrinsic_value,
            models.dni?.mc,
            'Financial institutions (Banks, Insurance) where cash flow lines are distorted.',
            'Net Income is vulnerable to non-recurring items and accounting choices.',
        );

        // 4. Mean P/E
        addItem(
            'mean_pe',
            'Mean P/E Ratio Valuation',
            'multiples',
            models.mean_pe?.intrinsic_value,
            models.mean_pe?.mc,
            'Mature, profitable companies with stable earnings predictability.',
            'Ignores future earnings growth rates and margin trajectory.',
        );

        // 5. PEG Ratio
        addItem(
            'peg',
            'PEG Ratio Fair Value (PEG=1.0)',
            'multiples',
            models.peg?.intrinsic_value,
            models.peg?.mc,
            'Profitable growth companies with positive, expanding earnings.',
            'Assumes earnings growth is linear and sustainable.',
        );

        // 6. Mean P/B
        addItem(
            'mean_pb',
            'Mean P/B Ratio Valuation',
            'multiples',
            models.mean_pb?.intrinsic_value,
            models.mean_pb?.mc,
            'Asset-heavy businesses, Banks (1.2–1.4x), REITs, and property developers.',
            'Understates high-ROE, asset-light, and tech businesses.',
        );

        // 7. Mean P/S
        addItem(
            'mean_ps',
            'Mean P/S Ratio Valuation',
            'multiples',
            models.mean_ps?.intrinsic_value,
            models.mean_ps?.mc,
            'Early-stage or cyclical growth companies not yet consistently profitable.',
            'Ignores profit margins and cash burn entirely.',
        );

        // 8. PSG
        addItem(
            'psg',
            'Price-to-Sales Growth (PSG)',
            'multiples',
            models.psg?.intrinsic_value,
            models.psg?.mc,
            'High-growth, unprofitable software and tech businesses.',
            'Assumes rapid revenue expansion will achieve operating leverage.',
        );

        // 9. Graham
        addItem(
            'graham',
            'Benjamin Graham Formula',
            'relative',
            models.graham?.intrinsic_value,
            models.graham?.mc,
            'Defensive value screening comparing EPS and growth to corporate bond yield.',
            'Formula multiplier is aggressive if growth inputs are elevated.',
        );

        // 10. DDM
        addItem(
            'ddm',
            'Dividend Discount Model (DDM)',
            'relative',
            models.ddm?.intrinsic_value,
            models.ddm?.mc,
            'Mature dividend payers with long track records of consistent dividend growth.',
            'Only reflects value returned as direct dividends.',
        );

        // 11. Peter Lynch
        addItem(
            'lynch',
            'Peter Lynch Fair Value',
            'relative',
            models.lynch?.intrinsic_value,
            models.lynch?.mc,
            'Fast rule-of-thumb valuation equating fair P/E multiple to growth rate + yield.',
            'Heuristic benchmark; does not account for cost of capital.',
        );

        // 12. EPV
        addItem(
            'epv',
            'Earnings Power Value (EPV Floor)',
            'cash_earnings',
            models.epv?.intrinsic_value,
            models.epv?.mc,
            'Conservative valuation of normalized sustainable operating earnings in perpetuity.',
            'Strictly a no-growth baseline floor.',
        );

        // Sort items
        if (sortMode === 'value_desc') {
            list.sort((a, b) => b.value - a.value);
        } else if (sortMode === 'upside_desc') {
            list.sort((a, b) => b.upsidePct - a.upsidePct);
        }

        return list;
    }, [models, currentPrice, recommendedMethod, sortMode, customModelValues]);

    const effectiveBlendedValue = customBlendedValue !== undefined ? customBlendedValue : intrinsicValue.average_intrinsic_value;
    const isBlendedCustom = customBlendedValue != null && intrinsicValue.average_intrinsic_value != null && Math.abs(customBlendedValue - intrinsicValue.average_intrinsic_value) > 0.001;

    // Compute scale domain: min and max across all points (including price & MC ranges)
    const { minBound, maxBound, spread, undervaluedCount } = useMemo(() => {
        if (items.length === 0) {
            return { minBound: 0, maxBound: 100, spread: 100, undervaluedCount: 0 };
        }

        const allValues: number[] = [];
        if (currentPrice > 0) allValues.push(currentPrice);

        items.forEach(item => {
            allValues.push(item.value);
            if (item.defaultValue) allValues.push(item.defaultValue);
            if (item.bear) allValues.push(item.bear);
            if (item.bull) allValues.push(item.bull);
        });

        if (effectiveBlendedValue) {
            allValues.push(effectiveBlendedValue);
        }
        if (intrinsicValue.average_intrinsic_value) {
            allValues.push(intrinsicValue.average_intrinsic_value);
        }

        const rawMin = Math.min(...allValues);
        const rawMax = Math.max(...allValues);
        const padding = Math.max((rawMax - rawMin) * 0.12, rawMax * 0.05, 5);

        const minBound = Math.max(0, rawMin - padding);
        const maxBound = rawMax + padding;
        const spread = maxBound - minBound || 1;

        const undervaluedCount = items.filter(i => i.value >= currentPrice).length;

        return { minBound, maxBound, spread, undervaluedCount };
    }, [items, currentPrice, effectiveBlendedValue, intrinsicValue.average_intrinsic_value]);

    // Position helper (0% to 100%)
    const getPosPct = (val: number) => {
        return Math.max(0, Math.min(100, ((val - minBound) / spread) * 100));
    };

    const currentPricePosPct = currentPrice > 0 ? getPosPct(currentPrice) : null;

    if (items.length === 0) return null;

    return (
        <div className="bg-card border border-border/70 rounded-2xl p-5 sm:p-6 shadow-xs space-y-6">
            {/* Header & Quick Summary */}
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                <div>
                    <div className="flex items-center gap-2">
                        <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-500 dark:text-indigo-400">
                            <Layers className="w-4 h-4" />
                        </span>
                        <h3 className="text-base font-bold text-foreground">
                            Intrinsic Value Comparison Spectrum
                        </h3>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                        Cross-method valuation distribution for {symbol} plotted against current stock price of{' '}
                        <span className="font-semibold text-foreground">{formatCurrency(currentPrice * fxRate, currency)}</span>.
                    </p>
                </div>

                {/* KPI stats & sorting controls */}
                <div className="flex items-center gap-2 flex-wrap">
                    <div className="flex items-center gap-1.5 bg-muted/60 px-3 py-1.5 rounded-xl text-xs">
                        <span className="text-muted-foreground">Range:</span>
                        <span className="font-semibold">{formatCurrency(Math.min(...items.map(i => i.value)) * fxRate, currency)}</span>
                        <span className="text-muted-foreground">–</span>
                        <span className="font-semibold">{formatCurrency(Math.max(...items.map(i => i.value)) * fxRate, currency)}</span>
                    </div>

                    <div className="flex items-center gap-1.5 bg-muted/60 px-3 py-1.5 rounded-xl text-xs">
                        <span className="text-muted-foreground">Consensus:</span>
                        <span className={cn(
                            "font-bold",
                            undervaluedCount > items.length / 2 ? "text-emerald-500" : "text-amber-500"
                        )}>
                            {undervaluedCount}/{items.length} Undervalued
                        </span>
                    </div>

                    {/* Sort buttons */}
                    <div className="flex items-center rounded-xl bg-muted/80 p-0.5 border border-border/50">
                        <button
                            onClick={() => setSortMode('default')}
                            title="Default Method Order"
                            className={cn(
                                "px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors cursor-pointer",
                                sortMode === 'default' ? "bg-background text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Default
                        </button>
                        <button
                            onClick={() => setSortMode('value_desc')}
                            title="Sort by Highest Fair Value"
                            className={cn(
                                "px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors cursor-pointer",
                                sortMode === 'value_desc' ? "bg-background text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Valuation
                        </button>
                        <button
                            onClick={() => setSortMode('upside_desc')}
                            title="Sort by Highest Upside %"
                            className={cn(
                                "px-2.5 py-1 text-[11px] font-semibold rounded-lg transition-colors cursor-pointer",
                                sortMode === 'upside_desc' ? "bg-background text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Upside %
                        </button>
                    </div>
                </div>
            </div>

            {/* Spectrum Chart Body */}
            <div className="relative pt-8 pb-4">
                {/* Axis Top Labels: Min, Spot, Max */}
                <div className="relative w-full h-6 text-[11px] text-muted-foreground select-none">
                    <span className="absolute left-0 top-0 font-mono">
                        {formatCurrency(minBound * fxRate, currency)}
                    </span>
                    {currentPricePosPct !== null && (
                        <div
                            className="absolute top-0 -translate-x-1/2 flex flex-col items-center z-20 pointer-events-none"
                            style={{ left: `${currentPricePosPct}%` }}
                        >
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-foreground text-background shadow-md whitespace-nowrap">
                                Current: {formatCurrency(currentPrice * fxRate, currency)}
                            </span>
                        </div>
                    )}
                    <span className="absolute right-0 top-0 font-mono">
                        {formatCurrency(maxBound * fxRate, currency)}
                    </span>
                </div>

                {/* Vertical Grid & Current Price Guide Line */}
                <div className="relative w-full space-y-3 mt-2">
                    {/* Background Grid Bands */}
                    <div className="absolute inset-0 pointer-events-none flex justify-between">
                        {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => (
                            <div key={idx} className="h-full border-r border-border/30 border-dashed" />
                        ))}
                    </div>

                    {/* Current Price Vertical Reference Rule */}
                    {currentPricePosPct !== null && (
                        <div
                            className="absolute top-0 bottom-0 w-px border-r-2 border-dashed border-foreground/60 z-10 pointer-events-none"
                            style={{ left: `${currentPricePosPct}%` }}
                        />
                    )}

                    {/* Blended / Recommended Anchor Row (if available) */}
                    {effectiveBlendedValue && (
                        <div className={cn(
                            "relative group p-3 rounded-xl transition-all border",
                            isBlendedCustom
                                ? "bg-gradient-to-r from-amber-500/10 via-indigo-500/10 to-transparent border-amber-500/40"
                                : "bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-transparent border-indigo-500/30"
                        )}>
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                                <div className="flex items-center gap-2 min-w-0 flex-wrap">
                                    <span className={cn(
                                        "inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-bold text-white",
                                        isBlendedCustom ? "bg-amber-600" : "bg-indigo-500"
                                    )}>
                                        <Sparkles className="w-2.5 h-2.5" /> {isBlendedCustom ? "CUSTOM BLENDED FAIR VALUE" : "BLENDED FAIR VALUE"}
                                    </span>
                                    <span className="text-xs font-bold truncate">Composite Intrinsic Value</span>
                                    {isBlendedCustom && intrinsicValue.average_intrinsic_value && (
                                        <span className="text-[10px] text-muted-foreground font-normal">
                                            (Default: {formatCurrency(intrinsicValue.average_intrinsic_value * fxRate, currency)})
                                        </span>
                                    )}
                                </div>
                                <div className="flex items-center gap-2 self-end sm:self-auto shrink-0">
                                    <span className={cn(
                                        "text-sm font-bold",
                                        isBlendedCustom ? "text-amber-600 dark:text-amber-400" : "text-indigo-600 dark:text-indigo-400"
                                    )}>
                                        {formatCurrency(effectiveBlendedValue * fxRate, currency)}
                                    </span>
                                    {currentPrice > 0 && (
                                        <span className={cn(
                                            "text-xs font-semibold px-1.5 py-0.5 rounded",
                                            effectiveBlendedValue >= currentPrice
                                                ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                                                : "bg-rose-500/15 text-rose-600 dark:text-rose-400"
                                        )}>
                                            {effectiveBlendedValue >= currentPrice ? '+' : ''}
                                            {(((effectiveBlendedValue - currentPrice) / currentPrice) * 100).toFixed(1)}%
                                        </span>
                                    )}
                                </div>
                            </div>

                            {/* Spectrum Bar Row */}
                            <div className="relative h-6 w-full bg-secondary/40 rounded-lg overflow-visible flex items-center">
                                {/* Bear-Bull Range for Blended if exists */}
                                {intrinsicValue.range?.bear && intrinsicValue.range?.bull && (
                                    <div
                                        className="absolute h-3 rounded-full bg-indigo-500/25 border border-indigo-500/40"
                                        style={{
                                            left: `${getPosPct(intrinsicValue.range.bear)}%`,
                                            width: `${Math.max(1, getPosPct(intrinsicValue.range.bull) - getPosPct(intrinsicValue.range.bear))}%`,
                                        }}
                                    />
                                )}

                                {/* Delta Connector from Current Price */}
                                {currentPricePosPct !== null && (
                                    <div
                                        className={cn(
                                            "absolute h-1 top-1/2 -translate-y-1/2 opacity-70",
                                            effectiveBlendedValue >= currentPrice
                                                ? "bg-emerald-500"
                                                : "bg-rose-500"
                                        )}
                                        style={{
                                            left: `${Math.min(currentPricePosPct, getPosPct(effectiveBlendedValue))}%`,
                                            width: `${Math.abs(getPosPct(effectiveBlendedValue) - currentPricePosPct)}%`,
                                        }}
                                    />
                                )}

                                {/* Intrinsic Value Pin / Pill */}
                                <div
                                    className="absolute -translate-x-1/2 top-1/2 -translate-y-1/2 z-20"
                                    style={{ left: `${getPosPct(effectiveBlendedValue)}%` }}
                                >
                                    <div className={cn(
                                        "flex items-center justify-center w-5 h-5 rounded-full text-white shadow-md ring-2 ring-background text-[10px] font-black",
                                        isBlendedCustom ? "bg-amber-600" : "bg-indigo-600"
                                    )}>
                                        ★
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Method Rows */}
                    {items.map(item => {
                        const isHovered = hoveredKey === item.key;
                        const itemPos = getPosPct(item.value);
                        const isUndervalued = item.value >= currentPrice;

                        return (
                            <div
                                key={item.key}
                                onMouseEnter={() => setHoveredKey(item.key)}
                                onMouseLeave={() => setHoveredKey(null)}
                                onClick={() => onSelectMethod?.(item.key)}
                                className={cn(
                                    "relative p-2.5 sm:p-3 rounded-xl border transition-all duration-200 cursor-pointer",
                                    item.isCustom
                                        ? "bg-amber-500/5 border-amber-500/40 shadow-xs"
                                        : item.isRecommended
                                            ? "bg-indigo-500/5 border-indigo-500/40 shadow-xs"
                                            : isHovered
                                                ? "bg-muted/70 border-border"
                                                : "bg-card/50 border-border/40 hover:bg-muted/40"
                                )}
                            >
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-1.5">
                                    <div className="flex items-center gap-1.5 min-w-0 flex-wrap">
                                        <div className={cn("w-2 h-2 rounded-full shrink-0", CATEGORY_STYLES[item.category].dot)} />
                                        <span className="text-xs font-semibold text-foreground truncate">
                                            {item.name}
                                        </span>
                                        {item.isRecommended && (
                                            <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[9px] font-extrabold bg-indigo-500 text-white">
                                                <Sparkles className="w-2 h-2" /> BEST-FIT
                                            </span>
                                        )}
                                        {item.isCustom && (
                                            <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[9px] font-extrabold bg-amber-500/20 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                                                CUSTOM
                                            </span>
                                        )}
                                        {item.isCustom && item.defaultValue != null && (
                                            <span className="text-[10px] text-muted-foreground">
                                                (Def: {formatCurrency(item.defaultValue * fxRate, currency)})
                                            </span>
                                        )}
                                    </div>

                                    <div className="flex items-center gap-2 self-end sm:self-auto shrink-0">
                                        <span className={cn(
                                            "text-xs font-bold tabular-nums",
                                            item.isCustom ? "text-amber-600 dark:text-amber-400" : "text-foreground"
                                        )}>
                                            {formatCurrency(item.value * fxRate, currency)}
                                        </span>
                                        <span className={cn(
                                            "text-[11px] font-bold px-1.5 py-0.2 rounded tabular-nums",
                                            isUndervalued
                                                ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                                                : "bg-rose-500/15 text-rose-600 dark:text-rose-400"
                                        )}>
                                            {isUndervalued ? '+' : ''}{item.upsidePct.toFixed(1)}%
                                        </span>
                                    </div>
                                </div>

                                {/* Graph Track */}
                                <div className="relative h-5 w-full bg-secondary/30 rounded-md overflow-visible flex items-center">
                                    {/* Monte Carlo Range (Bear to Bull) */}
                                    {item.bear && item.bull && (
                                        <div
                                            className={cn(
                                                "absolute h-2.5 rounded-full border opacity-80",
                                                isUndervalued
                                                    ? "bg-emerald-500/25 border-emerald-500/40"
                                                    : "bg-rose-500/25 border-rose-500/40"
                                            )}
                                            style={{
                                                left: `${getPosPct(item.bear)}%`,
                                                width: `${Math.max(1, getPosPct(item.bull) - getPosPct(item.bear))}%`,
                                            }}
                                            title={`Bear: ${formatCurrency(item.bear * fxRate, currency)} — Bull: ${formatCurrency(item.bull * fxRate, currency)}`}
                                        />
                                    )}

                                    {/* Delta Connector from Current Price Line */}
                                    {currentPricePosPct !== null && (
                                        <div
                                            className={cn(
                                                "absolute h-1 top-1/2 -translate-y-1/2 opacity-60 rounded-full",
                                                isUndervalued ? "bg-emerald-500" : "bg-rose-500"
                                            )}
                                            style={{
                                                left: `${Math.min(currentPricePosPct, itemPos)}%`,
                                                width: `${Math.abs(itemPos - currentPricePosPct)}%`,
                                            }}
                                        />
                                    )}

                                    {/* Point Estimate Dot */}
                                    <div
                                        className="absolute -translate-x-1/2 top-1/2 -translate-y-1/2 z-20 group-hover:scale-125 transition-transform"
                                        style={{ left: `${itemPos}%` }}
                                    >
                                        <div className={cn(
                                            "w-3.5 h-3.5 rounded-full border-2 border-background shadow-xs",
                                            item.isRecommended
                                                ? "bg-indigo-600 ring-2 ring-indigo-400"
                                                : isUndervalued
                                                    ? "bg-emerald-500"
                                                    : "bg-rose-500"
                                        )} />
                                    </div>
                                </div>

                                {/* Expanded Guidance on Hover */}
                                {isHovered && item.bestSuitedFor && (
                                    <div className="mt-2 pt-2 border-t border-border/40 text-[11px] text-muted-foreground flex flex-col sm:flex-row gap-2">
                                        <div>
                                            <span className="font-semibold text-emerald-600 dark:text-emerald-400">Best Suited: </span>
                                            <span>{item.bestSuitedFor}</span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Legend & Guidance Footer */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-border/50 text-xs text-muted-foreground">
                <div className="flex items-center gap-4 flex-wrap">
                    <div className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                        <span>Undervalued vs Market</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                        <span>Overvalued vs Market</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <div className="w-4 h-2 rounded bg-emerald-500/25 border border-emerald-500/40" />
                        <span>Monte Carlo Range (Bear–Bull)</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full bg-indigo-600 ring-1 ring-indigo-400" />
                        <span>Best-Fit Recommended</span>
                    </div>
                </div>

                <div className="text-[11px] italic">
                    Tip: Click any valuation row to inspect model parameters below.
                </div>
            </div>
        </div>
    );
}
