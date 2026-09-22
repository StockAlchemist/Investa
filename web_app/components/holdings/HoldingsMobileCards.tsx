import React, { useCallback, useLayoutEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronRight, ChevronUp, Layers } from 'lucide-react';
import { Holding, Lot } from '../../lib/api';
import { Card } from '../ui/card';
import WatchlistStar from '../WatchlistStar';
import { getCellClass, formatHoldingValue } from './holdingsUtils';
import { formatCompactNumber, formatCurrencyWhole } from '../../lib/utils';

/**
 * A symbol and its amount, where the amount gets shorter rather than the symbol
 * getting clipped.
 *
 * The web twin of `FittedMoney` / the `ViewThatFits` ladder in the native row
 * (macos_app HoldingsTableView.swift). A phone card cannot always fit eight
 * digits beside a fund's full ticker, and of the two the amount is the one that
 * can be said more briefly: `$61,705,355` becomes `$61.7M` and stays a number,
 * whereas `SCBRMS&P500` clipped to `SCBRMS&P5…` no longer names anything — the
 * ticker this rule is written about.
 *
 * CSS has no `minimumScaleFactor` and no `ViewThatFits`, so the fit is measured:
 * the symbol reports whether it is being clipped, and the amount drops a rung
 * when it is. Measuring the symbol rather than the amount is deliberate — the
 * amount never clips (it refuses to shrink), so it could not report anything.
 */
const SymbolAndAmount: React.FC<{
    symbol: string;
    value: unknown;
    currency: string;
    fallback: (val: unknown) => string;
    /** The day-change line under the amount. */
    children?: React.ReactNode;
    /** The expand chevron, which shares the amount's non-shrinking block. */
    trailing?: React.ReactNode;
}> = ({ symbol, value, currency, fallback, children, trailing }) => {
    const symbolRef = useRef<HTMLHeadingElement>(null);

    // The step down is one-way for a given row, and resets when the row starts
    // showing something else. Going compact gives the symbol back the width the
    // longer amount was using, so a rule that re-measured freely would find the
    // symbol now fits, step back up, clip again, and oscillate forever — the
    // hysteresis `ViewThatFits` gets for free by never re-proposing.
    const identity = `${symbol}|${String(value)}|${currency}`;
    const [fit, setFit] = useState({ id: identity, compact: false });
    const compact = fit.id === identity && fit.compact;

    const measure = useCallback(() => {
        const el = symbolRef.current;
        if (!el) return;
        // The text's own width, from a Range over its contents, against the
        // width the row is giving it. `scrollWidth`/`clientWidth` are rounded to
        // whole pixels, and a symbol overflowing by a pixel and a half reports
        // 129 against 128 — which any slack wide enough to absorb rounding noise
        // also absorbs. A Range measures in fractions and is never clipped, so
        // it answers the actual question: is the whole string being shown?
        const range = document.createRange();
        // jsdom has Range but no layout, so it has no getBoundingClientRect.
        // Nothing can be measured there, and nothing needs to be: with no
        // layout there is no clipping to detect, so the row keeps the longer
        // form rather than guessing.
        if (typeof range.getBoundingClientRect !== 'function') return;
        range.selectNodeContents(el);
        const natural = range.getBoundingClientRect().width;
        const available = el.getBoundingClientRect().width;
        const clipped = natural > available + 0.5;
        setFit(prev => {
            const wasCompact = prev.id === identity && prev.compact;
            const next = wasCompact || clipped;
            if (prev.id === identity && prev.compact === next) return prev;
            return { id: identity, compact: next };
        });
    }, [identity]);

    useLayoutEffect(() => {
        measure();
        const el = symbolRef.current;
        if (!el) return;
        // Remeasure once the webfont lands: the fallback face is narrower, so a
        // symbol measured against it fits and then stops fitting, and the box it
        // sits in never changes size — nothing else would notice.
        let live = true;
        document.fonts?.ready.then(() => { if (live) measure(); }).catch(() => {});
        if (typeof ResizeObserver === 'undefined') return () => { live = false; };
        // Rotation, an accessibility type size, a card expanding beside it —
        // the row is re-measured whenever its width changes, not only on mount.
        const ro = new ResizeObserver(measure);
        ro.observe(el);
        return () => { live = false; ro.disconnect(); };
    }, [measure]);

    const amount = typeof value === 'number'
        ? (compact ? formatCompactNumber(value, currency) : formatCurrencyWhole(value, currency))
        : fallback(value);

    return (
        <>
            <div className="flex min-w-0 items-center gap-3">
                <WatchlistStar symbol={symbol} size="md" />
                <h3 ref={symbolRef} className="truncate text-[17px] font-bold text-foreground leading-none">{symbol}</h3>
            </div>
            <div className="flex shrink-0 items-center gap-2">
                <div className="text-right">
                    <div className="text-[15px] font-bold tabular-nums text-foreground leading-none whitespace-nowrap">
                        {amount}
                    </div>
                    {children}
                </div>
                {trailing}
            </div>
        </>
    );
};

interface HoldingsMobileCardsProps {
    mobileViewMode: 'card' | 'table';
    visibleHoldings: Holding[];
    currency: string;
    openStockDetail: (symbol: string, currency?: string) => void;
    expandedCards: Set<string>;
    toggleCardExpansion: (key: string) => void;
    expandedLots: Set<string>;
    toggleLotExpansion: (key: string) => void;
    getExpansionKey: (holding: Holding) => string;
    getValue: (holding: Holding, header: string) => string | number | string[] | number[] | null;
    getLotValue: (lot: Lot, header: string, holdingPrice?: number) => string | number | null;
}

export const HoldingsMobileCards: React.FC<HoldingsMobileCardsProps> = ({
    mobileViewMode,
    visibleHoldings,
    currency,
    openStockDetail,
    expandedCards,
    toggleCardExpansion,
    expandedLots,
    toggleLotExpansion,
    getExpansionKey,
    getValue,
    getLotValue,
}) => {
    const formatValue = (val: unknown, field: string) => formatHoldingValue(val, field, currency);

    return (
        <div className={`${mobileViewMode === 'card' ? 'block' : 'hidden'} md:hidden space-y-4 p-4`}>
            {visibleHoldings.map((holding, idx) => {
                const expKey = getExpansionKey(holding);
                const isCardExpanded = expandedCards.has(expKey);
                const isLotExpanded = expandedLots.has(expKey);

                return (
                    <Card
                        key={`mobile-${holding.Symbol}-${idx}`}
                        /* p-3.5 is the native row's own inset (`.padding(14)` on
                           iosHoldingRow). It was p-0, so the icon's star badge and
                           the amount sat flush against the card's top edge with
                           nothing above them, and the row read as cramped. */
                        className="bg-card rounded-2xl border-none p-3.5 relative group cursor-pointer hover:border-cyan-500/50 transition-all active:scale-[0.98]"
                        onClick={() => openStockDetail(holding.Symbol, currency)}
                    >
                        <div className="space-y-3">
                            {/* Type sizes follow the native row (`iosHoldingRowHeader`
                                in macos_app HoldingsTableView.swift): the symbol at
                                headline, the amount one step below it at subheadline.
                                Both were text-xl here, which made this the loudest
                                type on the phone — larger than the portfolio total —
                                and left the two ends of the row no room to coexist.

                                `min-w-0` on the symbol and `shrink-0` on the amount
                                are what keep them apart: without them neither side
                                yields, so a long symbol runs under its own figure
                                instead of the row giving way.

                                Centred, not top-aligned (`HStack(alignment: .center)`
                                in the native header): the amount block is shorter
                                than the 40px icon beside it, so `items-start` left
                                the figure pinned to the very top of the card. */}
                            <div className="flex justify-between items-center gap-2">
                                <SymbolAndAmount
                                    symbol={holding.Symbol}
                                    value={getValue(holding, "Mkt Val")}
                                    currency={currency}
                                    fallback={(val) => formatValue(val, "Mkt Val")}
                                    trailing={
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                toggleCardExpansion(expKey);
                                            }}
                                            className="p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-colors border-none shrink-0"
                                            aria-expanded={isCardExpanded}
                                            title={isCardExpanded ? `Hide ${holding.Symbol} details` : `Show ${holding.Symbol} details`}
                                        >
                                            {isCardExpanded ? (
                                                <ChevronUp className="w-4 h-4 text-muted-foreground" />
                                            ) : (
                                                <ChevronDown className="w-4 h-4 text-muted-foreground" />
                                            )}
                                        </button>
                                    }
                                >
                                    {!isCardExpanded && (
                                        <div className={`text-[11px] font-medium tabular-nums mt-1 ${getCellClass(getValue(holding, "Day Chg %"), "Day Chg %")}`}>
                                            {formatValue(getValue(holding, "Day Chg %"), "Day Chg %")}
                                        </div>
                                    )}
                                </SymbolAndAmount>
                            </div>

                            {isCardExpanded && (
                                <div className="flex justify-between items-center bg-zinc-500/5 dark:bg-zinc-400/5 p-2 rounded-md">
                                    <div className="flex flex-col gap-1">
                                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                                            <span>{holding.Account}</span>
                                        </div>
                                        {holding.lots && holding.lots.length > 0 && (
                                            <div className="flex items-center gap-1 text-[10px] bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 px-1.5 py-0.5 rounded-full w-fit">
                                                <Layers className="w-2.5 h-2.5" />
                                                <span className="font-medium">{holding.lots.length} Lots</span>
                                            </div>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="text-right">
                                            <div className={`text-sm font-medium ${getCellClass(getValue(holding, "Day Chg"), "Day Chg")}`}>
                                                {formatValue(getValue(holding, "Day Chg"), "Day Chg")}
                                            </div>
                                            <div className={`text-xs ${getCellClass(getValue(holding, "Day Chg %"), "Day Chg %")}`}>
                                                {formatValue(getValue(holding, "Day Chg %"), "Day Chg %")}
                                            </div>
                                        </div>
                                        {holding.lots && holding.lots.length > 0 && (
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    toggleLotExpansion(expKey);
                                                }}
                                                className="p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-colors border-none"
                                            >
                                                {isLotExpanded ? (
                                                    <ChevronDown className="w-4 h-4 text-cyan-500" />
                                                ) : (
                                                    <ChevronRight className="w-4 h-4 text-muted-foreground" />
                                                )}
                                            </button>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>

                        {isCardExpanded && (
                            <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm mt-3 pt-3">
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Qty:</span>
                                    <span className="text-foreground font-medium">{formatValue(getValue(holding, "Quantity"), "Quantity")}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Price:</span>
                                    <span className="text-foreground font-medium">{formatValue(getValue(holding, "Price"), "Price")}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Avg Cost:</span>
                                    <span className="text-foreground font-medium">{formatValue(getValue(holding, "Avg Cost"), "Avg Cost")}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Div Yield:</span>
                                    <span className="text-foreground font-medium">{formatValue(getValue(holding, "Yield (Mkt) %"), "Yield (Mkt) %")}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">AI Score:</span>
                                    <div className="flex justify-end">
                                        {holding.ai_score !== null && holding.ai_score !== undefined ? (
                                            <div className={`px-1.5 py-0.5 rounded text-[10px] font-bold text-white ${holding.ai_score >= 8.0 ? 'bg-emerald-500' :
                                                holding.ai_score >= 6.0 ? 'bg-amber-500' : 'bg-red-500'
                                                }`}>
                                                {holding.ai_score.toFixed(1)}
                                            </div>
                                        ) : <span className="text-muted-foreground/30 leading-none">-</span>}
                                    </div>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Intrinsic:</span>
                                    <span className={`font-medium ${holding.intrinsic_value !== null && holding.intrinsic_value !== undefined && holding.Price !== undefined ? (
                                        holding.intrinsic_value > (holding.Price as number) ? 'text-up' :
                                            holding.intrinsic_value < (holding.Price as number) ? 'text-rose-500' : 'text-foreground'
                                    ) : 'text-foreground'
                                        }`}>
                                        {formatValue(holding.intrinsic_value, "Intrinsic Value")}
                                        {holding.margin_of_safety !== null && holding.margin_of_safety !== undefined && (
                                            <span className="text-[10px] opacity-70 ml-1">
                                                ({Math.abs(holding.margin_of_safety).toFixed(1)}%)
                                            </span>
                                        )}
                                    </span>
                                </div>
                                <div className="flex flex-col items-center justify-center col-span-2 bg-emerald-500/5 dark:bg-emerald-400/5 p-3 rounded-lg">
                                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Total Return</span>
                                    <span className={`text-base font-bold ${getCellClass(getValue(holding, "Total G/L"), "Total G/L")}`}>
                                        {formatValue(getValue(holding, "Total G/L"), "Total G/L")} ({formatValue(getValue(holding, "Total Ret %"), "Total Ret %")})
                                    </span>
                                </div>
                            </div>
                        )}

                        {isCardExpanded && isLotExpanded && holding.lots && holding.lots.length > 0 && (
                            <div className="mt-4 pt-3">
                                <h4 className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">Tax Lots</h4>
                                <div className="space-y-2">
                                    {holding.lots.map((lot, lotIdx) => {
                                        const holdingPrice = getValue(holding, "Price") as number;
                                        const gain = getLotValue(lot, "Unreal. G/L", holdingPrice);
                                        const gainPct = getLotValue(lot, "Unreal. G/L %", holdingPrice);
                                        return (
                                            <div key={`mobile-lot-${lotIdx}`} className="bg-secondary p-2 rounded text-xs">
                                                <div className="flex justify-between items-center mb-1">
                                                    <span className="font-medium text-foreground">
                                                        {formatValue(getLotValue(lot, "Symbol"), "Symbol")}
                                                    </span>
                                                    <span className={`font-medium ${getCellClass(gain, "Unreal. G/L")}`}>
                                                        {formatValue(gain, "Unreal. G/L")} ({formatValue(gainPct, "Unreal. G/L %")})
                                                    </span>
                                                </div>
                                                <div className="flex justify-between text-muted-foreground">
                                                    <span>Qty: {formatValue(getLotValue(lot, "Quantity"), "Quantity")}</span>
                                                    <span>Cost: {formatValue(getLotValue(lot, "Cost Basis"), "Cost Basis")}</span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </Card>
                );
            })}
        </div>
    );
};
