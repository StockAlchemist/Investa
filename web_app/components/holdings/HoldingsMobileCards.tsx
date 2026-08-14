import React from 'react';
import { ChevronDown, ChevronRight, ChevronUp, Layers } from 'lucide-react';
import { Holding, Lot } from '../../lib/api';
import { Card } from '../ui/card';
import WatchlistStar from '../WatchlistStar';
import { getCellClass, formatHoldingValue } from './holdingsUtils';

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
                        className="bg-card rounded-2xl border-none p-0 relative group cursor-pointer hover:border-cyan-500/50 transition-all active:scale-[0.98]"
                        onClick={() => openStockDetail(holding.Symbol, currency)}
                    >
                        <div className="space-y-3">
                            <div className="flex justify-between items-start">
                                <div className="flex items-center gap-3">
                                    <WatchlistStar symbol={holding.Symbol} size="md" />
                                    <h3 className="text-xl font-bold text-foreground leading-none">{holding.Symbol}</h3>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="text-right">
                                        <div className="text-xl font-bold text-foreground leading-none">
                                            {formatValue(getValue(holding, "Mkt Val"), "Mkt Val")}
                                        </div>
                                        {!isCardExpanded && (
                                            <div className={`text-xs font-medium mt-1 ${getCellClass(getValue(holding, "Day Chg %"), "Day Chg %")}`}>
                                                {formatValue(getValue(holding, "Day Chg %"), "Day Chg %")}
                                            </div>
                                        )}
                                    </div>
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
                                </div>
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
                                        holding.intrinsic_value > (holding.Price as number) ? 'text-emerald-500' :
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
