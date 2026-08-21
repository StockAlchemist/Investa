import React from 'react';
import { ChevronDown, ChevronRight, Layers, PenLine, Save, X } from 'lucide-react';
import { Holding, Lot } from '../../lib/api';
import { SortConfig, GroupingOption } from './types';
import { getCellClass, formatHoldingValue } from './holdingsUtils';
import { getHeatmapClass } from '../../lib/utils';
import { Skeleton } from '../ui/skeleton';
import { TrendSparkline } from '../ui/TrendSparkline';
import { InlineProgressBar } from '../ui/InlineProgressBar';
import { SemanticBadge } from '../ui/SemanticBadge';
import WatchlistStar from '../WatchlistStar';

interface HoldingsDesktopTableProps {
    mobileViewMode: 'card' | 'table';
    visibleColumns: string[];
    isLoading: boolean;
    draggedColumn: string | null;
    handleDragStart: (e: React.DragEvent<HTMLTableHeaderCellElement>, header: string) => void;
    handleDragOver: (e: React.DragEvent<HTMLTableHeaderCellElement>) => void;
    handleDrop: (e: React.DragEvent<HTMLTableHeaderCellElement>, header: string) => void;
    handleSort: (header: string) => void;
    sortConfig: SortConfig;
    groupBy: GroupingOption;
    groupedHoldings: { key: string; holdings: Holding[]; aggregates: Record<string, number> }[] | null;
    visibleHoldings: Holding[];
    expandedGroups: Set<string>;
    toggleGroup: (key: string) => void;
    expandedLots: Set<string>;
    toggleLotExpansion: (key: string) => void;
    getExpansionKey: (holding: Holding) => string;
    getValue: (holding: Holding, header: string) => string | number | string[] | number[] | null;
    getLotValue: (lot: Lot, header: string, holdingPrice?: number) => string | number | null;
    currency: string;
    openStockDetail: (symbol: string, currency?: string) => void;
    editingTags: { symbol: string; account: string; currentTags: string } | null;
    setEditingTags: (val: { symbol: string; account: string; currentTags: string } | null) => void;
    tagsInput: string;
    setTagsInput: (tags: string) => void;
    handleEditTags: (symbol: string, account: string, currentTags: string[]) => void;
    handleSaveTags: () => void;
}

export const HoldingsDesktopTable: React.FC<HoldingsDesktopTableProps> = ({
    mobileViewMode,
    visibleColumns,
    isLoading,
    draggedColumn,
    handleDragStart,
    handleDragOver,
    handleDrop,
    handleSort,
    sortConfig,
    groupBy,
    groupedHoldings,
    visibleHoldings,
    expandedGroups,
    toggleGroup,
    expandedLots,
    toggleLotExpansion,
    getExpansionKey,
    getValue,
    getLotValue,
    currency,
    openStockDetail,
    editingTags,
    setEditingTags,
    tagsInput,
    setTagsInput,
    handleEditTags,
    handleSaveTags,
}) => {
    const formatValue = (val: unknown, field: string) => formatHoldingValue(val, field, currency);

    return (
        <div className={`${mobileViewMode === 'table' ? 'block' : 'hidden'} md:block overflow-x-auto [overflow-y:clip]`}>
            <table className="min-w-full">
                <thead className="bg-secondary sticky top-0 z-30 font-semibold border-b">
                    <tr>
                        {visibleColumns.map(header => {
                            const isLeftAligned = ['Symbol', 'Account', 'Sector', 'Industry', 'Tags'].includes(header);
                            const isSticky = header === 'Symbol' || (header === 'Account' && visibleColumns.indexOf('Account') === 0);
                            return (
                                <th
                                    key={header}
                                    scope="col"
                                    draggable
                                    onDragStart={(e) => handleDragStart(e, header)}
                                    onDragOver={handleDragOver}
                                    onDrop={(e) => handleDrop(e, header)}
                                    className={`px-6 py-3 text-xs font-semibold text-muted-foreground transition-colors select-none whitespace-nowrap group hover:bg-accent/10 cursor-pointer ${draggedColumn === header ? 'opacity-50 bg-secondary' : ''} ${isLeftAligned ? 'text-left' : 'text-right'} ${isSticky ? 'sticky left-0 z-40 bg-secondary/95 backdrop-blur-md shadow-[2px_0_5px_-2px_rgba(0,0,0,0.3)]' : ''}`}
                                    onClick={() => handleSort(header)}
                                >
                                    <div className={`flex items-center gap-1 ${isLeftAligned ? 'justify-start' : 'justify-end'}`}>
                                        {header}
                                        {sortConfig.key === header && (
                                            <span className="text-cyan-500">{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>
                                        )}
                                    </div>
                                </th>
                            );
                        })}
                    </tr>
                </thead>
                <tbody className="divide-y-none">
                    {isLoading ? (
                        Array.from({ length: 5 }).map((_, i) => (
                            <tr key={`skeleton-${i}`}>
                                {visibleColumns.map((header, j) => (
                                    <td key={`skeleton-${i}-${j}`} className="px-6 py-4">
                                        <Skeleton className="h-6 w-full ml-auto" />
                                    </td>
                                ))}
                            </tr>
                        ))
                    ) : groupBy && groupedHoldings ? (
                        groupedHoldings.map((group) => (
                            <React.Fragment key={group.key}>
                                {/* Group Header Row */}
                                <tr
                                    className="bg-secondary/30 hover:bg-secondary/50 cursor-pointer transition-colors"
                                    onClick={() => toggleGroup(group.key)}
                                >
                                    <td colSpan={visibleColumns.length} className="px-4 py-3">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                {expandedGroups.has(group.key) ? (
                                                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                                                ) : (
                                                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                                                )}
                                                <span className="font-semibold text-foreground">{group.key}</span>
                                                <span className="text-xs text-muted-foreground bg-secondary px-1.5 py-0.5 rounded-full">
                                                    {group.holdings.length}
                                                </span>
                                            </div>

                                            {/* Group Summaries for Visible Columns */}
                                            <div className="flex items-center gap-6 pr-2">
                                                {visibleColumns.includes('Mkt Val') && (
                                                    <div className="text-sm">
                                                        <span className="text-xs text-muted-foreground mr-1">Mkt:</span>
                                                        <span className="font-medium tabular-nums">{formatValue(group.aggregates['Mkt Val'], 'Mkt Val')}</span>
                                                    </div>
                                                )}
                                                {visibleColumns.includes('Day Chg') && (
                                                    <div className="text-sm hidden sm:block">
                                                        <span className="text-xs text-muted-foreground mr-1">Day:</span>
                                                        <span className={`${getCellClass(group.aggregates['Day Chg'], 'Day Chg')} tabular-nums`}>
                                                            {formatValue(group.aggregates['Day Chg'], 'Day Chg')}
                                                        </span>
                                                    </div>
                                                )}
                                                {visibleColumns.includes('Day Chg %') && (
                                                    <div className="text-sm hidden sm:block">
                                                        <span className={`${getCellClass(group.aggregates['Day Chg'], 'Day Chg %')} tabular-nums`}>
                                                            {formatValue(group.aggregates['Day Chg %'], 'Day Chg %')}
                                                        </span>
                                                    </div>
                                                )}
                                                {visibleColumns.includes('Unreal. G/L') && (
                                                    <div className="text-sm hidden md:block">
                                                        <span className="text-xs text-muted-foreground mr-1">Unreal:</span>
                                                        <span className={`${getCellClass(group.aggregates['Unreal. G/L'], 'Unreal. G/L')} tabular-nums`}>
                                                            {formatValue(group.aggregates['Unreal. G/L'], 'Unreal. G/L')}
                                                        </span>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                                {/* Group Items */}
                                {expandedGroups.has(group.key) && group.holdings.map((holding, idx) => (
                                    <React.Fragment key={`${holding.Symbol}-${idx}`}>
                                        <tr className="hover:bg-accent/5 transition-colors">
                                            {visibleColumns.map(header => {
                                                const val = getValue(holding, header);
                                                const isNumeric = ['Quantity', 'Price', 'Mkt Val', 'Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Cost Basis', 'Avg Cost'].some(k => header.includes(k) || header === k);
                                                const isLeftAligned = ['Symbol', 'Account', 'Sector', 'Industry', 'Tags'].includes(header);
                                                const isHeatmap = ['Day Chg %', 'Unreal. G/L %', 'Total Ret %'].includes(header);
                                                const heatmapClass = isHeatmap ? getHeatmapClass(val as number) : '';

                                                return (
                                                    <td key={header} className={`px-6 py-3 whitespace-nowrap text-sm ${isLeftAligned ? 'text-left' : 'text-right'} ${isNumeric ? 'tabular-nums' : ''} ${getCellClass(val, header) || (header === 'Symbol' || header === 'Account' ? 'text-foreground font-medium' : 'text-muted-foreground')} ${header === 'Symbol' ? 'sticky left-0 z-20 bg-background/95 backdrop-blur-md supports-[backdrop-filter]:bg-background/80' : ''} ${heatmapClass}`}>
                                                        {header === '1M Trend' ? (
                                                            <div className="h-10 w-28 ml-auto">
                                                                {val && Array.isArray(val) && val.length > 1 ? (
                                                                    <TrendSparkline data={val as number[]} />
                                                                ) : (
                                                                    <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground/50">
                                                                        no data
                                                                    </div>
                                                                )}
                                                            </div>
                                                        ) : ['Contribution %', '% of Total', 'pct_of_total'].includes(header) ? (
                                                            <div className="w-24 ml-auto h-6">
                                                                <InlineProgressBar value={(val as number) || 0} max={100}>
                                                                    <span className={`text-xs font-medium relative z-10 ${((val as number) || 0) < 0 ? 'text-red-600 dark:text-red-500' : ''}`}>{formatValue(val, header)}</span>
                                                                </InlineProgressBar>
                                                            </div>
                                                        ) : header === 'Symbol' ? (
                                                            <div className="flex items-center gap-3">
                                                                <WatchlistStar
                                                                    symbol={holding.Symbol}
                                                                    className="text-muted-foreground hover:text-amber-400"
                                                                    onIconClick={() => openStockDetail(holding.Symbol, currency)}
                                                                />
                                                                <div className="flex flex-col">
                                                                    <div className="flex items-center gap-2">
                                                                        <span
                                                                            className="font-bold text-foreground hover:text-cyan-500 cursor-pointer transition-colors"
                                                                            onClick={() => openStockDetail(holding.Symbol, currency)}
                                                                        >
                                                                            {holding.Symbol}
                                                                        </span>
                                                                        {holding.lots && holding.lots.length > 0 && (
                                                                            <button
                                                                                onClick={(e) => {
                                                                                    e.stopPropagation();
                                                                                    toggleLotExpansion(getExpansionKey(holding));
                                                                                }}
                                                                                className="p-0.5 hover:bg-accent/20 rounded-md transition-colors"
                                                                                title={expandedLots.has(getExpansionKey(holding)) ? "Hide Lots" : "Show Lots"}
                                                                            >
                                                                                {expandedLots.has(getExpansionKey(holding)) ? (
                                                                                    <ChevronDown className="w-3.5 h-3.5 text-cyan-500" />
                                                                                ) : (
                                                                                    <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
                                                                                )}
                                                                            </button>
                                                                        )}
                                                                    </div>
                                                                    {holding.lots && holding.lots.length > 0 && (
                                                                        <div className="flex items-center gap-1 mt-0.5" title={`${holding.lots.length} tax lots`}>
                                                                            <Layers className="w-3 h-3 text-muted-foreground" />
                                                                            <span className="text-[10px] text-muted-foreground">{holding.lots.length} Lots</span>
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        ) : header === 'Tags' ? (
                                                            <div className="group/tags flex items-center gap-2 justify-start min-w-[100px]">
                                                                {editingTags?.symbol === holding.Symbol && editingTags?.account === (holding.Account || 'All') ? (
                                                                    <div className="flex items-center gap-1 animate-in fade-in zoom-in duration-200">
                                                                        <input
                                                                            type="text"
                                                                            value={tagsInput}
                                                                            onChange={(e) => setTagsInput(e.target.value)}
                                                                            className="w-32 h-7 text-xs bg-background rounded px-2 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                                                                            placeholder="e.g. Dividend, Tech"
                                                                            autoFocus
                                                                            onKeyDown={(e) => {
                                                                                if (e.key === 'Enter') handleSaveTags();
                                                                                if (e.key === 'Escape') setEditingTags(null);
                                                                            }}
                                                                        />
                                                                        <button onClick={handleSaveTags} className="p-1 hover:bg-emerald-500/10 text-emerald-600 rounded">
                                                                            <Save className="w-3.5 h-3.5" />
                                                                        </button>
                                                                        <button onClick={() => setEditingTags(null)} className="p-1 hover:bg-red-500/10 text-red-600 rounded">
                                                                            <X className="w-3.5 h-3.5" />
                                                                        </button>
                                                                    </div>
                                                                ) : (
                                                                    <>
                                                                        {val && Array.isArray(val) && val.length > 0 ? (
                                                                            <div className="flex flex-wrap gap-1">
                                                                                {(val as string[]).map((tag: string, i: number) => (
                                                                                    <SemanticBadge key={i} text={tag} />
                                                                                ))}
                                                                            </div>
                                                                        ) : (
                                                                            <span className="text-muted-foreground/30 text-xs italic group-hover/tags:opacity-100 opacity-0 transition-opacity">Add tags...</span>
                                                                        )}

                                                                        <button
                                                                            onClick={() => handleEditTags(holding.Symbol, holding.Account || 'All', Array.isArray(val) ? val as string[] : [])}
                                                                            className="opacity-0 group-hover/tags:opacity-100 p-1 hover:bg-secondary rounded-full transition-all text-muted-foreground hover:text-foreground"
                                                                        >
                                                                            <PenLine className="w-3 h-3" />
                                                                        </button>
                                                                    </>
                                                                )}
                                                            </div>
                                                        ) : header === 'AI Score' ? (
                                                            <div className="flex justify-end">
                                                                {val !== null && val !== undefined ? (
                                                                    <div
                                                                        className={`px-1.5 py-0.5 rounded text-[10px] font-bold text-white shadow-sm ${(val as number) >= 8.0 ? 'bg-emerald-500' :
                                                                            (val as number) >= 6.0 ? 'bg-amber-500' : 'bg-red-500'
                                                                            }`}
                                                                    >
                                                                        {(val as number).toFixed(1)}
                                                                    </div>
                                                                ) : (
                                                                    <span className="text-muted-foreground/30">-</span>
                                                                )}
                                                            </div>
                                                        ) : header === 'Intrinsic Value' ? (
                                                            <div className="flex flex-col items-end gap-1.5 min-w-[80px]">
                                                                <span className={
                                                                    val !== null && val !== undefined && (holding.Price !== undefined || holding.price !== undefined) ? (
                                                                        (val as number) > ((holding.Price || holding.price) as number) ? 'text-emerald-600 dark:text-emerald-400 font-medium' :
                                                                            (val as number) < ((holding.Price || holding.price) as number) ? 'text-rose-500 font-medium' : ''
                                                                    ) : ''
                                                                }>
                                                                    {formatValue(val, header)}
                                                                    {holding.margin_of_safety !== null && holding.margin_of_safety !== undefined && (
                                                                        <span className="text-[10px] opacity-70 ml-1.5 tabular-nums">
                                                                            ({Math.abs(holding.margin_of_safety).toFixed(1)}%)
                                                                        </span>
                                                                    )}
                                                                </span>
                                                                {holding.margin_of_safety !== null && holding.margin_of_safety !== undefined && (
                                                                    <div className="w-16 h-1 bg-secondary rounded-full overflow-hidden flex">
                                                                        {holding.margin_of_safety > 0 ? (
                                                                            <>
                                                                                <div className="w-1/2 bg-transparent" />
                                                                                <div className="h-full bg-emerald-500" style={{ width: `${Math.min(holding.margin_of_safety, 100) / 2}%` }} />
                                                                            </>
                                                                        ) : (
                                                                            <>
                                                                                <div className="h-full bg-transparent" style={{ width: `${50 - Math.min(Math.abs(holding.margin_of_safety), 100) / 2}%` }} />
                                                                                <div className="h-full bg-rose-500" style={{ width: `${Math.min(Math.abs(holding.margin_of_safety), 100) / 2}%` }} />
                                                                            </>
                                                                        )}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        ) : (
                                                            formatValue(val, header)
                                                        )}
                                                    </td>
                                                );
                                            })}
                                        </tr>
                                        {expandedLots.has(getExpansionKey(holding)) && holding.lots && holding.lots.length > 0 && (
                                            holding.lots.map((lot, lotIdx) => (
                                                <tr key={`${holding.Symbol}-lot-${lotIdx}`} className="bg-zinc-50/50 dark:bg-zinc-900/40">
                                                    {visibleColumns.map(header => {
                                                        const holdingPrice = getValue(holding, "Price") as number;
                                                        const val = getLotValue(lot, header, holdingPrice);
                                                        const isNumeric = ['Quantity', 'Price', 'Mkt Val', 'Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Cost Basis', 'Avg Cost'].some(k => header.includes(k) || header === k);

                                                        return (
                                                            <td key={header} className={`px-6 py-2 whitespace-nowrap text-xs text-right ${isNumeric ? 'tabular-nums' : ''} ${getCellClass(val, header) || (header === 'Symbol' ? 'pl-10 text-muted-foreground italic flex items-center justify-end gap-2' : 'text-muted-foreground')} ${header === 'Symbol' ? 'sticky left-0 z-20 bg-background/90 backdrop-blur-md shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]' : ''}`}>
                                                                {header === 'Symbol' && <span className="text-[10px] opacity-50">↳</span>}
                                                                {formatValue(val, header)}
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            ))
                                        )}
                                    </React.Fragment>
                                ))}
                            </React.Fragment>
                        ))
                    ) : (
                        visibleHoldings.map((holding, idx) => (
                            <React.Fragment key={`${holding.Symbol}-${idx}`}>
                                <tr className="hover:bg-accent/5 transition-colors">
                                    {visibleColumns.map(header => {
                                        const val = getValue(holding, header);
                                        const isNumeric = ['Quantity', 'Price', 'Mkt Val', 'Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Cost Basis', 'Avg Cost'].some(k => header.includes(k) || header === k);
                                        const isLeftAligned = ['Symbol', 'Account', 'Sector', 'Industry', 'Tags'].includes(header);
                                        const isHeatmap = ['Day Chg %', 'Unreal. G/L %', 'Total Ret %'].includes(header);
                                        const heatmapClass = isHeatmap ? getHeatmapClass(val as number) : '';

                                        return (
                                            <td key={header} className={`px-6 py-3 whitespace-nowrap text-sm ${isLeftAligned ? 'text-left' : 'text-right'} ${isNumeric ? 'tabular-nums' : ''} ${getCellClass(val, header) || (header === 'Symbol' || header === 'Account' ? 'text-foreground font-medium' : 'text-muted-foreground')} ${header === 'Symbol' ? 'sticky left-0 z-20 bg-background/90 backdrop-blur-lg shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]' : ''} ${heatmapClass}`}>
                                                {header === '1M Trend' ? (
                                                    <div className="h-10 w-28 ml-auto">
                                                        {val && Array.isArray(val) && val.length > 1 ? (
                                                            <TrendSparkline data={val as number[]} />
                                                        ) : (
                                                            <div className="h-full w-full flex items-center justify-center text-[10px] text-muted-foreground/30">
                                                                no data
                                                            </div>
                                                        )}
                                                    </div>
                                                ) : ['Contribution %', '% of Total', 'pct_of_total'].includes(header) ? (
                                                    <div className="w-24 ml-auto h-6">
                                                        <InlineProgressBar value={(val as number) || 0} max={100}>
                                                            <span className={`text-xs font-medium relative z-10 ${((val as number) || 0) < 0 ? 'text-red-600 dark:text-red-500' : ''}`}>{formatValue(val, header)}</span>
                                                        </InlineProgressBar>
                                                    </div>
                                                ) : header === 'Symbol' ? (
                                                    <div className="flex items-center justify-start gap-3">
                                                        <WatchlistStar symbol={val as string} size="md" />
                                                        <div className="flex flex-col">
                                                            <div className="flex items-center gap-2">
                                                                <button
                                                                    onClick={() => openStockDetail(val as string, currency)}
                                                                    className="font-semibold text-foreground hover:text-cyan-500 transition-colors cursor-pointer"
                                                                >
                                                                    {formatValue(val, header)}
                                                                </button>
                                                                {holding.lots && holding.lots.length > 0 && (
                                                                    <button
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            toggleLotExpansion(getExpansionKey(holding));
                                                                        }}
                                                                        className="p-0.5 hover:bg-accent/20 rounded-md transition-colors"
                                                                        title={expandedLots.has(getExpansionKey(holding)) ? "Hide Lots" : "Show Lots"}
                                                                    >
                                                                        {expandedLots.has(getExpansionKey(holding)) ? (
                                                                            <ChevronDown className="w-3.5 h-3.5 text-cyan-500" />
                                                                        ) : (
                                                                            <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
                                                                        )}
                                                                    </button>
                                                                )}
                                                            </div>
                                                            {holding.lots && holding.lots.length > 0 && (
                                                                <div className="flex items-center gap-1 mt-0.5 cursor-help" title={`${holding.lots.length} tax lots`}>
                                                                    <Layers className="w-3 h-3 text-muted-foreground hover:text-cyan-500 transition-colors" />
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                ) : header === 'Tags' ? (
                                                    <div className="flex items-center justify-end gap-2 group/tags min-w-[120px]">
                                                        <div className="flex flex-wrap gap-1 justify-end">
                                                            {Array.isArray(val) && val.length > 0 ? (
                                                                (val as string[]).map((tag: string, i: number) => (
                                                                    <SemanticBadge key={i} text={tag} />
                                                                ))
                                                            ) : (
                                                                <span className="text-muted-foreground italic text-xs opacity-0 group-hover/tags:opacity-50 transition-opacity">Add tag</span>
                                                            )}
                                                        </div>
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                const tags = Array.isArray(val) ? val as string[] : [];
                                                                const acc = formatValue(getValue(holding, "Account"), "Account") as string;
                                                                handleEditTags(holding.Symbol, acc, tags);
                                                            }}
                                                            className="text-muted-foreground hover:text-cyan-500 opacity-0 group-hover/tags:opacity-100 transition-opacity p-1"
                                                            title="Edit Tags"
                                                        >
                                                            <PenLine className="h-3 w-3" />
                                                        </button>
                                                    </div>
                                                ) : header === 'AI Score' ? (
                                                    <div className="flex justify-end">
                                                        {val !== null && val !== undefined ? (
                                                            <div
                                                                className={`px-2 py-0.5 rounded text-xs font-bold text-white shadow-sm ${(val as number) >= 8.0 ? 'bg-emerald-500' :
                                                                    (val as number) >= 6.0 ? 'bg-amber-500' : 'bg-red-500'
                                                                    }`}
                                                            >
                                                                {(val as number).toFixed(1)}
                                                            </div>
                                                        ) : (
                                                            <span className="text-muted-foreground/30">-</span>
                                                        )}
                                                    </div>
                                                ) : header === 'Intrinsic Value' ? (
                                                    <span className={
                                                        val !== null && val !== undefined && (holding.Price !== undefined || holding.price !== undefined) ? (
                                                            (val as number) > ((holding.Price || holding.price) as number) ? 'text-emerald-600 dark:text-emerald-400 font-medium' :
                                                                (val as number) < ((holding.Price || holding.price) as number) ? 'text-rose-500 font-medium' : ''
                                                        ) : ''
                                                    }>
                                                        {formatValue(val, header)}
                                                        {holding.margin_of_safety !== null && holding.margin_of_safety !== undefined && (
                                                            <span className="text-[10px] opacity-70 ml-1.5 tabular-nums">
                                                                ({Math.abs(holding.margin_of_safety).toFixed(1)}%)
                                                            </span>
                                                        )}
                                                    </span>
                                                ) : (
                                                    formatValue(val, header)
                                                )}
                                            </td>
                                        );
                                    })}
                                </tr>
                                {expandedLots.has(getExpansionKey(holding)) && holding.lots && holding.lots.length > 0 && (
                                    holding.lots.map((lot, lotIdx) => (
                                        <tr key={`${holding.Symbol}-lot-${lotIdx}`} className="bg-zinc-50/50 dark:bg-zinc-900/40">
                                            {visibleColumns.map(header => {
                                                const holdingPrice = getValue(holding, "Price") as number;
                                                const val = getLotValue(lot, header, holdingPrice);
                                                const isNumeric = ['Quantity', 'Price', 'Mkt Val', 'Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Cost Basis', 'Avg Cost'].some(k => header.includes(k) || header === k);

                                                return (
                                                    <td key={header} className={`px-6 py-2 whitespace-nowrap text-xs text-right border-none ${isNumeric ? 'tabular-nums' : ''} ${getCellClass(val, header) || (header === 'Symbol' ? 'pl-10 text-muted-foreground italic flex items-center justify-end gap-2' : 'text-muted-foreground')} ${header === 'Symbol' ? 'sticky left-0 z-20 bg-background/90 backdrop-blur-md shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]' : ''}`}>
                                                        {header === 'Symbol' && <span className="text-[10px] opacity-50">↳</span>}
                                                        {formatValue(val, header)}
                                                    </td>
                                                );
                                            })}
                                        </tr>
                                    ))
                                )}
                            </React.Fragment>
                        ))
                    )}
                </tbody>
            </table>
        </div>
    );
};
