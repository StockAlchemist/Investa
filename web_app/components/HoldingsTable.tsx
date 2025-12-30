"use client";

import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { exportToCSV } from '../lib/export';
import { Holding, Lot } from '../lib/api';

interface HoldingsTableProps {
    holdings: Holding[];
    currency: string;
}

// Mapping from UI Header to Data Key Prefix (or exact key)
// Based on src/utils.py
const COLUMN_DEFINITIONS: { [header: string]: string } = {
    "Account": "Account",
    "Symbol": "Symbol",
    "Sector": "Sector",
    "Industry": "Industry",
    "Quantity": "Quantity",
    "Day Chg": "Day Change", // Suffix added dynamically
    "Day Chg %": "Day Change %",
    "Avg Cost": "Avg Cost",
    "Price": "Price",
    "Cost Basis": "Cost Basis",
    "Mkt Val": "Market Value",
    "% of Total": "pct_of_total",
    "Unreal. G/L": "Unreal. Gain",
    "Unreal. G/L %": "Unreal. Gain %",
    "Real. G/L": "Realized Gain",
    "Divs": "Dividends",
    "Fees": "Commissions",
    "Total G/L": "Total Gain",
    "Total Ret %": "Total Return %",
    "IRR (%)": "IRR (%)",
    "Total Buy Cost": "Total Buy Cost",
    "Yield (Cost) %": "Div. Yield (Cost) %",
    "Yield (Mkt) %": "Div. Yield (Current) %",
    "FX G/L": "FX Gain/Loss",
    "FX G/L %": "FX Gain/Loss %",
    "Est. Income": "Est. Ann. Income",
};

const DEFAULT_VISIBLE_COLUMNS = [
    "Symbol", "Quantity", "Price", "Mkt Val", "Day Chg", "Day Chg %", "Unreal. G/L", "Unreal. G/L %"
];

type SortDirection = 'asc' | 'desc';

interface SortConfig {
    key: string;
    direction: SortDirection;
}

export default function HoldingsTable({ holdings, currency }: HoldingsTableProps) {
    const [visibleColumns, setVisibleColumns] = useState<string[]>(DEFAULT_VISIBLE_COLUMNS);
    const [showLots, setShowLots] = useState(false);
    const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'Mkt Val', direction: 'desc' });
    const [isColumnMenuOpen, setIsColumnMenuOpen] = useState(false);
    const [draggedColumn, setDraggedColumn] = useState<string | null>(null);
    const columnMenuRef = useRef<HTMLDivElement>(null);
    const isLoaded = useRef(false);
    const [visibleRows, setVisibleRows] = useState(10);

    // Initialize columns and sort from localStorage
    useEffect(() => {
        const savedColumns = localStorage.getItem('investa_holdings_columns');
        if (savedColumns) {
            try {
                const parsed = JSON.parse(savedColumns);
                if (Array.isArray(parsed) && parsed.length > 0) {
                    // eslint-disable-next-line react-hooks/set-state-in-effect
                    setVisibleColumns(parsed);
                }
            } catch (e) {
                console.error("Failed to parse saved columns", e);
            }
        }

        const savedSort = localStorage.getItem('investa_holdings_sort');
        if (savedSort) {
            try {
                const parsed = JSON.parse(savedSort);
                if (parsed.key && parsed.direction) {
                    setSortConfig(parsed);
                }
            } catch (e) {
                console.error("Failed to parse saved sort config", e);
            }
        }

        isLoaded.current = true;
    }, []);

    // Persist columns to localStorage on change
    useEffect(() => {
        if (!isLoaded.current) return;
        if (visibleColumns && visibleColumns.length > 0) {
            localStorage.setItem('investa_holdings_columns', JSON.stringify(visibleColumns));
        }
    }, [visibleColumns]);

    // Persist sort to localStorage on change
    useEffect(() => {
        if (!isLoaded.current) return;
        localStorage.setItem('investa_holdings_sort', JSON.stringify(sortConfig));
    }, [sortConfig]);

    // Close column menu when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (columnMenuRef.current && !columnMenuRef.current.contains(event.target as Node)) {
                setIsColumnMenuOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    // Helper to get value from holding object handling currency suffix
    const getValue = useCallback((holding: Holding, header: string) => {
        const prefix = COLUMN_DEFINITIONS[header];
        if (!prefix) return null;

        // Try exact match first
        if (holding[prefix] !== undefined) return holding[prefix];

        // Try with currency suffix (e.g., "Market Value (USD)")
        const keyWithCurrency = `${prefix} (${currency})`;
        if (holding[keyWithCurrency] !== undefined) return holding[keyWithCurrency];

        // Fallback: search for key starting with prefix
        const foundKey = Object.keys(holding).find(k => k.startsWith(prefix));
        return foundKey ? holding[foundKey] : null;
    }, [currency]);

    // In getLotValue
    const getLotValue = useCallback((lot: Lot, header: string, holdingPrice?: number) => {
        if (header === 'Quantity') return lot.Quantity;
        if (header === 'Cost Basis' || header === 'Total Buy Cost') return lot['Cost Basis'];
        if (header === 'Mkt Val') {
            if (lot['Market Value']) return lot['Market Value'];
            if (holdingPrice && lot.Quantity) return holdingPrice * lot.Quantity;
            return null;
        }
        if (header === 'Unreal. G/L' || header === 'Total G/L') {
            if (lot['Unreal. Gain']) return lot['Unreal. Gain'];
            // Calculate if missing
            const mktVal = lot['Market Value'] || (holdingPrice ? holdingPrice * lot.Quantity : 0);
            if (mktVal && lot['Cost Basis']) return mktVal - lot['Cost Basis'];
            return null;
        }
        if (header === 'Unreal. G/L %' || header === 'Total Ret %') {
            if (lot['Unreal. Gain %']) return lot['Unreal. Gain %'];
            // Calculate if missing
            const mktVal = lot['Market Value'] || (holdingPrice ? holdingPrice * lot.Quantity : 0);
            if (mktVal && lot['Cost Basis']) return ((mktVal - lot['Cost Basis']) / lot['Cost Basis']) * 100;
            return null;
        }
        // Calculated fields
        if ((header === 'Price' || header === 'Avg Cost') && lot.Quantity) return lot['Cost Basis'] / lot.Quantity;

        // Show Date in the first visible text column (usually Symbol or Account)
        // We will default to showing it in the "Symbol" column if present, else Account.
        if (header === 'Symbol') return `Lot: ${lot.Date}`;
        if (header === 'Account' && !visibleColumns.includes('Symbol')) return `Lot: ${lot.Date}`;

        return null;
    }, [visibleColumns]);

    const handleSort = (header: string) => {
        setSortConfig(current => ({
            key: header,
            direction: current.key === header && current.direction === 'desc' ? 'asc' : 'desc',
        }));
    };

    const sortedHoldings = useMemo(() => {
        if (!holdings) return [];
        return [...holdings].sort((a, b) => {
            const valA = getValue(a, sortConfig.key);
            const valB = getValue(b, sortConfig.key);

            if (valA === null || valA === undefined) return 1;
            if (valB === null || valB === undefined) return -1;

            if (typeof valA === 'string' && typeof valB === 'string') {
                return sortConfig.direction === 'asc'
                    ? valA.localeCompare(valB)
                    : valB.localeCompare(valA);
            }

            const numA = typeof valA === 'number' ? valA : 0;
            const numB = typeof valB === 'number' ? valB : 0;
            return sortConfig.direction === 'asc' ? (numA - numB) : (numB - numA);
        });
    }, [holdings, sortConfig, getValue]);

    if (!holdings || holdings.length === 0) {
        return <div className="p-4 text-center text-gray-500">No holdings found.</div>;
    }

    const toggleColumn = (header: string) => {
        setVisibleColumns(current =>
            current.includes(header)
                ? current.filter(c => c !== header)
                : [...current, header]
        );
    };

    // Drag and Drop Handlers
    const handleDragStart = (e: React.DragEvent<HTMLTableHeaderCellElement>, header: string) => {
        setDraggedColumn(header);
        e.dataTransfer.effectAllowed = 'move';
        // Set a transparent drag image or custom styling if needed
        // e.dataTransfer.setDragImage(e.currentTarget, 0, 0);
    };

    const handleDragOver = (e: React.DragEvent<HTMLTableHeaderCellElement>) => {
        e.preventDefault(); // Necessary to allow dropping
        e.dataTransfer.dropEffect = 'move';
    };

    const handleDrop = (e: React.DragEvent<HTMLTableHeaderCellElement>, targetHeader: string) => {
        e.preventDefault();
        if (!draggedColumn || draggedColumn === targetHeader) return;

        const currentIndex = visibleColumns.indexOf(draggedColumn);
        const targetIndex = visibleColumns.indexOf(targetHeader);

        if (currentIndex !== -1 && targetIndex !== -1) {
            const newColumns = [...visibleColumns];
            newColumns.splice(currentIndex, 1); // Remove from old pos
            newColumns.splice(targetIndex, 0, draggedColumn); // Insert at new pos
            setVisibleColumns(newColumns);
        }
        setDraggedColumn(null);
    };

    // Formatters
    const formatValue = (val: unknown, header: string) => {
        if (val === null || val === undefined) return '-';
        if (typeof val !== 'number') return val as string;

        if (header.includes('%') || header.includes('Yield') || header.includes('Ret')) {
            return `${val.toFixed(2)}%`;
        }

        if (['Price', 'Cost Basis', 'Avg Cost', 'Mkt Val', 'Day Chg', 'Unreal. G/L', 'Real. G/L', 'Divs', 'Fees', 'Total G/L', 'Total Buy Cost', 'FX G/L', 'Est. Income'].includes(header)) {
            return val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }

        if (header === 'Quantity') {
            return val.toLocaleString();
        }

        return val;
    };

    const getCellClass = (val: unknown, header: string) => {
        if (typeof val !== 'number') return '';
        if (['Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Real. G/L', 'Total G/L', 'Total Ret %', 'FX G/L', 'FX G/L %', 'IRR (%)'].includes(header)) {
            if (Math.abs(val) < 0.001) return 'text-muted-foreground';
            return val > 0 ? 'text-emerald-600 dark:text-emerald-400 font-medium' : 'text-rose-600 dark:text-rose-400 font-medium';
        }
        return '';
    };

    const visibleHoldings = sortedHoldings.slice(0, visibleRows);

    const handleShowMore = () => {
        setVisibleRows(prev => prev + 20);
    };

    const handleShowAll = () => {
        setVisibleRows(sortedHoldings.length);
    };

    return (
        <div className="bg-card backdrop-blur-md border border-border rounded-xl shadow-sm mt-4 overflow-hidden scrollbar-thin scrollbar-thumb-zinc-200 dark:scrollbar-thumb-zinc-800 scrollbar-track-transparent">
            <div className="p-4 border-b border-black/5 dark:border-white/5 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div className="flex items-center gap-4 w-full md:w-auto justify-between md:justify-start">
                    <h2 className="text-lg font-bold text-foreground">Holdings</h2>
                    <div className="text-sm text-muted-foreground">
                        Showing {visibleHoldings.length} of {sortedHoldings.length}
                    </div>
                </div>

                {/* Column Selector */}
                {/* Column Selector */}
                <div className="relative flex flex-wrap gap-2 w-full md:w-auto" ref={columnMenuRef}>
                    <button
                        onClick={() => setIsColumnMenuOpen(!isColumnMenuOpen)}
                        className="flex-1 md:flex-none px-3 py-1.5 text-sm font-medium text-foreground bg-secondary border border-border rounded-md shadow-sm hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center backdrop-blur-md"
                    >
                        Columns
                    </button>

                    <button
                        onClick={() => setShowLots(!showLots)}
                        className={`flex-1 md:flex-none px-3 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center backdrop-blur-md
                            ${showLots
                                ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/50'
                                : 'text-foreground bg-secondary border border-border hover:bg-accent/10'
                            }`}
                    >
                        {showLots ? 'Hide Lots' : 'Show Lots'}
                    </button>

                    <button
                        onClick={() => exportToCSV(holdings, 'holdings.csv')}
                        className="flex-1 md:flex-none px-3 py-1.5 text-sm font-medium text-foreground bg-secondary border border-border rounded-md shadow-sm hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center backdrop-blur-md"
                    >
                        Export CSV
                    </button>

                    {isColumnMenuOpen && (
                        <div className="absolute right-0 z-50 mt-2 w-56 origin-top-right bg-card border border-border rounded-md shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none max-h-96 overflow-y-auto backdrop-blur-xl">
                            <div className="py-1">
                                {Object.keys(COLUMN_DEFINITIONS).map(header => (
                                    <label key={header} className="flex items-center px-4 py-2 text-sm text-foreground hover:bg-black/5 dark:hover:bg-white/5 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={visibleColumns.includes(header)}
                                            onChange={() => toggleColumn(header)}
                                            className="h-4 w-4 text-cyan-600 focus:ring-cyan-500 border-black/20 dark:border-white/20 rounded bg-black/5 dark:bg-white/5"
                                        />
                                        <span className="ml-2">{header}</span>
                                    </label>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Desktop Table View */}
            <div className="hidden md:block overflow-x-auto">
                <table className="min-w-full divide-y divide-black/5 dark:divide-white/5">
                    <thead className="bg-secondary">
                        <tr>
                            {visibleColumns.map(header => (
                                <th
                                    key={header}
                                    scope="col"
                                    draggable
                                    onDragStart={(e) => handleDragStart(e, header)}
                                    onDragOver={handleDragOver}
                                    onDrop={(e) => handleDrop(e, header)}
                                    className={`px-6 py-3 text-right text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:bg-accent/10 transition-colors select-none whitespace-nowrap ${draggedColumn === header ? 'opacity-50 bg-secondary' : ''}`}
                                    onClick={() => handleSort(header)}
                                >
                                    <div className="flex items-center justify-end gap-1">
                                        {header}
                                        {sortConfig.key === header && (
                                            <span>{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>
                                        )}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-black/5 dark:divide-white/5">
                        {visibleHoldings.map((holding, idx) => (
                            <React.Fragment key={`${holding.Symbol}-${idx}`}>
                                <tr className="hover:bg-accent/5 transition-colors">
                                    {visibleColumns.map(header => {
                                        const val = getValue(holding, header);
                                        return (
                                            <td key={header} className={`px-6 py-4 whitespace-nowrap text-sm text-right ${getCellClass(val, header) || (header === 'Symbol' || header === 'Account' ? 'text-foreground font-medium' : 'text-muted-foreground')}`}>
                                                {formatValue(val, header)}
                                            </td>
                                        );
                                    })}
                                </tr>
                                {showLots && holding.lots && holding.lots.length > 0 && (
                                    holding.lots.map((lot, lotIdx) => (
                                        <tr key={`${holding.Symbol}-lot-${lotIdx}`} className="bg-zinc-50/50 dark:bg-zinc-900/40">
                                            {visibleColumns.map(header => {
                                                const holdingPrice = getValue(holding, "Price") as number;
                                                const val = getLotValue(lot, header, holdingPrice);
                                                return (
                                                    <td key={header} className={`px-6 py-2 whitespace-nowrap text-xs text-right border-t border-white/5 ${getCellClass(val, header) || (header === 'Symbol' ? 'pl-10 text-muted-foreground italic' : 'text-muted-foreground')}`}>
                                                        {formatValue(val, header)}
                                                    </td>
                                                );
                                            })}
                                        </tr>
                                    ))
                                )}
                            </React.Fragment>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Mobile Card View */}
            <div className="block md:hidden space-y-4 p-4">
                {visibleHoldings.map((holding, idx) => (
                    <div key={`mobile-${holding.Symbol}-${idx}`} className="bg-card rounded-lg border border-border shadow-sm p-4 backdrop-blur-md">
                        <div className="flex justify-between items-start mb-2">
                            <div>
                                <h3 className="text-lg font-bold text-foreground">{holding.Symbol}</h3>
                                <p className="text-xs text-muted-foreground">{holding.Account}</p>
                            </div>
                            <div className="text-right">
                                <div className="text-lg font-bold text-foreground">
                                    {formatValue(getValue(holding, "Mkt Val"), "Mkt Val")}
                                </div>
                                <div className={`text-sm ${getCellClass(getValue(holding, "Day Chg %"), "Day Chg %")}`}>
                                    {formatValue(getValue(holding, "Day Chg"), "Day Chg")} ({formatValue(getValue(holding, "Day Chg %"), "Day Chg %")})
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm mt-3 pt-3 border-t border-black/5 dark:border-white/10">
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
                            <div className="flex justify-between col-span-2 bg-secondary border border-border p-2 rounded backdrop-blur-sm">
                                <span className="text-muted-foreground">Total Return:</span>
                                <span className={`font-medium ${getCellClass(getValue(holding, "Total G/L"), "Total G/L")}`}>
                                    {formatValue(getValue(holding, "Total G/L"), "Total G/L")} ({formatValue(getValue(holding, "Total Ret %"), "Total Ret %")})
                                </span>
                            </div>
                        </div>

                        {/* Mobile Lots View */}
                        {showLots && holding.lots && holding.lots.length > 0 && (
                            <div className="mt-4 pt-3 border-t border-black/5 dark:border-white/10">
                                <h4 className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">Tax Lots</h4>
                                <div className="space-y-2">
                                    {holding.lots.map((lot, lotIdx) => {
                                        const holdingPrice = getValue(holding, "Price") as number;
                                        const gain = getLotValue(lot, "Unreal. G/L", holdingPrice);
                                        const gainPct = getLotValue(lot, "Unreal. G/L %", holdingPrice);
                                        return (
                                            <div key={`mobile-lot-${lotIdx}`} className="bg-secondary p-2 rounded text-xs border border-border">
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
                                        )
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>
            {visibleRows < sortedHoldings.length && (
                <div className="flex justify-center gap-4 p-4 border-t border-black/5 dark:border-white/10">
                    <button
                        onClick={handleShowMore}
                        className="px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm"
                    >
                        Show More
                    </button>
                    <button
                        onClick={handleShowAll}
                        className="px-4 py-2 bg-card text-foreground border border-border rounded-md hover:bg-secondary transition-colors text-sm font-medium shadow-sm"
                    >
                        Show All
                    </button>
                </div>
            )}
        </div>
    );
}
