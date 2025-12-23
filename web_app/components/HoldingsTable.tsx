"use client";

import React, { useState, useMemo, useEffect, useRef } from 'react';
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
    const getValue = (holding: Holding, header: string) => {
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
    };

    const getLotValue = (lot: Lot, header: string) => {
        if (header === 'Quantity') return lot.Quantity;
        if (header === 'Cost Basis' || header === 'Total Buy Cost') return lot['Cost Basis'];
        if (header === 'Mkt Val') return lot['Market Value'];
        if (header === 'Unreal. G/L') return lot['Unreal. Gain'];
        if (header === 'Unreal. G/L %') return lot['Unreal. Gain %'];
        // Calculated fields
        if ((header === 'Price' || header === 'Avg Cost') && lot.Quantity) return lot['Cost Basis'] / lot.Quantity;

        // Show Date in the first visible text column (usually Symbol or Account)
        // We will default to showing it in the "Symbol" column if present, else Account.
        if (header === 'Symbol') return `Lot: ${lot.Date}`;
        if (header === 'Account' && !visibleColumns.includes('Symbol')) return `Lot: ${lot.Date}`;

        return null;
    };

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

            return sortConfig.direction === 'asc' ? (valA - valB) : (valB - valA);
        });
    }, [holdings, sortConfig, currency]);

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
    const formatValue = (val: any, header: string) => {
        if (val === null || val === undefined) return '-';
        if (typeof val !== 'number') return val;

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

    const getCellClass = (val: any, header: string) => {
        if (typeof val !== 'number') return '';
        if (['Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Real. G/L', 'Total G/L', 'Total Ret %', 'FX G/L', 'FX G/L %', 'IRR (%)'].includes(header)) {
            return val >= 0 ? 'text-green-600 font-medium' : 'text-red-600 font-medium';
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
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 mt-4 overflow-hidden">
            <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div className="flex items-center gap-4 w-full md:w-auto justify-between md:justify-start">
                    <h2 className="text-lg font-bold text-gray-900 dark:text-white">Holdings</h2>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                        Showing {visibleHoldings.length} of {sortedHoldings.length}
                    </div>
                </div>

                {/* Column Selector */}
                {/* Column Selector */}
                <div className="relative flex flex-wrap gap-2 w-full md:w-auto" ref={columnMenuRef}>
                    <button
                        onClick={() => setIsColumnMenuOpen(!isColumnMenuOpen)}
                        className="flex-1 md:flex-none px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-600 text-center"
                    >
                        Columns
                    </button>

                    <button
                        onClick={() => setShowLots(!showLots)}
                        className={`flex-1 md:flex-none px-3 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 text-center
                            ${showLots
                                ? 'bg-indigo-100 text-indigo-700 border-indigo-200 dark:bg-indigo-900 dark:text-indigo-200 dark:border-indigo-700'
                                : 'text-gray-700 bg-white border-gray-300 hover:bg-gray-50 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-600'
                            }`}
                    >
                        {showLots ? 'Hide Lots' : 'Show Lots'}
                    </button>

                    <button
                        onClick={() => exportToCSV(holdings, 'holdings.csv')}
                        className="flex-1 md:flex-none px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-600 text-center"
                    >
                        Export CSV
                    </button>

                    {isColumnMenuOpen && (
                        <div className="absolute right-0 z-50 mt-2 w-56 origin-top-right bg-white dark:bg-gray-800 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none max-h-96 overflow-y-auto">
                            <div className="py-1">
                                {Object.keys(COLUMN_DEFINITIONS).map(header => (
                                    <label key={header} className="flex items-center px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={visibleColumns.includes(header)}
                                            onChange={() => toggleColumn(header)}
                                            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
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
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-900">
                        <tr>
                            {visibleColumns.map(header => (
                                <th
                                    key={header}
                                    scope="col"
                                    draggable
                                    onDragStart={(e) => handleDragStart(e, header)}
                                    onDragOver={handleDragOver}
                                    onDrop={(e) => handleDrop(e, header)}
                                    className={`px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 select-none whitespace-nowrap ${draggedColumn === header ? 'opacity-50 bg-gray-100 dark:bg-gray-700' : ''}`}
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
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                        {visibleHoldings.map((holding, idx) => (
                            <React.Fragment key={`${holding.Symbol}-${idx}`}>
                                <tr className="hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
                                    {visibleColumns.map(header => {
                                        const val = getValue(holding, header);
                                        return (
                                            <td key={header} className={`px-6 py-4 whitespace-nowrap text-sm text-right ${getCellClass(val, header)} ${header === 'Symbol' || header === 'Account' ? 'text-gray-900 dark:text-white font-medium' : 'text-gray-500 dark:text-gray-300'}`}>
                                                {formatValue(val, header)}
                                            </td>
                                        );
                                    })}
                                </tr>
                                {showLots && holding.lots && holding.lots.length > 0 && (
                                    holding.lots.map((lot, lotIdx) => (
                                        <tr key={`${holding.Symbol}-lot-${lotIdx}`} className="bg-gray-50/50 dark:bg-gray-800/30">
                                            {visibleColumns.map(header => {
                                                const val = getLotValue(lot, header);
                                                return (
                                                    <td key={header} className={`px-6 py-2 whitespace-nowrap text-xs text-right border-t border-gray-100 dark:border-gray-700 ${getCellClass(val, header)} ${header === 'Symbol' ? 'pl-10 text-gray-500 italic' : ''}`}>
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
                    <div key={`mobile-${holding.Symbol}-${idx}`} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-100 dark:border-gray-700 shadow-sm p-4">
                        <div className="flex justify-between items-start mb-2">
                            <div>
                                <h3 className="text-lg font-bold text-gray-900 dark:text-white">{holding.Symbol}</h3>
                                <p className="text-xs text-gray-500 dark:text-gray-400">{holding.Account}</p>
                            </div>
                            <div className="text-right">
                                <div className="text-lg font-bold text-gray-900 dark:text-white">
                                    {formatValue(getValue(holding, "Mkt Val"), "Mkt Val")}
                                </div>
                                <div className={`text-sm ${getCellClass(getValue(holding, "Day Chg %"), "Day Chg %")}`}>
                                    {formatValue(getValue(holding, "Day Chg %"), "Day Chg %")}
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">Qty:</span>
                                <span className="text-gray-900 dark:text-gray-200 font-medium">{formatValue(getValue(holding, "Quantity"), "Quantity")}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">Price:</span>
                                <span className="text-gray-900 dark:text-gray-200 font-medium">{formatValue(getValue(holding, "Price"), "Price")}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">Avg Cost:</span>
                                <span className="text-gray-900 dark:text-gray-200 font-medium">{formatValue(getValue(holding, "Avg Cost"), "Avg Cost")}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-500 dark:text-gray-400">Div Yield:</span>
                                <span className="text-gray-900 dark:text-gray-200 font-medium">{formatValue(getValue(holding, "Yield (Mkt) %"), "Yield (Mkt) %")}</span>
                            </div>
                            <div className="flex justify-between col-span-2 bg-gray-50 dark:bg-gray-700/50 p-2 rounded">
                                <span className="text-gray-500 dark:text-gray-400">Total Return:</span>
                                <span className={`font-medium ${getCellClass(getValue(holding, "Total G/L"), "Total G/L")}`}>
                                    {formatValue(getValue(holding, "Total G/L"), "Total G/L")} ({formatValue(getValue(holding, "Total Ret %"), "Total Ret %")})
                                </span>
                            </div>
                        </div>

                        {/* Mobile Lots View */}
                        {showLots && holding.lots && holding.lots.length > 0 && (
                            <div className="mt-4 pt-3 border-t border-gray-100 dark:border-gray-700">
                                <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider">Tax Lots</h4>
                                <div className="space-y-2">
                                    {holding.lots.map((lot, lotIdx) => (
                                        <div key={`mobile-lot-${lotIdx}`} className="bg-gray-50 dark:bg-gray-700/30 p-2 rounded text-xs">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="font-medium text-gray-700 dark:text-gray-300">
                                                    {formatValue(getLotValue(lot, "Symbol"), "Symbol")}
                                                </span>
                                                <span className={`font-medium ${getCellClass(getLotValue(lot, "Total G/L"), "Total G/L")}`}>
                                                    {formatValue(getLotValue(lot, "Total G/L"), "Total G/L")}
                                                </span>
                                            </div>
                                            <div className="flex justify-between text-gray-500 dark:text-gray-400">
                                                <span>Qty: {formatValue(getLotValue(lot, "Quantity"), "Quantity")}</span>
                                                <span>Cost: {formatValue(getLotValue(lot, "Cost Basis"), "Cost Basis")}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>
            {visibleRows < sortedHoldings.length && (
                <div className="flex justify-center gap-4 p-4 border-t border-gray-200 dark:border-gray-700">
                    <button
                        onClick={handleShowMore}
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm font-medium"
                    >
                        Show More
                    </button>
                    <button
                        onClick={handleShowAll}
                        className="px-4 py-2 bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors text-sm font-medium"
                    >
                        Show All
                    </button>
                </div>
            )}
        </div>
    );
}
