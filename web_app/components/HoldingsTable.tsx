import React, { useState, useMemo, useEffect, useRef } from 'react';
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
    const [visibleRows, setVisibleRows] = useState(10);

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
        if (['Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Real. G/L', 'Total G/L', 'Total Ret %', 'FX G/L', 'FX G/L %'].includes(header)) {
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
            <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center">
                <div className="flex items-center gap-4">
                    <h2 className="text-lg font-bold text-gray-900 dark:text-white">Holdings</h2>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                        Showing {visibleHoldings.length} of {sortedHoldings.length} holdings
                    </div>
                </div>

                {/* Column Selector */}
                <div className="relative" ref={columnMenuRef}>
                    <button
                        onClick={() => setIsColumnMenuOpen(!isColumnMenuOpen)}
                        className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-600"
                    >
                        Columns
                    </button>

                    <button
                        onClick={() => setShowLots(!showLots)}
                        className={`ml-2 px-3 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 
                            ${showLots
                                ? 'bg-indigo-100 text-indigo-700 border-indigo-200 dark:bg-indigo-900 dark:text-indigo-200 dark:border-indigo-700'
                                : 'text-gray-700 bg-white border-gray-300 hover:bg-gray-50 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-600'
                            }`}
                    >
                        {showLots ? 'Hide Lots' : 'Show Lots'}
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

            <div className="overflow-x-auto">
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
