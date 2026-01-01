"use client";

import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { exportToCSV } from '../lib/export';
import { Holding, Lot, addToWatchlist } from '../lib/api';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';
import { Star } from 'lucide-react';

import { Skeleton } from './ui/skeleton';

interface HoldingsTableProps {
    holdings: Holding[];
    currency: string;
    isLoading?: boolean;
    showClosed?: boolean;
    onToggleShowClosed?: (val: boolean) => void;
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
    "FX G/L %": "FX Gain/Loss %",
    "Est. Income": "Est. Ann. Income",
    "7d Trend": "sparkline_7d",
};

const DEFAULT_VISIBLE_COLUMNS = [
    "Symbol", "7d Trend", "Quantity", "Price", "Mkt Val", "Day Chg", "Day Chg %", "Unreal. G/L", "Unreal. G/L %"
];

type SortDirection = 'asc' | 'desc';

interface SortConfig {
    key: string;
    direction: SortDirection;
}

export default function HoldingsTable({ holdings, currency, isLoading = false, showClosed = false, onToggleShowClosed }: HoldingsTableProps) {
    const [visibleColumns, setVisibleColumns] = useState<string[]>(DEFAULT_VISIBLE_COLUMNS);
    const [showLots, setShowLots] = useState(false);
    const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'Mkt Val', direction: 'desc' });
    const [isColumnMenuOpen, setIsColumnMenuOpen] = useState(false);
    const [draggedColumn, setDraggedColumn] = useState<string | null>(null);
    const columnMenuRef = useRef<HTMLDivElement>(null);
    const isLoaded = useRef(false);
    const [visibleRows, setVisibleRows] = useState(10);

    // --- Search & Filter State ---
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedSectors, setSelectedSectors] = useState<Set<string>>(new Set());
    const [selectedAccounts, setSelectedAccounts] = useState<Set<string>>(new Set());
    const [isSectorMenuOpen, setIsSectorMenuOpen] = useState(false);
    const [isAccountMenuOpen, setIsAccountMenuOpen] = useState(false);
    const sectorMenuRef = useRef<HTMLDivElement>(null);
    const accountMenuRef = useRef<HTMLDivElement>(null);

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

    // Close menus when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (columnMenuRef.current && !columnMenuRef.current.contains(event.target as Node)) {
                setIsColumnMenuOpen(false);
            }
            if (sectorMenuRef.current && !sectorMenuRef.current.contains(event.target as Node)) {
                setIsSectorMenuOpen(false);
            }
            if (accountMenuRef.current && !accountMenuRef.current.contains(event.target as Node)) {
                setIsAccountMenuOpen(false);
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

    // --- Filter Logic ---
    const uniqueSectors = useMemo(() => Array.from(new Set(holdings.map(h => h.Sector).filter(Boolean) as string[])).sort(), [holdings]);
    const uniqueAccounts = useMemo(() => Array.from(new Set(holdings.map(h => h.Account).filter(Boolean) as string[])).sort(), [holdings]);

    const filteredHoldings = useMemo(() => {
        if (!holdings) return [];
        return holdings.filter(h => {
            const matchesSearch = h.Symbol.toLowerCase().includes(searchQuery.toLowerCase());
            const matchesSector = selectedSectors.size === 0 || (h.Sector && selectedSectors.has(h.Sector));
            const matchesAccount = selectedAccounts.size === 0 || (h.Account && selectedAccounts.has(h.Account));
            return matchesSearch && matchesSector && matchesAccount;
        });
    }, [holdings, searchQuery, selectedSectors, selectedAccounts]);

    const sortedHoldings = useMemo(() => {
        return [...filteredHoldings].sort((a, b) => { // Use filteredHoldings here
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
    }, [filteredHoldings, sortConfig, getValue]);

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

    const toggleSector = (sector: string) => {
        setSelectedSectors(prev => {
            const newSet = new Set(prev);
            if (newSet.has(sector)) newSet.delete(sector);
            else newSet.add(sector);
            return newSet;
        });
    };

    const toggleAccount = (account: string) => {
        setSelectedAccounts(prev => {
            const newSet = new Set(prev);
            if (newSet.has(account)) newSet.delete(account);
            else newSet.add(account);
            return newSet;
        });
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
            <div className="flex flex-col gap-4 p-4 border-b border-black/5 dark:border-white/5">
                {/* Header Row: Title & Count */}
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <h2 className="text-lg font-bold text-foreground">Holdings</h2>
                        <span className="text-xs font-medium text-muted-foreground bg-secondary px-2 py-1 rounded-full border border-border">
                            {filteredHoldings.length} / {holdings.length}
                        </span>
                    </div>

                    {/* Desktop Search (hidden on mobile) */}
                    <div className="hidden lg:relative lg:block lg:w-72">
                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <svg className="h-4 w-4 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                        </div>
                        <input
                            type="text"
                            placeholder="Search symbol..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-9 pr-4 py-1.5 text-sm bg-secondary border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-cyan-500 placeholder-muted-foreground transition-all"
                        />
                        {searchQuery && (
                            <button
                                onClick={() => setSearchQuery("")}
                                className="absolute inset-y-0 right-0 pr-3 flex items-center text-muted-foreground hover:text-foreground"
                            >
                                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        )}
                    </div>
                </div>

                {/* Mobile Search (hidden on desktop) */}
                <div className="relative w-full lg:hidden">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <svg className="h-4 w-4 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                    </div>
                    <input
                        type="text"
                        placeholder="Search symbol..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-9 pr-4 py-1.5 text-sm bg-secondary border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-cyan-500 placeholder-muted-foreground transition-all"
                    />
                    {searchQuery && (
                        <button
                            onClick={() => setSearchQuery("")}
                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-muted-foreground hover:text-foreground"
                        >
                            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    )}
                </div>

                {/* Filters & Actions Group - Consolidated into one row */}
                <div className="flex flex-wrap items-center gap-2">
                    {/* Sector Filter */}
                    <div className="relative" ref={sectorMenuRef}>
                        <button
                            onClick={() => setIsSectorMenuOpen(!isSectorMenuOpen)}
                            className={`px-3 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 backdrop-blur-md transition-colors
                                ${selectedSectors.size > 0 || isSectorMenuOpen
                                    ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/50'
                                    : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                                }`}
                        >
                            Sector {selectedSectors.size > 0 && `(${selectedSectors.size})`}
                        </button>
                        {isSectorMenuOpen && (
                            <div className="absolute left-0 z-50 mt-2 w-56 origin-top-left bg-card border border-border rounded-md shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none max-h-96 overflow-y-auto backdrop-blur-xl">
                                <div className="p-2 border-b border-border">
                                    <button onClick={() => setSelectedSectors(new Set())} className="text-xs text-cyan-500 hover:text-cyan-600 font-medium w-full text-left px-2">
                                        Clear Filter
                                    </button>
                                </div>
                                <div className="py-1">
                                    {uniqueSectors.map(sector => (
                                        <label key={sector} className="flex items-center px-4 py-2 text-sm text-foreground hover:bg-accent/10 cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={selectedSectors.has(sector)}
                                                onChange={() => toggleSector(sector)}
                                                className="h-4 w-4 text-cyan-600 focus:ring-cyan-500 border-border rounded bg-secondary"
                                            />
                                            <span className="ml-2 truncate">{sector}</span>
                                        </label>
                                    ))}
                                    {uniqueSectors.length === 0 && <div className="px-4 py-2 text-sm text-muted-foreground">No sectors found</div>}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Account Filter */}
                    <div className="relative" ref={accountMenuRef}>
                        <button
                            onClick={() => setIsAccountMenuOpen(!isAccountMenuOpen)}
                            className={`px-3 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 backdrop-blur-md transition-colors
                                ${selectedAccounts.size > 0 || isAccountMenuOpen
                                    ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/50'
                                    : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                                }`}
                        >
                            Account {selectedAccounts.size > 0 && `(${selectedAccounts.size})`}
                        </button>
                        {isAccountMenuOpen && (
                            <div className="absolute left-0 z-50 mt-2 w-56 origin-top-left bg-card border border-border rounded-md shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none max-h-96 overflow-y-auto backdrop-blur-xl">
                                <div className="p-2 border-b border-border">
                                    <button onClick={() => setSelectedAccounts(new Set())} className="text-xs text-cyan-500 hover:text-cyan-600 font-medium w-full text-left px-2">
                                        Clear Filter
                                    </button>
                                </div>
                                <div className="py-1">
                                    {uniqueAccounts.map(account => (
                                        <label key={account} className="flex items-center px-4 py-2 text-sm text-foreground hover:bg-accent/10 cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={selectedAccounts.has(account)}
                                                onChange={() => toggleAccount(account)}
                                                className="h-4 w-4 text-cyan-600 focus:ring-cyan-500 border-border rounded bg-secondary"
                                            />
                                            <span className="ml-2 truncate">{account}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Actions consolidated here */}
                    <div className="relative" ref={columnMenuRef}>
                        <button
                            onClick={() => setIsColumnMenuOpen(!isColumnMenuOpen)}
                            className="px-3 py-1.5 text-sm font-medium text-foreground bg-secondary border border-border rounded-md shadow-sm hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center backdrop-blur-md"
                        >
                            Columns
                        </button>
                        {isColumnMenuOpen && (
                            <div className="absolute left-0 sm:left-auto sm:right-0 z-50 mt-2 w-56 origin-top-left sm:origin-top-right bg-card border border-border rounded-md shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none max-h-96 overflow-y-auto backdrop-blur-xl">
                                <div className="py-1">
                                    {Object.keys(COLUMN_DEFINITIONS).map(header => (
                                        <label key={header} className="flex items-center px-4 py-2 text-sm text-foreground hover:bg-accent/10 cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={visibleColumns.includes(header)}
                                                onChange={() => toggleColumn(header)}
                                                className="h-4 w-4 text-cyan-600 focus:ring-cyan-500 border-border rounded bg-secondary"
                                            />
                                            <span className="ml-2">{header}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    <button
                        onClick={() => onToggleShowClosed?.(!showClosed)}
                        className={`px-3 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center backdrop-blur-md transition-colors
                            ${showClosed
                                ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/50'
                                : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                            }`}
                    >
                        {showClosed ? 'Hide Closed' : 'Show Closed'}
                    </button>

                    <button
                        onClick={() => setShowLots(!showLots)}
                        className={`px-3 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center backdrop-blur-md transition-colors
                            ${showLots
                                ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/50'
                                : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                            }`}
                    >
                        {showLots ? 'Hide Lots' : 'Show Lots'}
                    </button>

                    <button
                        onClick={() => exportToCSV(holdings, 'holdings.csv')}
                        className="px-3 py-1.5 text-sm font-medium text-foreground bg-secondary border border-border rounded-md shadow-sm hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center backdrop-blur-md"
                    >
                        Export
                    </button>
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
                        ) : visibleHoldings.map((holding, idx) => (
                            <React.Fragment key={`${holding.Symbol}-${idx}`}>
                                <tr className="hover:bg-accent/5 transition-colors">
                                    {visibleColumns.map(header => {
                                        const val = getValue(holding, header);
                                        return (
                                            <td key={header} className={`px-6 py-4 whitespace-nowrap text-sm text-right ${getCellClass(val, header) || (header === 'Symbol' || header === 'Account' ? 'text-foreground font-medium' : 'text-muted-foreground')}`}>
                                                {header === '7d Trend' ? (
                                                    <div className="h-8 w-24 ml-auto">
                                                        {val && Array.isArray(val) && val.length > 1 ? (
                                                            <ResponsiveContainer width="100%" height="100%">
                                                                <LineChart data={val.map(v => ({ value: v }))}>
                                                                    <YAxis hide domain={['dataMin', 'dataMax']} />
                                                                    <Line
                                                                        type="monotone"
                                                                        dataKey="value"
                                                                        stroke={val[val.length - 1] >= val[0] ? "#10b981" : "#f43f5e"}
                                                                        strokeWidth={2}
                                                                        dot={false}
                                                                        isAnimationActive={false}
                                                                    />
                                                                </LineChart>
                                                            </ResponsiveContainer>
                                                        ) : (
                                                            <div className="h-full w-full flex items-center justify-center text-[10px] text-muted-foreground/30">
                                                                no data
                                                            </div>
                                                        )}
                                                    </div>
                                                ) : header === 'Symbol' ? (
                                                    <div className="flex items-center justify-end gap-2">
                                                        <button
                                                            onClick={async (e) => {
                                                                e.stopPropagation();
                                                                try {
                                                                    await addToWatchlist(val as string);
                                                                    // We could add a visual feedback here if we had state
                                                                } catch (err) {
                                                                    console.error("Failed to add to watchlist", err);
                                                                }
                                                            }}
                                                            className="text-muted-foreground/30 hover:text-yellow-500 transition-colors"
                                                            title="Add to Watchlist"
                                                        >
                                                            <Star className="h-3 w-3" />
                                                        </button>
                                                        <span className="font-medium text-foreground">{formatValue(val, header)}</span>
                                                    </div>
                                                ) : (
                                                    formatValue(val, header)
                                                )}
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
