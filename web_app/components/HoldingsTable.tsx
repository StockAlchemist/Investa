"use client";

import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { exportToCSV } from '../lib/export';
import { Holding, Lot, addToWatchlist, removeFromWatchlist, WatchlistItem, updateHoldingTags } from '../lib/api';
import { useQueryClient, useMutation } from '@tanstack/react-query';
import { AreaChart, Area, Line, ResponsiveContainer, YAxis, ReferenceLine } from 'recharts';
import { Search, X, Filter, LayoutGrid, Eye, EyeOff, Layers, Download, Building2, UserCircle, Tag, PenLine, Save, Table as TableIcon, Settings2, ChevronDown, ChevronRight, ListFilter } from 'lucide-react';

import { Skeleton } from './ui/skeleton';
import { useStockModal } from '@/context/StockModalContext';
import WatchlistStar from './WatchlistStar';

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
    "Tags": "Tags",
    "Contribution %": "Contribution %",
};

const DEFAULT_VISIBLE_COLUMNS = [
    "Symbol", "7d Trend", "Quantity", "Price", "Mkt Val", "Day Chg", "Day Chg %", "Unreal. G/L", "Unreal. G/L %"
];

type SortDirection = 'asc' | 'desc';

interface SortConfig {
    key: string;
    direction: SortDirection;
}

type GroupingOption = 'Market' | 'Currency' | 'Sector' | 'Industry' | 'quoteType' | 'Country' | null;

const GROUPING_LABEL_MAP: Record<string, string> = {
    'Market': 'Market',
    'Currency': 'Currency',
    'Sector': 'Sector',
    'Industry': 'Industry',
    'quoteType': 'Investment Type',
    'Country': 'Country',
};

const INVESTMENT_TYPE_MAP: Record<string, string> = {
    'EQUITY': 'Stocks',
    'ETF': 'ETFs',
    'CASH': 'Cash',
    'MUTUALFUND': 'Mutual Funds',
};

const CURRENCY_MAP: Record<string, string> = {
    'USD': 'US Dollar',
    'THB': 'Thai Baht',
    'EUR': 'Euro',
    'GBP': 'British Pound',
    'SGD': 'Singapore Dollar',
    'JPY': 'Japanese Yen',
    'HKD': 'Hong Kong Dollar',
};

const normalizeMarketName = (market: string): string => {
    if (!market) return 'Unknown';
    const m = market.toUpperCase();
    if (m.includes('NASDAQ') || m === 'NMS' || m === 'NGM' || m === 'NCM') return 'NASDAQ';
    if (m === 'NYQ' || m === 'NYSE' || m.includes('NEW YORK')) return 'NYSE';
    if (m === 'ASE' || m === 'AMEX') return 'AMEX';
    if (m === 'PCX' || m === 'ARCA' || m.includes('ARCA')) return 'NYSE Arca';
    return market; // Return original if no match
};

export default function HoldingsTable({ holdings, currency, isLoading = false, showClosed = false, onToggleShowClosed }: HoldingsTableProps) {
    const queryClient = useQueryClient();

    // Watchlist state now handled by WatchlistStar context
    /*
    const addWatchlistMutation = useMutation({
        mutationFn: ({ symbol, note }: { symbol: string, note: string }) => addToWatchlist(symbol, note),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlist'] });
        },
    });

    const removeWatchlistMutation = useMutation({
        mutationFn: (symbol: string) => removeFromWatchlist(symbol),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['watchlist'] });
        },
    });

    const toggleWatchlist = (symbol: string) => {
        const isInWatchlist = watchlist.some(item => item.Symbol === symbol);
        if (isInWatchlist) {
            removeWatchlistMutation.mutate(symbol);
        } else {
            addWatchlistMutation.mutate({ symbol, note: '' });
        }
    };
    */

    const [visibleColumns, setVisibleColumns] = useState<string[]>(DEFAULT_VISIBLE_COLUMNS);
    const [showLots, setShowLots] = useState(false);
    const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'Mkt Val', direction: 'desc' });
    const [isColumnMenuOpen, setIsColumnMenuOpen] = useState(false);
    const [draggedColumn, setDraggedColumn] = useState<string | null>(null);
    const columnMenuRef = useRef<HTMLDivElement>(null);
    const isLoaded = useRef(false);
    const [visibleRows, setVisibleRows] = useState(10);
    const [mobileViewMode, setMobileViewMode] = useState<'card' | 'table'>('table');
    const { openStockDetail } = useStockModal();

    // --- Search & Filter State ---
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedSectors, setSelectedSectors] = useState<Set<string>>(new Set());
    const [selectedAccounts, setSelectedAccounts] = useState<Set<string>>(new Set());
    const [isSectorMenuOpen, setIsSectorMenuOpen] = useState(false);
    const [isAccountMenuOpen, setIsAccountMenuOpen] = useState(false);
    const sectorMenuRef = useRef<HTMLDivElement>(null);
    const accountMenuRef = useRef<HTMLDivElement>(null);
    const groupByMenuRef = useRef<HTMLDivElement>(null);

    // --- Grouping State ---
    const [groupBy, setGroupBy] = useState<GroupingOption>(null);
    const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
    const [isGroupByMenuOpen, setIsGroupByMenuOpen] = useState(false);

    // --- Tags Editing State ---
    const [editingTags, setEditingTags] = useState<{ symbol: string, account: string, currentTags: string } | null>(null);
    const [tagsInput, setTagsInput] = useState("");

    const updateTagsMutation = useMutation({
        mutationFn: ({ account, symbol, tags }: { account: string, symbol: string, tags: string }) => updateHoldingTags(account, symbol, tags),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
            queryClient.invalidateQueries({ queryKey: ['summary'] });
            setEditingTags(null);
        },
    });

    const handleEditTags = (symbol: string, account: string, currentTags: string[]) => {
        setEditingTags({ symbol, account, currentTags: currentTags.join(", ") });
        setTagsInput(currentTags.join(", "));
    };

    const handleSaveTags = () => {
        if (editingTags) {
            updateTagsMutation.mutate({
                account: editingTags.account,
                symbol: editingTags.symbol,
                tags: tagsInput
            });
        }
    };


    const [isInitialized, setIsInitialized] = useState(false);

    // Initialize state from localStorage
    useEffect(() => {
        // Columns
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

        // Sort
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

        // Show Lots
        const savedShowLots = localStorage.getItem('investa_holdings_show_lots');
        if (savedShowLots) {
            setShowLots(savedShowLots === 'true');
        }

        setIsInitialized(true);
    }, []);

    // Persist columns to localStorage on change
    useEffect(() => {
        if (!isInitialized) return;
        localStorage.setItem('investa_holdings_columns', JSON.stringify(visibleColumns));
    }, [visibleColumns, isInitialized]);

    // Persist sort to localStorage on change
    useEffect(() => {
        if (!isInitialized) return;
        localStorage.setItem('investa_holdings_sort', JSON.stringify(sortConfig));
    }, [sortConfig, isInitialized]);

    // Persist showLots to localStorage on change
    useEffect(() => {
        if (!isInitialized) return;
        localStorage.setItem('investa_holdings_show_lots', String(showLots));
    }, [showLots, isInitialized]);

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
            if (groupByMenuRef.current && !groupByMenuRef.current.contains(event.target as Node)) {
                setIsGroupByMenuOpen(false);
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

    const aggregatedHoldings = useMemo(() => {
        if (visibleColumns.includes('Account')) {
            return filteredHoldings;
        }

        const grouped = new Map<string, Holding>();

        // Helper to safely get numeric value
        const getRawVal = (h: Holding, key: string): number => {
            const val = getValue(h, key);
            return typeof val === 'number' ? val : 0;
        };

        filteredHoldings.forEach(h => {
            if (!grouped.has(h.Symbol)) {
                // Clone holding (shallow copy + lots copy)
                grouped.set(h.Symbol, { ...h, lots: h.lots ? [...h.lots] : [] });
            } else {
                const current = grouped.get(h.Symbol)!;

                // Merge lots
                if (h.lots) {
                    current.lots = (current.lots || []).concat(h.lots);
                }

                // Sum numeric fields
                const keysToSum = [
                    "Quantity", "Mkt Val", "Cost Basis", "Day Chg",
                    "Unreal. G/L", "Real. G/L", "Divs", "Fees",
                    "Total G/L", "Total Buy Cost", "Est. Income",
                    "Contribution %", "FX G/L"
                ];

                keysToSum.forEach(header => {
                    const def = COLUMN_DEFINITIONS[header];
                    if (def) {
                        const valA = getRawVal(current, header);
                        const valB = getRawVal(h, header);
                        (current as any)[def] = valA + valB;
                    }
                });

                // Merge Tags
                const currentTags = Array.isArray(getValue(current, "Tags")) ? getValue(current, "Tags") as string[] : [];
                const newTags = Array.isArray(getValue(h, "Tags")) ? getValue(h, "Tags") as string[] : [];
                const mergedTags = Array.from(new Set([...currentTags, ...newTags]));
                (current as any)["Tags"] = mergedTags;
            }
        });

        return Array.from(grouped.values()).map(h => {
            // Recalculate derived fields
            const qty = getRawVal(h, 'Quantity');
            const mktVal = getRawVal(h, 'Mkt Val');
            const costBasis = getRawVal(h, 'Cost Basis');
            const dayChg = getRawVal(h, 'Day Chg');
            const unrealGl = getRawVal(h, 'Unreal. G/L');
            const estIncome = getRawVal(h, 'Est. Income');
            const totalGl = getRawVal(h, 'Total G/L');

            if (qty !== 0) {
                (h as any)['Price'] = mktVal / qty;
                (h as any)['Avg Cost'] = costBasis / qty;
            }

            if (mktVal - dayChg !== 0) {
                (h as any)['Day Change %'] = (dayChg / (mktVal - dayChg)) * 100;
            } else {
                (h as any)['Day Change %'] = 0;
            }

            if (costBasis !== 0) {
                (h as any)['Unreal. Gain %'] = (unrealGl / costBasis) * 100;
                (h as any)['Div. Yield (Cost) %'] = (estIncome / costBasis) * 100;
                (h as any)['Total Return %'] = (totalGl / costBasis) * 100;
            }

            if (mktVal !== 0) {
                (h as any)['Div. Yield (Current) %'] = (estIncome / mktVal) * 100;
            }

            return h;
        });

    }, [filteredHoldings, visibleColumns, getValue]);

    // --- Grouping Logic ---
    const groupedHoldings = useMemo(() => {
        if (!groupBy) return null;

        const groups = new Map<string, {
            key: string;
            holdings: Holding[];
            aggregates: any;
        }>();

        aggregatedHoldings.forEach(h => {
            // Determine group key
            let groupKey = 'Other';
            if (groupBy === 'Market') {
                // Use 'fullExchangeName' or 'exchange' if available, then fallback to 'Market' or 'Unknown'
                const rawExchange = (h as any)['fullExchangeName'] || (h as any)['exchange'] || (h as any)['Market'] || 'Unknown';
                const exchange = normalizeMarketName(rawExchange);
                if (h.Symbol === 'AAPL') console.log(`DEBUG_MARKET: AAPL -> Raw: ${rawExchange}, Norm: ${exchange}`, h);
                groupKey = exchange;
            } else if (groupBy === 'quoteType') {
                const rawType = (h as any)['quoteType'] || 'Other';
                groupKey = INVESTMENT_TYPE_MAP[rawType] || rawType;
            } else if (groupBy === 'Country') {
                // Prioritize 'geography' over 'Country'
                groupKey = (h as any)['geography'] || (h as any)['Country'] || 'Unknown';
            } else if (groupBy === 'Currency') {
                const rawCurrency = (h as any)['Local Currency'] || 'Unknown';
                groupKey = CURRENCY_MAP[rawCurrency] || rawCurrency;
            } else {
                const val = getValue(h, groupBy);
                groupKey = val ? String(val) : 'Other';
            }

            if (!groups.has(groupKey)) {
                groups.set(groupKey, {
                    key: groupKey,
                    holdings: [],
                    aggregates: {
                        'Mkt Val': 0,
                        'Day Chg': 0,
                        'Cost Basis': 0,
                        'Unreal. G/L': 0,
                        'Real. G/L': 0,
                        'Divs': 0,
                        'Fees': 0,
                        'Total G/L': 0,
                    }
                });
            }

            const group = groups.get(groupKey)!;
            group.holdings.push(h);

            // Accumulate Aggregates
            const getNum = (key: string) => {
                const val = getValue(h, key);
                return typeof val === 'number' ? val : 0;
            };

            group.aggregates['Mkt Val'] += getNum('Mkt Val');
            group.aggregates['Day Chg'] += getNum('Day Chg');
            group.aggregates['Cost Basis'] += getNum('Cost Basis');
            group.aggregates['Unreal. G/L'] += getNum('Unreal. G/L');
            group.aggregates['Real. G/L'] += getNum('Real. G/L');
            group.aggregates['Divs'] += getNum('Divs');
            group.aggregates['Fees'] += getNum('Fees');
            group.aggregates['Total G/L'] += getNum('Total G/L');
        });

        // Calculate Percentages for Groups and Sort Items
        return Array.from(groups.values()).map(g => {
            // Calculate Group Percentages
            if (g.aggregates['Mkt Val'] !== 0 && g.aggregates['Mkt Val'] - g.aggregates['Day Chg'] !== 0) {
                g.aggregates['Day Chg %'] = (g.aggregates['Day Chg'] / (g.aggregates['Mkt Val'] - g.aggregates['Day Chg'])) * 100;
            }
            if (g.aggregates['Cost Basis'] !== 0) {
                g.aggregates['Unreal. G/L %'] = (g.aggregates['Unreal. G/L'] / g.aggregates['Cost Basis']) * 100;
                g.aggregates['Total Ret %'] = (g.aggregates['Total G/L'] / g.aggregates['Cost Basis']) * 100;
            }

            // Sort items within group
            g.holdings.sort((a, b) => {
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

            return g;
        }).sort((a, b) => b.aggregates['Mkt Val'] - a.aggregates['Mkt Val']); // Sort groups by Mkt Val largely

    }, [aggregatedHoldings, groupBy, getValue, sortConfig]);

    const toggleGroup = (groupKey: string) => {
        setExpandedGroups(prev => {
            const next = new Set(prev);
            if (next.has(groupKey)) next.delete(groupKey);
            else next.add(groupKey);
            return next;
        });
    };

    // Auto-expand all groups when grouping changes
    useEffect(() => {
        if (groupedHoldings) {
            setExpandedGroups(new Set(groupedHoldings.map(g => g.key)));
        }
    }, [groupBy]); // This might cause re-expansion on data updates, which might be annoying if user collapsed. 
    // Better: Only expand if expandedGroups is empty? 
    // Or just depend on `groupBy` changing. 
    // If we put `groupedHoldings` dependency, it triggers on every price update.
    // The implementation above with [groupBy] roughly works but `groupedHoldings` is derived. 
    // Best to verify if `groupBy` changed. 

    // Actually, we can just use a separate effect for when `groupBy` changes identity.
    // The current effect depends on `groupBy` implicitly because of the component render, 
    // but cleaner to make it explicit or use a ref to track prev groupBy.
    // For simplicity, let's leave it manual or auto-expand on setGroupBy.

    const handleSetGroupBy = (option: GroupingOption) => {
        setGroupBy(option);
        // Expanded groups will be handled by effect or reset
    };

    const sortedHoldings = useMemo(() => {
        return [...aggregatedHoldings].sort((a, b) => {
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
    }, [aggregatedHoldings, sortConfig, getValue]);

    const totalItemsCount = useMemo(() => {
        if (!holdings) return 0;
        if (groupBy) {
            return groupedHoldings?.length || 0; // Count groups? or items? Usually items.
            // If grouped, existing count logic (aggregatedHoldings.length) is still valid for "Total Items" 
            // but maybe we want to show "X Groups". 
            // Let's stick to total holdings count.
        }
        if (visibleColumns.includes('Account')) {
            return holdings.length;
        }
        return new Set(holdings.map(h => h.Symbol)).size;
    }, [holdings, visibleColumns, groupBy, groupedHoldings]);

    if (!isLoading && (!holdings || holdings.length === 0)) {
        return <div className="p-4 text-center text-muted-foreground">No holdings found.</div>;
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
        if (['Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Real. G/L', 'Total G/L', 'Total Ret %', 'FX G/L', 'FX G/L %', 'IRR (%)', 'Contribution %'].includes(header)) {
            if (Math.abs(val) < 0.001) return 'text-muted-foreground';
            return val > 0 ? 'text-emerald-600 dark:text-emerald-400 font-medium' : 'text-red-600 dark:text-red-500 font-medium';
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



    // ... (existing code)

    return (
        <>
            <div className="bg-card border border-border rounded-xl shadow-sm mt-4 overflow-hidden scrollbar-thin scrollbar-thumb-zinc-200 dark:scrollbar-thumb-zinc-800 scrollbar-track-transparent">
                <div className="flex flex-col gap-4 p-4 border-b border-black/5 dark:border-white/5">
                    {/* Header Row: Title, Count & Search */}
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                        <div className="flex items-center gap-3">
                            <h2 className="text-lg font-bold text-foreground">Holdings</h2>
                            <span className="text-xs font-medium text-muted-foreground bg-secondary px-2 py-0.5 rounded-full border border-border">
                                {groupBy
                                    ? `${aggregatedHoldings.length} items in ${groupedHoldings?.length || 0} groups`
                                    : `${aggregatedHoldings.length} / ${totalItemsCount}`
                                }
                            </span>
                        </div>

                        <div className="relative w-full sm:w-64 lg:w-80">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <Search className="h-4 w-4 text-muted-foreground" />
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
                                    <X className="h-4 w-4" />
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Filters & Actions Group - Consolidated and Compact */}
                    <div className="flex flex-wrap items-center gap-1.5">

                        {/* Group By Filter */}
                        <div className="relative" ref={groupByMenuRef}>
                            <button
                                onClick={() => setIsGroupByMenuOpen(!isGroupByMenuOpen)}
                                className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors
                                ${groupBy
                                        ? 'bg-[#0097b2] text-white shadow-sm border-transparent'
                                        : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                                    }`}
                            >
                                <ListFilter className="w-3.5 h-3.5" />
                                <span className="hidden sm:inline">{groupBy ? `By ${GROUPING_LABEL_MAP[groupBy]}` : 'Group'}</span>
                            </button>
                            {isGroupByMenuOpen && (
                                <div className="absolute left-0 z-50 mt-1.5 w-48 origin-top-left bg-white dark:bg-zinc-950 border border-border rounded-md shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none">
                                    <div className="py-1">
                                        <label className="flex items-center px-4 py-2 text-sm text-foreground hover:bg-accent/10 cursor-pointer">
                                            <input
                                                type="radio"
                                                name="grouping"
                                                checked={groupBy === null}
                                                onChange={() => handleSetGroupBy(null)}
                                                className="h-4 w-4 text-cyan-600 focus:ring-cyan-500 border-border rounded-full bg-secondary"
                                            />
                                            <span className="ml-2">Do not group</span>
                                        </label>
                                        <hr className="my-1 border-border" />
                                        {Object.entries(GROUPING_LABEL_MAP).map(([key, label]) => (
                                            <label key={key} className="flex items-center px-4 py-2 text-sm text-foreground hover:bg-accent/10 cursor-pointer">
                                                <input
                                                    type="radio"
                                                    name="grouping"
                                                    checked={groupBy === key}
                                                    onChange={() => handleSetGroupBy(key as GroupingOption)}
                                                    className="h-4 w-4 text-cyan-600 focus:ring-cyan-500 border-border rounded-full bg-secondary"
                                                />
                                                <span className="ml-2">{label}</span>
                                            </label>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Sector Filter */}
                        <div className="relative" ref={sectorMenuRef}>
                            <button
                                onClick={() => setIsSectorMenuOpen(!isSectorMenuOpen)}
                                className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors
                                ${selectedSectors.size > 0 || isSectorMenuOpen
                                        ? 'bg-[#0097b2] text-white shadow-sm border-transparent'
                                        : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                                    }`}
                            >
                                <Building2 className="w-3.5 h-3.5" />
                                <span className="hidden sm:inline">Sector {selectedSectors.size > 0 && `(${selectedSectors.size})`}</span>
                                {selectedSectors.size > 0 && <span className="sm:hidden text-[10px] absolute -top-1 -right-1 bg-cyan-500 text-white rounded-full w-4 h-4 flex items-center justify-center border border-card">{selectedSectors.size}</span>}
                            </button>
                            {isSectorMenuOpen && (
                                <div className="absolute left-0 z-50 mt-1.5 w-56 origin-top-left bg-white dark:bg-zinc-950 border border-border rounded-md shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none max-h-96 overflow-y-auto">
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
                                className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors
                                ${selectedAccounts.size > 0 || isAccountMenuOpen
                                        ? 'bg-[#0097b2] text-white shadow-sm border-transparent'
                                        : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                                    }`}
                            >
                                <UserCircle className="w-3.5 h-3.5" />
                                <span className="hidden sm:inline">Account {selectedAccounts.size > 0 && `(${selectedAccounts.size})`}</span>
                                {selectedAccounts.size > 0 && <span className="sm:hidden text-[10px] absolute -top-1 -right-1 bg-cyan-500 text-white rounded-full w-4 h-4 flex items-center justify-center border border-card">{selectedAccounts.size}</span>}
                            </button>
                            {isAccountMenuOpen && (
                                <div className="absolute left-0 z-50 mt-1.5 w-56 origin-top-left bg-white dark:bg-zinc-950 border border-border rounded-md shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none max-h-96 overflow-y-auto">
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

                        {/* Columns Selector */}
                        <div className="relative" ref={columnMenuRef}>
                            <button
                                onClick={() => setIsColumnMenuOpen(!isColumnMenuOpen)}
                                className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center transition-colors
                                ${isColumnMenuOpen
                                        ? 'bg-[#0097b2] text-white shadow-sm border-transparent'
                                        : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                                    }`}
                            >
                                <Settings2 className="w-3.5 h-3.5" />
                                <span className="hidden sm:inline">Columns</span>
                            </button>
                            {isColumnMenuOpen && (
                                <div className="absolute left-0 sm:left-auto sm:right-0 z-50 mt-1.5 w-56 origin-top-left sm:origin-top-right bg-white dark:bg-zinc-950 border border-border rounded-md shadow-xl ring-1 ring-black ring-opacity-5 focus:outline-none max-h-96 overflow-y-auto">
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

                        {/* Show/Hide Closed Toggle */}
                        <button
                            onClick={() => onToggleShowClosed?.(!showClosed)}
                            className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center transition-colors
                            ${showClosed
                                    ? 'bg-[#0097b2] text-white shadow-sm border-transparent'
                                    : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                                }`}
                            title={showClosed ? 'Hide Closed Positions' : 'Show Closed Positions'}
                        >
                            {showClosed ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                            <span className="hidden sm:inline">Closed</span>
                        </button>

                        {/* Show/Hide Lots Toggle */}
                        <button
                            onClick={() => setShowLots(!showLots)}
                            className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center transition-colors
                            ${showLots
                                    ? 'bg-[#0097b2] text-white shadow-sm border-transparent'
                                    : 'text-foreground bg-secondary border-border hover:bg-accent/10'
                                }`}
                            title={showLots ? 'Hide Tax Lots' : 'Show Tax Lots'}
                        >
                            <Layers className="w-3.5 h-3.5" />
                            <span className="hidden sm:inline">Lots</span>
                        </button>

                        {/* Export Button */}
                        <button
                            onClick={() => exportToCSV(holdings, 'holdings.csv')}
                            className="flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium text-foreground bg-secondary border border-border rounded-md shadow-sm hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center ml-auto"
                            title="Export to CSV"
                        >
                            <Download className="w-3.5 h-3.5" />
                        </button>

                        {/* Mobile View Toggle */}
                        <button
                            onClick={() => setMobileViewMode(current => current === 'card' ? 'table' : 'card')}
                            className="md:hidden flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center transition-colors text-foreground bg-secondary border-border hover:bg-accent/10"
                            title={mobileViewMode === 'card' ? 'Switch to Table View' : 'Switch to Card View'}
                        >
                            {mobileViewMode === 'card' ? <TableIcon className="w-3.5 h-3.5" /> : <LayoutGrid className="w-3.5 h-3.5" />}
                            <span className="sr-only">{mobileViewMode === 'card' ? 'Table' : 'Card'}</span>
                        </button>
                    </div>
                </div>

                {/* Desktop Table View (also visible on mobile if toggled) */}
                <div className={`${mobileViewMode === 'table' ? 'block' : 'hidden'} md:block overflow-x-auto`}>
                    <table className="min-w-full divide-y divide-black/5 dark:divide-white/5">
                        <thead className="bg-secondary/50 font-semibold border-b border-border">
                            <tr>
                                {visibleColumns.map(header => {
                                    const isLeftAligned = ['Symbol', 'Account', 'Sector', 'Industry', 'Tags'].includes(header);
                                    return (
                                        <th
                                            key={header}
                                            scope="col"
                                            draggable
                                            onDragStart={(e) => handleDragStart(e, header)}
                                            onDragOver={handleDragOver}
                                            onDrop={(e) => handleDrop(e, header)}
                                            className={`px-6 py-3 text-xs font-semibold text-muted-foreground transition-colors select-none whitespace-nowrap group hover:bg-accent/10 cursor-pointer ${draggedColumn === header ? 'opacity-50 bg-secondary' : ''} ${isLeftAligned ? 'text-left' : 'text-right'}`}
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
                        <tbody className="divide-y divide-border/50">
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
                                                        <span className="text-xs text-muted-foreground bg-secondary border border-border px-1.5 py-0.5 rounded-full">
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
                                            <tr key={`${holding.Symbol}-${idx}`} className="hover:bg-accent/5 transition-colors">
                                                {visibleColumns.map(header => {
                                                    const val = getValue(holding, header);
                                                    const isNumeric = ['Quantity', 'Price', 'Mkt Val', 'Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Cost Basis', 'Avg Cost'].some(k => header.includes(k) || header === k);
                                                    const isLeftAligned = ['Symbol', 'Account', 'Sector', 'Industry', 'Tags'].includes(header);

                                                    return (
                                                        <td key={header} className={`px-6 py-3 whitespace-nowrap text-sm ${isLeftAligned ? 'text-left' : 'text-right'} ${isNumeric ? 'tabular-nums' : ''} ${getCellClass(val, header) || (header === 'Symbol' || header === 'Account' ? 'text-foreground font-medium' : 'text-muted-foreground')}`}>
                                                            {header === '7d Trend' ? (
                                                                <div className="h-10 w-28 ml-auto">
                                                                    {val && Array.isArray(val) && val.length > 1 ? (
                                                                        <ResponsiveContainer width="100%" height="100%" minWidth={50} minHeight={30}>
                                                                            <AreaChart data={val.map((v, i) => ({ value: v, index: i }))}>
                                                                                <defs>
                                                                                    {(() => {
                                                                                        const baseline = val[0];
                                                                                        const min = Math.min(...val);
                                                                                        const max = Math.max(...val);
                                                                                        const range = max - min;
                                                                                        const off = range <= 0 ? 0 : (max - baseline) / range;

                                                                                        return (
                                                                                            <>
                                                                                                <linearGradient id={`splitFill-${holding.Symbol}`} x1="0" y1="0" x2="0" y2="1">
                                                                                                    <stop offset={off} stopColor="#10b981" stopOpacity={0.15} />
                                                                                                    <stop offset={off} stopColor="#ef4444" stopOpacity={0.15} />
                                                                                                </linearGradient>
                                                                                                <linearGradient id={`splitStroke-${holding.Symbol}`} x1="0" y1="0" x2="0" y2="1">
                                                                                                    <stop offset={off} stopColor="#10b981" stopOpacity={1} />
                                                                                                    <stop offset={off} stopColor="#ef4444" stopOpacity={1} />
                                                                                                </linearGradient>
                                                                                            </>
                                                                                        );
                                                                                    })()}
                                                                                </defs>
                                                                                <YAxis hide domain={['dataMin', 'dataMax']} />
                                                                                <ReferenceLine y={val[0]} stroke="#71717a" strokeDasharray="2 2" strokeOpacity={0.3} />
                                                                                <Area
                                                                                    type="monotone"
                                                                                    dataKey="value"
                                                                                    baseValue={val[0]}
                                                                                    stroke={`url(#splitStroke-${holding.Symbol})`}
                                                                                    fill={`url(#splitFill-${holding.Symbol})`}
                                                                                    strokeWidth={1.5}
                                                                                    isAnimationActive={false}
                                                                                    dot={(props: any) => {
                                                                                        const { cx, cy, index } = props;
                                                                                        if (index === val.length - 1) {
                                                                                            const color = val[val.length - 1] >= val[0] ? "#10b981" : "#ef4444";
                                                                                            return (
                                                                                                <circle key="dot" cx={cx} cy={cy} r={2} fill={color} stroke="none" />
                                                                                            );
                                                                                        }
                                                                                        return null;
                                                                                    }}
                                                                                />
                                                                            </AreaChart>
                                                                        </ResponsiveContainer>
                                                                    ) : (
                                                                        <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground/50">
                                                                            no data
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            ) : header === 'Symbol' ? (
                                                                <div className="flex items-center gap-3">
                                                                    {/* Watchlist Star */}
                                                                    <WatchlistStar
                                                                        symbol={holding.Symbol}
                                                                        className="text-muted-foreground hover:text-amber-400"
                                                                    />

                                                                    {/* Logo */}


                                                                    <div className="flex flex-col">
                                                                        <span
                                                                            className="font-bold text-foreground hover:text-cyan-500 cursor-pointer transition-colors"
                                                                            onClick={() => openStockDetail(holding.Symbol)}
                                                                        >
                                                                            {holding.Symbol}
                                                                        </span>
                                                                        {showLots && holding.lots && holding.lots.length > 0 && (
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
                                                                                className="w-32 h-7 text-xs bg-background border border-cyan-500/50 rounded px-2 focus:outline-none focus:ring-1 focus:ring-cyan-500 shadow-sm"
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
                                                                                    {val.map((tag: string, i: number) => (
                                                                                        <span key={i} className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-cyan-500/10 text-cyan-700 dark:text-cyan-400 border border-cyan-500/20">
                                                                                            {tag}
                                                                                        </span>
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
                                                            ) : (
                                                                formatValue(val, header)
                                                            )}
                                                        </td>
                                                    );
                                                })}
                                            </tr>
                                        ))}
                                    </React.Fragment>
                                ))
                            ) : (
                                // Existing flat list rendering
                                visibleHoldings.map((holding, idx) => (
                                    <React.Fragment key={`${holding.Symbol}-${idx}`}>
                                        <tr className="hover:bg-accent/5 transition-colors">
                                            {visibleColumns.map(header => {
                                                const val = getValue(holding, header);
                                                // Use tabular-nums for numeric columns to ensure digit alignment
                                                const isNumeric = ['Quantity', 'Price', 'Mkt Val', 'Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Cost Basis', 'Avg Cost'].some(k => header.includes(k) || header === k);
                                                const isLeftAligned = ['Symbol', 'Account', 'Sector', 'Industry', 'Tags'].includes(header);

                                                return (
                                                    <td key={header} className={`px-6 py-3 whitespace-nowrap text-sm ${isLeftAligned ? 'text-left' : 'text-right'} ${isNumeric ? 'tabular-nums' : ''} ${getCellClass(val, header) || (header === 'Symbol' || header === 'Account' ? 'text-foreground font-medium' : 'text-muted-foreground')}`}>
                                                        {header === '7d Trend' ? (
                                                            <div className="h-10 w-28 ml-auto">
                                                                {val && Array.isArray(val) && val.length > 1 ? (
                                                                    <ResponsiveContainer width="100%" height="100%" minWidth={50} minHeight={30}>
                                                                        <AreaChart data={val.map((v, i) => ({ value: v, index: i }))}>
                                                                            <defs>
                                                                                {(() => {
                                                                                    const baseline = val[0];
                                                                                    const min = Math.min(...val);
                                                                                    const max = Math.max(...val);
                                                                                    const range = max - min;
                                                                                    const off = range <= 0 ? 0 : (max - baseline) / range;

                                                                                    return (
                                                                                        <>
                                                                                            <linearGradient id={`splitFill-${holding.Symbol}`} x1="0" y1="0" x2="0" y2="1">
                                                                                                <stop offset={off} stopColor="#10b981" stopOpacity={0.15} />
                                                                                                <stop offset={off} stopColor="#ef4444" stopOpacity={0.15} />
                                                                                            </linearGradient>
                                                                                            <linearGradient id={`splitStroke-${holding.Symbol}`} x1="0" y1="0" x2="0" y2="1">
                                                                                                <stop offset={off} stopColor="#10b981" stopOpacity={1} />
                                                                                                <stop offset={off} stopColor="#ef4444" stopOpacity={1} />
                                                                                            </linearGradient>
                                                                                        </>
                                                                                    );
                                                                                })()}
                                                                            </defs>
                                                                            <YAxis hide domain={['dataMin', 'dataMax']} />
                                                                            <ReferenceLine y={val[0]} stroke="#71717a" strokeDasharray="2 2" strokeOpacity={0.3} />
                                                                            <Area
                                                                                type="monotone"
                                                                                dataKey="value"
                                                                                baseValue={val[0]}
                                                                                stroke={`url(#splitStroke-${holding.Symbol})`}
                                                                                fill={`url(#splitFill-${holding.Symbol})`}
                                                                                strokeWidth={1.5}
                                                                                isAnimationActive={false}
                                                                                dot={(props: any) => {
                                                                                    const { cx, cy, index } = props;
                                                                                    if (index === val.length - 1) {
                                                                                        // Calculate color for dot based on value vs baseline
                                                                                        const color = val[val.length - 1] >= val[0] ? "#10b981" : "#ef4444";
                                                                                        return (
                                                                                            <circle key="dot" cx={cx} cy={cy} r={2} fill={color} stroke="none" />
                                                                                        );
                                                                                    }
                                                                                    return <React.Fragment key={index} />;
                                                                                }}
                                                                            />
                                                                        </AreaChart>
                                                                    </ResponsiveContainer>
                                                                ) : (
                                                                    <div className="h-full w-full flex items-center justify-center text-[10px] text-muted-foreground/30">
                                                                        no data
                                                                    </div>
                                                                )}
                                                            </div>
                                                        ) : header === 'Symbol' ? (
                                                            <div className="flex items-center justify-start gap-3">
                                                                <WatchlistStar symbol={val as string} size="md" />
                                                                <button
                                                                    onClick={() => openStockDetail(val as string, currency)}
                                                                    className="font-semibold text-foreground hover:text-cyan-500 transition-colors cursor-pointer"
                                                                >
                                                                    {formatValue(val, header)}
                                                                </button>
                                                            </div>
                                                        ) : header === 'Tags' ? (
                                                            <div className="flex items-center justify-end gap-2 group/tags min-w-[120px]">
                                                                <div className="flex flex-wrap gap-1 justify-end">
                                                                    {Array.isArray(val) && val.length > 0 ? (
                                                                        val.map((tag: string, i: number) => (
                                                                            <span key={i} className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300">
                                                                                {tag}
                                                                            </span>
                                                                        ))
                                                                    ) : (
                                                                        <span className="text-muted-foreground italic text-xs opacity-0 group-hover/tags:opacity-50 transition-opacity">Add tag</span>
                                                                    )}
                                                                </div>
                                                                <button
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        const tags = Array.isArray(val) ? val : [];
                                                                        const acc = formatValue(getValue(holding, "Account"), "Account") as string;
                                                                        handleEditTags(holding.Symbol, acc, tags);
                                                                    }}
                                                                    className="text-muted-foreground hover:text-cyan-500 opacity-0 group-hover/tags:opacity-100 transition-opacity p-1"
                                                                    title="Edit Tags"
                                                                >
                                                                    <PenLine className="h-3 w-3" />
                                                                </button>
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
                                                        const isNumeric = ['Quantity', 'Price', 'Mkt Val', 'Day Chg', 'Day Chg %', 'Unreal. G/L', 'Unreal. G/L %', 'Cost Basis', 'Avg Cost'].some(k => header.includes(k) || header === k);

                                                        return (
                                                            <td key={header} className={`px-6 py-2 whitespace-nowrap text-xs text-right border-t border-dashed border-border/40 ${isNumeric ? 'tabular-nums' : ''} ${getCellClass(val, header) || (header === 'Symbol' ? 'pl-10 text-muted-foreground italic flex items-center justify-end gap-2' : 'text-muted-foreground')}`}>
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

                {/* Mobile Card View */}
                <div className={`${mobileViewMode === 'card' ? 'block' : 'hidden'} md:hidden space-y-4 p-4`}>
                    {visibleHoldings.map((holding, idx) => (
                        <div
                            key={`mobile-${holding.Symbol}-${idx}`}
                            className="bg-card rounded-lg border border-border shadow-sm p-4 cursor-pointer hover:border-cyan-500/50 transition-all active:scale-[0.98]"
                            onClick={() => openStockDetail(holding.Symbol, currency)}
                        >
                            <div className="flex justify-between items-start mb-2">
                                <div className="flex items-center gap-3">
                                    <WatchlistStar symbol={holding.Symbol} size="md" />
                                    <div>
                                        <h3 className="text-lg font-bold text-foreground">{holding.Symbol}</h3>
                                        <p className="text-xs text-muted-foreground">{holding.Account}</p>
                                    </div>
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
                                <div className="flex justify-between col-span-2 bg-secondary border border-border p-2 rounded">
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
                {!groupBy && visibleRows < sortedHoldings.length && (
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

            {/* Edit Tags Modal */}
            {
                editingTags && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
                        <div className="bg-card border border-border rounded-lg shadow-xl w-full max-w-sm p-4 space-y-4">
                            <div className="flex justify-between items-center">
                                <h3 className="text-lg font-semibold flex items-center gap-2">
                                    <Tag className="w-4 h-4" />
                                    Edit Tags
                                </h3>
                                <button onClick={() => setEditingTags(null)} className="text-muted-foreground hover:text-foreground">
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                            <div className="space-y-2">
                                <div className="text-sm text-muted-foreground">
                                    Tags for <span className="font-medium text-foreground">{editingTags.symbol}</span> ({editingTags.account})
                                </div>
                                <input
                                    type="text"
                                    value={tagsInput}
                                    onChange={(e) => setTagsInput(e.target.value)}
                                    placeholder="Enter tags separated by commas..."
                                    className="w-full px-3 py-2 bg-secondary border border-border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                                    autoFocus
                                />
                                <p className="text-xs text-muted-foreground">
                                    Separate multiple tags with commas (e.g. "Long Term, High Risk").
                                </p>
                            </div>
                            <div className="flex justify-end gap-2">
                                <button
                                    onClick={() => setEditingTags(null)}
                                    className="px-3 py-1.5 text-sm bg-secondary text-foreground rounded hover:bg-accent/50 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleSaveTags}
                                    disabled={updateTagsMutation.isPending}
                                    className="px-3 py-1.5 text-sm bg-[#0097b2] text-white rounded hover:bg-[#0086a0] transition-colors flex items-center gap-2 disabled:opacity-50"
                                >
                                    {updateTagsMutation.isPending ? "Saving..." : <><Save className="w-3 h-3" /> Save</>}
                                </button>
                            </div>
                        </div>
                    </div>
                )
            }
        </>
    );
}
