import React, { RefObject } from 'react';
import {
    Search,
    X,
    ListFilter,
    UserCircle,
    Settings2,
    Check,
    ChevronsUpDown,
    Layers,
    Download,
    Table as TableIcon,
    LayoutGrid
} from 'lucide-react';
import { GroupingOption } from './types';
import { GROUPING_LABEL_MAP, COLUMN_GROUPS, DEFAULT_VISIBLE_COLUMNS } from './constants';
import { exportToCSV } from '../../lib/export';
import { Holding } from '../../lib/api';

interface HoldingsToolbarProps {
    totalItemsCount: number;
    aggregatedCount: number;
    groupedCount: number;
    groupBy: GroupingOption;
    searchQuery: string;
    setSearchQuery: (q: string) => void;
    // Group dropdown
    isGroupByMenuOpen: boolean;
    setIsGroupByMenuOpen: (open: boolean) => void;
    groupByMenuRef: RefObject<HTMLDivElement | null>;
    handleSetGroupBy: (option: GroupingOption) => void;
    // Account dropdown
    isAccountMenuOpen: boolean;
    setIsAccountMenuOpen: (open: boolean) => void;
    accountMenuRef: RefObject<HTMLDivElement | null>;
    selectedAccounts: Set<string>;
    setSelectedAccounts: (accs: Set<string>) => void;
    uniqueAccounts: string[];
    toggleAccount: (acc: string) => void;
    // Columns dropdown
    isColumnMenuOpen: boolean;
    setIsColumnMenuOpen: (open: boolean) => void;
    columnMenuRef: RefObject<HTMLDivElement | null>;
    visibleColumns: string[];
    setVisibleColumns: (cols: string[]) => void;
    toggleColumn: (col: string) => void;
    // Lots & cards toggles
    mobileViewMode: 'card' | 'table';
    setMobileViewMode: React.Dispatch<React.SetStateAction<'card' | 'table'>>;
    expandedCards: Set<string>;
    toggleAllCards: () => void;
    expandedLots: Set<string>;
    toggleAllLots: () => void;
    // Export
    holdings: Holding[];
}

export const HoldingsToolbar: React.FC<HoldingsToolbarProps> = ({
    totalItemsCount,
    aggregatedCount,
    groupedCount,
    groupBy,
    searchQuery,
    setSearchQuery,
    isGroupByMenuOpen,
    setIsGroupByMenuOpen,
    groupByMenuRef,
    handleSetGroupBy,
    isAccountMenuOpen,
    setIsAccountMenuOpen,
    accountMenuRef,
    selectedAccounts,
    setSelectedAccounts,
    uniqueAccounts,
    toggleAccount,
    isColumnMenuOpen,
    setIsColumnMenuOpen,
    columnMenuRef,
    visibleColumns,
    setVisibleColumns,
    toggleColumn,
    mobileViewMode,
    setMobileViewMode,
    expandedCards,
    toggleAllCards,
    expandedLots,
    toggleAllLots,
    holdings,
}) => {
    return (
        <div className="flex flex-col gap-4 p-5">
            {/* Header Row: Title, Count & Search */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                    <h2 className="section-label text-sm font-bold text-foreground">Holdings</h2>
                    <span className="text-[10px] font-bold text-slate-600 dark:text-slate-400 bg-muted/50 border border-border/60 px-2 py-0.5 rounded-full tracking-wide">
                        {groupBy
                            ? `${aggregatedCount} items · ${groupedCount} groups`
                            : `${aggregatedCount} / ${totalItemsCount}`
                        }
                    </span>
                </div>

                <div className="relative w-full sm:w-64 lg:w-80">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Search className="h-3.5 w-3.5 text-muted-foreground/50" />
                    </div>
                    <input
                        type="text"
                        aria-label="Search holdings by symbol or name"
                        placeholder="Search symbol..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-9 pr-4 py-1.5 text-sm bg-muted/40 dark:bg-white/[0.04] border border-border/60 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500/50 placeholder-muted-foreground/40 transition-all"
                    />
                    {searchQuery && (
                        <button
                            type="button"
                            aria-label="Clear search query"
                            onClick={() => setSearchQuery("")}
                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-muted-foreground hover:text-foreground"
                        >
                            <X className="h-3.5 w-3.5" />
                        </button>
                    )}
                </div>
            </div>

            {/* Filters & Actions Group */}
            <div className="flex flex-wrap items-center gap-1.5">
                {/* Group By Filter */}
                <div className="relative" ref={groupByMenuRef}>
                    <button
                        onClick={() => setIsGroupByMenuOpen(!isGroupByMenuOpen)}
                        className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors
                        ${groupBy
                                ? 'bg-[#0097b2] text-white border-none'
                                : 'text-foreground bg-secondary border-none hover:bg-accent/10'
                            }`}
                    >
                        <ListFilter className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">{groupBy ? `By ${GROUPING_LABEL_MAP[groupBy]}` : 'Group'}</span>
                    </button>
                    {isGroupByMenuOpen && (
                        <div className="absolute left-0 z-50 mt-1.5 w-48 origin-top-left bg-white dark:bg-zinc-950 rounded-md focus:outline-none shadow-lg border border-border">
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
                                <hr className="my-1 opacity-10" />
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

                {/* Account Filter */}
                <div className="relative" ref={accountMenuRef}>
                    <button
                        onClick={() => setIsAccountMenuOpen(!isAccountMenuOpen)}
                        className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors
                        ${selectedAccounts.size > 0 || isAccountMenuOpen
                                ? 'bg-[#0097b2] text-white'
                                : 'text-foreground bg-secondary hover:bg-accent/10'
                            }`}
                    >
                        <UserCircle className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">Account {selectedAccounts.size > 0 && `(${selectedAccounts.size})`}</span>
                        {selectedAccounts.size > 0 && <span className="sm:hidden text-[10px] absolute -top-1 -right-1 bg-cyan-500 text-white rounded-full w-4 h-4 flex items-center justify-center">{selectedAccounts.size}</span>}
                    </button>
                    {isAccountMenuOpen && (
                        <div className="absolute left-0 z-50 mt-1.5 w-56 origin-top-left bg-white dark:bg-zinc-950 rounded-md border border-border shadow-lg focus:outline-none max-h-96 overflow-y-auto">
                            <div className="p-2">
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
                        className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium rounded-md focus:outline-none transition-colors
                        ${isColumnMenuOpen
                                ? 'bg-primary/10 text-primary border border-primary/30'
                                : 'text-foreground bg-secondary hover:bg-accent/10 border border-transparent'
                            }`}
                    >
                        <Settings2 className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">Columns</span>
                        <span className="hidden sm:inline text-[10px] font-bold bg-primary/15 text-primary px-1.5 rounded-full leading-4">
                            {visibleColumns.length}
                        </span>
                    </button>
                    {isColumnMenuOpen && (
                        <div
                            style={{ backgroundColor: 'var(--menu-solid)' }}
                            className="absolute left-0 sm:left-auto sm:right-0 z-50 mt-1.5 w-72 origin-top-left sm:origin-top-right border border-border rounded-xl shadow-xl overflow-hidden"
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-muted/30">
                                <span className="text-xs font-bold text-foreground">Visible Columns</span>
                                <button
                                    onClick={() => setVisibleColumns(DEFAULT_VISIBLE_COLUMNS)}
                                    className="text-[10px] font-semibold text-primary hover:underline"
                                >
                                    Reset
                                </button>
                            </div>
                            {/* Column groups */}
                            {COLUMN_GROUPS.map(group => (
                                <div key={group.label} className="px-2 py-1.5">
                                    <p className="text-[9px] font-bold uppercase tracking-widest text-muted-foreground/60 px-1.5 mb-1">{group.label}</p>
                                    <div className="grid grid-cols-2 gap-0.5">
                                        {group.cols.map(header => {
                                            const isSelected = visibleColumns.includes(header);
                                            return (
                                                <label
                                                    key={header}
                                                    className={`flex items-center gap-2 px-1.5 py-1 rounded-md cursor-pointer group transition-colors ${
                                                        isSelected
                                                            ? 'bg-primary/10 hover:bg-primary/15'
                                                            : 'hover:bg-muted/60'
                                                    }`}
                                                >
                                                    <span className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-colors ${
                                                        isSelected
                                                            ? 'bg-primary border-primary text-primary-foreground'
                                                            : 'border-border group-hover:border-primary/50'
                                                    }`}>
                                                        {isSelected && (
                                                            <Check className="w-3 h-3" strokeWidth={3} />
                                                        )}
                                                    </span>
                                                    <input
                                                        type="checkbox"
                                                        checked={isSelected}
                                                        onChange={() => toggleColumn(header)}
                                                        className="sr-only"
                                                    />
                                                    <span className={`text-xs truncate ${
                                                        isSelected ? 'text-primary font-semibold' : 'text-foreground'
                                                    }`}>{header}</span>
                                                </label>
                                            );
                                        })}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Expand/Collapse All Cards — card view is mobile-only */}
                {mobileViewMode === 'card' && (
                    <button
                        onClick={toggleAllCards}
                        className={`md:hidden flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center transition-colors
                        ${expandedCards.size > 0
                                ? 'bg-[#0097b2] text-white'
                                : 'text-foreground bg-secondary hover:bg-accent/10'
                            }`}
                        title={expandedCards.size > 0 ? 'Collapse All Details' : 'Expand All Details'}
                    >
                        <ChevronsUpDown className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">Details</span>
                    </button>
                )}

                {/* Toggle All Lots Helper */}
                <button
                    onClick={toggleAllLots}
                    className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center transition-colors
                    ${expandedLots.size > 0
                            ? 'bg-[#0097b2] text-white'
                            : 'text-foreground bg-secondary hover:bg-accent/10'
                        }`}
                    title={expandedLots.size > 0 ? 'Collapse All Tax Lots' : 'Show All Tax Lots'}
                >
                    <Layers className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Lots</span>
                </button>

                {/* Export Button */}
                <button
                    onClick={() => exportToCSV(holdings, 'holdings.csv')}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium text-foreground bg-secondary rounded-md hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center ml-auto"
                    title="Export to CSV"
                >
                    <Download className="w-3.5 h-3.5" />
                </button>

                {/* Mobile View Toggle */}
                <button
                    onClick={() => setMobileViewMode(current => current === 'card' ? 'table' : 'card')}
                    className="md:hidden flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 text-center transition-colors text-foreground bg-secondary hover:bg-accent/10"
                    title={mobileViewMode === 'card' ? 'Switch to Table View' : 'Switch to Card View'}
                >
                    {mobileViewMode === 'card' ? <TableIcon className="w-3.5 h-3.5" /> : <LayoutGrid className="w-3.5 h-3.5" />}
                    <span className="sr-only">{mobileViewMode === 'card' ? 'Table' : 'Card'}</span>
                </button>
            </div>
        </div>
    );
};
