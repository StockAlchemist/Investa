import React from 'react';
import {
    Search,
    X,
    Filter,
    Plus,
    FileText,
    Download,
    Table as TableIcon,
    LayoutGrid
} from 'lucide-react';
import { DatePreset, DATE_PRESETS } from './types';
import { exportToCSV } from '../../lib/export';
import { Transaction } from '../../lib/api';

interface TransactionsToolbarProps {
    symbolFilter: string;
    setSymbolFilter: (s: string) => void;
    accountFilter: string;
    setAccountFilter: (a: string) => void;
    uniqueAccounts: string[];
    filterTypes: string[];
    toggleFilterType: (t: string) => void;
    availableTypes: string[];
    datePreset: DatePreset;
    setDatePreset: (p: DatePreset) => void;
    customFrom: string;
    setCustomFrom: (f: string) => void;
    customTo: string;
    setCustomTo: (t: string) => void;
    resetFilters: () => void;
    hasActiveFilters: boolean;
    viewMode: 'table' | 'cards';
    setViewMode: (m: 'table' | 'cards') => void;
    onOpenAddModal: () => void;
    onOpenImportModal: () => void;
    filteredTransactions: Transaction[];
}

export const TransactionsToolbar: React.FC<TransactionsToolbarProps> = ({
    symbolFilter,
    setSymbolFilter,
    accountFilter,
    setAccountFilter,
    uniqueAccounts,
    filterTypes,
    toggleFilterType,
    availableTypes,
    datePreset,
    setDatePreset,
    customFrom,
    setCustomFrom,
    customTo,
    setCustomTo,
    resetFilters,
    hasActiveFilters,
    viewMode,
    setViewMode,
    onOpenAddModal,
    onOpenImportModal,
    filteredTransactions,
}) => {
    return (
        <div className="flex flex-col gap-4">
            {/* Top Action Row */}
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                    <button
                        onClick={onOpenAddModal}
                        className="flex items-center gap-1.5 px-3.5 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white rounded-xl text-xs font-semibold shadow-sm transition-all cursor-pointer"
                    >
                        <Plus className="w-3.5 h-3.5" />
                        Add Transaction
                    </button>
                    <button
                        onClick={onOpenImportModal}
                        className="flex items-center gap-1.5 px-3 py-2 bg-secondary hover:bg-accent/10 text-foreground border border-border/60 rounded-xl text-xs font-medium transition-all cursor-pointer"
                        title="Import PDF or CSV statements from brokers"
                    >
                        <FileText className="w-3.5 h-3.5 text-cyan-500" />
                        Import Statement
                    </button>
                    <button
                        onClick={() => exportToCSV(filteredTransactions, 'transactions.csv')}
                        className="flex items-center gap-1.5 px-3 py-2 bg-secondary hover:bg-accent/10 text-foreground border border-border/60 rounded-xl text-xs font-medium transition-all cursor-pointer"
                        title="Export filtered transactions to CSV"
                    >
                        <Download className="w-3.5 h-3.5" />
                        Export
                    </button>
                </div>

                <div className="flex items-center gap-1 bg-secondary p-1 rounded-xl border border-border/60">
                    <button
                        onClick={() => setViewMode('table')}
                        className={`p-1.5 rounded-lg transition-all ${viewMode === 'table' ? 'bg-background text-cyan-500 shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                        title="Table View"
                    >
                        <TableIcon className="w-4 h-4" />
                    </button>
                    <button
                        onClick={() => setViewMode('cards')}
                        className={`p-1.5 rounded-lg transition-all ${viewMode === 'cards' ? 'bg-background text-cyan-500 shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                        title="Card View"
                    >
                        <LayoutGrid className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Filter Bar */}
            <div className="bg-card/40 border border-border/60 rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex flex-wrap items-center gap-3">
                    {/* Search Symbol */}
                    <div className="relative flex-1 min-w-[160px] max-w-xs">
                        <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                        <input
                            type="text"
                            placeholder="Filter by symbol..."
                            value={symbolFilter}
                            onChange={e => setSymbolFilter(e.target.value)}
                            className="w-full pl-8 pr-8 py-1.5 bg-background border border-border/60 rounded-xl text-xs text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                        />
                        {symbolFilter && (
                            <button
                                onClick={() => setSymbolFilter('')}
                                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                            >
                                <X className="w-3.5 h-3.5" />
                            </button>
                        )}
                    </div>

                    {/* Account Dropdown */}
                    {uniqueAccounts.length > 1 && (
                        <select
                            value={accountFilter}
                            onChange={e => setAccountFilter(e.target.value)}
                            className="px-3 py-1.5 bg-background border border-border/60 rounded-xl text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-cyan-500"
                        >
                            <option value="">All Accounts</option>
                            {uniqueAccounts.map(acc => (
                                <option key={acc} value={acc}>{acc}</option>
                            ))}
                        </select>
                    )}

                    {/* Date Presets */}
                    <div className="flex flex-wrap items-center gap-1">
                        {DATE_PRESETS.map(preset => (
                            <button
                                key={preset.key}
                                onClick={() => setDatePreset(preset.key)}
                                className={`px-2.5 py-1 text-[11px] font-medium rounded-lg transition-all ${
                                    datePreset === preset.key
                                        ? 'bg-cyan-500/15 text-cyan-500 font-semibold'
                                        : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                                }`}
                            >
                                {preset.label}
                            </button>
                        ))}
                    </div>

                    {/* Reset Filters */}
                    {hasActiveFilters && (
                        <button
                            onClick={resetFilters}
                            className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-cyan-500 transition-colors ml-auto"
                        >
                            <Filter className="w-3 h-3" />
                            Clear Filters
                        </button>
                    )}
                </div>

                {/* Custom Date Inputs */}
                {datePreset === 'custom' && (
                    <div className="flex items-center gap-2 pt-1 text-xs text-muted-foreground">
                        <span>From:</span>
                        <input
                            type="date"
                            value={customFrom}
                            onChange={e => setCustomFrom(e.target.value)}
                            className="px-2.5 py-1 bg-background border border-border/60 rounded-lg text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-cyan-500"
                        />
                        <span>To:</span>
                        <input
                            type="date"
                            value={customTo}
                            onChange={e => setCustomTo(e.target.value)}
                            className="px-2.5 py-1 bg-background border border-border/60 rounded-lg text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-cyan-500"
                        />
                    </div>
                )}

                {/* Type Filter Chips */}
                {availableTypes.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 pt-1 border-t border-border/30">
                        <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground/60 mr-1">Type:</span>
                        {availableTypes.map(t => {
                            const isSelected = filterTypes.includes(t);
                            return (
                                <button
                                    key={t}
                                    onClick={() => toggleFilterType(t)}
                                    className={`px-2 py-0.5 text-[11px] rounded-full transition-all border ${
                                        isSelected
                                            ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-500 font-semibold'
                                            : 'bg-secondary/60 border-border/40 text-muted-foreground hover:text-foreground'
                                    }`}
                                >
                                    {t}
                                </button>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
};
