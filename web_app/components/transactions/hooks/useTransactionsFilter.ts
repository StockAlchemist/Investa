import { useState, useMemo } from 'react';
import { Transaction } from '../../../lib/api';
import { SortableKey, DatePreset } from '../types';
import { dupKey, canonicalType, computeDateRange } from '../transactionsUtils';

interface UseTransactionsFilterProps {
    transactions: Transaction[];
}

export function useTransactionsFilter({ transactions }: UseTransactionsFilterProps) {
    const [symbolFilter, setSymbolFilter] = useState('');
    const [accountFilter, setAccountFilter] = useState('');
    const [filterTypes, setFilterTypes] = useState<string[]>([]);
    const [datePreset, setDatePreset] = useState<DatePreset>('all');
    const [customFrom, setCustomFrom] = useState('');
    const [customTo, setCustomTo] = useState('');
    const [sortBy, setSortBy] = useState<SortableKey>('Date');
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
    const [currentPage, setCurrentPage] = useState(1);
    const [pageSize, setPageSize] = useState(25);
    const [viewMode, setViewMode] = useState<'table' | 'cards'>('table');

    // Duplicate detection: flag transactions that share the same dupKey
    const duplicateKeys = useMemo(() => {
        const counts = new Map<string, number>();
        transactions.forEach(t => {
            const k = dupKey(t);
            counts.set(k, (counts.get(k) ?? 0) + 1);
        });
        const dups = new Set<string>();
        counts.forEach((count, key) => {
            if (count > 1) dups.add(key);
        });
        return dups;
    }, [transactions]);

    const uniqueAccounts = useMemo(() => {
        const set = new Set<string>();
        transactions.forEach(t => { if (t.Account) set.add(t.Account); });
        return Array.from(set).sort();
    }, [transactions]);

    const availableTypes = useMemo(() => {
        const set = new Set<string>();
        transactions.forEach(t => { if (t.Type) set.add(canonicalType(t.Type)); });
        return Array.from(set).sort();
    }, [transactions]);

    const dateRange = useMemo(
        () => computeDateRange(datePreset, customFrom, customTo),
        [datePreset, customFrom, customTo],
    );

    const filteredTransactions = useMemo(() => {
        return transactions.filter(t => {
            if (symbolFilter && !(t.Symbol || '').toLowerCase().includes(symbolFilter.toLowerCase())) {
                return false;
            }
            if (accountFilter && t.Account !== accountFilter) {
                return false;
            }
            if (filterTypes.length > 0 && !filterTypes.includes(canonicalType(t.Type || ''))) {
                return false;
            }
            if (dateRange.from || dateRange.to) {
                const d = (t.Date || '').split('T')[0].split(' ')[0];
                if (dateRange.from && d < dateRange.from) return false;
                if (dateRange.to && d > dateRange.to) return false;
            }
            return true;
        });
    }, [transactions, symbolFilter, accountFilter, filterTypes, dateRange]);

    const sortedTransactions = useMemo(() => {
        const copy = [...filteredTransactions];
        copy.sort((a, b) => {
            const aVal: unknown = a[sortBy as keyof Transaction];
            const bVal: unknown = b[sortBy as keyof Transaction];

            if (sortBy === 'Date') {
                const aDate = new Date(String(aVal || '')).getTime() || 0;
                const bDate = new Date(String(bVal || '')).getTime() || 0;
                return sortDirection === 'asc' ? aDate - bDate : bDate - aDate;
            }

            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
            }

            const aStr = String(aVal || '').toLowerCase();
            const bStr = String(bVal || '').toLowerCase();
            if (aStr < bStr) return sortDirection === 'asc' ? -1 : 1;
            if (aStr > bStr) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
        return copy;
    }, [filteredTransactions, sortBy, sortDirection]);

    const totalPages = Math.ceil(sortedTransactions.length / pageSize) || 1;
    const paginatedTransactions = useMemo(() => {
        const start = (currentPage - 1) * pageSize;
        return sortedTransactions.slice(start, start + pageSize);
    }, [sortedTransactions, currentPage, pageSize]);

    const handleSort = (key: SortableKey) => {
        if (sortBy === key) {
            setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
        } else {
            setSortBy(key);
            setSortDirection('desc');
        }
    };

    const toggleFilterType = (type: string) => {
        setFilterTypes(prev =>
            prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
        );
        setCurrentPage(1);
    };

    const resetFilters = () => {
        setSymbolFilter('');
        setAccountFilter('');
        setFilterTypes([]);
        setDatePreset('all');
        setCustomFrom('');
        setCustomTo('');
        setCurrentPage(1);
    };

    const hasActiveFilters = Boolean(
        symbolFilter || accountFilter || filterTypes.length > 0 || datePreset !== 'all' || customFrom || customTo
    );

    return {
        symbolFilter,
        setSymbolFilter,
        accountFilter,
        setAccountFilter,
        filterTypes,
        setFilterTypes,
        toggleFilterType,
        datePreset,
        setDatePreset,
        customFrom,
        setCustomFrom,
        customTo,
        setCustomTo,
        sortBy,
        sortDirection,
        handleSort,
        currentPage,
        setCurrentPage,
        pageSize,
        setPageSize,
        viewMode,
        setViewMode,
        resetFilters,
        hasActiveFilters,
        uniqueAccounts,
        availableTypes,
        duplicateKeys,
        filteredTransactions,
        sortedTransactions,
        paginatedTransactions,
        totalPages,
    };
}
