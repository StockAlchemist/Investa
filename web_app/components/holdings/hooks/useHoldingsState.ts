import { useState, useEffect, useRef } from 'react';
import { SortConfig } from '../types';
import { DEFAULT_VISIBLE_COLUMNS, RENAMED_COLUMNS } from '../constants';

export function useHoldingsState() {
    const [visibleColumns, setVisibleColumns] = useState<string[]>(() => {
        if (typeof window !== 'undefined') {
            try {
                const saved = localStorage.getItem('investa_holdings_columns');
                if (saved) {
                    const parsed = JSON.parse(saved);
                    if (Array.isArray(parsed) && parsed.length > 0) {
                        return parsed.map((c: string) => RENAMED_COLUMNS[c] ?? c);
                    }
                }
            } catch (e) {
                console.error("Failed to parse saved columns", e);
            }
        }
        return DEFAULT_VISIBLE_COLUMNS;
    });

    const [sortConfig, setSortConfig] = useState<SortConfig>(() => {
        if (typeof window !== 'undefined') {
            try {
                const saved = localStorage.getItem('investa_holdings_sort');
                if (saved) {
                    const parsed = JSON.parse(saved);
                    if (parsed.key && parsed.direction) return parsed;
                }
            } catch (e) {
                console.error("Failed to parse saved sort config", e);
            }
        }
        return { key: 'Mkt Val', direction: 'desc' };
    });

    const [expandedLots, setExpandedLots] = useState<Set<string>>(() => {
        if (typeof window !== 'undefined') {
            try {
                const saved = localStorage.getItem('investa_holdings_expanded_lots');
                if (saved) {
                    const parsed = JSON.parse(saved);
                    if (Array.isArray(parsed)) return new Set(parsed);
                }
            } catch (e) {
                console.error("Failed to parse saved expanded lots", e);
            }
        }
        return new Set();
    });

    const [expandedCards, setExpandedCards] = useState<Set<string>>(() => {
        if (typeof window !== 'undefined') {
            try {
                const saved = localStorage.getItem('investa_holdings_expanded_cards');
                if (saved) {
                    const parsed = JSON.parse(saved);
                    if (Array.isArray(parsed)) return new Set(parsed);
                }
            } catch (e) {
                console.error("Failed to parse saved expanded cards", e);
            }
        }
        return new Set();
    });

    const [isColumnMenuOpen, setIsColumnMenuOpen] = useState(false);
    const [draggedColumn, setDraggedColumn] = useState<string | null>(null);
    const [mobileViewMode, setMobileViewMode] = useState<'card' | 'table'>('table');
    const [isAccountMenuOpen, setIsAccountMenuOpen] = useState(false);
    const [isGroupByMenuOpen, setIsGroupByMenuOpen] = useState(false);

    const columnMenuRef = useRef<HTMLDivElement>(null);
    const accountMenuRef = useRef<HTMLDivElement>(null);
    const groupByMenuRef = useRef<HTMLDivElement>(null);

    // Persist columns to localStorage on change
    useEffect(() => {
        try {
            localStorage.setItem('investa_holdings_columns', JSON.stringify(visibleColumns));
        } catch {
            // Ignore quota or private mode errors
        }
    }, [visibleColumns]);

    // Persist sort to localStorage on change
    useEffect(() => {
        try {
            localStorage.setItem('investa_holdings_sort', JSON.stringify(sortConfig));
        } catch {
            // Ignore
        }
    }, [sortConfig]);

    // Persist expandedLots to localStorage on change
    useEffect(() => {
        try {
            localStorage.setItem('investa_holdings_expanded_lots', JSON.stringify(Array.from(expandedLots)));
        } catch {
            // Ignore
        }
    }, [expandedLots]);

    // Persist expandedCards to localStorage on change
    useEffect(() => {
        try {
            localStorage.setItem('investa_holdings_expanded_cards', JSON.stringify(Array.from(expandedCards)));
        } catch {
            // Ignore
        }
    }, [expandedCards]);

    // Close menus when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (columnMenuRef.current && !columnMenuRef.current.contains(event.target as Node)) {
                setIsColumnMenuOpen(false);
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

    const handleSort = (header: string) => {
        setSortConfig(current => ({
            key: header,
            direction: current.key === header && current.direction === 'desc' ? 'asc' : 'desc',
        }));
    };

    const toggleColumn = (header: string) => {
        setVisibleColumns(prev => {
            if (prev.includes(header)) {
                if (prev.length <= 1) return prev;
                return prev.filter(c => c !== header);
            } else {
                return [...prev, header];
            }
        });
    };

    const toggleLotExpansion = (key: string) => {
        setExpandedLots(prev => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    const toggleCardExpansion = (key: string) => {
        setExpandedCards(prev => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    const handleDragStart = (e: React.DragEvent<HTMLTableHeaderCellElement>, header: string) => {
        setDraggedColumn(header);
        e.dataTransfer.setData('text/plain', header);
        e.dataTransfer.effectAllowed = 'move';
    };

    const handleDragOver = (e: React.DragEvent<HTMLTableHeaderCellElement>) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    };

    const handleDrop = (e: React.DragEvent<HTMLTableHeaderCellElement>, targetHeader: string) => {
        e.preventDefault();
        if (!draggedColumn || draggedColumn === targetHeader) return;

        setVisibleColumns(prev => {
            const newCols = [...prev];
            const dragIdx = newCols.indexOf(draggedColumn);
            const dropIdx = newCols.indexOf(targetHeader);
            if (dragIdx === -1 || dropIdx === -1) return prev;

            newCols.splice(dragIdx, 1);
            newCols.splice(dropIdx, 0, draggedColumn);
            return newCols;
        });

        setDraggedColumn(null);
    };

    return {
        visibleColumns,
        setVisibleColumns,
        expandedLots,
        setExpandedLots,
        expandedCards,
        setExpandedCards,
        sortConfig,
        setSortConfig,
        handleSort,
        toggleColumn,
        toggleLotExpansion,
        toggleCardExpansion,
        isColumnMenuOpen,
        setIsColumnMenuOpen,
        columnMenuRef,
        isAccountMenuOpen,
        setIsAccountMenuOpen,
        accountMenuRef,
        isGroupByMenuOpen,
        setIsGroupByMenuOpen,
        groupByMenuRef,
        draggedColumn,
        handleDragStart,
        handleDragOver,
        handleDrop,
        mobileViewMode,
        setMobileViewMode,
    };
}
