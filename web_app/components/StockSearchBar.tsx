'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Search, X, TrendingUp, BarChart3, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { fetchSymbolSearch, type SymbolSearchResult } from '@/lib/api';
import { useStockModal } from '@/context/StockModalContext';
import StockIcon from './StockIcon';

// Map yfinance typeDisp to a short badge label + icon
function TypeBadge({ type }: { type: string }) {
    const t = type.toLowerCase();
    if (t === 'equity' || t === 'stock') return (
        <span className="text-[9px] font-bold uppercase tracking-wide text-indigo-500 bg-indigo-500/10 px-1.5 py-0.5 rounded">Equity</span>
    );
    if (t === 'etf') return (
        <span className="text-[9px] font-bold uppercase tracking-wide text-cyan-500 bg-cyan-500/10 px-1.5 py-0.5 rounded">ETF</span>
    );
    if (t === 'mutualfund' || t === 'mutual fund') return (
        <span className="text-[9px] font-bold uppercase tracking-wide text-purple-500 bg-purple-500/10 px-1.5 py-0.5 rounded">Fund</span>
    );
    if (t === 'index') return (
        <span className="text-[9px] font-bold uppercase tracking-wide text-amber-500 bg-amber-500/10 px-1.5 py-0.5 rounded">Index</span>
    );
    if (t === 'crypto' || t === 'cryptocurrency') return (
        <span className="text-[9px] font-bold uppercase tracking-wide text-orange-500 bg-orange-500/10 px-1.5 py-0.5 rounded">Crypto</span>
    );
    if (t) return (
        <span className="text-[9px] font-bold uppercase tracking-wide text-muted-foreground bg-muted px-1.5 py-0.5 rounded">{type}</span>
    );
    return null;
}

interface StockSearchBarProps {
    currency: string;
    placeholder?: string;
    fullWidth?: boolean;
}

export function StockSearchBar({ currency, placeholder = 'Search symbol…', fullWidth = false }: StockSearchBarProps) {
    const { openStockDetail } = useStockModal();
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SymbolSearchResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [open, setOpen] = useState(false);
    const [activeIdx, setActiveIdx] = useState(0);

    const inputRef = useRef<HTMLInputElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);


    // Debounced search
    const runSearch = useCallback((q: string) => {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        if (!q.trim()) {
            setResults([]);
            setLoading(false);
            return;
        }
        setLoading(true);
        debounceRef.current = setTimeout(async () => {
            try {
                const data = await fetchSymbolSearch(q.trim());
                setResults(data);
                setActiveIdx(0);
            } catch {
                setResults([]);
            } finally {
                setLoading(false);
            }
        }, 280);
    }, []);

    useEffect(() => {
        runSearch(query);
    }, [query, runSearch]);

    // Close dropdown on outside click
    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    // ⌘K shortcut
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                inputRef.current?.focus();
                setOpen(true);
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);

    const openModal = (symbol: string) => {
        openStockDetail(symbol, currency);
        setOpen(false);
        setQuery('');
        setResults([]);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (!open) return;
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setActiveIdx(i => Math.min(i + 1, results.length - 1));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setActiveIdx(i => Math.max(i - 1, 0));
        } else if (e.key === 'Enter') {
            e.preventDefault();
            const hit = results[activeIdx];
            if (hit) {
                openModal(hit.symbol);
            } else if (query.trim()) {
                openModal(query.trim().toUpperCase());
            }
        } else if (e.key === 'Escape') {
            setOpen(false);
        }
    };

    const hasResults = results.length > 0;
    const isSearching = query.trim().length > 0;

    return (
        <div ref={containerRef} className={cn('relative', fullWidth ? 'w-full' : 'w-48 sm:w-64')}>
            <div className="relative flex items-center">
                <Search className="absolute left-3 w-4 h-4 text-muted-foreground pointer-events-none" />
                <input
                    ref={inputRef}
                    type="text"
                    className={cn(
                        'w-full pl-9 pr-8 py-1.5 text-xs rounded-xl border border-border/80 bg-muted/40 dark:bg-zinc-900/60 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary/60 transition-all',
                        open && 'ring-2 ring-primary/40 border-primary/60',
                    )}
                    placeholder={placeholder}
                    value={query}
                    onChange={e => {
                        setQuery(e.target.value);
                        setOpen(true);
                        runSearch(e.target.value);
                    }}
                    onFocus={() => {
                        if (query.trim()) setOpen(true);
                    }}
                    onKeyDown={handleKeyDown}
                    autoComplete="off"
                    spellCheck={false}
                />
                {loading && (
                    <Loader2 className="absolute right-2.5 w-3.5 h-3.5 text-muted-foreground animate-spin pointer-events-none" />
                )}
                {!loading && query && (
                    <button
                        onClick={() => {
                            setQuery('');
                            setResults([]);
                            setOpen(false);
                        }}
                        className="absolute right-2 text-muted-foreground hover:text-foreground p-0.5"
                    >
                        <X className="w-3.5 h-3.5" />
                    </button>
                )}
            </div>

            {/* Dropdown */}
            {open && isSearching && (
                <div className="absolute top-full left-0 mt-1.5 w-72 z-50 rounded-xl border border-border bg-white dark:bg-zinc-900 shadow-2xl overflow-hidden animate-in fade-in slide-in-from-top-1 duration-150">
                    {hasResults ? (
                        <ul className="py-1 max-h-72 overflow-y-auto">
                            {results.map((r, i) => (
                                <li
                                    key={r.symbol}
                                    onMouseEnter={() => setActiveIdx(i)}
                                    onClick={() => openModal(r.symbol)}
                                    className={cn(
                                        'flex items-center gap-3 px-3 py-2.5 cursor-pointer transition-colors',
                                        activeIdx === i ? 'bg-muted' : 'hover:bg-muted/60',
                                    )}
                                >
                                    <div className="w-7 h-7 shrink-0">
                                        <StockIcon symbol={r.symbol} size={28} />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-1.5">
                                            <span className="text-sm font-bold text-foreground">{r.symbol}</span>
                                            <TypeBadge type={r.type} />
                                        </div>
                                        {r.name && (
                                            <p className="text-[11px] text-muted-foreground truncate">{r.name}</p>
                                        )}
                                    </div>
                                    <TrendingUp className="w-3.5 h-3.5 text-muted-foreground/50 shrink-0" />
                                </li>
                            ))}
                        </ul>
                    ) : (
                        /* No API results — offer direct lookup */
                        <button
                            onClick={() => openModal(query.trim().toUpperCase())}
                            className="flex items-center gap-3 w-full px-3 py-3 text-left bg-white dark:bg-zinc-900 hover:bg-muted transition-colors"
                        >
                            <BarChart3 className="w-4 h-4 text-primary shrink-0" />
                            <span className="text-sm font-bold text-foreground">
                                {query.trim().toUpperCase()}
                            </span>
                        </button>
                    )}
                    <div className="px-3 py-1.5 border-t border-border/60 flex items-center justify-between">
                        <span className="text-[10px] text-muted-foreground">↑↓ navigate · Enter to open</span>
                        <span className="text-[10px] text-muted-foreground">Esc to close</span>
                    </div>
                </div>
            )}
        </div>
    );
}
