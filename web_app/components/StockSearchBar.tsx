'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Search, X, TrendingUp, BarChart3, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { fetchSymbolSearch, type SymbolSearchResult } from '@/lib/api';
import StockIcon from './StockIcon';

const StockDetailModal = dynamic(() => import('@/components/StockDetailModal'), { ssr: false });

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
}

export function StockSearchBar({ currency, placeholder = 'Search symbol…' }: StockSearchBarProps) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<SymbolSearchResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [open, setOpen] = useState(false);
    const [activeIdx, setActiveIdx] = useState(0);
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

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
        setSelectedSymbol(symbol);
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
            inputRef.current?.blur();
        }
    };

    const hasResults = results.length > 0;

    return (
        <>
            <div ref={containerRef} className="relative">
                {/* Input */}
                <div className={cn(
                    'flex items-center gap-2 h-7 rounded-md border transition-all duration-200',
                    open
                        ? 'w-48 sm:w-56 border-primary/40 bg-background ring-1 ring-primary/20'
                        : 'w-32 sm:w-40 border-border/60 bg-muted/30',
                )}>
                    {loading
                        ? <Loader2 className="w-3 h-3 ml-2.5 shrink-0 text-muted-foreground animate-spin" />
                        : <Search className="w-3 h-3 ml-2.5 shrink-0 text-muted-foreground" />
                    }
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={e => { setQuery(e.target.value); setOpen(true); }}
                        onFocus={() => setOpen(true)}
                        onKeyDown={handleKeyDown}
                        placeholder={placeholder}
                        className="flex-1 min-w-0 bg-transparent text-xs text-foreground placeholder:text-muted-foreground focus:outline-none pr-1"
                        autoComplete="off"
                        spellCheck={false}
                    />
                    {query && (
                        <button
                            onClick={() => { setQuery(''); setResults([]); inputRef.current?.focus(); }}
                            className="mr-1.5 text-muted-foreground hover:text-foreground transition-colors"
                        >
                            <X className="w-3 h-3" />
                        </button>
                    )}
                    {!query && (
                        <kbd className="hidden lg:inline mr-2 px-1 py-0.5 rounded border border-border/60 bg-background/80 text-[10px] font-mono text-muted-foreground leading-none">
                            ⌘K
                        </kbd>
                    )}
                </div>

                {/* Dropdown */}
                {open && (hasResults || (query.trim().length > 0 && !loading)) && (
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

            {/* Stock detail popup */}
            {selectedSymbol && (
                <StockDetailModal
                    symbol={selectedSymbol}
                    isOpen={!!selectedSymbol}
                    onClose={() => setSelectedSymbol(null)}
                    currency={currency}
                />
            )}
        </>
    );
}
