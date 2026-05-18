'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import {
    LayoutDashboard, PieChart, TrendingUp, ArrowLeftRight,
    DollarSign, BarChart3, Search, Star, Globe, Sparkles,
    Settings, ChevronRight, Loader2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { fetchSymbolSearch, type SymbolSearchResult } from '@/lib/api';
import StockIcon from '@/components/StockIcon';

const StockDetailModal = dynamic(() => import('@/components/StockDetailModal'), { ssr: false });

interface CommandPaletteProps {
    isOpen: boolean;
    onClose: () => void;
    onNavigate: (tab: string) => void;
    currency: string;
}

const NAV_COMMANDS = [
    { id: 'performance',   label: 'Dashboard',     icon: LayoutDashboard, section: 'Portfolio' },
    { id: 'allocation',    label: 'Portfolio',      icon: PieChart,        section: 'Portfolio' },
    { id: 'asset_change',  label: 'Performance',    icon: TrendingUp,      section: 'Portfolio' },
    { id: 'transactions',  label: 'Transactions',   icon: ArrowLeftRight,  section: 'Portfolio' },
    { id: 'dividend',      label: 'Income',         icon: DollarSign,      section: 'Portfolio' },
    { id: 'capital_gains', label: 'Capital Gains',  icon: BarChart3,       section: 'Portfolio' },
    { id: 'screener',      label: 'Screener',       icon: Search,          section: 'Tools' },
    { id: 'watchlist',     label: 'Watchlist',      icon: Star,            section: 'Tools' },
    { id: 'markets',       label: 'Markets',        icon: Globe,           section: 'Tools' },
    { id: 'ai_review',     label: 'AI Insights',    icon: Sparkles,        section: 'Tools' },
    { id: 'settings',      label: 'Settings',       icon: Settings,        section: 'Settings' },
];

function TypeBadge({ type }: { type: string }) {
    const t = type.toLowerCase();
    if (t === 'equity' || t === 'stock')
        return <span className="text-[9px] font-bold uppercase tracking-wide text-indigo-500 bg-indigo-500/10 px-1.5 py-0.5 rounded">Equity</span>;
    if (t === 'etf')
        return <span className="text-[9px] font-bold uppercase tracking-wide text-cyan-500 bg-cyan-500/10 px-1.5 py-0.5 rounded">ETF</span>;
    if (t === 'mutualfund' || t === 'mutual fund')
        return <span className="text-[9px] font-bold uppercase tracking-wide text-purple-500 bg-purple-500/10 px-1.5 py-0.5 rounded">Fund</span>;
    if (t === 'index')
        return <span className="text-[9px] font-bold uppercase tracking-wide text-amber-500 bg-amber-500/10 px-1.5 py-0.5 rounded">Index</span>;
    if (t === 'crypto' || t === 'cryptocurrency')
        return <span className="text-[9px] font-bold uppercase tracking-wide text-orange-500 bg-orange-500/10 px-1.5 py-0.5 rounded">Crypto</span>;
    if (t)
        return <span className="text-[9px] font-bold uppercase tracking-wide text-muted-foreground bg-muted px-1.5 py-0.5 rounded">{type}</span>;
    return null;
}

export default function CommandPalette({ isOpen, onClose, onNavigate, currency }: CommandPaletteProps) {
    const [query, setQuery] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [stockResults, setStockResults] = useState<SymbolSearchResult[]>([]);
    const [stockLoading, setStockLoading] = useState(false);
    const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        if (isOpen) {
            setQuery('');
            setSelectedIndex(0);
            setStockResults([]);
            setTimeout(() => inputRef.current?.focus(), 50);
        }
    }, [isOpen]);

    // Debounced stock search
    const runStockSearch = useCallback((q: string) => {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        if (!q.trim()) { setStockResults([]); setStockLoading(false); return; }
        setStockLoading(true);
        debounceRef.current = setTimeout(async () => {
            try {
                const data = await fetchSymbolSearch(q.trim());
                setStockResults(data);
            } catch {
                setStockResults([]);
            } finally {
                setStockLoading(false);
            }
        }, 280);
    }, []);

    useEffect(() => {
        setSelectedIndex(0);
        runStockSearch(query);
    }, [query, runStockSearch]);

    const filteredNav = NAV_COMMANDS.filter(cmd =>
        cmd.label.toLowerCase().includes(query.toLowerCase()) ||
        cmd.section.toLowerCase().includes(query.toLowerCase())
    );

    // Flat list for keyboard navigation: nav items first, then stock results
    const totalCount = filteredNav.length + stockResults.length;

    const openStock = useCallback((symbol: string) => {
        setSelectedSymbol(symbol);
        onClose();
    }, [onClose]);

    useEffect(() => {
        if (!isOpen) return;
        const handler = (e: KeyboardEvent) => {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                setSelectedIndex(i => Math.min(i + 1, totalCount - 1));
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                setSelectedIndex(i => Math.max(i - 1, 0));
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (selectedIndex < filteredNav.length) {
                    onNavigate(filteredNav[selectedIndex].id);
                    onClose();
                } else {
                    const stockIdx = selectedIndex - filteredNav.length;
                    const hit = stockResults[stockIdx];
                    if (hit) openStock(hit.symbol);
                    else if (query.trim()) openStock(query.trim().toUpperCase());
                }
            } else if (e.key === 'Escape') {
                e.preventDefault();
                onClose();
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [isOpen, filteredNav, stockResults, selectedIndex, totalCount, query, onNavigate, onClose, openStock]);

    // Group nav by section for display
    const sections: Record<string, typeof filteredNav> = {};
    for (const cmd of filteredNav) {
        (sections[cmd.section] ??= []).push(cmd);
    }

    let globalIndex = 0;

    // NOTE: We deliberately DO NOT early-return on `!isOpen` here. Clicking a
    // stock result triggers both setSelectedSymbol(...) and onClose() in the
    // same React batch — flipping `isOpen` to false. If we early-returned, the
    // <StockDetailModal/> below would be discarded along with the palette and
    // never render. Render the modal independently of the palette's open state.
    return (
        <>
            {isOpen && (
            <div className="fixed inset-0 z-[200] flex items-start justify-center pt-[15vh]">
                {/* Backdrop */}
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />

                {/* Modal */}
                <div className="relative w-full max-w-lg rounded-2xl border border-border bg-white dark:bg-zinc-900 shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">

                    {/* Search input */}
                    <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
                        {stockLoading
                            ? <Loader2 className="w-4 h-4 text-muted-foreground shrink-0 animate-spin" />
                            : <Search className="w-4 h-4 text-muted-foreground shrink-0" />
                        }
                        <input
                            ref={inputRef}
                            type="text"
                            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
                            placeholder="Go to page or search a stock symbol…"
                            value={query}
                            onChange={e => setQuery(e.target.value)}
                            autoComplete="off"
                            spellCheck={false}
                        />
                        <kbd className="px-1.5 py-0.5 rounded border border-border bg-muted text-[10px] font-mono text-muted-foreground shrink-0">
                            ESC
                        </kbd>
                    </div>

                    {/* Results */}
                    <div className="max-h-[55vh] overflow-y-auto py-2">
                        {totalCount === 0 && !stockLoading ? (
                            <p className="px-4 py-8 text-center text-sm text-muted-foreground">
                                {query ? `No results for "${query}"` : 'Start typing to search…'}
                            </p>
                        ) : (
                            <>
                                {/* Navigation section */}
                                {Object.entries(sections).map(([section, cmds]) => (
                                    <div key={section}>
                                        <p className="px-4 pt-3 pb-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/60">
                                            {section}
                                        </p>
                                        {cmds.map(cmd => {
                                            const idx = globalIndex++;
                                            const active = idx === selectedIndex;
                                            const Icon = cmd.icon;
                                            return (
                                                <button
                                                    key={cmd.id}
                                                    className={cn(
                                                        'w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors',
                                                        active ? 'bg-indigo-600 text-white' : 'text-foreground hover:bg-muted',
                                                    )}
                                                    onClick={() => { onNavigate(cmd.id); onClose(); }}
                                                    onMouseEnter={() => setSelectedIndex(idx)}
                                                >
                                                    <Icon className={cn('w-4 h-4 shrink-0', active ? 'text-white' : 'text-muted-foreground')} />
                                                    <span className="flex-1 text-left font-medium">{cmd.label}</span>
                                                    {active && <ChevronRight className="w-4 h-4 text-white/70 shrink-0" />}
                                                </button>
                                            );
                                        })}
                                    </div>
                                ))}

                                {/* Stock results section */}
                                {stockResults.length > 0 && (
                                    <div>
                                        <p className="px-4 pt-3 pb-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/60">
                                            Stocks
                                        </p>
                                        {stockResults.map(r => {
                                            const idx = globalIndex++;
                                            const active = idx === selectedIndex;
                                            return (
                                                <button
                                                    key={r.symbol}
                                                    className={cn(
                                                        'w-full flex items-center gap-3 px-4 py-2 transition-colors',
                                                        active ? 'bg-indigo-600 text-white' : 'text-foreground hover:bg-muted',
                                                    )}
                                                    onClick={() => openStock(r.symbol)}
                                                    onMouseEnter={() => setSelectedIndex(idx)}
                                                >
                                                    <div className="w-7 h-7 shrink-0">
                                                        <StockIcon symbol={r.symbol} size={28} />
                                                    </div>
                                                    <div className="flex-1 min-w-0 text-left">
                                                        <div className="flex items-center gap-1.5">
                                                            <span className={cn('text-sm font-bold', active ? 'text-white' : 'text-foreground')}>
                                                                {r.symbol}
                                                            </span>
                                                            <TypeBadge type={r.type} />
                                                        </div>
                                                        {r.name && (
                                                            <p className={cn('text-[11px] truncate', active ? 'text-white/70' : 'text-muted-foreground')}>
                                                                {r.name}
                                                            </p>
                                                        )}
                                                    </div>
                                                    {active && <ChevronRight className="w-4 h-4 text-white/70 shrink-0" />}
                                                </button>
                                            );
                                        })}
                                    </div>
                                )}
                            </>
                        )}
                    </div>

                    {/* Footer */}
                    <div className="border-t border-border px-4 py-2 flex items-center justify-between text-[10px] text-muted-foreground">
                        <span>↑↓ navigate · Enter to open</span>
                        <span className="font-semibold">Investa</span>
                    </div>
                </div>
            </div>
            )}

            {/* Stock detail modal — kept outside the `isOpen` guard so it
                survives the palette closing on stock selection. */}
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
