'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import {
    LayoutDashboard, PieChart, TrendingUp, ArrowLeftRight,
    DollarSign, BarChart3, Search, Star, Globe, Sparkles, Trophy,
    Settings, ChevronRight, Loader2, Layers,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { fetchSymbolSearch, type SymbolSearchResult } from '@/lib/api';
import { useStockModal } from '@/context/StockModalContext';
import StockIcon from '@/components/StockIcon';

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
    { id: 'buffett_rank',  label: 'Rankings',       icon: Trophy,          section: 'Tools' },
    { id: 'strategies',    label: 'Strategies',     icon: Layers,          section: 'Tools' },
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
    const { openStockDetail } = useStockModal();
    const [query, setQuery] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [stockResults, setStockResults] = useState<SymbolSearchResult[]>([]);
    const [stockLoading, setStockLoading] = useState(false);
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
    const searchStocks = useCallback((q: string) => {
        if (debounceRef.current) clearTimeout(debounceRef.current);
        if (!q.trim()) {
            setStockResults([]);
            setStockLoading(false);
            return;
        }
        setStockLoading(true);
        debounceRef.current = setTimeout(async () => {
            try {
                const results = await fetchSymbolSearch(q.trim());
                setStockResults(results);
            } catch {
                setStockResults([]);
            } finally {
                setStockLoading(false);
            }
        }, 250);
    }, []);

    useEffect(() => {
        searchStocks(query);
    }, [query, searchStocks]);

    const filteredNav = NAV_COMMANDS.filter(cmd =>
        cmd.label.toLowerCase().includes(query.toLowerCase()) ||
        cmd.section.toLowerCase().includes(query.toLowerCase())
    );

    // Flat list for keyboard navigation: nav items first, then stock results
    const totalCount = filteredNav.length + stockResults.length;

    const openStock = useCallback((symbol: string) => {
        openStockDetail(symbol, currency);
        onClose();
    }, [openStockDetail, currency, onClose]);

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

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[200] flex items-start justify-center pt-[15vh]">
            {/* Backdrop */}
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />

            {/* Modal */}
            <div className="relative w-full max-w-lg rounded-2xl border border-border bg-white dark:bg-zinc-900 shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">

                {/* Search input */}
                <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
                    <Search className="w-4 h-4 text-muted-foreground shrink-0" />
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        placeholder="Search sections or stock symbols…"
                        className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
                        autoComplete="off"
                        spellCheck={false}
                    />
                    {stockLoading && <Loader2 className="w-3.5 h-3.5 text-muted-foreground animate-spin shrink-0" />}
                    <kbd className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-border bg-muted text-muted-foreground">
                        ESC
                    </kbd>
                </div>

                {/* Results list */}
                <div className="max-h-80 overflow-y-auto p-2 space-y-3">
                    {totalCount === 0 && (
                        <div className="py-8 text-center text-xs text-muted-foreground">
                            No results found for &ldquo;{query}&rdquo;
                        </div>
                    )}

                    {/* Navigation sections */}
                    {Object.entries(sections).map(([section, items]) => (
                        <div key={section} className="space-y-0.5">
                            <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60 px-2 py-1">
                                {section}
                            </p>
                            {items.map(cmd => {
                                const Icon = cmd.icon;
                                const itemIndex = globalIndex++;
                                const active = itemIndex === selectedIndex;
                                return (
                                    <button
                                        key={cmd.id}
                                        onClick={() => { onNavigate(cmd.id); onClose(); }}
                                        onMouseEnter={() => setSelectedIndex(itemIndex)}
                                        className={cn(
                                            'w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium text-left transition-colors',
                                            active
                                                ? 'bg-primary text-primary-foreground'
                                                : 'text-foreground hover:bg-muted',
                                        )}
                                    >
                                        <div className="flex items-center gap-2.5">
                                            <Icon className={cn('w-4 h-4', active ? 'text-primary-foreground' : 'text-muted-foreground')} />
                                            <span>{cmd.label}</span>
                                        </div>
                                        {active && <ChevronRight className="w-3.5 h-3.5 opacity-70" />}
                                    </button>
                                );
                            })}
                        </div>
                    ))}

                    {/* Stock search results */}
                    {stockResults.length > 0 && (
                        <div className="space-y-0.5 pt-1 border-t border-border">
                            <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60 px-2 py-1">
                                Stocks &amp; Assets
                            </p>
                            {stockResults.map(r => {
                                const itemIndex = globalIndex++;
                                const active = itemIndex === selectedIndex;
                                return (
                                    <button
                                        key={r.symbol}
                                        onClick={() => openStock(r.symbol)}
                                        onMouseEnter={() => setSelectedIndex(itemIndex)}
                                        className={cn(
                                            'w-full flex items-center gap-3 px-2.5 py-2 rounded-lg text-left transition-colors',
                                            active
                                                ? 'bg-primary text-white'
                                                : 'hover:bg-muted',
                                        )}
                                    >
                                        <div className="w-6 h-6 shrink-0">
                                            <StockIcon symbol={r.symbol} size={24} />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-1.5">
                                                <span className={cn('text-xs font-bold', active ? 'text-white' : 'text-foreground')}>
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
                </div>

                {/* Footer */}
                <div className="border-t border-border px-4 py-2 flex items-center justify-between text-[10px] text-muted-foreground">
                    <span>↑↓ navigate · Enter to open</span>
                    <span className="font-semibold">Investa</span>
                </div>
            </div>
        </div>
    );
}
