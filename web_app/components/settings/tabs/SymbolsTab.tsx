import React, { useState, useMemo } from 'react';
import { ArrowLeftRight, Trash2, Search } from 'lucide-react';
import { Settings as SettingsType, updateSettings } from '../../../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { useAuth } from '../../../context/AuthContext';
import {
    cardClassName,
    cardHeadClassName,
    sectionTitleClassName,
    countBadgeClassName,
    labelClassName,
    inputClassName,
    primaryButtonClassName,
    chipActiveClassName
} from '../constants';

interface SymbolsTabProps {
    settings: SettingsType | null;
}

export const SymbolsTab: React.FC<SymbolsTabProps> = ({ settings }) => {
    const queryClient = useQueryClient();
    const { user } = useAuth();

    const [mapFrom, setMapFrom] = useState('');
    const [mapTo, setMapTo] = useState('');
    const [mappingSearch, setMappingSearch] = useState('');
    const [excludeSymbol, setExcludeSymbol] = useState('');

    const symbolMap = settings?.user_symbol_map || {};
    const excluded = settings?.user_excluded_symbols || [];

    const sortedMapEntries = useMemo(() => {
        return Object.entries(symbolMap)
            .sort((a, b) => a[0].localeCompare(b[0]))
            .filter(([from, to]) => {
                if (!mappingSearch.trim()) return true;
                const q = mappingSearch.toLowerCase();
                return from.toLowerCase().includes(q) || to.toLowerCase().includes(q);
            });
    }, [symbolMap, mappingSearch]);

    const addMapping = async () => {
        if (!mapFrom || !mapTo) return;
        try {
            const updated = { ...symbolMap, [mapFrom]: mapTo };
            await updateSettings({ user_symbol_map: updated });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
            setMapFrom('');
            setMapTo('');
        } catch {
            alert("Failed to save mapping");
        }
    };

    const removeMapping = async (from: string) => {
        try {
            const updated = { ...symbolMap };
            delete updated[from];
            await updateSettings({ user_symbol_map: updated });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
        } catch {
            alert("Failed to remove mapping");
        }
    };

    const addExcluded = async () => {
        if (!excludeSymbol) return;
        try {
            const updated = Array.from(new Set([...excluded, excludeSymbol]));
            await updateSettings({ user_excluded_symbols: updated });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
            setExcludeSymbol('');
        } catch {
            alert("Failed to exclude symbol");
        }
    };

    const removeExcluded = async (symbol: string) => {
        try {
            const updated = excluded.filter(s => s !== symbol);
            await updateSettings({ user_excluded_symbols: updated });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
        } catch {
            alert("Failed to remove excluded symbol");
        }
    };

    return (
        <div className="space-y-6 max-w-4xl">
            {/* Symbol Mapping Section */}
            <div className={cardClassName}>
                <div className={cardHeadClassName}>
                    <h3 className={sectionTitleClassName}>Add Symbol Mapping</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-4">Resolve custom or broker-specific tickers to a real Yahoo Finance symbol.</p>
                <div className="flex flex-col md:flex-row gap-4 items-end">
                    <div className="flex-1 w-full space-y-1">
                        <label className={labelClassName}>Portfolio Symbol</label>
                        <input
                            type="text"
                            value={mapFrom}
                            onChange={(e) => setMapFrom(e.target.value.toUpperCase())}
                            placeholder="e.g. MY-FUND"
                            className={inputClassName}
                        />
                    </div>
                    <div className="hidden md:flex pb-3 text-muted-foreground">
                        <ArrowLeftRight className="w-5 h-5 opacity-40" />
                    </div>
                    <div className="flex-1 w-full space-y-1">
                        <label className={labelClassName}>Yahoo Finance Ticker</label>
                        <input
                            type="text"
                            value={mapTo}
                            onChange={(e) => setMapTo(e.target.value.toUpperCase())}
                            placeholder="e.g. VTSAX"
                            className={inputClassName}
                        />
                    </div>
                    <button
                        type="button"
                        onClick={addMapping}
                        disabled={!mapFrom || !mapTo}
                        className={`${primaryButtonClassName} w-full md:w-auto`}
                    >
                        Map Symbol
                    </button>
                </div>
            </div>

            {/* Active Mappings Table */}
            <div className={`${cardClassName} !p-0`}>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-6 py-4 border-b border-border">
                    <div className="flex items-center gap-2.5">
                        <h3 className={sectionTitleClassName}>Active Mappings</h3>
                        <span className={countBadgeClassName}>{Object.entries(symbolMap).length}</span>
                    </div>
                    {Object.entries(symbolMap).length > 0 && (
                        <div className="relative max-w-xs w-full">
                            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                            <input
                                type="text"
                                placeholder="Filter mappings..."
                                value={mappingSearch}
                                onChange={(e) => setMappingSearch(e.target.value)}
                                className="w-full h-8 pl-8 pr-3 text-xs rounded-control border border-input bg-background text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            />
                        </div>
                    )}
                </div>
                <table className="min-w-full text-sm">
                    <thead className="bg-muted/60 border-b border-border">
                        <tr>
                            <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Portfolio Symbol</th>
                            <th className="px-6 py-3 text-center font-semibold text-muted-foreground uppercase tracking-wider text-xs w-16"></th>
                            <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Mapped Ticker</th>
                            <th className="px-6 py-3 text-right font-semibold text-muted-foreground uppercase tracking-wider text-xs">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                        {sortedMapEntries.length === 0 ? (
                            <tr>
                                <td colSpan={4} className="px-6 py-12 text-center text-muted-foreground">
                                    {Object.entries(symbolMap).length === 0 ? "No symbol mappings defined." : "No mappings match your search."}
                                </td>
                            </tr>
                        ) : (
                            sortedMapEntries.map(([from, to]: [string, string]) => (
                                <tr key={from} className="hover:bg-muted/50 transition-colors group">
                                    <td className="px-6 py-4 font-bold text-foreground font-mono">{from}</td>
                                    <td className="px-6 py-4 text-center text-muted-foreground">
                                        <ArrowLeftRight className="w-4 h-4 inline opacity-50" />
                                    </td>
                                    <td className="px-6 py-4 text-primary-ink font-mono font-medium">{to}</td>
                                    <td className="px-6 py-4 text-right">
                                        <button
                                            type="button"
                                            onClick={() => removeMapping(from)}
                                            className="p-2 text-down hover:bg-down/12 rounded-lg transition-colors opacity-0 group-hover:opacity-100 cursor-pointer"
                                            title={`Remove mapping for ${from}`}
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>

            {/* Excluded Symbols Section */}
            <div className={cardClassName}>
                <div className={cardHeadClassName}>
                    <h3 className={sectionTitleClassName}>Exclude a Symbol</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-4">Excluded symbols are skipped during portfolio calculations and data fetches.</p>
                <div className="flex gap-3 items-end">
                    <div className="flex-1 space-y-1.5">
                        <label className={labelClassName}>Symbol to Exclude</label>
                        <input
                            type="text"
                            value={excludeSymbol}
                            onChange={(e) => setExcludeSymbol(e.target.value.toUpperCase())}
                            placeholder="e.g. TEST-SYM"
                            className={inputClassName}
                        />
                    </div>
                    <button
                        type="button"
                        onClick={addExcluded}
                        disabled={!excludeSymbol}
                        className={primaryButtonClassName}
                    >
                        Exclude
                    </button>
                </div>
            </div>

            <div className={cardClassName}>
                <div className={cardHeadClassName}>
                    <h3 className={sectionTitleClassName}>Excluded Symbols</h3>
                    <span className={countBadgeClassName}>{excluded.length}</span>
                </div>
                {excluded.length === 0 ? (
                    <div className="py-10 text-center text-sm text-muted-foreground border border-dashed border-border rounded-inset">
                        No excluded symbols.
                    </div>
                ) : (
                    <div className="flex flex-wrap gap-2">
                        {excluded.map((sym, idx) => (
                            <div
                                key={sym + idx}
                                className={`group ${chipActiveClassName} hover:border-primary/40`}
                            >
                                <span className="font-bold">{sym}</span>
                                <button
                                    type="button"
                                    onClick={() => removeExcluded(sym)}
                                    className="opacity-40 group-hover:opacity-100 text-down hover:text-down transition-opacity cursor-pointer"
                                    aria-label={`Remove ${sym}`}
                                >
                                    <Trash2 className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};
