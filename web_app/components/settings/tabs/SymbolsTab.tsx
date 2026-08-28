import React, { useState } from 'react';
import { Map as MapIcon, ArrowRight, Trash2, XCircle } from 'lucide-react';
import { Settings as SettingsType, updateSettings } from '../../../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { useAuth } from '../../../context/AuthContext';
import {
    cardClassName,
    sectionTitleClassName,
    labelClassName,
    inputClassName,
    primaryButtonClassName
} from '../constants';

interface SymbolsTabProps {
    settings: SettingsType | null;
}

export const SymbolsTab: React.FC<SymbolsTabProps> = ({ settings }) => {
    const queryClient = useQueryClient();
    const { user } = useAuth();

    const [mapFrom, setMapFrom] = useState('');
    const [mapTo, setMapTo] = useState('');
    const [excludeSymbol, setExcludeSymbol] = useState('');

    const symbolMap = settings?.user_symbol_map || {};
    const excluded = settings?.user_excluded_symbols || [];

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
        <div className="space-y-8 max-w-4xl">
            {/* Symbol Mapping Section */}
            <div className={cardClassName}>
                <div className="mb-2">
                    <h3 className={sectionTitleClassName}>
                        <MapIcon className="w-5 h-5 text-blue-500" />
                        Add Symbol Mapping
                    </h3>
                </div>
                <p className="text-sm text-muted-foreground mb-5">Resolve custom or broker-specific tickers to a real Yahoo Finance symbol.</p>
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
                        <ArrowRight className="w-5 h-5 opacity-50" />
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
                        Map
                    </button>
                </div>
            </div>

            {/* Active Mappings Table */}
            <div className={`${cardClassName} !p-0`}>
                <div className="flex items-center justify-between px-6 py-4 border-b border-black/5 dark:border-white/5 bg-white/30 dark:bg-black/20">
                    <h3 className={sectionTitleClassName}>
                        <MapIcon className="w-5 h-5 text-blue-500" />
                        Active Mappings
                        <span className="text-xs font-medium text-muted-foreground bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-full ml-1">
                            {Object.entries(symbolMap).length}
                        </span>
                    </h3>
                </div>
                <table className="min-w-full text-sm">
                    <thead className="bg-black/5 dark:bg-white/5 border-b border-black/10 dark:border-white/10">
                        <tr>
                            <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Portfolio Symbol</th>
                            <th className="px-6 py-3 text-center font-semibold text-muted-foreground uppercase tracking-wider text-xs w-16"></th>
                            <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Mapped Ticker</th>
                            <th className="px-6 py-3 text-right font-semibold text-muted-foreground uppercase tracking-wider text-xs">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-black/5 dark:divide-white/5">
                        {Object.entries(symbolMap).length === 0 ? (
                            <tr>
                                <td colSpan={4} className="px-6 py-12 text-center text-muted-foreground">
                                    No symbol mappings defined.
                                </td>
                            </tr>
                        ) : (
                            Object.entries(symbolMap)
                                .sort((a, b) => a[0].localeCompare(b[0]))
                                .map(([from, to]: [string, string]) => (
                                    <tr key={from} className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors group">
                                        <td className="px-6 py-4 font-bold text-foreground">{from}</td>
                                        <td className="px-6 py-4 text-center text-muted-foreground">
                                            <ArrowRight className="w-4 h-4 inline opacity-50" />
                                        </td>
                                        <td className="px-6 py-4 text-blue-600 dark:text-blue-400 font-mono font-medium">{to}</td>
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
                <div className="mb-2">
                    <h3 className={sectionTitleClassName}>
                        <XCircle className="w-5 h-5 text-rose-500" />
                        Exclude a Symbol
                    </h3>
                </div>
                <p className="text-sm text-muted-foreground mb-5">Excluded symbols are skipped during portfolio calculations and data fetches.</p>
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
                        className="px-6 py-2.5 bg-rose-500 hover:bg-rose-600 text-white rounded-xl font-medium shadow-sm transition-colors disabled:opacity-50 cursor-pointer"
                    >
                        Exclude
                    </button>
                </div>
            </div>

            <div className={cardClassName}>
                <h3 className={`${sectionTitleClassName} mb-5`}>
                    <XCircle className="w-5 h-5 text-rose-500" />
                    Excluded Symbols
                    <span className="text-xs font-medium text-muted-foreground bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-full ml-1">
                        {excluded.length}
                    </span>
                </h3>
                {excluded.length === 0 ? (
                    <div className="py-10 text-center text-muted-foreground border border-dashed border-black/10 dark:border-white/10 rounded-xl">
                        No excluded symbols.
                    </div>
                ) : (
                    <div className="flex flex-wrap gap-2">
                        {excluded.map((sym, idx) => (
                            <div
                                key={sym + idx}
                                className="group inline-flex items-center gap-2 bg-rose-500/10 border border-rose-500/30 hover:border-rose-500/50 px-3 py-1.5 rounded-lg transition-colors"
                            >
                                <span className="font-bold font-mono text-down text-sm">{sym}</span>
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
