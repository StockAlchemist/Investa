"use client";

import React, { useState, useEffect } from 'react';
import { fetchSettings, updateSettings, triggerRefresh, fetchHoldings, Settings as SettingsType, ManualOverride, ManualOverrideData } from '../lib/api';
import { COUNTRIES, ALL_INDUSTRIES } from '../lib/constants';

type Tab = 'overrides' | 'mapping' | 'excluded';

// --- Constants (Mirrored from config.py) ---
const ASSET_TYPES = [
    "",
    "STOCK",
    "ETF",
    "MUTUALFUND",
    "CURRENCY",
    "INDEX",
    "FUTURE",
    "OPTION",
    "CRYPTOCURRENCY",
    "Other",
];

const SECTORS = [
    "",
    "Other",
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
    "Exchange-Traded Fund",
];

export default function Settings() {
    const [settings, setSettings] = useState<SettingsType | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<Tab>('overrides');

    // Webhook state
    const [refreshSecret, setRefreshSecret] = useState('');
    const [refreshStatus, setRefreshStatus] = useState<string | null>(null);

    // Form states
    const [overrideSymbol, setOverrideSymbol] = useState('');
    const [overridePrice, setOverridePrice] = useState('');
    const [overrideAssetType, setOverrideAssetType] = useState('');
    const [overrideSector, setOverrideSector] = useState('');
    const [overrideGeo, setOverrideGeo] = useState('');
    const [overrideIndustry, setOverrideIndustry] = useState('');

    const [mapFrom, setMapFrom] = useState('');
    const [mapTo, setMapTo] = useState('');

    const [excludeSymbol, setExcludeSymbol] = useState('');

    // Dynamic lists based on portfolio data
    const [portfolioCountries, setPortfolioCountries] = useState<string[]>([]);

    // Common input/select classes
    const inputClassName = "w-full rounded-md border border-border bg-secondary text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500 px-3 py-2 text-sm outline-none focus:ring-1";
    const labelClassName = "block text-xs font-medium text-muted-foreground mb-1 uppercase tracking-wide";

    useEffect(() => {
        loadSettings();
    }, []);

    const loadSettings = async (isRefreshing = false) => {
        try {
            if (!isRefreshing) setLoading(true);
            const [data, holdings] = await Promise.all([
                fetchSettings(),
                fetchHoldings()
            ]);

            setSettings(data);

            // Extract unique used countries from holdings
            const usedCountries = new Set<string>();
            holdings.forEach(h => {
                // Ensure Country exists and is not empty or "N/A"
                if (h.Country && h.Country !== 'N/A') {
                    usedCountries.add(h.Country);
                }
            });
            setPortfolioCountries(Array.from(usedCountries).sort());

            setError(null);
        } catch (err) {
            setError('Failed to load settings');
            console.error(err);
        } finally {
            if (!isRefreshing) setLoading(false);
        }
    };



    const handleEdit = (symbol: string, data: ManualOverride) => {
        setOverrideSymbol(symbol);

        if (typeof data === 'number') {
            setOverridePrice(data.toString());
            setOverrideAssetType('');
            setOverrideSector('');
            setOverrideGeo('');
            setOverrideIndustry('');
        } else {
            setOverridePrice(data.price ? data.price.toString() : '');
            setOverrideAssetType(data.asset_type || '');
            setOverrideSector(data.sector || '');
            setOverrideGeo(data.geography || '');
            setOverrideIndustry(data.industry || '');
        }

        // Scroll to top of settings container to show the form
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // --- Actions ---

    const addOverride = async () => {
        const hasMetadata = overrideAssetType || overrideSector || overrideGeo || overrideIndustry;
        if (!settings || !overrideSymbol || (!overridePrice && !hasMetadata)) return;

        let price: number | null = null;
        if (overridePrice) {
            const parsed = parseFloat(overridePrice);
            if (isNaN(parsed)) return;
            price = parsed;
        }

        const currentOverrides = settings.manual_overrides || {};

        // Reconstruct the full dictionary in the format the API expects
        // API expects Record<string, ManualOverride>
        const newOverrides: Record<string, ManualOverride> = { ...currentOverrides };

        const newData: ManualOverrideData = {
            price: price !== null ? price : 0,
            asset_type: overrideAssetType || undefined,
            sector: overrideSector || undefined,
            geography: overrideGeo || undefined,
            industry: overrideIndustry || undefined
        };

        // Handling the price=0 case if user meant "no price override".
        if (price === null) {
            const existing = currentOverrides[overrideSymbol.toUpperCase()];
            if (existing) {
                if (typeof existing === 'number') {
                    newData.price = existing;
                } else {
                    newData.price = existing.price;
                }
            }
        }

        newOverrides[overrideSymbol.toUpperCase()] = newData;

        const cleanedOverrides: Record<string, ManualOverride> = {};
        Object.entries(newOverrides).forEach(([k, v]) => {
            if (typeof v === 'number') {
                cleanedOverrides[k] = v;
            } else {
                const { ...rest } = v;
                if ('currency' in rest) delete rest.currency;
                // Clean undefined
                if (rest.asset_type === undefined) delete rest.asset_type;
                if (rest.sector === undefined) delete rest.sector;
                if (rest.geography === undefined) delete rest.geography;
                if (rest.industry === undefined) delete rest.industry;

                cleanedOverrides[k] = rest;
            }
        });

        try {
            await updateSettings({ manual_price_overrides: cleanedOverrides });

            // Reset form
            setOverrideSymbol('');
            setOverridePrice('');
            setOverrideAssetType('');
            setOverrideSector('');
            setOverrideGeo('');
            setOverrideIndustry('');

            await loadSettings(true);
            await loadSettings(true);
        } catch {
            alert('Failed to save override');
        }
    };

    const removeOverride = async (symbol: string) => {
        if (!settings) return;
        const currentOverrides = settings.manual_overrides || {};
        const cleanedOverrides: Record<string, ManualOverride> = {};

        Object.entries(currentOverrides).forEach(([k, v]) => {
            if (k !== symbol) {
                if (typeof v === 'number') {
                    cleanedOverrides[k] = v;
                } else {
                    const { ...rest } = v;
                    if ('currency' in rest) delete rest.currency;
                    cleanedOverrides[k] = rest;
                }
            }
        });

        try {
            await updateSettings({ manual_price_overrides: cleanedOverrides });
            await loadSettings(true);
        } catch {
            alert('Failed to remove override');
        }
    };

    const addMapping = async () => {
        if (!settings || !mapFrom || !mapTo) return;

        const currentMap = { ...settings.user_symbol_map };
        currentMap[mapFrom.toUpperCase()] = mapTo.toUpperCase();

        try {
            await updateSettings({ user_symbol_map: currentMap });
            setMapFrom('');
            setMapTo('');
            await loadSettings(true);
        } catch {
            alert('Failed to save mapping');
        }
    };

    const removeMapping = async (fromSymbol: string) => {
        if (!settings) return;
        const currentMap = { ...settings.user_symbol_map };
        delete currentMap[fromSymbol];

        try {
            await updateSettings({ user_symbol_map: currentMap });
            await loadSettings(true);
        } catch {
            alert('Failed to remove mapping');
        }
    };

    const addExcluded = async () => {
        if (!settings || !excludeSymbol) return;

        const currentList = [...(settings.user_excluded_symbols || [])];
        const sym = excludeSymbol.toUpperCase();
        if (!currentList.includes(sym)) {
            currentList.push(sym);
        }

        try {
            await updateSettings({ user_excluded_symbols: currentList });
            setExcludeSymbol('');
            await loadSettings(true);
        } catch {
            alert('Failed to add excluded symbol');
        }
    };

    const removeExcluded = async (symbol: string) => {
        if (!settings) return;
        const currentList = (settings.user_excluded_symbols || []).filter(s => s !== symbol);

        try {
            await updateSettings({ user_excluded_symbols: currentList });
            await loadSettings(true);
        } catch {
            alert('Failed to remove excluded symbol');
        }
    };

    const handleRefresh = async () => {
        try {
            const res = await triggerRefresh(refreshSecret);
            setRefreshStatus(res.message || null);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            setRefreshStatus(`Error: ${message}`);
        }
    }

    if (loading) return <div className="p-12 text-center text-gray-500">Loading settings...</div>;
    if (error) return <div className="p-12 text-center text-red-500">{error}</div>;

    const overrides = settings?.manual_overrides || {};
    const symbolMap = settings?.user_symbol_map || {};
    // Sort excluded symbols alphabetically
    const excluded = (settings?.user_excluded_symbols || []).slice().sort((a, b) => a.localeCompare(b));

    // Construct Geography Options: Portfolio Countries first, then ALL countries
    const availableCountries = COUNTRIES.filter(c => !portfolioCountries.includes(c));

    return (
        <div className="space-y-8 pb-20 max-w-6xl mx-auto">

            {/* Symbol Settings Section */}
            <div className="bg-card backdrop-blur-md shadow-sm rounded-xl overflow-hidden border border-border">
                <div className="px-6 py-4 border-b border-border">
                    <h2 className="text-xl font-bold text-foreground">Symbol Settings</h2>
                    <p className="text-sm text-muted-foreground mt-1">
                        Manage price overrides, custom symbol mappings, and exclusions.
                    </p>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-border bg-secondary">
                    <button
                        onClick={() => setActiveTab('overrides')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 ${activeTab === 'overrides'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/10 dark:hover:border-white/10'
                            }`}
                    >
                        Manual Overrides
                    </button>
                    <button
                        onClick={() => setActiveTab('mapping')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 ${activeTab === 'mapping'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/20 dark:hover:border-white/20'
                            }`}
                    >
                        Symbol Mapping
                    </button>
                    <button
                        onClick={() => setActiveTab('excluded')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 ${activeTab === 'excluded'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/20 dark:hover:border-white/20'
                            }`}
                    >
                        Excluded Symbols
                    </button>
                </div>

                <div className="p-6 min-h-[400px]">

                    {/* Manual Price Overrides Tab */}
                    {activeTab === 'overrides' && (
                        <div>
                            <p className="mb-4 text-sm text-gray-500 dark:text-gray-400">
                                Set manual prices or metadata (Asset Type, Sector, etc.) to override automatic data.
                            </p>
                            <div className="flex flex-col gap-4 mb-6 bg-secondary p-4 rounded-lg border border-border">
                                <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
                                    <div className="col-span-1">
                                        <label className={labelClassName}>Symbol</label>
                                        <input
                                            type="text"
                                            value={overrideSymbol}
                                            onChange={(e) => setOverrideSymbol(e.target.value.toUpperCase())}
                                            placeholder="e.g. AAPL"
                                            className={inputClassName}
                                        />
                                    </div>
                                    <div className="col-span-1">
                                        <label className={labelClassName}>Price</label>
                                        <input
                                            type="number"
                                            step="0.0001"
                                            value={overridePrice}
                                            onChange={(e) => setOverridePrice(e.target.value)}
                                            placeholder="0.00"
                                            className={inputClassName}
                                        />
                                    </div>
                                    <div className="col-span-1">
                                        <label className={labelClassName}>Asset Type</label>
                                        <select
                                            value={overrideAssetType}
                                            onChange={(e) => setOverrideAssetType(e.target.value)}
                                            className={inputClassName}
                                        >
                                            {ASSET_TYPES.map(t => <option key={t} value={t} className="bg-white dark:bg-black text-foreground">{t || "Select..."}</option>)}
                                        </select>
                                    </div>
                                    <div className="col-span-1">
                                        <label className={labelClassName}>Sector</label>
                                        <select
                                            value={overrideSector}
                                            onChange={(e) => setOverrideSector(e.target.value)}
                                            className={inputClassName}
                                        >
                                            {SECTORS.map(s => <option key={s} value={s} className="bg-white dark:bg-black text-foreground">{s || "Select..."}</option>)}
                                        </select>
                                    </div>
                                    <div className="col-span-1">
                                        <label className={labelClassName}>Geography</label>
                                        <select
                                            value={overrideGeo}
                                            onChange={(e) => setOverrideGeo(e.target.value)}
                                            className={inputClassName}
                                        >
                                            <option value="" className="bg-white dark:bg-black text-foreground">Select...</option>

                                            {/* Portfolio Countries Section */}
                                            {portfolioCountries.length > 0 && (
                                                <optgroup label="In Portfolio" className="bg-white dark:bg-black text-foreground">
                                                    {portfolioCountries.map(c => (
                                                        <option key={c} value={c} className="bg-white dark:bg-black text-foreground">{c}</option>
                                                    ))}
                                                </optgroup>
                                            )}

                                            {/* All Countries Section */}
                                            <optgroup label="All Countries" className="bg-white dark:bg-black text-foreground">
                                                {availableCountries.map(c => (
                                                    <option key={c} value={c} className="bg-white dark:bg-black text-foreground">{c}</option>
                                                ))}
                                            </optgroup>
                                        </select>
                                    </div>
                                    <div className="col-span-1">
                                        <label className={labelClassName}>Industry</label>
                                        <select
                                            value={overrideIndustry}
                                            onChange={(e) => setOverrideIndustry(e.target.value)}
                                            className={inputClassName}
                                        >
                                            <option value="" className="bg-white dark:bg-black text-foreground">Select...</option>
                                            {ALL_INDUSTRIES.map(i => (
                                                <option key={i} value={i} className="bg-white dark:bg-black text-foreground">{i}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>
                                <div className="flex justify-end mt-2">
                                    <button
                                        type="button"
                                        onClick={addOverride}
                                        disabled={!overrideSymbol || (!overridePrice && !overrideAssetType && !overrideSector && !overrideGeo && !overrideIndustry)}
                                        className="w-full md:w-auto px-6 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Set Override
                                    </button>
                                </div>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-border text-sm">
                                    <thead className="bg-secondary">
                                        <tr>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground uppercase tracking-wider">Symbol</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground uppercase tracking-wider">Price</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground uppercase tracking-wider">Asset Type</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground uppercase tracking-wider">Sector</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground uppercase tracking-wider">Geography</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-muted-foreground uppercase tracking-wider">Industry</th>
                                            <th scope="col" className="px-4 py-3 text-right font-medium text-muted-foreground uppercase tracking-wider">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-border">
                                        {Object.entries(overrides).length === 0 ? (
                                            <tr>
                                                <td colSpan={7} className="px-6 py-12 text-center text-muted-foreground italic">
                                                    No manual overrides defined.
                                                </td>
                                            </tr>
                                        ) : (
                                            Object.entries(overrides)
                                                .sort((a, b) => a[0].localeCompare(b[0]))
                                                .map(([symbol, data]) => {
                                                    const isObj = typeof data !== 'number';
                                                    const price = isObj ? (data as ManualOverrideData).price : (data as number);
                                                    const assetType = isObj ? (data as ManualOverrideData).asset_type : '';
                                                    const sector = isObj ? (data as ManualOverrideData).sector : '';
                                                    const geo = isObj ? (data as ManualOverrideData).geography : '';
                                                    const industry = isObj ? (data as ManualOverrideData).industry : '';
                                                    const currency = isObj ? (data as ManualOverrideData).currency : 'USD';

                                                    return (
                                                        <tr key={symbol} className="hover:bg-accent/5 transition-colors">
                                                            <td className="px-4 py-4 whitespace-nowrap font-medium text-foreground">{symbol}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-muted-foreground font-mono">
                                                                {price === 0
                                                                    ? <span className="text-muted-foreground">-</span>
                                                                    : `${currency === 'THB' ? '฿' : '$'}${price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`
                                                                }
                                                            </td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-muted-foreground">{assetType || '-'}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-muted-foreground">{sector || '-'}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-muted-foreground">{geo || '-'}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-muted-foreground">{industry || '-'}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-right font-medium">
                                                                <button
                                                                    type="button"
                                                                    onClick={() => handleEdit(symbol, data)}
                                                                    className="text-cyan-500 hover:text-cyan-400 hover:bg-cyan-500/10 px-3 py-1 rounded transition-colors mr-2"
                                                                >
                                                                    Edit
                                                                </button>
                                                                <button
                                                                    type="button"
                                                                    onClick={() => removeOverride(symbol)}
                                                                    className="text-rose-500 hover:text-rose-400 hover:bg-rose-500/10 px-3 py-1 rounded transition-colors"
                                                                >
                                                                    Remove
                                                                </button>
                                                            </td>
                                                        </tr>
                                                    );
                                                })
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Symbol Mapping Tab */}
                    {activeTab === 'mapping' && (
                        <div>
                            <p className="mb-4 text-sm text-muted-foreground">
                                Map standard symbols in your portfolio to specific Yahoo Finance tickers for data retrieval.
                            </p>
                            <div className="flex flex-col md:flex-row gap-4 mb-6 bg-secondary p-4 rounded-lg border border-border">
                                <div className="flex-1">
                                    <label className={labelClassName}>Portfolio Symbol</label>
                                    <input
                                        type="text"
                                        value={mapFrom}
                                        onChange={(e) => setMapFrom(e.target.value.toUpperCase())}
                                        placeholder="e.g. MY-FUND"
                                        className={inputClassName}
                                    />
                                </div>
                                <div className="flex items-center justify-center pt-6">
                                    <span className="text-muted-foreground font-bold">→</span>
                                </div>
                                <div className="flex-1">
                                    <label className={labelClassName}>Yahoo Finance Symbol</label>
                                    <input
                                        type="text"
                                        value={mapTo}
                                        onChange={(e) => setMapTo(e.target.value.toUpperCase())}
                                        placeholder="e.g. VTSAX"
                                        className={inputClassName}
                                    />
                                </div>
                                <div className="flex items-end">
                                    <button
                                        type="button"
                                        onClick={addMapping}
                                        disabled={!mapFrom || !mapTo}
                                        className="w-full md:w-auto px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Add Mapping
                                    </button>
                                </div>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-border">
                                    <thead className="bg-secondary">
                                        <tr>
                                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Portfolio Symbol</th>
                                            <th scope="col" className="px-6 py-3 text-center text-xs font-medium text-muted-foreground uppercase tracking-wider">Mapped To</th>
                                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">YFinance Ticker</th>
                                            <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-muted-foreground uppercase tracking-wider">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-border">
                                        {Object.entries(symbolMap).length === 0 ? (
                                            <tr>
                                                <td colSpan={4} className="px-6 py-12 text-center text-muted-foreground italic">
                                                    No symbol mappings defined.
                                                </td>
                                            </tr>
                                        ) : (
                                            Object.entries(symbolMap)
                                                .sort((a, b) => a[0].localeCompare(b[0]))
                                                .map(([from, to]: [string, string]) => (
                                                    <tr key={from} className="hover:bg-accent/5 transition-colors">
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-foreground">{from}</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-center text-muted-foreground">→</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-foreground font-mono bg-accent/10 px-2 rounded w-min">{to}</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                                            <button
                                                                type="button"
                                                                onClick={() => removeMapping(from)}
                                                                className="text-rose-500 hover:text-rose-400 hover:bg-rose-500/10 px-3 py-1 rounded transition-colors"
                                                            >
                                                                Remove
                                                            </button>
                                                        </td>
                                                    </tr>
                                                ))
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Excluded Symbols Tab */}
                    {activeTab === 'excluded' && (
                        <div>
                            <p className="mb-4 text-sm text-muted-foreground">
                                Symbols listed here will be ignored by the market data provider and analysis.
                            </p>
                            <div className="flex flex-col md:flex-row gap-4 mb-6 bg-secondary p-4 rounded-lg border border-border">
                                <div className="flex-1">
                                    <label className={labelClassName}>Symbol to Exclude</label>
                                    <input
                                        type="text"
                                        value={excludeSymbol}
                                        onChange={(e) => setExcludeSymbol(e.target.value.toUpperCase())}
                                        placeholder="e.g. TEST-SYM"
                                        className={inputClassName}
                                    />
                                </div>
                                <div className="flex items-end">
                                    <button
                                        type="button"
                                        onClick={addExcluded}
                                        disabled={!excludeSymbol}
                                        className="w-full md:w-auto px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-rose-600 hover:bg-rose-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-rose-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                    >
                                        Exclude Symbol
                                    </button>
                                </div>
                            </div>

                            <div className="overflow-hidden bg-secondary border border-border rounded-lg">
                                {excluded.length === 0 ? (
                                    <div className="p-8 text-center text-muted-foreground italic">
                                        No excluded symbols.
                                    </div>
                                ) : (
                                    <ul className="divide-y divide-border">
                                        {excluded.map((sym, idx) => (
                                            <li key={sym + idx} className="px-6 py-4 flex items-center justify-between hover:bg-accent/5 transition-colors">
                                                <span className="text-sm font-medium text-foreground">{sym}</span>
                                                <button
                                                    type="button"
                                                    onClick={() => removeExcluded(sym)}
                                                    className="text-muted-foreground hover:text-rose-500 hover:bg-rose-500/10 p-2 rounded-full transition-all"
                                                    title="Remove from exclusion list"
                                                >
                                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                                                    </svg>
                                                </button>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Webhook Connection (Existing) */}
            <div className="bg-card backdrop-blur-md shadow-sm rounded-xl p-6 border border-border border-l-4 border-l-cyan-500">
                <h2 className="text-lg font-bold mb-4 text-foreground">Webhook Integration</h2>
                <div className="space-y-4">
                    <p className="text-sm text-muted-foreground">
                        Trigger a data refresh externally (e.g., from a shortcut) using: <code className="bg-black/10 dark:bg-white/10 px-2 py-1 rounded text-xs text-cyan-500 dark:text-cyan-400">POST /api/webhook/refresh</code>
                    </p>
                    <div className="flex gap-2 max-w-md">
                        <input
                            type="text"
                            placeholder="Webhook Secret"
                            value={refreshSecret}
                            onChange={(e) => setRefreshSecret(e.target.value)}
                            className="flex-1 rounded-md border border-border bg-secondary shadow-sm focus:border-cyan-500 focus:ring-cyan-500 sm:text-sm text-foreground px-3 py-2 outline-none focus:ring-1"
                        />
                        <button
                            type="button"
                            onClick={handleRefresh}
                            className="px-4 py-2 border border-border rounded-md shadow-sm text-sm font-medium text-foreground bg-secondary hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors"
                        >
                            Test
                        </button>
                    </div>
                    {refreshStatus && (
                        <p className={`text-sm ${refreshStatus.startsWith('Error') ? 'text-rose-400' : 'text-emerald-400'}`}>
                            {refreshStatus}
                        </p>
                    )}
                </div>
            </div>
        </div >
    );
}
