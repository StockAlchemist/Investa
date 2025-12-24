"use client";

import React, { useState, useEffect } from 'react';
import { fetchSettings, updateSettings, triggerRefresh, fetchHoldings, Settings as SettingsType, ManualOverride, ManualOverrideData } from '../lib/api';
import { COUNTRIES, ALL_INDUSTRIES, SECTOR_INDUSTRY_MAP } from '../lib/constants';

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

    useEffect(() => {
        loadSettings();
    }, []);

    const loadSettings = async () => {
        try {
            setLoading(true);
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
            setLoading(false);
        }
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

            await loadSettings();
        } catch (err) {
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
            await loadSettings();
        } catch (err) {
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
            await loadSettings();
        } catch (err) {
            alert('Failed to save mapping');
        }
    };

    const removeMapping = async (fromSymbol: string) => {
        if (!settings) return;
        const currentMap = { ...settings.user_symbol_map };
        delete currentMap[fromSymbol];

        try {
            await updateSettings({ user_symbol_map: currentMap });
            await loadSettings();
        } catch (err) {
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
            await loadSettings();
        } catch (err) {
            alert('Failed to add excluded symbol');
        }
    };

    const removeExcluded = async (symbol: string) => {
        if (!settings) return;
        const currentList = (settings.user_excluded_symbols || []).filter(s => s !== symbol);

        try {
            await updateSettings({ user_excluded_symbols: currentList });
            await loadSettings();
        } catch (err) {
            alert('Failed to remove excluded symbol');
        }
    };

    const handleRefresh = async () => {
        try {
            const res = await triggerRefresh(refreshSecret);
            setRefreshStatus(res.message);
        } catch (err: any) {
            setRefreshStatus(`Error: ${err.message}`);
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
            <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">Symbol Settings</h2>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        Manage price overrides, custom symbol mappings, and exclusions.
                    </p>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
                    <button
                        onClick={() => setActiveTab('overrides')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors ${activeTab === 'overrides'
                            ? 'bg-white dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 border-t-2 border-indigo-600'
                            : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                            }`}
                    >
                        Manual Overrides
                    </button>
                    <button
                        onClick={() => setActiveTab('mapping')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors ${activeTab === 'mapping'
                            ? 'bg-white dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 border-t-2 border-indigo-600'
                            : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                            }`}
                    >
                        Symbol Mapping
                    </button>
                    <button
                        onClick={() => setActiveTab('excluded')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors ${activeTab === 'excluded'
                            ? 'bg-white dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 border-t-2 border-indigo-600'
                            : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
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
                            <div className="flex flex-col gap-4 mb-6 bg-gray-50 dark:bg-gray-700/30 p-4 rounded-lg border border-gray-100 dark:border-gray-700">
                                <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
                                    <div className="col-span-1">
                                        <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">Symbol</label>
                                        <input
                                            type="text"
                                            value={overrideSymbol}
                                            onChange={(e) => setOverrideSymbol(e.target.value.toUpperCase())}
                                            placeholder="e.g. AAPL"
                                            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2 text-sm"
                                        />
                                    </div>
                                    <div className="col-span-1">
                                        <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">Price</label>
                                        <input
                                            type="number"
                                            step="0.0001"
                                            value={overridePrice}
                                            onChange={(e) => setOverridePrice(e.target.value)}
                                            placeholder="0.00"
                                            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2 text-sm"
                                        />
                                    </div>
                                    <div className="col-span-1">
                                        <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">Asset Type</label>
                                        <select
                                            value={overrideAssetType}
                                            onChange={(e) => setOverrideAssetType(e.target.value)}
                                            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2 text-sm"
                                        >
                                            {ASSET_TYPES.map(t => <option key={t} value={t}>{t || "Select..."}</option>)}
                                        </select>
                                    </div>
                                    <div className="col-span-1">
                                        <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">Sector</label>
                                        <select
                                            value={overrideSector}
                                            onChange={(e) => setOverrideSector(e.target.value)}
                                            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2 text-sm"
                                        >
                                            {SECTORS.map(s => <option key={s} value={s}>{s || "Select..."}</option>)}
                                        </select>
                                    </div>
                                    <div className="col-span-1">
                                        <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">Geography</label>
                                        <select
                                            value={overrideGeo}
                                            onChange={(e) => setOverrideGeo(e.target.value)}
                                            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2 text-sm"
                                        >
                                            <option value="">Select...</option>

                                            {/* Portfolio Countries Section */}
                                            {portfolioCountries.length > 0 && (
                                                <optgroup label="In Portfolio">
                                                    {portfolioCountries.map(c => (
                                                        <option key={c} value={c}>{c}</option>
                                                    ))}
                                                </optgroup>
                                            )}

                                            {/* All Countries Section */}
                                            <optgroup label="All Countries">
                                                {availableCountries.map(c => (
                                                    <option key={c} value={c}>{c}</option>
                                                ))}
                                            </optgroup>
                                        </select>
                                    </div>
                                    <div className="col-span-1">
                                        <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">Industry</label>
                                        <select
                                            value={overrideIndustry}
                                            onChange={(e) => setOverrideIndustry(e.target.value)}
                                            className="w-full rounded-md border border-gray-300 bg-white text-gray-900 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2 text-sm"
                                        >
                                            <option value="">Select...</option>
                                            {ALL_INDUSTRIES.map(i => (
                                                <option key={i} value={i}>{i}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>
                                <div className="flex justify-end mt-2">
                                    <button
                                        onClick={addOverride}
                                        disabled={!overrideSymbol || (!overridePrice && !overrideAssetType && !overrideSector && !overrideGeo && !overrideIndustry)}
                                        className="w-full md:w-auto px-6 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                    >
                                        Set Override
                                    </button>
                                </div>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700 text-sm">
                                    <thead className="bg-gray-50 dark:bg-gray-800">
                                        <tr>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Symbol</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Price</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Asset Type</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Sector</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Geography</th>
                                            <th scope="col" className="px-4 py-3 text-left font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Industry</th>
                                            <th scope="col" className="px-4 py-3 text-right font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                                        {Object.entries(overrides).length === 0 ? (
                                            <tr>
                                                <td colSpan={7} className="px-6 py-12 text-center text-gray-500 dark:text-gray-400 italic">
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
                                                        <tr key={symbol} className="hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
                                                            <td className="px-4 py-4 whitespace-nowrap font-medium text-gray-900 dark:text-white">{symbol}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-gray-500 dark:text-gray-300 font-mono">
                                                                {price === 0
                                                                    ? <span className="text-gray-400">-</span>
                                                                    : `${currency === 'THB' ? '฿' : '$'}${price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`
                                                                }
                                                            </td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-gray-500 dark:text-gray-300">{assetType || '-'}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-gray-500 dark:text-gray-300">{sector || '-'}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-gray-500 dark:text-gray-300">{geo || '-'}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-gray-500 dark:text-gray-300">{industry || '-'}</td>
                                                            <td className="px-4 py-4 whitespace-nowrap text-right font-medium">
                                                                <button
                                                                    onClick={() => removeOverride(symbol)}
                                                                    className="text-red-600 hover:text-red-900 dark:hover:text-red-400 bg-red-50 dark:bg-transparent px-3 py-1 rounded hover:bg-red-100 transition-colors"
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
                            <p className="mb-4 text-sm text-gray-500 dark:text-gray-400">
                                Map standard symbols in your portfolio to specific Yahoo Finance tickers for data retrieval.
                            </p>
                            <div className="flex flex-col md:flex-row gap-4 mb-6 bg-gray-50 dark:bg-gray-700/30 p-4 rounded-lg border border-gray-100 dark:border-gray-700">
                                <div className="flex-1">
                                    <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">Portfolio Symbol</label>
                                    <input
                                        type="text"
                                        value={mapFrom}
                                        onChange={(e) => setMapFrom(e.target.value.toUpperCase())}
                                        placeholder="e.g. MY-FUND"
                                        className="w-full rounded-md border border-gray-300 bg-white text-gray-900 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2 text-sm"
                                    />
                                </div>
                                <div className="flex items-center justify-center pt-6">
                                    <span className="text-gray-400 font-bold">→</span>
                                </div>
                                <div className="flex-1">
                                    <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">Yahoo Finance Symbol</label>
                                    <input
                                        type="text"
                                        value={mapTo}
                                        onChange={(e) => setMapTo(e.target.value.toUpperCase())}
                                        placeholder="e.g. VTSAX"
                                        className="w-full rounded-md border border-gray-300 bg-white text-gray-900 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2 text-sm"
                                    />
                                </div>
                                <div className="flex items-end">
                                    <button
                                        onClick={addMapping}
                                        disabled={!mapFrom || !mapTo}
                                        className="w-full md:w-auto px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                    >
                                        Add Mapping
                                    </button>
                                </div>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                                    <thead className="bg-gray-50 dark:bg-gray-800">
                                        <tr>
                                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Portfolio Symbol</th>
                                            <th scope="col" className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Mapped To</th>
                                            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">YFinance Ticker</th>
                                            <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                                        {Object.entries(symbolMap).length === 0 ? (
                                            <tr>
                                                <td colSpan={4} className="px-6 py-12 text-center text-gray-500 dark:text-gray-400 italic">
                                                    No symbol mappings defined.
                                                </td>
                                            </tr>
                                        ) : (
                                            Object.entries(symbolMap)
                                                .sort((a, b) => a[0].localeCompare(b[0]))
                                                .map(([from, to]: [string, string]) => (
                                                    <tr key={from} className="hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{from}</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-center text-gray-400">→</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300 font-mono bg-gray-50 dark:bg-gray-900 px-2 rounded w-min">{to}</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                                            <button
                                                                onClick={() => removeMapping(from)}
                                                                className="text-red-600 hover:text-red-900 dark:hover:text-red-400 bg-red-50 dark:bg-transparent px-3 py-1 rounded hover:bg-red-100 transition-colors"
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
                            <p className="mb-4 text-sm text-gray-500 dark:text-gray-400">
                                Symbols listed here will be ignored by the market data provider and analysis.
                            </p>
                            <div className="flex flex-col md:flex-row gap-4 mb-6 bg-gray-50 dark:bg-gray-700/30 p-4 rounded-lg border border-gray-100 dark:border-gray-700">
                                <div className="flex-1">
                                    <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">Symbol to Exclude</label>
                                    <input
                                        type="text"
                                        value={excludeSymbol}
                                        onChange={(e) => setExcludeSymbol(e.target.value.toUpperCase())}
                                        placeholder="e.g. TEST-SYM"
                                        className="w-full rounded-md border border-gray-300 bg-white text-gray-900 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2 text-sm"
                                    />
                                </div>
                                <div className="flex items-end">
                                    <button
                                        onClick={addExcluded}
                                        disabled={!excludeSymbol}
                                        className="w-full md:w-auto px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-rose-600 hover:bg-rose-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-rose-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                                    >
                                        Exclude Symbol
                                    </button>
                                </div>
                            </div>

                            <div className="overflow-hidden bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
                                {excluded.length === 0 ? (
                                    <div className="p-8 text-center text-gray-500 dark:text-gray-400 italic">
                                        No excluded symbols.
                                    </div>
                                ) : (
                                    <ul className="divide-y divide-gray-200 dark:divide-gray-700">
                                        {excluded.map((sym, idx) => (
                                            <li key={sym + idx} className="px-6 py-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
                                                <span className="text-sm font-medium text-gray-900 dark:text-white">{sym}</span>
                                                <button
                                                    onClick={() => removeExcluded(sym)}
                                                    className="text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 p-2 rounded-full transition-all"
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
            <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6 border-l-4 border-indigo-500">
                <h2 className="text-lg font-bold mb-4 text-gray-900 dark:text-white">Webhook Integration</h2>
                <div className="space-y-4">
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                        Trigger a data refresh externally (e.g., from a shortcut) using: <code className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded text-xs">POST /api/webhook/refresh</code>
                    </p>
                    <div className="flex gap-2 max-w-md">
                        <input
                            type="text"
                            placeholder="Webhook Secret"
                            value={refreshSecret}
                            onChange={(e) => setRefreshSecret(e.target.value)}
                            className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2"
                        />
                        <button
                            onClick={handleRefresh}
                            className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:bg-gray-700 dark:text-white dark:border-gray-600 dark:hover:bg-gray-600"
                        >
                            Test
                        </button>
                    </div>
                    {refreshStatus && (
                        <p className={`text-sm ${refreshStatus.startsWith('Error') ? 'text-red-600' : 'text-green-600'}`}>
                            {refreshStatus}
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
}
