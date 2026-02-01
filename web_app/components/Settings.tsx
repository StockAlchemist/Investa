"use client";

import React, { useState, useMemo } from 'react';
import { Pencil, Trash2, Loader2 } from 'lucide-react';
import { updateSettings, triggerRefresh, clearCache, deleteUser, changePassword, syncIbkr, Settings as SettingsType, ManualOverride, ManualOverrideData, Holding } from '../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { COUNTRIES, ALL_INDUSTRIES } from '../lib/constants';
import AccountGroupManager from './AccountGroupManager';
import ManualValuationSettings from './ManualValuationSettings';

import { useAuth } from '../context/AuthContext';

type Tab = 'overrides' | 'mapping' | 'excluded' | 'groups' | 'currencies' | 'yield' | 'valuation' | 'account' | 'advanced';

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

interface SettingsProps {
    settings: SettingsType | null;
    holdings: Holding[];
    availableAccounts: string[];
    initialTab?: Tab;
}

export default function Settings({ settings, holdings, availableAccounts, initialTab }: SettingsProps) {
    const queryClient = useQueryClient();
    const { logout, user } = useAuth();
    // Removed local settings state, loading, error, portfolioCountries, availableAccounts (now props/derived)

    const [activeTab, setActiveTab] = useState<Tab>(initialTab || 'overrides');
    const [confirmClear, setConfirmClear] = useState(false);
    const [clearStatus, setClearStatus] = useState<string | null>(null);

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
    const [overrideExchange, setOverrideExchange] = useState('');

    const [mapFrom, setMapFrom] = useState('');
    const [mapTo, setMapTo] = useState('');

    const [excludeSymbol, setExcludeSymbol] = useState('');
    const [newCurrency, setNewCurrency] = useState('');

    // Password State
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [passwordStatus, setPasswordStatus] = useState<{ type: 'success' | 'error', message: string } | null>(null);
    const [isChangingPassword, setIsChangingPassword] = useState(false);

    // IBKR State
    const [ibkrToken, setIbkrToken] = useState(settings?.ibkr_token || '');
    const [ibkrQueryId, setIbkrQueryId] = useState(settings?.ibkr_query_id || '');
    const [isSyncing, setIsSyncing] = useState(false);
    const [syncStatus, setSyncStatus] = useState<string | null>(null);
    const [isSavingIbkr, setIsSavingIbkr] = useState(false);

    // Common input/select classes
    const inputClassName = "w-full rounded-md border border-border bg-secondary text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500 px-3 py-2 text-sm outline-none focus:ring-1";
    const labelClassName = "block text-xs font-medium text-muted-foreground mb-1 uppercase tracking-wide";

    // Derive portfolioCountries from holdings prop
    const portfolioCountries = useMemo(() => {
        const usedCountries = new Set<string>();
        holdings.forEach(h => {
            if (h.Country && h.Country !== 'N/A') {
                usedCountries.add(h.Country);
            }
        });
        return Array.from(usedCountries).sort();
    }, [holdings]);



    const handleEdit = (symbol: string, data: ManualOverride) => {
        setOverrideSymbol(symbol);

        if (typeof data === 'number') {
            setOverridePrice(data.toString());
            setOverrideAssetType('');
            setOverrideSector('');
            setOverrideGeo('');
            setOverrideIndustry('');
            setOverrideExchange('');
        } else {
            setOverridePrice(data.price ? data.price.toString() : '');
            setOverrideAssetType(data.asset_type || '');
            setOverrideSector(data.sector || '');
            setOverrideGeo(data.geography || '');
            setOverrideIndustry(data.industry || '');
            setOverrideExchange(data.exchange || '');
        }

        // Scroll to top of settings container to show the form
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // --- Actions ---

    const addOverride = async () => {
        const hasMetadata = overrideAssetType || overrideSector || overrideGeo || overrideIndustry || overrideExchange;
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
            industry: overrideIndustry || undefined,
            exchange: overrideExchange || undefined
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
                if (rest.exchange === undefined) delete rest.exchange;

                cleanedOverrides[k] = rest;
            }
        });

        try {
            await updateSettings({ manual_price_overrides: cleanedOverrides });
            await queryClient.invalidateQueries({ queryKey: ['settings'] });

            // Reset form
            setOverrideSymbol('');
            setOverridePrice('');
            setOverrideAssetType('');
            setOverrideSector('');
            setOverrideGeo('');
            setOverrideIndustry('');
            setOverrideExchange('');

            // Removed loadSettings call, parent will update
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
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
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
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
            setMapFrom('');
            setMapTo('');
            // Removed loadSettings call
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
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
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
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
            setExcludeSymbol('');
            // Removed loadSettings call
        } catch {
            alert('Failed to add excluded symbol');
        }
    };

    const removeExcluded = async (symbol: string) => {
        if (!settings) return;
        const currentList = (settings.user_excluded_symbols || []).filter(s => s !== symbol);

        try {
            await updateSettings({ user_excluded_symbols: currentList });
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
        } catch {
            alert('Failed to remove excluded symbol');
        }
    };

    const addCurrency = async () => {
        if (!settings || !newCurrency) return;
        const currentList = [...(settings.available_currencies || ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY'])];
        const curr = newCurrency.toUpperCase();
        if (!currentList.includes(curr)) {
            currentList.push(curr);
        }
        try {
            await updateSettings({ available_currencies: currentList });
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
            setNewCurrency('');
        } catch {
            alert('Failed to add currency');
        }
    }

    const removeCurrency = async (curr: string) => {
        if (!settings) return;
        const currentList = (settings.available_currencies || ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY']).filter(c => c !== curr);
        try {
            await updateSettings({ available_currencies: currentList });
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
        } catch {
            alert('Failed to remove currency');
        }
    }

    const updateAccountCurrency = async (account: string, currency: string) => {
        if (!settings) return;
        const currentMap = { ...settings.account_currency_map };
        currentMap[account] = currency;

        try {
            await updateSettings({ account_currency_map: currentMap });
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
        } catch {
            alert('Failed to update account currency');
        }
    }

    const updateAccountYield = async (account: string, rate: number, threshold: number) => {
        if (!settings) return;
        const currentRates = { ...(settings.account_interest_rates || {}) };
        const currentThresholds = { ...(settings.interest_free_thresholds || {}) };

        currentRates[account] = rate;
        currentThresholds[account] = threshold;

        try {
            await updateSettings({
                account_interest_rates: currentRates,
                interest_free_thresholds: currentThresholds
            });
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
        } catch {
            alert('Failed to update yield settings');
        }
    }

    const handleRefresh = async () => {
        try {
            const res = await triggerRefresh(refreshSecret);
            setRefreshStatus(res.message || null);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            setRefreshStatus(`Error: ${message}`);
        }
    }

    const handleClearCache = async () => {
        if (!confirmClear) {
            setConfirmClear(true);
            // Auto-reset after 3 seconds
            setTimeout(() => setConfirmClear(false), 3000);
            return;
        }

        try {
            setConfirmClear(false);
            setClearStatus("Clearing...");
            const res = await clearCache();
            setClearStatus(res.message || "Cache cleared successfully.");
            await queryClient.invalidateQueries(); // Refresh all data in the UI
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            setClearStatus(`Error: ${message}`);
        }
    };

    const handleDeleteAccount = async () => {
        if (!window.confirm("Are you sure you want to delete your account? This action cannot be undone and will delete all your data.")) return;
        if (!window.confirm("Please confirm again: DELETE ACCOUNT PERMANENTLY?")) return;

        try {
            await deleteUser();
            logout();
        } catch (err) {
            alert("Failed to delete account: " + String(err));
        }
    };

    const handleChangePassword = async (e: React.FormEvent) => {
        e.preventDefault();
        setPasswordStatus(null);

        if (newPassword !== confirmPassword) {
            setPasswordStatus({ type: 'error', message: "New passwords do not match" });
            return;
        }

        if (newPassword.length < 4) {
            setPasswordStatus({ type: 'error', message: "Password must be at least 4 characters" });
            return;
        }

        setIsChangingPassword(true);
        try {
            const res = await changePassword(currentPassword, newPassword);
            setPasswordStatus({ type: 'success', message: res.message || "Password changed successfully" });
            setCurrentPassword('');
            setNewPassword('');
            setConfirmPassword('');
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            setPasswordStatus({ type: 'error', message: message });
        } finally {
            setIsChangingPassword(false);
        }
    };

    const handleSaveIbkr = async () => {
        setIsSavingIbkr(true);
        try {
            await updateSettings({
                ibkr_token: ibkrToken,
                ibkr_query_id: ibkrQueryId
            });
            await queryClient.invalidateQueries({ queryKey: ['settings'] });
            setSyncStatus("Settings saved successfully.");
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Failed to save IBKR settings";
            setSyncStatus(`Error: ${message}`);
        } finally {
            setIsSavingIbkr(false);
            setTimeout(() => setSyncStatus(null), 5000);
        }
    }

    const handleSyncIbkr = async () => {
        setIsSyncing(true);
        setSyncStatus("Syncing with IBKR...");
        try {
            const res = await syncIbkr();
            setSyncStatus(res.message || "Sync complete");
            await queryClient.invalidateQueries(); // Refresh all data
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Sync failed";
            setSyncStatus(`Error: ${message}`);
        } finally {
            setIsSyncing(false);
            // Don't clear status immediately so user can see result
        }
    }

    const overrides = settings?.manual_overrides || {};
    const symbolMap = settings?.user_symbol_map || {};
    // Sort excluded symbols alphabetically
    const excluded = (settings?.user_excluded_symbols || []).slice().sort((a, b) => a.localeCompare(b));
    const availableCurrencies = (settings?.available_currencies || ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY']).slice().sort();
    const accountCurrencyMap = settings?.account_currency_map || {};

    // Construct Geography Options: Portfolio Countries first, then ALL countries
    const availableCountries = COUNTRIES.filter(c => !portfolioCountries.includes(c));

    return (
        <div className="space-y-8 pb-20 max-w-6xl mx-auto">

            {/* Symbol Settings Section */}
            <div className="bg-white dark:bg-zinc-950 shadow-sm rounded-xl overflow-hidden border border-border">
                <div className="px-6 py-4 border-b border-border flex justify-between items-center">
                    <div>
                        <h2 className="text-2xl font-bold leading-tight tracking-tight bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent w-fit pb-1">Settings</h2>
                        <p className="text-sm text-muted-foreground mt-1">
                            Manage application settings, preferences, and account.
                        </p>
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-border bg-secondary overflow-x-auto">
                    <button
                        onClick={() => setActiveTab('groups')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 whitespace-nowrap ${activeTab === 'groups'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/10 dark:hover:border-white/10'
                            }`}
                    >
                        Account Groups
                    </button>
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
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 whitespace-nowrap ${activeTab === 'mapping'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/20 dark:hover:border-white/20'
                            }`}
                    >
                        Symbol Mapping
                    </button>
                    <button
                        onClick={() => setActiveTab('excluded')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 whitespace-nowrap ${activeTab === 'excluded'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/20 dark:hover:border-white/20'
                            }`}
                    >
                        Excluded Symbols
                    </button>
                    <button
                        onClick={() => setActiveTab('currencies')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 whitespace-nowrap ${activeTab === 'currencies'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/20 dark:hover:border-white/20'
                            }`}
                    >
                        Currencies
                    </button>
                    <button
                        onClick={() => setActiveTab('yield')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 whitespace-nowrap ${activeTab === 'yield'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/20 dark:hover:border-white/20'
                            }`}
                    >
                        Cash Yield
                    </button>
                    <button
                        onClick={() => setActiveTab('valuation')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 whitespace-nowrap ${activeTab === 'valuation'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/20 dark:hover:border-white/20'
                            }`}
                    >
                        Valuation Overrides
                    </button>
                    <button
                        onClick={() => setActiveTab('advanced')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 whitespace-nowrap ${activeTab === 'advanced'
                            ? 'border-cyan-500 text-cyan-500 dark:text-cyan-400'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/20 dark:hover:border-white/20'
                            }`}
                    >
                        Advanced
                    </button>
                    <button
                        onClick={() => setActiveTab('account')}
                        className={`px-6 py-3 text-sm font-medium focus:outline-none transition-colors border-b-2 whitespace-nowrap ${activeTab === 'account'
                            ? 'border-red-500 text-red-500'
                            : 'border-transparent text-muted-foreground hover:text-foreground hover:border-black/20 dark:hover:border-white/20'
                            }`}
                    >
                        Account
                    </button>
                </div>

                <div className="p-6 min-h-[400px]">

                    {/* Loading State */}
                    {!settings && (
                        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                            <Loader2 className="w-8 h-8 animate-spin mb-2 text-cyan-500" />
                            <p>Loading settings...</p>
                        </div>
                    )}

                    {/* Account Groups Tab */}
                    {activeTab === 'groups' && settings && (
                        <AccountGroupManager
                            availableAccounts={availableAccounts}
                            onUpdate={() => queryClient.invalidateQueries({ queryKey: ['settings'] })}
                        />
                    )}

                    {/* Cash Yield Tab */}
                    {activeTab === 'yield' && settings && (
                        <div>
                            <p className="mb-4 text-sm text-gray-500 dark:text-gray-400">
                                Configure annual interest rates and interest-free thresholds for cash balances in each account.
                            </p>
                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-border text-sm">
                                    <thead className="bg-secondary/50 font-semibold border-b border-border">
                                        <tr>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Account</th>
                                            <th scope="col" className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Cash Balance</th>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Annual Interest Rate (%)</th>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Interest-Free Threshold</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-border">
                                        {availableAccounts
                                            .filter(account => {
                                                // Calculate cash balance for this account
                                                const accountCash = holdings
                                                    .filter(h => h.Account === account && (h.Symbol === '$CASH' || h.Symbol === 'Cash' || h.Symbol.includes('Cash')))
                                                    .reduce((sum, h) => {
                                                        // Use Market Value (Display) or approximate from Quantity * Price if not available
                                                        // Holding has dynamic keys, but usually "Market Value" is standard or "Market Value (USD)"
                                                        // Let's try to find a key starting with "Market Value"
                                                        const mvKey = Object.keys(h).find(k => k.startsWith('Market Value'));
                                                        const val = mvKey ? (h[mvKey] as number) : 0;
                                                        return sum + (val || 0);
                                                    }, 0);
                                                return Math.abs(accountCash) > 0.01; // Filter out zero/near-zero balance
                                            })
                                            .map(account => {
                                                const rate = settings.account_interest_rates?.[account] ?? 0;
                                                const threshold = settings.interest_free_thresholds?.[account] ?? 0;

                                                const accountCash = holdings
                                                    .filter(h => h.Account === account && (h.Symbol === '$CASH' || h.Symbol === 'Cash' || h.Symbol.includes('Cash')))
                                                    .reduce((sum, h) => {
                                                        const mvKey = Object.keys(h).find(k => k.startsWith('Market Value'));
                                                        const val = mvKey ? (h[mvKey] as number) : 0;
                                                        return sum + (val || 0);
                                                    }, 0);

                                                return (
                                                    <tr key={account} className="hover:bg-accent/5 transition-colors">
                                                        <td className="px-4 py-3 whitespace-nowrap font-medium text-foreground">{account}</td>
                                                        <td className="px-4 py-3 whitespace-nowrap text-right font-mono text-muted-foreground">
                                                            {accountCash.toLocaleString(undefined, { style: 'currency', currency: 'USD' })}
                                                        </td>
                                                        <td className="px-4 py-3 whitespace-nowrap">
                                                            <div className="relative min-w-[110px] max-w-[160px]">
                                                                <input
                                                                    type="number"
                                                                    step="0.01"
                                                                    defaultValue={rate}
                                                                    onBlur={(e) => {
                                                                        const val = parseFloat(e.target.value);
                                                                        if (!isNaN(val) && val !== rate) {
                                                                            updateAccountYield(account, val, threshold);
                                                                        }
                                                                    }}
                                                                    className={`${inputClassName} pr-7 w-full`}
                                                                />
                                                                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground text-xs pointer-events-none">%</span>
                                                            </div>
                                                        </td>
                                                        <td className="px-4 py-3 whitespace-nowrap">
                                                            <div className="min-w-[110px] max-w-[160px]">
                                                                <input
                                                                    type="number"
                                                                    step="100"
                                                                    defaultValue={threshold}
                                                                    onBlur={(e) => {
                                                                        const val = parseFloat(e.target.value);
                                                                        if (!isNaN(val) && val !== threshold) {
                                                                            updateAccountYield(account, rate, val);
                                                                        }
                                                                    }}
                                                                    className={`${inputClassName} w-full`}
                                                                />
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Valuation Overrides Tab */}
                    {activeTab === 'valuation' && settings && (
                        <ManualValuationSettings settings={settings} />
                    )}

                    {/* Account Tab */}
                    {activeTab === 'account' && (
                        <div className="max-w-2xl">
                            <h3 className="text-lg font-medium mb-4">Account Information</h3>
                            <div className="bg-secondary p-4 rounded-lg border border-border mb-8">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className={labelClassName}>Username</label>
                                        <p className="font-mono text-lg">{user?.username}</p>
                                    </div>
                                    <div>
                                        <label className={labelClassName}>User ID</label>
                                        <p className="font-mono text-lg">{user?.id}</p>
                                    </div>
                                </div>
                            </div>

                            <h3 className="text-lg font-medium mb-4">Security</h3>
                            <div className="bg-secondary p-6 rounded-lg border border-border mb-8">
                                <form onSubmit={handleChangePassword} className="space-y-4 max-w-md">
                                    <h4 className="font-medium mb-4">Change Password</h4>

                                    <div>
                                        <label className={labelClassName}>Current Password</label>
                                        <input
                                            type="password"
                                            value={currentPassword}
                                            onChange={(e) => setCurrentPassword(e.target.value)}
                                            className={inputClassName}
                                            required
                                        />
                                    </div>

                                    <div>
                                        <label className={labelClassName}>New Password</label>
                                        <input
                                            type="password"
                                            value={newPassword}
                                            onChange={(e) => setNewPassword(e.target.value)}
                                            className={inputClassName}
                                            required
                                        />
                                    </div>

                                    <div>
                                        <label className={labelClassName}>Confirm New Password</label>
                                        <input
                                            type="password"
                                            value={confirmPassword}
                                            onChange={(e) => setConfirmPassword(e.target.value)}
                                            className={inputClassName}
                                            required
                                        />
                                    </div>

                                    {passwordStatus && (
                                        <div className={`text-sm p-3 rounded ${passwordStatus.type === 'success' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-red-500/10 text-red-500'}`}>
                                            {passwordStatus.message}
                                        </div>
                                    )}

                                    <button
                                        type="submit"
                                        disabled={isChangingPassword}
                                        className="px-4 py-2 bg-cyan-600 text-white rounded hover:bg-cyan-700 transition-colors shadow-sm disabled:opacity-50 flex items-center gap-2"
                                    >
                                        {isChangingPassword && <Loader2 className="w-4 h-4 animate-spin" />}
                                        Change Password
                                    </button>
                                </form>
                            </div>

                            <h3 className="text-lg font-medium mb-4 text-red-500">Danger Zone</h3>
                            <div className="bg-red-50 dark:bg-red-950/20 p-4 rounded-lg border border-red-200 dark:border-red-900 mb-8">
                                <h4 className="font-medium text-red-700 dark:text-red-400 mb-2">Delete Account</h4>
                                <p className="text-sm text-red-600/80 dark:text-red-400/80 mb-4">
                                    Once you delete your account, there is no going back. Please be certain.
                                    This will permanently delete your user profile, portfolio data, and settings.
                                </p>
                                <button
                                    onClick={handleDeleteAccount}
                                    className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors shadow-sm"
                                >
                                    Delete Account
                                </button>
                            </div>

                            <h3 className="text-lg font-medium mb-4">Device</h3>
                            <div className="bg-secondary p-6 rounded-lg border border-border">
                                <h4 className="font-medium mb-2 text-foreground">Sign Out</h4>
                                <div className="flex items-center justify-between">
                                    <p className="text-sm text-muted-foreground">
                                        Sign out of your account on this device.
                                    </p>
                                    <button
                                        type="button"
                                        onClick={() => logout()}
                                        className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-rose-600 hover:bg-rose-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-rose-500 transition-colors"
                                    >
                                        Sign Out
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

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
                                        <label className={labelClassName}>Country</label>
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
                                    <div className="col-span-1">
                                        <label className={labelClassName}>Market</label>
                                        <input
                                            type="text"
                                            value={overrideExchange}
                                            onChange={(e) => setOverrideExchange(e.target.value)}
                                            placeholder="e.g. NASDAQ"
                                            className={inputClassName}
                                        />
                                    </div>
                                </div>
                                <div className="flex justify-end mt-2">
                                    <button
                                        type="button"
                                        onClick={addOverride}
                                        disabled={!overrideSymbol || (!overridePrice && !overrideAssetType && !overrideSector && !overrideGeo && !overrideIndustry && !overrideExchange)}
                                        className="w-full md:w-auto px-6 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Set Override
                                    </button>
                                </div>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-border text-sm">
                                    <thead className="bg-secondary/50 font-semibold border-b border-border">
                                        <tr>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Symbol</th>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Price</th>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Asset Type</th>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Sector</th>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Country</th>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Industry</th>
                                            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Market</th>
                                            <th scope="col" className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-border">
                                        {Object.entries(overrides).length === 0 ? (
                                            <tr>
                                                <td colSpan={8} className="px-6 py-12 text-center text-muted-foreground italic">
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
                                                    const exchange = isObj ? (data as ManualOverrideData).exchange : '';
                                                    const currency = isObj ? (data as ManualOverrideData).currency : 'USD';

                                                    return (
                                                        <tr key={symbol} className="hover:bg-accent/5 transition-colors">
                                                            <td className="px-4 py-3 whitespace-nowrap font-medium text-foreground">{symbol}</td>
                                                            <td className="px-4 py-3 whitespace-nowrap text-muted-foreground font-mono tabular-nums">
                                                                {price === 0
                                                                    ? <span className="text-muted-foreground">-</span>
                                                                    : `${currency === 'THB' ? '฿' : '$'}${price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`
                                                                }
                                                            </td>
                                                            <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">{assetType || '-'}</td>
                                                            <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">{sector || '-'}</td>
                                                            <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">{geo || '-'}</td>
                                                            <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">{industry || '-'}</td>
                                                            <td className="px-4 py-3 whitespace-nowrap text-muted-foreground">{exchange || '-'}</td>
                                                            <td className="px-4 py-3 whitespace-nowrap text-right font-medium">
                                                                <button
                                                                    type="button"
                                                                    onClick={() => handleEdit(symbol, data)}
                                                                    className="text-cyan-500 hover:text-cyan-400 hover:bg-cyan-500/10 p-2 rounded transition-colors mr-1"
                                                                    title="Edit"
                                                                >
                                                                    <Pencil className="w-4 h-4" />
                                                                </button>
                                                                <button
                                                                    type="button"
                                                                    onClick={() => removeOverride(symbol)}
                                                                    className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 rounded transition-colors"
                                                                    title="Remove"
                                                                >
                                                                    <Trash2 className="w-4 h-4" />
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
                                    <thead className="bg-secondary/50 font-semibold border-b border-border">
                                        <tr>
                                            <th scope="col" className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground">Portfolio Symbol</th>
                                            <th scope="col" className="px-6 py-3 text-center text-xs font-semibold text-muted-foreground">Mapped To</th>
                                            <th scope="col" className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground">YFinance Ticker</th>
                                            <th scope="col" className="px-6 py-3 text-right text-xs font-semibold text-muted-foreground">Actions</th>
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
                                                        <td className="px-6 py-3 whitespace-nowrap text-sm font-medium text-foreground">{from}</td>
                                                        <td className="px-6 py-3 whitespace-nowrap text-center text-muted-foreground">→</td>
                                                        <td className="px-6 py-3 whitespace-nowrap text-sm text-foreground font-mono bg-accent/10 px-2 rounded w-min">{to}</td>
                                                        <td className="px-6 py-3 whitespace-nowrap text-right text-sm font-medium">
                                                            <button
                                                                type="button"
                                                                onClick={() => removeMapping(from)}
                                                                className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 rounded transition-colors"
                                                                title="Remove"
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
                                        className="w-full md:w-auto px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
                                                    className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 rounded transition-colors"
                                                    title="Remove from exclusion list"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Currencies Tab */}
                    {activeTab === 'currencies' && (
                        <div>
                            <p className="mb-4 text-sm text-gray-500 dark:text-gray-400">
                                Manage available currencies and assign default currencies to accounts.
                            </p>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                {/* Available Currencies */}
                                <div>
                                    <h3 className="text-md font-bold mb-2 text-foreground">Available Currencies</h3>
                                    <div className="flex gap-2 mb-4">
                                        <input
                                            type="text"
                                            value={newCurrency}
                                            onChange={(e) => setNewCurrency(e.target.value.toUpperCase())}
                                            placeholder="e.g. SGD"
                                            className={inputClassName}
                                            maxLength={3}
                                        />
                                        <button
                                            type="button"
                                            onClick={addCurrency}
                                            disabled={!newCurrency}
                                            className="px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium shadow-sm focus:outline-none disabled:opacity-50"
                                        >
                                            Add
                                        </button>
                                    </div>
                                    <div className="bg-secondary border border-border rounded-lg overflow-hidden">
                                        <ul className="divide-y divide-border">
                                            {availableCurrencies.map(curr => (
                                                <li key={curr} className="px-4 py-3 flex items-center justify-between hover:bg-accent/5">
                                                    <span className="font-medium text-foreground">{curr}</span>
                                                    <button
                                                        onClick={() => removeCurrency(curr)}
                                                        className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 rounded transition-colors"
                                                        title="Remove"
                                                    >
                                                        <Trash2 className="w-4 h-4" />
                                                    </button>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                </div>

                                {/* Account Currency Map */}
                                <div>
                                    <h3 className="text-md font-bold mb-2 text-foreground">Account Default Currencies</h3>
                                    <div className="bg-secondary border border-border rounded-lg overflow-hidden">
                                        <table className="min-w-full divide-y divide-border">
                                            <thead className="bg-secondary/50 font-semibold border-b border-border">
                                                <tr>
                                                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Account</th>
                                                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Currency</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-border">
                                                {availableAccounts.map(account => (
                                                    <tr key={account} className="hover:bg-accent/5">
                                                        <td className="px-4 py-3 text-sm font-medium text-foreground">{account}</td>
                                                        <td className="px-4 py-2">
                                                            <select
                                                                value={accountCurrencyMap[account] || 'USD'} // Default to USD visually if missing, logic handles load
                                                                onChange={(e) => updateAccountCurrency(account, e.target.value)}
                                                                className={inputClassName}
                                                            >
                                                                {availableCurrencies.map(curr => (
                                                                    <option key={curr} value={curr} className="bg-white dark:bg-black text-foreground">{curr}</option>
                                                                ))}
                                                            </select>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Advanced Settings Tab */}
                    {activeTab === 'advanced' && (
                        <div className="space-y-8">
                            {/* Webhook Connection */}
                            <div className="bg-secondary p-6 rounded-lg border border-border border-l-4 border-l-cyan-500">
                                <h3 className="text-lg font-bold mb-4 text-foreground">Webhook Integration</h3>
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
                                            className="flex-1 rounded-md border border-border bg-background shadow-sm focus:border-cyan-500 focus:ring-cyan-500 sm:text-sm text-foreground px-3 py-2 outline-none focus:ring-1"
                                        />
                                        <button
                                            type="button"
                                            onClick={handleRefresh}
                                            className="px-4 py-2 border border-border rounded-md shadow-sm text-sm font-medium text-foreground bg-background hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors"
                                        >
                                            Test
                                        </button>
                                    </div>
                                    {refreshStatus && (
                                        <p className={`text-sm ${refreshStatus.startsWith('Error') ? 'text-red-500' : 'text-emerald-400'}`}>
                                            {refreshStatus}
                                        </p>
                                    )}
                                </div>
                            </div>

                            {/* IBKR Integration */}
                            <div className="bg-secondary p-6 rounded-lg border border-border border-l-4 border-l-[#0097b2]">
                                <div className="flex justify-between items-start mb-4">
                                    <div>
                                        <h3 className="text-lg font-bold text-foreground">Interactive Brokers (IBKR)</h3>
                                        <p className="text-sm text-muted-foreground mt-1">
                                            Sync transactions via the IBKR Flex Web Service. Requires an <span className="font-semibold">Activity Flex Query</span>.
                                        </p>
                                    </div>
                                    <div className="bg-[#0097b2]/10 text-[#0097b2] px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider">
                                        Integration
                                    </div>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                                    <div className="space-y-2">
                                        <label className={labelClassName}>Flex Token</label>
                                        <input
                                            type="password"
                                            placeholder="Your IBKR Flex Token"
                                            value={ibkrToken}
                                            onChange={(e) => setIbkrToken(e.target.value)}
                                            className={inputClassName}
                                        />
                                        <p className="text-[10px] text-muted-foreground italic">
                                            Enable Flex Web Service in IBKR Settings to get a token.
                                        </p>
                                    </div>
                                    <div className="space-y-2">
                                        <label className={labelClassName}>Query ID</label>
                                        <input
                                            type="text"
                                            placeholder="e.g. 123456"
                                            value={ibkrQueryId}
                                            onChange={(e) => setIbkrQueryId(e.target.value)}
                                            className={inputClassName}
                                        />
                                        <p className="text-[10px] text-muted-foreground italic">
                                            The ID of your Activity Flex Query template.
                                        </p>
                                    </div>
                                </div>

                                <div className="flex flex-wrap items-center gap-4">
                                    <button
                                        type="button"
                                        onClick={handleSaveIbkr}
                                        disabled={isSavingIbkr || isSyncing}
                                        className="px-4 py-2 border border-border rounded-md shadow-sm text-sm font-medium text-foreground bg-background hover:bg-zinc-100 dark:hover:bg-zinc-900 focus:outline-none transition-colors disabled:opacity-50"
                                    >
                                        {isSavingIbkr ? <Loader2 className="w-4 h-4 animate-spin" /> : "Save Credentials"}
                                    </button>

                                    <div className="h-6 w-px bg-border hidden md:block" />

                                    <button
                                        type="button"
                                        onClick={handleSyncIbkr}
                                        disabled={isSyncing || !ibkrToken || !ibkrQueryId}
                                        className="px-6 py-2 bg-[#0097b2] text-white rounded-md shadow-sm text-sm font-medium hover:bg-[#0086a0] focus:outline-none transition-all disabled:opacity-50 disabled:grayscale flex items-center gap-2"
                                    >
                                        {isSyncing ? <Loader2 className="w-4 h-4 animate-spin" /> : "Sync Now"}
                                    </button>

                                    {syncStatus && (
                                        <p className={`text-sm font-medium ${syncStatus.startsWith('Error') ? 'text-red-500' : 'text-emerald-500'}`}>
                                            {syncStatus}
                                        </p>
                                    )}
                                </div>
                            </div>


                            {/* Cache Management Section */}
                            <div className="bg-secondary p-6 rounded-lg border border-border border-l-4 border-l-red-500">
                                <h3 className="text-lg font-bold mb-4 text-foreground">Cache Management</h3>
                                <div className="space-y-4">
                                    <p className="text-sm text-muted-foreground">
                                        Identify and resolve data discrepancies by clearing all local caches. This includes historical performance data, market quotes, and metadata.
                                    </p>
                                    <div className="flex items-center gap-4">
                                        <button
                                            type="button"
                                            onClick={handleClearCache}
                                            className={`px-4 py-2 border rounded-md shadow-sm text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${confirmClear
                                                ? 'bg-red-600 text-white border-transparent hover:bg-red-700 focus:ring-red-500'
                                                : 'border-red-500/50 text-red-500 bg-red-500/5 hover:bg-red-500/10 focus:ring-red-500'}`}
                                        >
                                            {confirmClear ? "Click to Confirm" : "Clear All Cache"}
                                        </button>
                                        {clearStatus && (
                                            <p className={`text-sm ${clearStatus.startsWith('Error') ? 'text-red-500' : 'text-emerald-400'}`}>
                                                {clearStatus}
                                            </p>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
