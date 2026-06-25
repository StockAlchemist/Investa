"use client";

import React, { useState, useMemo } from 'react';
import { 
    Pencil, Trash2, Loader2, Users, Sliders, Map as MapIcon, XCircle, 
    DollarSign, Activity, Settings as SettingsIcon, 
    UserCircle, LogOut, Save, ArrowRight, ShieldAlert, Smartphone, CheckCircle, AlertCircle, Info, LineChart, Plus
} from 'lucide-react';
import { updateSettings, triggerRefresh, clearCache, deleteUser, changePassword, syncIbkr, updateUserProfile, Settings as SettingsType, ManualOverride, ManualOverrideData, Holding } from '../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { cn } from '../lib/utils';
import { COUNTRIES, ALL_INDUSTRIES } from '../lib/constants';
import AccountGroupManager from './AccountGroupManager';
import ManualValuationSettings from './ManualValuationSettings';
import YieldSettings from './YieldSettings';

import { useAuth } from '../context/AuthContext';

type Tab = 'accounts' | 'symbols' | 'overrides' | 'advanced' | 'account';

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

const PRESET_BENCHMARKS = [
    "S&P 500",
    "Dow Jones",
    "NASDAQ",
    "Russell 2000",
    "SPY (S&P 500 ETF)",
    "QQQ (Nasdaq 100 ETF)",
    "DIA (Dow Jones ETF)",
    "S&P 500 Total Return",
];

interface SettingsProps {
    settings: SettingsType | null;
    holdings: Holding[];
    availableAccounts: string[];
    initialTab?: Tab;
    benchmarks: string[];
    onBenchmarksChange: (benchmarks: string[]) => void;
}

const TABS: { id: Tab, label: string, description: string, icon: React.ElementType, color: string }[] = [
    { id: 'accounts', label: 'Accounts', description: 'Account groups, per-account currency/cash/closure settings, and cash-yield assumptions.', icon: Users, color: 'text-indigo-500 dark:text-indigo-400' },
    { id: 'symbols', label: 'Symbols', description: 'Map portfolio symbols to their Yahoo Finance ticker and manage excluded symbols.', icon: MapIcon, color: 'text-blue-500 dark:text-blue-400' },
    { id: 'overrides', label: 'Overrides', description: 'Manually override price/metadata and DCF valuation inputs for specific symbols.', icon: Sliders, color: 'text-emerald-500 dark:text-emerald-400' },
    { id: 'advanced', label: 'Advanced Settings', description: 'Webhook integration, Interactive Brokers sync, and system cache.', icon: SettingsIcon, color: 'text-zinc-500 dark:text-zinc-400' },
    { id: 'account', label: 'Profile & Security', description: 'Manage your user profile, password, and login.', icon: UserCircle, color: 'text-cyan-500 dark:text-cyan-400' },
];

export default function Settings({ settings, holdings, availableAccounts, initialTab, benchmarks, onBenchmarksChange }: SettingsProps) {
    const queryClient = useQueryClient();
    const { logout, user, refreshUser } = useAuth();

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
    const [customBenchmark, setCustomBenchmark] = useState('');

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

    // Common styling variables for new design
    const inputClassName = "w-full rounded-xl border border-black/10 dark:border-white/10 bg-white/50 dark:bg-black/20 backdrop-blur-sm text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500/50 px-4 py-2.5 text-sm outline-none focus:ring-2 transition-all hover:border-black/20 dark:hover:border-white/20";
    const compactInputClassName = "w-full rounded-lg border border-black/10 dark:border-white/10 bg-white/60 dark:bg-black/30 text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500/40 px-3 py-2 text-sm outline-none focus:ring-2 transition-all hover:border-black/20 dark:hover:border-white/20";
    const labelClassName = "block text-[11px] font-bold text-muted-foreground mb-1.5 uppercase tracking-wider";
    const cardClassName = "bg-white/60 dark:bg-zinc-900/60 backdrop-blur-xl p-6 rounded-2xl border border-white/40 dark:border-white/10 shadow-lg relative overflow-hidden";
    const sectionTitleClassName = "text-lg font-bold text-foreground flex items-center gap-2";
    const primaryButtonClassName = "px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white rounded-xl font-medium shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 focus:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2";

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
        const newOverrides: Record<string, ManualOverride> = { ...currentOverrides };

        const newData: ManualOverrideData = {
            price: price !== null ? price : 0,
            asset_type: overrideAssetType || undefined,
            sector: overrideSector || undefined,
            geography: overrideGeo || undefined,
            industry: overrideIndustry || undefined,
            exchange: overrideExchange || undefined
        };

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
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });

            setOverrideSymbol('');
            setOverridePrice('');
            setOverrideAssetType('');
            setOverrideSector('');
            setOverrideGeo('');
            setOverrideIndustry('');
            setOverrideExchange('');
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
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
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
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            setMapFrom('');
            setMapTo('');
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
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
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
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            setExcludeSymbol('');
        } catch {
            alert('Failed to add excluded symbol');
        }
    };

    const removeExcluded = async (symbol: string) => {
        if (!settings) return;
        const currentList = (settings.user_excluded_symbols || []).filter(s => s !== symbol);

        try {
            await updateSettings({ user_excluded_symbols: currentList });
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
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
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
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
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
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
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
        } catch {
            alert('Failed to update account currency');
        }
    }

    const updateAccountCashMode = async (account: string, mode: string) => {
        if (!settings) return;
        const currentMap = { ...(settings.account_cash_mode_map || {}) };
        currentMap[account] = mode;

        try {
            await updateSettings({ account_cash_mode_map: currentMap });
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
        } catch {
            alert('Failed to update cash management mode');
        }
    }

    const updateAccountClosureDate = async (account: string, isoDate: string | null) => {
        if (!settings) return;
        const currentMap = { ...(settings.account_closure_dates || {}) };
        if (isoDate) currentMap[account] = isoDate;
        else delete currentMap[account];

        try {
            await updateSettings({ account_closure_dates: currentMap });
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
        } catch {
            alert('Failed to update account closure date');
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
            setTimeout(() => setConfirmClear(false), 3000);
            return;
        }

        try {
            setConfirmClear(false);
            setClearStatus("Clearing...");
            const res = await clearCache();
            setClearStatus(res.message || "Cache cleared successfully.");
            await queryClient.invalidateQueries();
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
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
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
            await queryClient.invalidateQueries();
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Sync failed";
            setSyncStatus(`Error: ${message}`);
        } finally {
            setIsSyncing(false);
        }
    }

    const overrides = settings?.manual_overrides || {};
    const symbolMap = settings?.user_symbol_map || {};
    const excluded = (settings?.user_excluded_symbols || []).slice().sort((a, b) => a.localeCompare(b));
    const availableCurrencies = (settings?.available_currencies || ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY']).slice().sort();
    const accountCurrencyMap = settings?.account_currency_map || {};

    const availableCountries = COUNTRIES.filter(c => !portfolioCountries.includes(c));

    const activeTabObj = TABS.find(t => t.id === activeTab);

    return (
        <div className="pb-20 max-w-7xl mx-auto px-4 md:px-8">
            <div className="mb-10">
                <h2 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent w-fit pb-1 drop-shadow-sm">Settings</h2>
                <p className="text-muted-foreground mt-2 text-base">
                    Manage application settings, preferences, and account configurations.
                </p>
            </div>

            <div className="flex flex-col lg:flex-row gap-8">
                {/* Sidebar Navigation */}
                <div className="w-full lg:w-72 shrink-0 space-y-2">
                    {TABS.map((tab) => {
                        const Icon = tab.icon;
                        const isActive = activeTab === tab.id;
                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl font-medium transition-all duration-200 text-sm ${
                                    isActive 
                                    ? `bg-white/80 dark:bg-white/10 text-foreground shadow-md backdrop-blur-md border border-white/40 dark:border-white/5` 
                                    : `text-muted-foreground hover:bg-white/40 dark:hover:bg-white/5 hover:text-foreground border border-transparent`
                                }`}
                            >
                                <Icon className={`w-5 h-5 ${isActive ? tab.color : 'opacity-70'}`} />
                                {tab.label}
                                {isActive && (
                                    <div className="ml-auto w-1.5 h-1.5 rounded-full bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
                                )}
                            </button>
                        );
                    })}
                </div>

                {/* Main Content Area */}
                <div className="flex-1 bg-white/40 dark:bg-zinc-950/40 backdrop-blur-2xl shadow-xl rounded-3xl border border-white/60 dark:border-white/10 overflow-hidden relative min-h-[600px]">
                    <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent opacity-50" />
                    
                    {/* Header for active tab */}
                    <div className="px-8 py-6 border-b border-black/5 dark:border-white/5 flex items-start gap-4 bg-white/20 dark:bg-black/20">
                        {activeTabObj && (
                            <>
                                <div className={`p-2.5 rounded-xl bg-white dark:bg-zinc-900 shadow-sm border border-black/5 dark:border-white/5 ${activeTabObj.color} shrink-0`}>
                                    <activeTabObj.icon className="w-6 h-6" />
                                </div>
                                <div className="min-w-0">
                                    <h3 className="text-xl font-bold text-foreground leading-tight">{activeTabObj.label}</h3>
                                    <p className="text-sm text-muted-foreground mt-1">{activeTabObj.description}</p>
                                </div>
                            </>
                        )}
                    </div>

                    <div className="p-8">
                        {!settings && (
                            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground animate-in fade-in zoom-in duration-500">
                                <Loader2 className="w-10 h-10 animate-spin mb-4 text-cyan-500" />
                                <p className="font-medium">Loading settings...</p>
                            </div>
                        )}

                        {/* Content Switching with subtle animation */}
                        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 fill-mode-both space-y-10">
                            {/* Accounts tab — Account Groups section */}
                            {activeTab === 'accounts' && settings && (
                                <AccountGroupManager
                                    availableAccounts={availableAccounts}
                                    settings={settings}
                                    onUpdate={() => queryClient.invalidateQueries({ queryKey: ['settings', user?.username] })}
                                />
                            )}

                            {/* Account Tab */}
                            {activeTab === 'account' && (
                                <div className="space-y-8 max-w-3xl">
                                    <div className={cardClassName}>
                                        <div className="mb-2">
                                            <h3 className={sectionTitleClassName}>
                                                <UserCircle className="w-5 h-5 text-cyan-500" />
                                                Profile Information
                                            </h3>
                                        </div>
                                        <p className="text-sm text-muted-foreground mb-6">Identifiers and display name shown across the app.</p>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div className="space-y-1">
                                                <label className={labelClassName}>Username</label>
                                                <p className="font-mono text-lg bg-black/5 dark:bg-white/5 px-4 py-2.5 rounded-xl border border-black/5 dark:border-white/5">{user?.username}</p>
                                            </div>
                                            <div className="space-y-1">
                                                <label className={labelClassName}>User ID</label>
                                                <p className="font-mono text-lg bg-black/5 dark:bg-white/5 px-4 py-2.5 rounded-xl border border-black/5 dark:border-white/5">{user?.id}</p>
                                            </div>
                                            <div className="md:col-span-2 space-y-1">
                                                <label className={labelClassName}>Alias (Display Name)</label>
                                                <input
                                                    type="text"
                                                    defaultValue={user?.alias || ''}
                                                    placeholder="e.g. My Portfolio"
                                                    className={inputClassName}
                                                    onBlur={async (e) => {
                                                        const newAlias = e.target.value.trim();
                                                        if (newAlias !== (user?.alias || '')) {
                                                            try {
                                                                await updateUserProfile({ alias: newAlias });
                                                                await refreshUser();
                                                            } catch {
                                                                alert("Failed to update alias");
                                                            }
                                                        }
                                                    }}
                                                />
                                                <p className="text-[11px] text-muted-foreground mt-2 pl-1 flex items-center gap-1.5">
                                                    <Info className="w-3.5 h-3.5" />
                                                    This name will be displayed in the user menu. Leave empty to use username.
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className={cardClassName}>
                                        <div className="mb-2">
                                            <h3 className={sectionTitleClassName}>
                                                <ShieldAlert className="w-5 h-5 text-amber-500" />
                                                Security
                                            </h3>
                                        </div>
                                        <p className="text-sm text-muted-foreground mb-6">Change your login password.</p>
                                        <form onSubmit={handleChangePassword} className="space-y-5">
                                            <div className="space-y-1">
                                                <label className={labelClassName}>Current Password</label>
                                                <input
                                                    type="password"
                                                    value={currentPassword}
                                                    onChange={(e) => setCurrentPassword(e.target.value)}
                                                    className={inputClassName}
                                                    required
                                                />
                                            </div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                                                <div className="space-y-1">
                                                    <label className={labelClassName}>New Password</label>
                                                    <input
                                                        type="password"
                                                        value={newPassword}
                                                        onChange={(e) => setNewPassword(e.target.value)}
                                                        className={inputClassName}
                                                        required
                                                    />
                                                </div>
                                                <div className="space-y-1">
                                                    <label className={labelClassName}>Confirm Password</label>
                                                    <input
                                                        type="password"
                                                        value={confirmPassword}
                                                        onChange={(e) => setConfirmPassword(e.target.value)}
                                                        className={inputClassName}
                                                        required
                                                    />
                                                </div>
                                            </div>

                                            {passwordStatus && (
                                                <div className={`text-sm p-4 rounded-xl flex items-center gap-3 animate-in fade-in ${passwordStatus.type === 'success' ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20'}`}>
                                                    {passwordStatus.type === 'success' ? <CheckCircle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
                                                    {passwordStatus.message}
                                                </div>
                                            )}

                                            <button
                                                type="submit"
                                                disabled={isChangingPassword}
                                                className={primaryButtonClassName}
                                            >
                                                {isChangingPassword ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
                                                Change Password
                                            </button>
                                        </form>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="bg-white/60 dark:bg-zinc-900/60 backdrop-blur-xl p-6 rounded-2xl border border-white/40 dark:border-white/10 shadow-sm flex flex-col justify-between">
                                            <div>
                                                <div className="flex items-center gap-3 mb-2">
                                                    <Smartphone className="w-5 h-5 text-foreground" />
                                                    <h4 className="font-bold text-foreground">Sign Out Device</h4>
                                                </div>
                                                <p className="text-sm text-muted-foreground mb-6">
                                                    End your current session on this device.
                                                </p>
                                            </div>
                                            <button
                                                type="button"
                                                onClick={() => logout()}
                                                className="w-full py-2.5 rounded-xl border-2 border-border bg-transparent hover:bg-secondary font-medium transition-colors text-foreground flex items-center justify-center gap-2"
                                            >
                                                <LogOut className="w-4 h-4" />
                                                Sign Out
                                            </button>
                                        </div>

                                        <div className="bg-red-50/50 dark:bg-red-950/20 backdrop-blur-xl p-6 rounded-2xl border border-red-200 dark:border-red-900/50 shadow-sm flex flex-col justify-between relative overflow-hidden group">
                                            <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-red-500 to-rose-500 transform origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-500" />
                                            <div>
                                                <h4 className="font-bold text-red-700 dark:text-red-400 mb-2">Delete Account</h4>
                                                <p className="text-sm text-red-600/80 dark:text-red-400/80 mb-6 leading-relaxed">
                                                    Permanently delete your profile, portfolio data, and settings. This action is irreversible.
                                                </p>
                                            </div>
                                            <button
                                                onClick={handleDeleteAccount}
                                                className="w-full py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-xl font-medium shadow-sm transition-colors focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                                            >
                                                Delete Account Permanently
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Manual Price Overrides Tab */}
                            {activeTab === 'overrides' && (
                                <div className="space-y-8">
                                    <div className={cardClassName}>
                                        <div className="mb-2">
                                            <h3 className={sectionTitleClassName}>
                                                <Sliders className="w-5 h-5 text-emerald-500" />
                                                Add / Edit Override
                                            </h3>
                                        </div>
                                        <p className="text-sm text-muted-foreground mb-5">Set a manual price, asset type, or any metadata field for a symbol.</p>
                                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
                                            <div className="space-y-1.5">
                                                <label className={labelClassName}>Symbol</label>
                                                <input
                                                    type="text"
                                                    value={overrideSymbol}
                                                    onChange={(e) => setOverrideSymbol(e.target.value.toUpperCase())}
                                                    placeholder="AAPL"
                                                    className={inputClassName}
                                                />
                                            </div>
                                            <div className="space-y-1.5">
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
                                            <div className="space-y-1.5">
                                                <label className={labelClassName}>Asset Type</label>
                                                <select
                                                    value={overrideAssetType}
                                                    onChange={(e) => setOverrideAssetType(e.target.value)}
                                                    className={inputClassName}
                                                >
                                                    {ASSET_TYPES.map(t => <option key={t} value={t} className="bg-background text-foreground">{t || "Select..."}</option>)}
                                                </select>
                                            </div>
                                            <div className="space-y-1.5">
                                                <label className={labelClassName}>Sector</label>
                                                <select
                                                    value={overrideSector}
                                                    onChange={(e) => setOverrideSector(e.target.value)}
                                                    className={inputClassName}
                                                >
                                                    {SECTORS.map(s => <option key={s} value={s} className="bg-background text-foreground">{s || "Select..."}</option>)}
                                                </select>
                                            </div>
                                            <div className="space-y-1.5">
                                                <label className={labelClassName}>Country</label>
                                                <select
                                                    value={overrideGeo}
                                                    onChange={(e) => setOverrideGeo(e.target.value)}
                                                    className={inputClassName}
                                                >
                                                    <option value="" className="bg-background text-foreground">Select...</option>
                                                    {portfolioCountries.length > 0 && (
                                                        <optgroup label="In Portfolio" className="bg-muted text-foreground">
                                                            {portfolioCountries.map(c => <option key={c} value={c} className="bg-background">{c}</option>)}
                                                        </optgroup>
                                                    )}
                                                    <optgroup label="All Countries" className="bg-muted text-foreground">
                                                        {availableCountries.map(c => <option key={c} value={c} className="bg-background">{c}</option>)}
                                                    </optgroup>
                                                </select>
                                            </div>
                                            <div className="space-y-1.5">
                                                <label className={labelClassName}>Industry</label>
                                                <select
                                                    value={overrideIndustry}
                                                    onChange={(e) => setOverrideIndustry(e.target.value)}
                                                    className={inputClassName}
                                                >
                                                    <option value="" className="bg-background text-foreground">Select...</option>
                                                    {ALL_INDUSTRIES.map(i => <option key={i} value={i} className="bg-background text-foreground">{i}</option>)}
                                                </select>
                                            </div>
                                            <div className="space-y-1.5">
                                                <label className={labelClassName}>Market</label>
                                                <input
                                                    type="text"
                                                    value={overrideExchange}
                                                    onChange={(e) => setOverrideExchange(e.target.value)}
                                                    placeholder="NASDAQ"
                                                    className={inputClassName}
                                                />
                                            </div>
                                        </div>
                                        <div className="flex justify-end mt-6">
                                            <button
                                                type="button"
                                                onClick={addOverride}
                                                disabled={!overrideSymbol || (!overridePrice && !overrideAssetType && !overrideSector && !overrideGeo && !overrideIndustry && !overrideExchange)}
                                                className={primaryButtonClassName}
                                            >
                                                <Save className="w-4 h-4" />
                                                Save Override
                                            </button>
                                        </div>
                                    </div>

                                    <div className={`${cardClassName} !p-0`}>
                                        <div className="flex items-center justify-between px-6 py-4 border-b border-black/5 dark:border-white/5 bg-white/30 dark:bg-black/20">
                                            <h3 className={sectionTitleClassName}>
                                                <Sliders className="w-5 h-5 text-emerald-500" />
                                                Active Overrides
                                                <span className="text-xs font-medium text-muted-foreground bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-full ml-1">{Object.entries(overrides).length}</span>
                                            </h3>
                                        </div>
                                        <div className="overflow-x-auto">
                                            <table className="min-w-full text-sm">
                                                <thead className="bg-black/5 dark:bg-white/5 border-b border-black/10 dark:border-white/10">
                                                    <tr>
                                                        <th className="sticky left-0 z-20 px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs bg-zinc-100 dark:bg-zinc-800 shadow-[1px_0_0_0_rgba(0,0,0,0.06)] dark:shadow-[1px_0_0_0_rgba(255,255,255,0.08)]">Symbol</th>
                                                        <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Price</th>
                                                        <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Asset Type</th>
                                                        <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Sector</th>
                                                        <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Country</th>
                                                        <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Industry</th>
                                                        <th className="px-6 py-3 text-left font-semibold text-muted-foreground uppercase tracking-wider text-xs">Market</th>
                                                        <th className="px-6 py-3 text-right font-semibold text-muted-foreground uppercase tracking-wider text-xs">Actions</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-black/5 dark:divide-white/5">
                                                    {Object.entries(overrides).length === 0 ? (
                                                        <tr>
                                                            <td colSpan={8} className="px-6 py-12 text-center text-muted-foreground">
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
                                                                    <tr key={symbol} className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors group">
                                                                        <td className="sticky left-0 z-10 px-6 py-4 whitespace-nowrap font-bold text-foreground bg-white dark:bg-zinc-900 group-hover:bg-zinc-50 dark:group-hover:bg-zinc-800 transition-colors shadow-[1px_0_0_0_rgba(0,0,0,0.06)] dark:shadow-[1px_0_0_0_rgba(255,255,255,0.08)]">{symbol}</td>
                                                                        <td className="px-6 py-4 whitespace-nowrap text-muted-foreground font-mono">
                                                                            {price === 0
                                                                                ? <span className="opacity-50">-</span>
                                                                                : <span className="text-emerald-600 dark:text-emerald-400 font-medium">{currency === 'THB' ? '฿' : '$'}{price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</span>
                                                                            }
                                                                        </td>
                                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                                            {assetType ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{assetType}</span> : <span className="opacity-50">-</span>}
                                                                        </td>
                                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                                            {sector ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{sector}</span> : <span className="opacity-50">-</span>}
                                                                        </td>
                                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                                            {geo ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{geo}</span> : <span className="opacity-50">-</span>}
                                                                        </td>
                                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                                            {industry ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{industry}</span> : <span className="opacity-50">-</span>}
                                                                        </td>
                                                                        <td className="px-6 py-4 whitespace-nowrap">
                                                                            {exchange ? <span className="bg-black/5 dark:bg-white/5 px-2.5 py-1 rounded-md text-xs font-medium text-foreground">{exchange}</span> : <span className="opacity-50">-</span>}
                                                                        </td>
                                                                        <td className="px-6 py-4 whitespace-nowrap text-right">
                                                                            <div className="flex justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                                                <button
                                                                                    type="button"
                                                                                    onClick={() => handleEdit(symbol, data)}
                                                                                    className="p-2 text-cyan-500 hover:bg-cyan-500/10 rounded-lg transition-colors"
                                                                                >
                                                                                    <Pencil className="w-4 h-4" />
                                                                                </button>
                                                                                <button
                                                                                    type="button"
                                                                                    onClick={() => removeOverride(symbol)}
                                                                                    className="p-2 text-red-500 hover:bg-red-500/10 rounded-lg transition-colors"
                                                                                >
                                                                                    <Trash2 className="w-4 h-4" />
                                                                                </button>
                                                                            </div>
                                                                        </td>
                                                                    </tr>
                                                                );
                                                            })
                                                    )}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Overrides tab — Valuation Overrides section */}
                            {activeTab === 'overrides' && settings && (
                                <ManualValuationSettings settings={settings} />
                            )}

                            {/* Symbols tab — Symbol Mapping section */}
                            {activeTab === 'symbols' && (
                                <div className="space-y-8 max-w-4xl">
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

                                    <div className={`${cardClassName} !p-0`}>
                                        <div className="flex items-center justify-between px-6 py-4 border-b border-black/5 dark:border-white/5 bg-white/30 dark:bg-black/20">
                                            <h3 className={sectionTitleClassName}>
                                                <MapIcon className="w-5 h-5 text-blue-500" />
                                                Active Mappings
                                                <span className="text-xs font-medium text-muted-foreground bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-full ml-1">{Object.entries(symbolMap).length}</span>
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
                                                                <td className="px-6 py-4 text-center text-muted-foreground"><ArrowRight className="w-4 h-4 inline opacity-50" /></td>
                                                                <td className="px-6 py-4 text-blue-600 dark:text-blue-400 font-mono font-medium">{to}</td>
                                                                <td className="px-6 py-4 text-right">
                                                                    <button
                                                                        type="button"
                                                                        onClick={() => removeMapping(from)}
                                                                        className="p-2 text-red-500 hover:bg-red-500/10 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
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

                            {/* Symbols tab — Excluded Symbols section */}
                            {activeTab === 'symbols' && (
                                <div className="space-y-8 max-w-4xl">
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
                                                className="px-6 py-2.5 bg-rose-500 hover:bg-rose-600 text-white rounded-xl font-medium shadow-sm transition-colors disabled:opacity-50"
                                            >
                                                Exclude
                                            </button>
                                        </div>
                                    </div>

                                    <div className={cardClassName}>
                                        <h3 className={`${sectionTitleClassName} mb-5`}>
                                            <XCircle className="w-5 h-5 text-rose-500" />
                                            Excluded Symbols
                                            <span className="text-xs font-medium text-muted-foreground bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-full ml-1">{excluded.length}</span>
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
                                                        <span className="font-bold font-mono text-rose-700 dark:text-rose-300 text-sm">{sym}</span>
                                                        <button
                                                            type="button"
                                                            onClick={() => removeExcluded(sym)}
                                                            className="opacity-40 group-hover:opacity-100 text-red-500 hover:text-red-600 transition-opacity"
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
                            )}

                            {/* Accounts tab — Account Settings (currencies, per-account) section */}
                            {activeTab === 'accounts' && (
                                <div className="space-y-8">
                                    {/* Available Currencies Section */}
                                    <div className={cardClassName}>
                                        <div className="mb-2">
                                            <h3 className={sectionTitleClassName}>
                                                <DollarSign className="w-5 h-5 text-amber-500" />
                                                Available Currencies
                                                <span className="text-xs font-medium text-muted-foreground bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-full ml-1">{availableCurrencies.length}</span>
                                            </h3>
                                        </div>
                                        <p className="text-sm text-muted-foreground mb-5">Currencies you can assign to accounts below.</p>

                                        {availableCurrencies.length > 0 && (
                                            <div className="flex flex-wrap gap-2 mb-5">
                                                {availableCurrencies.map(curr => (
                                                    <div
                                                        key={curr}
                                                        className="group inline-flex items-center gap-2 bg-amber-500/10 border border-amber-500/30 hover:border-amber-500/50 px-3 py-1.5 rounded-lg transition-colors"
                                                    >
                                                        <span className="font-bold font-mono text-amber-700 dark:text-amber-300 text-sm">{curr}</span>
                                                        <button
                                                            onClick={() => removeCurrency(curr)}
                                                            className="opacity-40 group-hover:opacity-100 text-red-500 hover:text-red-600 transition-opacity"
                                                            aria-label={`Remove ${curr}`}
                                                        >
                                                            <Trash2 className="w-3.5 h-3.5" />
                                                        </button>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        <div className="flex gap-3 items-end max-w-md pt-5 border-t border-black/5 dark:border-white/5">
                                            <div className="flex-1 space-y-1.5">
                                                <label className={labelClassName}>Add a Currency</label>
                                                <input
                                                    type="text"
                                                    value={newCurrency}
                                                    onChange={(e) => setNewCurrency(e.target.value.toUpperCase())}
                                                    placeholder="e.g. SGD"
                                                    className={inputClassName}
                                                    maxLength={3}
                                                />
                                            </div>
                                            <button
                                                type="button"
                                                onClick={addCurrency}
                                                disabled={!newCurrency}
                                                className="px-5 py-2.5 bg-amber-500 hover:bg-amber-600 text-white rounded-xl font-medium shadow-sm transition-colors disabled:opacity-50"
                                            >
                                                Add
                                            </button>
                                        </div>
                                    </div>

                                    {/* Account Preferences Section */}
                                    {(() => {
                                        // "All Accounts" is a synthetic value attached to split transactions
                                        // (see TransactionModal), not a real configurable account.
                                        const configurableAccounts = availableAccounts.filter(a => a !== 'All Accounts');
                                        return (
                                    <div className={cardClassName}>
                                        <div className="mb-2">
                                            <h3 className={sectionTitleClassName}>
                                                <SettingsIcon className="w-5 h-5 text-zinc-500" />
                                                Account Preferences
                                                <span className="text-xs font-medium text-muted-foreground bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-full ml-1">{configurableAccounts.length}</span>
                                            </h3>
                                        </div>
                                        <p className="text-sm text-muted-foreground mb-5">Configure currency, cash management mode, and closure date for each account.</p>

                                        {configurableAccounts.length === 0 ? (
                                            <div className="text-center text-muted-foreground py-12 border border-dashed border-black/10 dark:border-white/10 rounded-xl">
                                                No accounts found.
                                            </div>
                                        ) : (
                                            <div className="space-y-3">
                                                {configurableAccounts.map(account => {
                                                    const closureDate = settings?.account_closure_dates?.[account] || '';
                                                    const isEffectivelyClosed = closureDate && closureDate <= new Date().toISOString().slice(0, 10);
                                                    return (
                                                        <div
                                                            key={account}
                                                            className={cn(
                                                                "bg-white/60 dark:bg-black/20 border border-black/5 dark:border-white/10 rounded-xl p-4 transition-all hover:border-black/15 dark:hover:border-white/20",
                                                                isEffectivelyClosed && "opacity-70"
                                                            )}
                                                        >
                                                            <div className="flex items-center gap-2 mb-4 pb-3 border-b border-black/5 dark:border-white/5">
                                                                <span className={cn(
                                                                    "font-bold text-foreground",
                                                                    isEffectivelyClosed && "line-through"
                                                                )}>{account}</span>
                                                                {isEffectivelyClosed && (
                                                                    <span className="text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 bg-zinc-500/15 text-zinc-600 dark:text-zinc-400 rounded-full">
                                                                        Closed
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                                                                <div className="space-y-1.5">
                                                                    <label className={labelClassName}>Default Currency</label>
                                                                    <select
                                                                        value={accountCurrencyMap[account] || 'USD'}
                                                                        onChange={(e) => updateAccountCurrency(account, e.target.value)}
                                                                        className={compactInputClassName}
                                                                    >
                                                                        {availableCurrencies.map(curr => (
                                                                            <option key={curr} value={curr} className="bg-background text-foreground">{curr}</option>
                                                                        ))}
                                                                    </select>
                                                                </div>
                                                                <div className="space-y-1.5">
                                                                    <label className={labelClassName}>Cash Management</label>
                                                                    <select
                                                                        value={settings?.account_cash_mode_map?.[account] || 'Manual'}
                                                                        onChange={(e) => updateAccountCashMode(account, e.target.value)}
                                                                        className={compactInputClassName}
                                                                    >
                                                                        <option value="Manual" className="bg-background text-foreground">Manual</option>
                                                                        <option value="Auto" className="bg-background text-foreground">Auto</option>
                                                                    </select>
                                                                </div>
                                                                <div className="space-y-1.5">
                                                                    <label className={labelClassName}>Closure Date</label>
                                                                    <div className="flex items-center gap-2">
                                                                        <input
                                                                            type="date"
                                                                            value={closureDate}
                                                                            onChange={(e) => updateAccountClosureDate(account, e.target.value || null)}
                                                                            className={compactInputClassName}
                                                                        />
                                                                        {closureDate && (
                                                                            <button
                                                                                type="button"
                                                                                onClick={() => updateAccountClosureDate(account, null)}
                                                                                className="shrink-0 p-2 text-muted-foreground hover:text-red-500 hover:bg-red-500/10 rounded-lg transition-colors"
                                                                                title="Clear closure date"
                                                                                aria-label="Clear closure date"
                                                                            >
                                                                                <Trash2 className="w-4 h-4" />
                                                                            </button>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </div>
                                        );
                                    })()}
                                </div>
                            )}

                            {/* Accounts tab — Cash Yield section */}
                            {activeTab === 'accounts' && settings && (
                                <YieldSettings
                                    settings={settings}
                                    availableAccounts={availableAccounts}
                                    holdings={holdings}
                                    onSettingsUpdated={() => {
                                        queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
                                        queryClient.invalidateQueries({ queryKey: ['portfolio'] });
                                    }}
                                />
                            )}

                            {/* Advanced Settings Tab */}
                            {activeTab === 'advanced' && (
                                <div className="space-y-8 max-w-4xl">
                                    {/* Benchmarks Section */}
                                    <div className={`${cardClassName} border-l-4 border-l-purple-500`}>
                                        <div className="mb-4">
                                            <h3 className={sectionTitleClassName}>
                                                <LineChart className="w-5 h-5 text-purple-500" />
                                                Benchmarks
                                            </h3>
                                            <p className="text-sm text-muted-foreground mt-1">
                                                Select indices and specific symbols to compare your portfolio performance against.
                                            </p>
                                        </div>
                                        
                                        <div className="space-y-4">
                                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                                                {PRESET_BENCHMARKS.map(benchmark => (
                                                    <label
                                                        key={benchmark}
                                                        className={cn(
                                                            "flex items-center gap-2 p-3 rounded-xl border cursor-pointer transition-all",
                                                            benchmarks.includes(benchmark)
                                                                ? "bg-purple-500/10 border-purple-500/50 text-foreground"
                                                                : "bg-black/5 dark:bg-white/5 border-transparent text-muted-foreground hover:bg-black/10 dark:hover:bg-white/10"
                                                        )}
                                                    >
                                                        <input
                                                            type="checkbox"
                                                            checked={benchmarks.includes(benchmark)}
                                                            onChange={(e) => {
                                                                if (e.target.checked) {
                                                                    onBenchmarksChange([...benchmarks, benchmark]);
                                                                } else {
                                                                    onBenchmarksChange(benchmarks.filter(b => b !== benchmark));
                                                                }
                                                            }}
                                                            className="rounded border-none bg-secondary text-purple-500 focus:ring-purple-500"
                                                        />
                                                        <span className="text-sm font-medium">{benchmark}</span>
                                                    </label>
                                                ))}
                                            </div>

                                            <div className="pt-4 border-t border-black/5 dark:border-white/5">
                                                <label className={labelClassName}>Custom Ticker</label>
                                                <div className="flex flex-wrap gap-3">
                                                    <div className="flex flex-1 sm:flex-none gap-2 min-w-[200px] max-w-xs">
                                                        <input
                                                            type="text"
                                                            placeholder="e.g. AAPL"
                                                            value={customBenchmark}
                                                            onChange={(e) => setCustomBenchmark(e.target.value.toUpperCase())}
                                                            onKeyDown={(e) => {
                                                                if (e.key === 'Enter') {
                                                                    e.preventDefault();
                                                                    if (customBenchmark && !benchmarks.includes(customBenchmark)) {
                                                                        onBenchmarksChange([...benchmarks, customBenchmark]);
                                                                        setCustomBenchmark('');
                                                                    }
                                                                }
                                                            }}
                                                            className={inputClassName}
                                                        />
                                                        <button
                                                            type="button"
                                                            onClick={() => {
                                                                if (customBenchmark && !benchmarks.includes(customBenchmark)) {
                                                                    onBenchmarksChange([...benchmarks, customBenchmark]);
                                                                    setCustomBenchmark('');
                                                                }
                                                            }}
                                                            className="p-2.5 bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 rounded-xl transition-colors text-foreground"
                                                        >
                                                            <Plus className="w-5 h-5" />
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>

                                            {benchmarks.filter(b => !PRESET_BENCHMARKS.includes(b)).length > 0 && (
                                                <div className="flex flex-wrap gap-2 pt-2">
                                                    {benchmarks.filter(b => !PRESET_BENCHMARKS.includes(b)).map(b => (
                                                        <span
                                                            key={b}
                                                            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-purple-500/10 text-purple-600 dark:text-purple-400 rounded-lg text-sm font-medium"
                                                        >
                                                            {b}
                                                            <button
                                                                type="button"
                                                                onClick={() => onBenchmarksChange(benchmarks.filter(item => item !== b))}
                                                                className="hover:bg-purple-500/20 p-0.5 rounded-md transition-colors"
                                                            >
                                                                <XCircle className="w-3.5 h-3.5" />
                                                            </button>
                                                        </span>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Webhook Connection */}
                                    <div className={`${cardClassName} border-l-4 border-l-cyan-500`}>
                                        <div className="mb-2">
                                            <h3 className={sectionTitleClassName}>
                                                <Activity className="w-5 h-5 text-cyan-500" />
                                                Webhook Integration
                                            </h3>
                                        </div>
                                        <p className="text-sm text-muted-foreground mb-5 leading-relaxed">
                                            Trigger a background data refresh externally by sending a POST request to{' '}
                                            <code className="inline-block bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-md text-xs text-cyan-600 dark:text-cyan-400 font-mono border border-black/10 dark:border-white/10 align-middle">POST /api/webhook/refresh</code>
                                        </p>
                                        <div className="space-y-3">
                                            <div className="flex gap-3 max-w-md">
                                                <input
                                                    type="text"
                                                    placeholder="Enter Webhook Secret"
                                                    value={refreshSecret}
                                                    onChange={(e) => setRefreshSecret(e.target.value)}
                                                    className={inputClassName}
                                                />
                                                <button
                                                    type="button"
                                                    onClick={handleRefresh}
                                                    className="px-6 py-2.5 border border-border rounded-xl font-medium text-foreground bg-background hover:bg-secondary transition-colors"
                                                >
                                                    Test
                                                </button>
                                            </div>
                                            {refreshStatus && (
                                                <p className={`text-sm font-medium animate-in fade-in ${refreshStatus.startsWith('Error') ? 'text-red-500' : 'text-emerald-500'}`}>
                                                    {refreshStatus}
                                                </p>
                                            )}
                                        </div>
                                    </div>

                                    {/* IBKR Integration */}
                                    <div className={`${cardClassName} border-l-4 border-l-blue-500`}>
                                        <div className="mb-2">
                                            <h3 className={sectionTitleClassName}>
                                                <Sliders className="w-5 h-5 text-blue-500" />
                                                Interactive Brokers Sync
                                            </h3>
                                        </div>
                                        <p className="text-sm text-muted-foreground mb-5">
                                            Sync transactions using IBKR Flex Web Service. Requires an active Activity Flex Query.
                                        </p>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
                                            <div className="space-y-1.5">
                                                <label className={labelClassName}>Flex Token</label>
                                                <input
                                                    type="password"
                                                    placeholder="Your IBKR Flex Token"
                                                    value={ibkrToken}
                                                    onChange={(e) => setIbkrToken(e.target.value)}
                                                    className={inputClassName}
                                                />
                                            </div>
                                            <div className="space-y-1.5">
                                                <label className={labelClassName}>Query ID</label>
                                                <input
                                                    type="text"
                                                    placeholder="e.g. 123456"
                                                    value={ibkrQueryId}
                                                    onChange={(e) => setIbkrQueryId(e.target.value)}
                                                    className={inputClassName}
                                                />
                                            </div>
                                        </div>

                                        <div className="flex flex-wrap items-center gap-4 bg-black/5 dark:bg-white/5 p-4 rounded-xl border border-black/5 dark:border-white/5">
                                            <button
                                                type="button"
                                                onClick={handleSaveIbkr}
                                                disabled={isSavingIbkr || isSyncing}
                                                className="px-5 py-2.5 border border-border rounded-xl text-sm font-medium hover:bg-secondary transition-colors disabled:opacity-50 flex items-center gap-2"
                                            >
                                                {isSavingIbkr ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4"/>}
                                                Save Credentials
                                            </button>

                                            <div className="h-6 w-px bg-border hidden md:block" />

                                            <button
                                                type="button"
                                                onClick={handleSyncIbkr}
                                                disabled={isSyncing || !ibkrToken || !ibkrQueryId}
                                                className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-medium transition-all disabled:opacity-50 flex items-center gap-2 shadow-sm"
                                            >
                                                {isSyncing ? <Loader2 className="w-4 h-4 animate-spin" /> : "Sync Transactions Now"}
                                            </button>

                                            {syncStatus && (
                                                <p className={`text-sm font-medium animate-in fade-in ${syncStatus.startsWith('Error') ? 'text-red-500' : 'text-emerald-500'}`}>
                                                    {syncStatus}
                                                </p>
                                            )}
                                        </div>
                                    </div>

                                    {/* Cache Management Section */}
                                    <div className={`${cardClassName} border-l-4 border-l-red-500`}>
                                        <div className="mb-2">
                                            <h3 className={sectionTitleClassName}>
                                                <ShieldAlert className="w-5 h-5 text-red-500" />
                                                System Cache
                                            </h3>
                                        </div>
                                        <p className="text-sm text-muted-foreground mb-5 leading-relaxed">
                                            Clear local caches to resolve data discrepancies. This drops historical performance data, market quotes, and metadata, forcing a fresh download on the next load.
                                        </p>
                                        <div className="space-y-6">
                                            <div className="flex items-center gap-4 flex-wrap">
                                                <button
                                                    type="button"
                                                    onClick={handleClearCache}
                                                    className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 border ${confirmClear
                                                        ? 'bg-red-600 text-white border-transparent hover:bg-red-700 focus:ring-red-500 scale-105'
                                                        : 'border-red-500/50 text-red-600 dark:text-red-400 bg-red-500/5 hover:bg-red-500/10 focus:ring-red-500'}`}
                                                >
                                                    {confirmClear ? "Click again to Confirm" : "Clear System Cache"}
                                                </button>
                                                {clearStatus && (
                                                    <p className={`text-sm font-medium animate-in fade-in ${clearStatus.startsWith('Error') ? 'text-red-500' : 'text-emerald-500'}`}>
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
            </div>
        </div>
    );
}
