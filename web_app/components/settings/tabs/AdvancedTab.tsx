import React, { useState, useEffect } from 'react';
import { LineChart, Plus, XCircle, Activity, Sliders, Save, Loader2, ShieldAlert } from 'lucide-react';
import { Settings as SettingsType, triggerRefresh, clearCache, syncIbkr, updateSettings } from '../../../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { useAuth } from '../../../context/AuthContext';
import { cn } from '../../../lib/utils';
import {
    PRESET_BENCHMARKS,
    cardClassName,
    sectionTitleClassName,
    labelClassName,
    inputClassName
} from '../constants';

interface AdvancedTabProps {
    settings: SettingsType | null;
    benchmarks: string[];
    onBenchmarksChange: (benchmarks: string[]) => void;
}

export const AdvancedTab: React.FC<AdvancedTabProps> = ({
    settings,
    benchmarks,
    onBenchmarksChange,
}) => {
    const queryClient = useQueryClient();
    const { user } = useAuth();

    const [customBenchmark, setCustomBenchmark] = useState('');
    const [refreshSecret, setRefreshSecret] = useState('');
    const [refreshStatus, setRefreshStatus] = useState<string | null>(null);

    const [ibkrToken, setIbkrToken] = useState('');
    const [ibkrQueryId, setIbkrQueryId] = useState('');
    const [isSavingIbkr, setIsSavingIbkr] = useState(false);
    const [isSyncing, setIsSyncing] = useState(false);
    const [syncStatus, setSyncStatus] = useState<string | null>(null);

    const [confirmClear, setConfirmClear] = useState(false);
    const [clearStatus, setClearStatus] = useState<string | null>(null);

    useEffect(() => {
        if (settings) {
            setIbkrToken(settings.ibkr_token || '');
            setIbkrQueryId(settings.ibkr_query_id || '');
        }
    }, [settings]);

    const handleRefresh = async () => {
        try {
            const res = await triggerRefresh(refreshSecret);
            setRefreshStatus(res.message || null);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            setRefreshStatus(`Error: ${message}`);
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
    };

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
    };

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

    return (
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
                                    className="p-2.5 bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 rounded-xl transition-colors text-foreground cursor-pointer"
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
                                        className="hover:bg-purple-500/20 p-0.5 rounded-md transition-colors cursor-pointer"
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
                            className="px-6 py-2.5 border border-border rounded-xl font-medium text-foreground bg-background hover:bg-secondary transition-colors cursor-pointer"
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
                        className="px-5 py-2.5 border border-border rounded-xl text-sm font-medium hover:bg-secondary transition-colors disabled:opacity-50 flex items-center gap-2 cursor-pointer"
                    >
                        {isSavingIbkr ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4"/>}
                        Save Credentials
                    </button>

                    <div className="h-6 w-px bg-border hidden md:block" />

                    <button
                        type="button"
                        onClick={handleSyncIbkr}
                        disabled={isSyncing || !ibkrToken || !ibkrQueryId}
                        className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-medium transition-all disabled:opacity-50 flex items-center gap-2 shadow-sm cursor-pointer"
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
                            className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 border cursor-pointer ${confirmClear
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
    );
};
