import React, { useState, useEffect } from 'react';
import { Plus, XCircle, Save, Loader2, Eye, EyeOff } from 'lucide-react';
import { Settings as SettingsType, triggerRefresh, clearCache, syncIbkr, updateSettings } from '../../../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { useAuth } from '../../../context/AuthContext';
import { cn } from '../../../lib/utils';
import {
    PRESET_BENCHMARKS,
    cardClassName,
    cardHeadClassName,
    sectionTitleClassName,
    countBadgeClassName,
    secondaryButtonClassName,
    labelClassName,
    inputClassName,
    primaryButtonClassName
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

    const [geminiApiKey, setGeminiApiKey] = useState('');
    const [fmpApiKey, setFmpApiKey] = useState('');
    const [secThApiKey, setSecThApiKey] = useState('');
    const [botApiKey, setBotApiKey] = useState('');
    const [tiingoApiKey, setTiingoApiKey] = useState('');
    const [showApiKeys, setShowApiKeys] = useState(false);
    const [isSavingApiKeys, setIsSavingApiKeys] = useState(false);
    const [apiKeyStatus, setApiKeyStatus] = useState<string | null>(null);
    // Which key fields the user has actually typed into. The server sends
    // masked previews, never the real keys, and treats "" as "clear" - so a
    // save must post only the fields that were edited. Posting all five would
    // wipe every stored key whenever the settings fetch has not resolved.
    const [editedApiKeys, setEditedApiKeys] = useState<Set<string>>(new Set());

    const markApiKeyEdited = (field: string) => {
        setEditedApiKeys((prev) => (prev.has(field) ? prev : new Set(prev).add(field)));
    };

    const [confirmClear, setConfirmClear] = useState(false);
    const [clearStatus, setClearStatus] = useState<string | null>(null);

    useEffect(() => {
        if (!settings) return;
        setIbkrToken(settings.ibkr_token || '');
        setIbkrQueryId(settings.ibkr_query_id || '');
        // A refetch must not overwrite a key the user is midway through
        // typing - saving an unrelated section refetches settings, and that
        // would silently discard a pasted-but-unsaved key.
        const seed = (field: string, set: (v: string) => void, value?: string | null) => {
            if (!editedApiKeys.has(field)) set(value || '');
        };
        seed('gemini_api_key', setGeminiApiKey, settings.gemini_api_key);
        seed('fmp_api_key', setFmpApiKey, settings.fmp_api_key);
        seed('sec_th_api_key', setSecThApiKey, settings.sec_th_api_key);
        seed('bot_api_key', setBotApiKey, settings.bot_api_key);
        seed('tiingo_api_key', setTiingoApiKey, settings.tiingo_api_key);
        // eslint-disable-next-line react-hooks/exhaustive-deps
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

    const handleSaveApiKeys = async () => {
        const values: Record<string, string> = {
            gemini_api_key: geminiApiKey,
            fmp_api_key: fmpApiKey,
            sec_th_api_key: secThApiKey,
            bot_api_key: botApiKey,
            tiingo_api_key: tiingoApiKey,
        };
        const payload: Record<string, string> = {};
        editedApiKeys.forEach((field) => { payload[field] = values[field]; });
        if (Object.keys(payload).length === 0) {
            setApiKeyStatus("No API key changes to save.");
            setTimeout(() => setApiKeyStatus(null), 5000);
            return;
        }

        setIsSavingApiKeys(true);
        try {
            await updateSettings(payload);
            setEditedApiKeys(new Set());
            await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            setApiKeyStatus("API keys saved successfully.");
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Failed to save API keys";
            setApiKeyStatus(`Error: ${message}`);
        } finally {
            setIsSavingApiKeys(false);
            setTimeout(() => setApiKeyStatus(null), 5000);
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
        <div className="space-y-6 max-w-4xl">
            {/* Benchmarks Section */}
            <div className={cardClassName}>
                <div className={cardHeadClassName}>
                    <h3 className={sectionTitleClassName}>Benchmarks</h3>
                    <span className={countBadgeClassName}>{benchmarks.length}</span>
                </div>
                <p className="text-xs text-muted-foreground mb-4">
                    Compared against your returns on the Performance tab.
                </p>
                
                <div className="space-y-4">
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                        {PRESET_BENCHMARKS.map(benchmark => (
                            <label
                                key={benchmark}
                                className={cn(
                                    "flex items-center gap-2 h-11 px-3 rounded-control border cursor-pointer transition-colors",
                                    benchmarks.includes(benchmark)
                                        ? "bg-primary/12 border-primary/30 text-foreground"
                                        : "bg-background border-border text-muted-foreground hover:bg-muted"
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
                                    className="rounded border-none bg-secondary text-primary focus:ring-ring"
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
                                    className={`${secondaryButtonClassName} px-2.5`}
                                >
                                    <Plus className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    </div>

                    {benchmarks.filter(b => !PRESET_BENCHMARKS.includes(b)).length > 0 && (
                        <div className="flex flex-wrap gap-2 pt-2">
                            {benchmarks.filter(b => !PRESET_BENCHMARKS.includes(b)).map(b => (
                                <span
                                    key={b}
                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary/12 text-primary-ink rounded-control text-sm font-medium"
                                >
                                    {b}
                                    <button
                                        type="button"
                                        onClick={() => onBenchmarksChange(benchmarks.filter(item => item !== b))}
                                        className="hover:bg-primary/20 p-0.5 rounded-md transition-colors cursor-pointer"
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
            <div className={cardClassName}>
                <div className={cardHeadClassName}>
                    <h3 className={sectionTitleClassName}>Webhook Integration</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-4 leading-relaxed">
                    Trigger a background data refresh externally by sending a POST request to{' '}
                    <code className="inline-block bg-muted px-2 py-0.5 rounded-md text-xs text-primary-ink font-mono border border-border align-middle">POST /api/webhook/refresh</code>
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
                            className={secondaryButtonClassName}
                        >
                            Test
                        </button>
                    </div>
                    {refreshStatus && (
                        <p className={`text-sm font-medium animate-in fade-in ${refreshStatus.startsWith('Error') ? 'text-down' : 'text-up'}`}>
                            {refreshStatus}
                        </p>
                    )}
                </div>
            </div>

            {/* IBKR Integration */}
            <div className={cardClassName}>
                <div className={cardHeadClassName}>
                    <h3 className={sectionTitleClassName}>Interactive Brokers Sync</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-4">
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

                <div className="flex flex-wrap items-center gap-4 card-inset p-4">
                    <button
                        type="button"
                        onClick={handleSaveIbkr}
                        disabled={isSavingIbkr || isSyncing}
                        className={secondaryButtonClassName}
                    >
                        {isSavingIbkr ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4"/>}
                        Save Credentials
                    </button>

                    <div className="h-6 w-px bg-border hidden md:block" />

                    <button
                        type="button"
                        onClick={handleSyncIbkr}
                        disabled={isSyncing || !ibkrToken || !ibkrQueryId}
                        className={primaryButtonClassName}
                    >
                        {isSyncing ? <Loader2 className="w-4 h-4 animate-spin" /> : "Sync Transactions Now"}
                    </button>

                    {syncStatus && (
                        <p className={`text-sm font-medium animate-in fade-in ${syncStatus.startsWith('Error') ? 'text-down' : 'text-up'}`}>
                            {syncStatus}
                        </p>
                    )}
                </div>
            </div>

            {/* API Keys Configuration */}
            <div className={cardClassName}>
                <div className={cardHeadClassName}>
                    <h3 className={sectionTitleClassName}>API Keys</h3>
                    <button
                        type="button"
                        onClick={() => setShowApiKeys(!showApiKeys)}
                        className="ml-auto inline-flex items-center gap-1.5 h-7 px-2.5 rounded-control text-xs font-medium border border-border bg-background hover:bg-muted text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                    >
                        {showApiKeys ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                        {showApiKeys ? "Hide Keys" : "Show Keys"}
                    </button>
                </div>
                <p className="text-xs text-muted-foreground mb-4 leading-relaxed">
                    Configure external API keys stored in <code className="inline-block bg-muted px-1.5 py-0.5 rounded text-xs font-mono text-primary-ink border border-border align-middle">.env</code> for AI stock analysis, valuation models, and supplementary market data feeds. Stored keys show only their last four characters &mdash; retype a field to replace it, or clear it to remove the key.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
                    <div className="space-y-1.5">
                        <label className={labelClassName}>Google Gemini API Key</label>
                        <input
                            type={showApiKeys ? "text" : "password"}
                            placeholder="AI stock analysis & screener"
                            value={geminiApiKey}
                            onChange={(e) => { setGeminiApiKey(e.target.value); markApiKeyEdited('gemini_api_key'); }}
                            className={inputClassName}
                        />
                    </div>
                    <div className="space-y-1.5">
                        <label className={labelClassName}>Financial Modeling Prep (FMP)</label>
                        <input
                            type={showApiKeys ? "text" : "password"}
                            placeholder="Financial statements & intrinsic models"
                            value={fmpApiKey}
                            onChange={(e) => { setFmpApiKey(e.target.value); markApiKeyEdited('fmp_api_key'); }}
                            className={inputClassName}
                        />
                    </div>
                    <div className="space-y-1.5">
                        <label className={labelClassName}>Thai SEC Open API</label>
                        <input
                            type={showApiKeys ? "text" : "password"}
                            placeholder="SSF / RMF mutual fund daily NAVs"
                            value={secThApiKey}
                            onChange={(e) => { setSecThApiKey(e.target.value); markApiKeyEdited('sec_th_api_key'); }}
                            className={inputClassName}
                        />
                    </div>
                    <div className="space-y-1.5">
                        <label className={labelClassName}>Bank of Thailand (BOT)</label>
                        <input
                            type={showApiKeys ? "text" : "password"}
                            placeholder="Historical THB exchange rates"
                            value={botApiKey}
                            onChange={(e) => { setBotApiKey(e.target.value); markApiKeyEdited('bot_api_key'); }}
                            className={inputClassName}
                        />
                    </div>
                    <div className="space-y-1.5 md:col-span-2">
                        <label className={labelClassName}>Tiingo API Key</label>
                        <input
                            type={showApiKeys ? "text" : "password"}
                            placeholder="Corporate actions & stock split verification"
                            value={tiingoApiKey}
                            onChange={(e) => { setTiingoApiKey(e.target.value); markApiKeyEdited('tiingo_api_key'); }}
                            className={inputClassName}
                        />
                    </div>
                </div>

                <div className="flex flex-wrap items-center gap-4 card-inset p-4">
                    <button
                        type="button"
                        onClick={handleSaveApiKeys}
                        disabled={isSavingApiKeys || editedApiKeys.size === 0}
                        className={primaryButtonClassName}
                    >
                        {isSavingApiKeys ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4"/>}
                        Save API Keys
                    </button>

                    {apiKeyStatus && (
                        <p className={`text-sm font-medium animate-in fade-in ${apiKeyStatus.startsWith('Error') ? 'text-down' : 'text-up'}`}>
                            {apiKeyStatus}
                        </p>
                    )}
                </div>
            </div>

            {/* Cache Management Section */}
            <div className={cardClassName}>
                <div className={cardHeadClassName}>
                    <h3 className={sectionTitleClassName}>System Cache</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-4 leading-relaxed">
                    Clear local caches to resolve data discrepancies. This drops historical performance data, market quotes, and metadata, forcing a fresh download on the next load.
                </p>
                <div className="space-y-6">
                    <div className="flex items-center gap-4 flex-wrap">
                        <button
                            type="button"
                            onClick={handleClearCache}
                            className={`h-9 px-3.5 rounded-control text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-down focus-visible:ring-offset-2 ring-offset-background border cursor-pointer ${confirmClear
                                ? 'bg-down text-white border-transparent hover:opacity-90'
                                : 'border-down/50 text-down bg-down/5 hover:bg-down/12'}`}
                        >
                            {confirmClear ? "Click again to Confirm" : "Clear System Cache"}
                        </button>
                        {clearStatus && (
                            <p className={`text-sm font-medium animate-in fade-in ${clearStatus.startsWith('Error') ? 'text-down' : 'text-up'}`}>
                                {clearStatus}
                            </p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
