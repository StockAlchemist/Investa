
"use client";

import React, { useState, useEffect } from 'react';
import { fetchSettings, saveManualOverride, triggerRefresh, Settings as SettingsType } from '../lib/api';

export default function Settings() {
    const [settings, setSettings] = useState<SettingsType | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [refreshSecret, setRefreshSecret] = useState('');
    const [refreshStatus, setRefreshStatus] = useState<string | null>(null);

    // Manual Override Form State
    const [overrideSymbol, setOverrideSymbol] = useState('');
    const [overridePrice, setOverridePrice] = useState('');

    useEffect(() => {
        loadSettings();
    }, []);

    const loadSettings = async () => {
        try {
            setLoading(true);
            const data = await fetchSettings();
            setSettings(data);
            setError(null);
        } catch (err) {
            setError('Failed to load settings');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleManualOverrideSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!overrideSymbol || !overridePrice) return;

        try {
            await saveManualOverride(overrideSymbol, parseFloat(overridePrice));
            setOverrideSymbol('');
            setOverridePrice('');
            await loadSettings(); // Reload to show new override
        } catch (err) {
            console.error(err);
            alert('Failed to save override');
        }
    };

    const handleDeleteOverride = async (symbol: string) => {
        try {
            await saveManualOverride(symbol, null);
            await loadSettings();
        } catch (err) {
            console.error(err);
            alert('Failed to delete override');
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

    if (loading) return <div className="p-4">Loading settings...</div>;
    if (error) return <div className="p-4 text-red-500">{error}</div>;

    // matches API: settings.manual_overrides is the dictionary of overrides
    const overrides = settings?.manual_overrides || {};

    return (
        <div className="space-y-6 pb-20">
            <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Manual Price Overrides</h2>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                    Manually set the price for specific symbols. This overrides the market data provider.
                </p>

                {/* List Existing Overrides */}
                <div className="mb-6">
                    <h3 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-2">Active Overrides</h3>
                    {Object.keys(overrides).length === 0 ? (
                        <p className="text-sm text-gray-500 italic">No manual overrides set.</p>
                    ) : (
                        <ul className="divide-y divide-gray-200 dark:divide-gray-700 border dark:border-gray-700 rounded-md">
                            {Object.entries(overrides).map(([symbol, data]: [string, any]) => (
                                <li key={symbol} className="p-3 flex justify-between items-center hover:bg-gray-50 dark:hover:bg-gray-700">
                                    <div>
                                        <span className="font-bold text-gray-900 dark:text-white mr-2">{symbol}</span>
                                        <span className="text-sm text-gray-500 dark:text-gray-400">
                                            Current Override: <span className="font-mono text-gray-800 dark:text-gray-200">${data.price}</span>
                                            {data.updated_at && <span className="ml-2 text-xs">({new Date(data.updated_at).toLocaleDateString()})</span>}
                                        </span>
                                    </div>
                                    <button
                                        onClick={() => handleDeleteOverride(symbol)}
                                        className="text-red-600 hover:text-red-800 text-sm font-medium px-2 py-1 rounded hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
                                    >
                                        Remove
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                {/* Add New Override Form */}
                <form onSubmit={handleManualOverrideSubmit} className="flex gap-4 items-end bg-gray-50 dark:bg-gray-700/30 p-4 rounded-md border border-gray-200 dark:border-gray-600">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Symbol</label>
                        <input
                            type="text"
                            value={overrideSymbol}
                            onChange={(e) => setOverrideSymbol(e.target.value.toUpperCase())}
                            placeholder="AAPL"
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Price</label>
                        <input
                            type="number"
                            step="0.01"
                            min="0"
                            value={overridePrice}
                            onChange={(e) => setOverridePrice(e.target.value)}
                            placeholder="150.00"
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2"
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={!overrideSymbol || !overridePrice}
                        className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Set Override
                    </button>
                </form>
            </div>

            <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Webhook Integration</h2>
                <div className="space-y-4">
                    <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-md">
                        <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
                            <strong>Endpoint:</strong> <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded">POST /api/webhook/refresh</code>
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-300">
                            Send a POST request with the following body to invalidate the cache and force a data refresh.
                        </p>
                        <pre className="mt-2 bg-gray-900 text-gray-100 p-3 rounded-md text-xs overflow-x-auto">
                            {`{
  "secret": "YOUR_WEBHOOK_SECRET"
}`}
                        </pre>
                    </div>

                    {/* Manual Test Tool */}
                    <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Manual Test</h3>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                placeholder="Enter Webhook Secret"
                                value={refreshSecret}
                                onChange={(e) => setRefreshSecret(e.target.value)}
                                className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white px-3 py-2"
                            />
                            <button
                                onClick={handleRefresh}
                                className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:bg-gray-700 dark:text-white dark:border-gray-600 dark:hover:bg-gray-600"
                            >
                                Trigger Refresh
                            </button>
                        </div>
                        {refreshStatus && (
                            <p className={`mt-2 text-sm ${refreshStatus.startsWith('Error') ? 'text-red-600' : 'text-green-600'}`}>
                                {refreshStatus}
                            </p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
