import React, { useEffect, useState } from 'react';
import { fetchSettings, Settings as SettingsType } from '../lib/api';

export default function Settings() {
    const [settings, setSettings] = useState<SettingsType | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function loadSettings() {
            try {
                const data = await fetchSettings();
                setSettings(data);
            } catch (err) {
                setError('Failed to load settings');
                console.error(err);
            } finally {
                setLoading(false);
            }
        }
        loadSettings();
    }, []);

    if (loading) return <div className="p-8 text-center text-gray-500">Loading settings...</div>;
    if (error) return <div className="p-8 text-center text-red-500">{error}</div>;
    if (!settings) return null;

    return (
        <div className="space-y-8 pb-10">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white border-b pb-2 dark:border-gray-700">
                    Manual Overrides (Metadata)
                </h2>
                <div className="overflow-x-auto">
                    <pre className="text-xs bg-gray-50 dark:bg-gray-900 p-4 rounded border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300">
                        {JSON.stringify(settings.manual_overrides, null, 2)}
                    </pre>
                </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white border-b pb-2 dark:border-gray-700">
                    User Symbol Map
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(settings.user_symbol_map).map(([internal, yf]) => (
                        <div key={internal} className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700">
                            <span className="font-medium text-gray-700 dark:text-gray-300">{internal}</span>
                            <span className="text-gray-500 dark:text-gray-500">→</span>
                            <span className="font-mono text-blue-600 dark:text-blue-400">{yf}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white border-b pb-2 dark:border-gray-700">
                    Excluded Symbols
                </h2>
                <div className="flex flex-wrap gap-2">
                    {settings.user_excluded_symbols.map((symbol) => (
                        <span key={symbol} className="px-3 py-1 bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300 rounded-full text-sm font-medium border border-red-200 dark:border-red-800">
                            {symbol}
                        </span>
                    ))}
                </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white border-b pb-2 dark:border-gray-700">
                    Account Currencies
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(settings.account_currency_map).map(([account, currency]) => (
                        <div key={account} className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700">
                            <span className="font-medium text-gray-700 dark:text-gray-300">{account}</span>
                            <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 rounded text-xs font-bold">
                                {currency}
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
