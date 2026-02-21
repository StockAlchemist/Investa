import React, { useState, useEffect } from 'react';
import { Save, AlertCircle } from 'lucide-react';
import { Settings as SettingsType } from '../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { updateSettings } from '../lib/api';

interface YieldSettingsProps {
    settings: SettingsType;
    availableAccounts: string[];
    holdings: Record<string, any>[]; // Using Record<string, any>[] instead of any[] to satisfy linter
    onSettingsUpdated: () => void;
}

export default function YieldSettings({ settings, availableAccounts, holdings, onSettingsUpdated }: YieldSettingsProps) {
    const queryClient = useQueryClient();

    // Local State for Form Values
    const [localRates, setLocalRates] = useState<Record<string, number>>({});
    const [localThresholds, setLocalThresholds] = useState<Record<string, number>>({});

    const [isSaving, setIsSaving] = useState(false);
    const [saveStatus, setSaveStatus] = useState<'success' | 'error' | null>(null);
    const [statusMessage, setStatusMessage] = useState('');

    // Initialize state from settings props
    useEffect(() => {
        if (settings) {
            setLocalRates(settings.account_interest_rates || {});
            setLocalThresholds(settings.interest_free_thresholds || {});
        }
    }, [settings]);

    const handleRateChange = (account: string, value: string) => {
        const numVal = parseFloat(value);
        setLocalRates(prev => ({
            ...prev,
            [account]: isNaN(numVal) ? 0 : numVal
        }));
        setSaveStatus(null);
    };

    const handleThresholdChange = (account: string, value: string) => {
        // Strip non-digits
        const cleanValue = value.replace(/\D/g, '');
        const numVal = cleanValue === '' ? 0 : parseInt(cleanValue, 10);

        setLocalThresholds(prev => ({
            ...prev,
            [account]: numVal
        }));
        setSaveStatus(null);
    };

    const handleSave = async () => {
        setIsSaving(true);
        setSaveStatus(null);
        setStatusMessage('');

        try {
            await updateSettings({
                account_interest_rates: localRates,
                interest_free_thresholds: localThresholds
            });

            await userQueryClientInvalidation();
            onSettingsUpdated();

            setSaveStatus('success');
            setStatusMessage('Yield settings saved successfully.');

            // Clear success message after 3 seconds
            setTimeout(() => {
                setSaveStatus(null);
                setStatusMessage('');
            }, 3000);

        } catch (error) {
            console.error("Error saving yield settings:", error);
            setSaveStatus('error');
            setStatusMessage('Failed to save settings. Please try again.');
        } finally {
            setIsSaving(false);
        }
    };

    const userQueryClientInvalidation = async () => {
        await queryClient.invalidateQueries({ queryKey: ['settings'] });
        await queryClient.invalidateQueries({ queryKey: ['portfolio'] });
        await queryClient.invalidateQueries({ queryKey: ['dividends'] }); // Invalidate dividends too
    };

    // Filter useful accounts (with cash)
    const activeAccounts = availableAccounts.filter(account => {
        const accountCash = holdings
            .filter(h => h.Account === account && (h.Symbol === '$CASH' || h.Symbol === 'Cash' || h.Symbol.includes('Cash')))
            .reduce((sum, h) => {
                const mvKey = Object.keys(h).find(k => k.startsWith('Market Value'));
                const val = mvKey ? (h[mvKey] as number) : 0;
                return sum + (val || 0);
            }, 0);

        // Show account if it has cash OR has existing settings
        return Math.abs(accountCash) > 0.01 ||
            (settings.account_interest_rates && settings.account_interest_rates[account] > 0) ||
            (settings.interest_free_thresholds && settings.interest_free_thresholds[account] > 0);
    });

    const inputClassName = "w-full rounded-md border border-border bg-background text-foreground shadow-sm focus:border-cyan-500 focus:ring-cyan-500 px-3 py-2 text-sm outline-none focus:ring-1";

    return (
        <div className="max-w-4xl space-y-6">
            <div className="flex justify-between items-center mb-4">
                <div>
                    <h3 className="text-lg font-medium">Cash Yield Settings</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                        Configure annual interest rates and interest-free thresholds for your cash balances.
                    </p>
                </div>
                <button
                    onClick={handleSave}
                    disabled={isSaving}
                    className="flex items-center gap-2 px-4 py-2 bg-[#0097b2] text-white rounded-md hover:bg-[#0086a0] transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium shadow-sm"
                >
                    {isSaving ? (
                        <>Saving...</>
                    ) : (
                        <>
                            <Save size={16} />
                            Save Changes
                        </>
                    )}
                </button>
            </div>

            {saveStatus && (
                <div className={`p-4 rounded-md border text-sm flex items-center gap-2 ${saveStatus === 'success'
                    ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                    : 'bg-red-500/10 border-red-500/20 text-red-600 dark:text-red-400'
                    }`}>
                    {saveStatus === 'success' ? <Save size={16} /> : <AlertCircle size={16} />}
                    {statusMessage}
                </div>
            )}

            <div className="bg-card border border-border rounded-lg overflow-hidden shadow-sm">
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-border">
                        <thead className="bg-secondary/50">
                            <tr>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Account</th>
                                <th scope="col" className="px-6 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Cash Balance</th>
                                <th scope="col" className="px-6 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Est. Annual Interest</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Annual Rate (%)</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Exempt Threshold</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border bg-card">
                            {activeAccounts.length === 0 ? (
                                <tr>
                                    <td colSpan={5} className="px-6 py-8 text-center text-muted-foreground text-sm">
                                        No accounts with cash balances found.
                                    </td>
                                </tr>
                            ) : (
                                activeAccounts.map(account => {
                                    const accountCash = holdings
                                        .filter(h => h.Account === account && (h.Symbol === '$CASH' || h.Symbol === 'Cash' || h.Symbol.includes('Cash')))
                                        .reduce((sum, h) => {
                                            const mvKey = Object.keys(h).find(k => k.startsWith('Market Value'));
                                            const val = mvKey ? (h[mvKey] as number) : 0;
                                            return sum + (val || 0);
                                        }, 0);

                                    return (
                                        <tr key={account} className="hover:bg-accent/5 transition-colors">
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-foreground">{account}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-mono text-muted-foreground">
                                                {accountCash.toLocaleString(undefined, { style: 'currency', currency: 'USD' })}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                                                {(() => {
                                                    const rate = (localRates[account] || 0) / 100;
                                                    const threshold = localThresholds[account] || 0;
                                                    const interest = Math.max(0, accountCash - threshold) * rate;
                                                    return interest.toLocaleString(undefined, { style: 'currency', currency: 'USD' });
                                                })()}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="relative max-w-[150px]">
                                                    <input
                                                        type="number"
                                                        step="0.01"
                                                        value={localRates[account] || ''} // Use empty string for 0 display if preferred, or maintain 0. But controlled usually needs value.
                                                        onChange={(e) => handleRateChange(account, e.target.value)}
                                                        placeholder="0.00"
                                                        className={`${inputClassName} pr-8`}
                                                    />
                                                    <span className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground text-xs pointer-events-none">%</span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="max-w-[150px]">
                                                    <input
                                                        type="text"
                                                        value={localThresholds[account] ? localThresholds[account].toLocaleString('en-US') : ''}
                                                        onChange={(e) => handleThresholdChange(account, e.target.value)}
                                                        placeholder="0"
                                                        className={inputClassName}
                                                    />
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
    );
}
