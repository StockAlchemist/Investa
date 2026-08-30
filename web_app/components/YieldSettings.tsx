import React, { useState, useEffect, useMemo } from 'react';
import { Save, AlertCircle, CheckCircle, Activity } from 'lucide-react';
import { Settings as SettingsType, Holding } from '../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { updateSettings } from '../lib/api';
import { useAuth } from '../context/AuthContext';

interface YieldSettingsProps {
    settings: SettingsType;
    availableAccounts: string[];
    holdings: Holding[];
    onSettingsUpdated: () => void;
}

export default function YieldSettings({ settings, availableAccounts, holdings, onSettingsUpdated }: YieldSettingsProps) {
    const queryClient = useQueryClient();
    const { user } = useAuth();

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
        await queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
        await queryClient.invalidateQueries({ queryKey: ['portfolio', user?.username] });
        await queryClient.invalidateQueries({ queryKey: ['dividends', user?.username] }); 
    };

    // Pre-calculate cash balances and fx rates to avoid expensive loops on every keystroke
    const accountCashData = useMemo(() => {
        const data: Record<string, { balance: number, currency: string, fxRate: number }> = {};
        let defaultCurrency = 'USD';

        for (const h of holdings) {
            if (h.Account && (h.Symbol === '$CASH' || h.Symbol === 'Cash' || h.Symbol.includes('Cash'))) {
                const mvKey = Object.keys(h).find(k => k.startsWith('Market Value ('));
                const val = mvKey ? (h[mvKey] as number) : 0;
                
                let curr = 'USD';
                if (mvKey) {
                    const match = mvKey.match(/\(([^)]+)\)/);
                    if (match) curr = match[1];
                }
                defaultCurrency = curr;

                if (!data[h.Account]) {
                    data[h.Account] = { balance: 0, currency: curr, fxRate: h.fx_rate || 1 };
                }
                data[h.Account].balance += val || 0;
            }
        }
        return { data, defaultCurrency };
    }, [holdings]);

    const accountCashBalances = useMemo(() => {
        const balances: Record<string, number> = {};
        for (const [acc, info] of Object.entries(accountCashData.data)) {
            balances[acc] = info.balance;
        }
        return balances;
    }, [accountCashData]);

    // Filter useful accounts (with cash or existing settings)
    const activeAccounts = useMemo(() => {
        return availableAccounts.filter(account => {
            const accountCash = accountCashBalances[account] || 0;

            // Show account if it has cash OR has existing settings
            return Math.abs(accountCash) > 0.01 ||
                (settings.account_interest_rates && settings.account_interest_rates[account] > 0) ||
                (settings.interest_free_thresholds && settings.interest_free_thresholds[account] > 0);
        });
    }, [availableAccounts, accountCashBalances, settings]);

    const inputClassName = "w-full h-9 rounded-control border border-input bg-background text-foreground px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background transition-colors";
    const primaryButtonClassName = "h-8 px-3 rounded-control bg-primary text-primary-foreground hover:bg-primary-hover text-xs font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ring-offset-background disabled:opacity-50 disabled:pointer-events-none transition-colors inline-flex items-center justify-center gap-1.5 shrink-0 cursor-pointer";

    return (
        <div className="card-standard p-6 space-y-5">
            <div className="flex items-center gap-2.5">
                <h3 className="section-label">Cash Yield Assumptions</h3>
                <span className="text-xs text-muted-foreground ml-auto hidden sm:block">Applied to idle cash above the exempt threshold</span>
                <button
                    onClick={handleSave}
                    disabled={isSaving}
                    className={primaryButtonClassName}
                >
                    {isSaving ? (
                        <>Saving...</>
                    ) : (
                        <>
                            <Save className="w-3.5 h-3.5" />
                            Save Changes
                        </>
                    )}
                </button>
            </div>

            {saveStatus && (
                <div className={`p-4 rounded-xl text-sm flex items-center gap-3 animate-in fade-in slide-in-from-top-2 border ${saveStatus === 'success'
                    ? 'bg-up/12 border-up/25 text-up'
                    : 'bg-down/12 border-down/25 text-down'
                    }`}>
                    {saveStatus === 'success' ? <CheckCircle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
                    <span className="font-medium">{statusMessage}</span>
                </div>
            )}

            <div className="card-inset overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-border">
                        <thead className="bg-black/5 dark:bg-white/5">
                            <tr>
                                <th scope="col" className="px-6 py-4 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Account</th>
                                <th scope="col" className="px-6 py-4 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Cash Balance</th>
                                <th scope="col" className="px-6 py-4 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Est. Annual Interest</th>
                                <th scope="col" className="px-6 py-4 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Annual Rate (%)</th>
                                <th scope="col" className="px-6 py-4 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                                    Exempt Threshold {accountCashData.defaultCurrency !== 'USD' && `(${accountCashData.defaultCurrency})`}
                                </th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-black/5 dark:divide-white/5">
                            {activeAccounts.length === 0 ? (
                                <tr>
                                    <td colSpan={5} className="px-6 py-12 text-center text-muted-foreground">
                                        <div className="flex flex-col items-center justify-center gap-3">
                                            <Activity className="w-8 h-8 opacity-50" />
                                            <p>No accounts with cash balances found.</p>
                                        </div>
                                    </td>
                                </tr>
                            ) : (
                                activeAccounts.map(account => {
                                    const accountCash = accountCashBalances[account] || 0;
                                    const accData = accountCashData.data[account] || { balance: 0, currency: accountCashData.defaultCurrency, fxRate: 1 };
                                    const displayCurrency = accData.currency;
                                    const fxRate = accData.fxRate || 1;
                                    
                                    // Threshold is stored in settings in USD. Convert to display currency for UI.
                                    const thresholdUSD = localThresholds[account] || 0;
                                    const thresholdDisplay = Math.round(thresholdUSD * fxRate);

                                    return (
                                        <tr key={account} className="hover:bg-muted/50 transition-colors group">
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-foreground">{account}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-mono text-muted-foreground font-medium">
                                                {accountCash.toLocaleString(undefined, { style: 'currency', currency: displayCurrency })}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-mono text-up font-bold">
                                                {(() => {
                                                    const rate = (localRates[account] || 0) / 100;
                                                    const interest = Math.max(0, accountCash - thresholdDisplay) * rate;
                                                    return interest.toLocaleString(undefined, { style: 'currency', currency: displayCurrency });
                                                })()}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="relative max-w-[150px]">
                                                    <input
                                                        type="number"
                                                        step="0.01"
                                                        value={localRates[account] || ''} 
                                                        onChange={(e) => handleRateChange(account, e.target.value)}
                                                        placeholder="0.00"
                                                        className={`${inputClassName} pr-8 font-mono`}
                                                    />
                                                    <span className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground text-xs pointer-events-none font-bold">%</span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="max-w-[150px]">
                                                    <input
                                                        type="text"
                                                        value={thresholdDisplay ? thresholdDisplay.toLocaleString('en-US') : ''}
                                                        onChange={(e) => {
                                                            const cleanValue = e.target.value.replace(/\D/g, '');
                                                            const numValDisplay = cleanValue === '' ? 0 : parseInt(cleanValue, 10);
                                                            // Convert back to USD for saving
                                                            const numValUSD = Math.round(numValDisplay / fxRate);
                                                            setLocalThresholds(prev => ({ ...prev, [account]: numValUSD }));
                                                            setSaveStatus(null);
                                                        }}
                                                        placeholder="0"
                                                        className={`${inputClassName} font-mono`}
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
