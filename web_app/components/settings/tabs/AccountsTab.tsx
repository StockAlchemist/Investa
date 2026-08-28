import React, { useState } from 'react';
import { Trash2, DollarSign, Settings as SettingsIcon } from 'lucide-react';
import { Settings as SettingsType, Holding, updateSettings } from '../../../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { useAuth } from '../../../context/AuthContext';
import { cn } from '../../../lib/utils';
import AccountGroupManager from '../../AccountGroupManager';
import YieldSettings from '../../YieldSettings';
import {
    cardClassName,
    sectionTitleClassName,
    labelClassName,
    inputClassName,
    compactInputClassName
} from '../constants';

interface AccountsTabProps {
    settings: SettingsType | null;
    availableAccounts: string[];
    holdings: Holding[];
}

export const AccountsTab: React.FC<AccountsTabProps> = ({
    settings,
    availableAccounts,
    holdings,
}) => {
    const queryClient = useQueryClient();
    const { user } = useAuth();
    const [newCurrency, setNewCurrency] = useState('');

    const availableCurrencies = settings?.available_currencies || ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'HKD', 'NZD', 'THB'];
    const accountCurrencyMap = settings?.account_currency_map || {};

    const addCurrency = async () => {
        if (!newCurrency || availableCurrencies.includes(newCurrency)) return;
        try {
            const updated = [...availableCurrencies, newCurrency];
            await updateSettings({ available_currencies: updated });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            setNewCurrency('');
        } catch {
            alert("Failed to add currency");
        }
    };

    const removeCurrency = async (curr: string) => {
        if (availableCurrencies.length <= 1) {
            alert("Must keep at least one available currency");
            return;
        }
        try {
            const updated = availableCurrencies.filter(c => c !== curr);
            await updateSettings({ available_currencies: updated });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
        } catch {
            alert("Failed to remove currency");
        }
    };

    const updateAccountCurrency = async (account: string, currency: string) => {
        try {
            const updated = { ...accountCurrencyMap, [account]: currency };
            await updateSettings({ account_currency_map: updated });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            queryClient.invalidateQueries({ queryKey: ['portfolio'] });
        } catch {
            alert("Failed to update account currency");
        }
    };

    const updateAccountCashMode = async (account: string, mode: string) => {
        try {
            const currentMap = settings?.account_cash_mode_map || {};
            const updated = { ...currentMap, [account]: mode };
            await updateSettings({ account_cash_mode_map: updated });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            queryClient.invalidateQueries({ queryKey: ['portfolio'] });
        } catch {
            alert("Failed to update account cash mode");
        }
    };

    const updateAccountClosureDate = async (account: string, date: string | null) => {
        try {
            const currentMap = { ...(settings?.account_closure_dates || {}) };
            if (date) {
                currentMap[account] = date;
            } else {
                delete currentMap[account];
            }
            await updateSettings({ account_closure_dates: currentMap });
            queryClient.invalidateQueries({ queryKey: ['settings', user?.username] });
            queryClient.invalidateQueries({ queryKey: ['portfolio'] });
        } catch {
            alert("Failed to update account closure date");
        }
    };

    const configurableAccounts = availableAccounts.filter(a => a !== 'All Accounts');

    return (
        <div className="space-y-10">
            {/* Account Groups section */}
            {settings && (
                <AccountGroupManager
                    availableAccounts={availableAccounts}
                    settings={settings}
                    onUpdate={() => queryClient.invalidateQueries({ queryKey: ['settings', user?.username] })}
                />
            )}

            <div className="space-y-8">
                {/* Available Currencies Section */}
                <div className={cardClassName}>
                    <div className="mb-2">
                        <h3 className={sectionTitleClassName}>
                            <DollarSign className="w-5 h-5 text-amber-500" />
                            Available Currencies
                            <span className="text-xs font-medium text-muted-foreground bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-full ml-1">
                                {availableCurrencies.length}
                            </span>
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
                                        className="opacity-40 group-hover:opacity-100 text-down hover:text-down transition-opacity cursor-pointer"
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
                            className="px-5 py-2.5 bg-amber-500 hover:bg-amber-600 text-white rounded-xl font-medium shadow-sm transition-colors disabled:opacity-50 cursor-pointer"
                        >
                            Add
                        </button>
                    </div>
                </div>

                {/* Account Preferences Section */}
                <div className={cardClassName}>
                    <div className="mb-2">
                        <h3 className={sectionTitleClassName}>
                            <SettingsIcon className="w-5 h-5 text-zinc-500" />
                            Account Preferences
                            <span className="text-xs font-medium text-muted-foreground bg-black/5 dark:bg-white/10 px-2 py-0.5 rounded-full ml-1">
                                {configurableAccounts.length}
                            </span>
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
                                                    aria-label="Default Currency"
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
                                                    aria-label="Cash Management"
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
                                                            className="shrink-0 p-2 text-muted-foreground hover:text-down hover:bg-down/12 rounded-lg transition-colors cursor-pointer"
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
            </div>

            {/* Cash Yield section */}
            {settings && (
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
        </div>
    );
};
