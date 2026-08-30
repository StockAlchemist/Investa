import React, { useState } from 'react';
import { Trash2 } from 'lucide-react';
import { Settings as SettingsType, Holding, updateSettings } from '../../../lib/api';
import { useQueryClient } from '@tanstack/react-query';
import { useAuth } from '../../../context/AuthContext';
import { cn } from '../../../lib/utils';
import AccountGroupManager from '../../AccountGroupManager';
import YieldSettings from '../../YieldSettings';
import {
    cardClassName,
    cardHeadClassName,
    sectionTitleClassName,
    countBadgeClassName,
    labelClassName,
    inputClassName,
    compactInputClassName,
    primaryButtonClassName,
    insetClassName,
    chipActiveClassName
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
        <div className="space-y-6">
            {/* Account Groups section */}
            {settings && (
                <AccountGroupManager
                    availableAccounts={availableAccounts}
                    settings={settings}
                    onUpdate={() => queryClient.invalidateQueries({ queryKey: ['settings', user?.username] })}
                />
            )}

            <>
                {/* Available Currencies Section */}
                <div className={cardClassName}>
                    <div className={cardHeadClassName}>
                        <h3 className={sectionTitleClassName}>Available Currencies</h3>
                        <span className={countBadgeClassName}>{availableCurrencies.length}</span>
                    </div>
                    <p className="text-xs text-muted-foreground mb-4">Currencies you can assign to accounts below.</p>

                    {availableCurrencies.length > 0 && (
                        <div className="flex flex-wrap gap-2 mb-5">
                            {availableCurrencies.map(curr => (
                                <div
                                    key={curr}
                                    className={`group ${chipActiveClassName} hover:border-primary/40`}
                                >
                                    <span className="font-bold tabular-nums">{curr}</span>
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

                    <div className="flex gap-3 items-end max-w-md pt-5 border-t border-border">
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
                            className={primaryButtonClassName}
                        >
                            Add
                        </button>
                    </div>
                </div>

                {/* Account Preferences Section */}
                <div className={cardClassName}>
                    <div className={cardHeadClassName}>
                        <h3 className={sectionTitleClassName}>Account Preferences</h3>
                        <span className={countBadgeClassName}>{configurableAccounts.length}</span>
                    </div>
                    <p className="text-xs text-muted-foreground mb-4">Base currency, cash automation and closure date, per account.</p>

                    {configurableAccounts.length === 0 ? (
                        <div className="text-center text-sm text-muted-foreground py-12 border border-dashed border-border rounded-inset">
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
                                            insetClassName,
                                            "transition-colors hover:border-primary/30",
                                            isEffectivelyClosed && "opacity-70"
                                        )}
                                    >
                                        <div className="flex items-center gap-2 mb-4 pb-3 border-b border-border">
                                            <span className={cn(
                                                "font-bold text-foreground",
                                                isEffectivelyClosed && "line-through"
                                            )}>{account}</span>
                                            {isEffectivelyClosed && (
                                                <span className="text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 bg-muted text-muted-foreground rounded-full">
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
            </>

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
