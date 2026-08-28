'use client';

import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';
import { SettingsProps, Tab } from './settings/types';
import { TABS } from './settings/constants';
import { AccountsTab } from './settings/tabs/AccountsTab';
import { SymbolsTab } from './settings/tabs/SymbolsTab';
import { OverridesTab } from './settings/tabs/OverridesTab';
import { AdvancedTab } from './settings/tabs/AdvancedTab';
import { ProfileSecurityTab } from './settings/tabs/ProfileSecurityTab';

export type { SettingsProps, Tab };

export default function Settings({
    settings,
    holdings,
    availableAccounts,
    initialTab = 'accounts',
    benchmarks,
    onBenchmarksChange,
}: SettingsProps) {
    const [activeTab, setActiveTab] = useState<Tab>(initialTab);
    const activeTabObj = TABS.find(t => t.id === activeTab);

    return (
        <div className="pb-20 max-w-7xl mx-auto px-4 md:px-8">
            <div className="mb-8 space-y-1">
                <h2 className="text-2xl font-bold tracking-tight text-foreground">Settings</h2>
                <p className="text-muted-foreground text-sm">
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
                                className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl font-medium transition-all duration-200 text-sm cursor-pointer ${
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
                        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 fill-mode-both">
                            {activeTab === 'accounts' && (
                                <AccountsTab
                                    settings={settings}
                                    availableAccounts={availableAccounts}
                                    holdings={holdings}
                                />
                            )}

                            {activeTab === 'symbols' && (
                                <SymbolsTab settings={settings} />
                            )}

                            {activeTab === 'overrides' && (
                                <OverridesTab settings={settings} holdings={holdings} />
                            )}

                            {activeTab === 'advanced' && (
                                <AdvancedTab
                                    settings={settings}
                                    benchmarks={benchmarks}
                                    onBenchmarksChange={onBenchmarksChange}
                                />
                            )}

                            {activeTab === 'account' && (
                                <ProfileSecurityTab />
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
