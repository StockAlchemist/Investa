'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { SettingsProps, Tab } from './settings/types';
import { TABS } from './settings/constants';
import { AccountsTab } from './settings/tabs/AccountsTab';
import { SymbolsTab } from './settings/tabs/SymbolsTab';
import { OverridesTab } from './settings/tabs/OverridesTab';
import { AdvancedTab } from './settings/tabs/AdvancedTab';
import { AppearanceTab } from './settings/tabs/AppearanceTab';
import { ProfileSecurityTab } from './settings/tabs/ProfileSecurityTab';

export type { SettingsProps, Tab };

/**
 * Settings is a tab like any other: a column of standard cards under the page
 * header, with a category rail built from the sidebar's own NavItem anatomy.
 *
 * It used to be the app's one visual exception — five per-category colours, a
 * rounded-3xl translucent panel wrapping the whole tab, and a banner that
 * repeated the tab's name under a 44px coloured icon. All three are gone; the
 * cards, the section labels and the indigo accent are the ones every other tab
 * already draws (see globals.css `.card-standard` and Theme.swift `CardStyle`).
 */
export default function Settings({
    settings,
    holdings,
    availableAccounts,
    initialTab = 'accounts',
    benchmarks,
    onBenchmarksChange,
}: SettingsProps) {
    const [activeTab, setActiveTab] = useState<Tab>(initialTab);
    const activeChipRef = useRef<HTMLButtonElement>(null);

    // Below lg the rail is a horizontal scroller. Keep the selected category in
    // view — Settings can open on a tab the user did not pick (the user menu
    // opens Profile & Security), which would otherwise start off-screen with
    // nothing on screen saying which category is showing.
    useEffect(() => {
        activeChipRef.current?.scrollIntoView({ block: 'nearest', inline: 'center' });
    }, [activeTab]);

    // Counts ride the rail so a category says how much it holds before it opens.
    const counts: Partial<Record<Tab, number>> = {
        accounts: settings ? Object.keys(settings.account_groups ?? {}).length : undefined,
        symbols: settings ? Object.keys(settings.user_symbol_map ?? {}).length : undefined,
        overrides: settings ? Object.keys(settings.manual_overrides ?? {}).length : undefined,
    };

    return (
        <div className="pb-20 max-w-7xl mx-auto px-4 md:px-8">
            {/* Below md the PageHeader title is hidden, so the screen names itself here. */}
            <h2 className="md:hidden text-2xl font-bold tracking-tight text-foreground mb-5">Settings</h2>

            <div className="flex flex-col lg:flex-row gap-6 lg:gap-8 items-start">
                {/* Category rail — vertical at lg+, one horizontal row of chips below it. */}
                <nav className="w-full lg:w-[216px] lg:shrink-0" aria-label="Settings categories">
                    <p className="hidden lg:block section-label px-3 mb-2">Categories</p>
                    <div className="flex lg:flex-col gap-2 lg:gap-0.5 overflow-x-auto lg:overflow-visible pb-1 lg:pb-0 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                        {TABS.map((tab) => {
                            const Icon = tab.icon;
                            const isActive = activeTab === tab.id;
                            const count = counts[tab.id];
                            return (
                                <button
                                    key={tab.id}
                                    ref={isActive ? activeChipRef : undefined}
                                    onClick={() => setActiveTab(tab.id)}
                                    title={tab.description}
                                    aria-current={isActive ? 'page' : undefined}
                                    className={cn(
                                        'group/item relative flex items-center gap-2.5 h-9 shrink-0 whitespace-nowrap rounded-lg px-3 text-sm font-medium transition-all duration-150 cursor-pointer border lg:border-0 lg:w-full',
                                        isActive
                                            ? 'bg-primary/15 text-primary-ink font-semibold border-primary/30'
                                            : 'text-muted-foreground border-border lg:border-transparent hover:bg-muted hover:text-foreground',
                                    )}
                                >
                                    {isActive && (
                                        <span className="hidden lg:block absolute left-0 inset-y-[6px] w-[3px] bg-primary rounded-r-full" />
                                    )}
                                    <Icon className="w-4 h-4 shrink-0" />
                                    <span className="truncate">{tab.label}</span>
                                    {count !== undefined && count > 0 && (
                                        <span className="lg:ml-auto text-[11px] font-bold tabular-nums opacity-75">
                                            {count}
                                        </span>
                                    )}
                                </button>
                            );
                        })}
                    </div>
                </nav>

                {/* Cards sit on the page ground — no panel, no glass, no banner. */}
                <div className="flex-1 min-w-0">
                    {!settings && (
                        <div className="flex flex-col items-center justify-center py-20 text-muted-foreground animate-in fade-in duration-300">
                            <Loader2 className="w-8 h-8 animate-spin mb-3 text-primary" />
                            <p className="text-sm font-medium">Loading settings…</p>
                        </div>
                    )}

                    <div className="animate-in fade-in duration-300 fill-mode-both">
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

                        {activeTab === 'appearance' && (
                            <AppearanceTab />
                        )}

                        {activeTab === 'account' && (
                            <ProfileSecurityTab />
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
