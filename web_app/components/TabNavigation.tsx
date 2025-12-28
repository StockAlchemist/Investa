import React from 'react';
import { cn } from '@/lib/utils';

interface TabNavigationProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
}

const TABS = [
    { id: 'performance', label: 'Performance' },
    { id: 'transactions', label: 'Transactions' },
    { id: 'allocation', label: 'Allocation' },
    { id: 'asset_change', label: 'Asset Change' },
    { id: 'capital_gains', label: 'Cap. Gains' },
    { id: 'dividend', label: 'Dividend' },
    { id: 'settings', label: 'Settings' },
];

export default function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
    return (
        <div className="w-full overflow-x-auto no-scrollbar py-4">
            <div className="flex min-w-max p-1 bg-black/5 dark:bg-white/5 backdrop-blur-md border border-black/5 dark:border-white/10 rounded-xl">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => onTabChange(tab.id)}
                        className={cn(
                            "py-2 px-4 text-sm font-medium rounded-lg transition-all whitespace-nowrap outline-none focus-visible:ring-2 focus-visible:ring-cyan-500",
                            activeTab === tab.id
                                ? "bg-black/10 dark:bg-white/10 text-foreground shadow-sm border border-black/5 dark:border-white/10"
                                : "text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5"
                        )}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>
        </div>
    );
}
