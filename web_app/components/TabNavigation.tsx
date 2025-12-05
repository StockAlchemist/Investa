import React from 'react';

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
    { id: 'intraday', label: 'Intraday' },
    { id: 'rebalancing', label: 'Rebalancing' },
    { id: 'analysis', label: 'Analysis' },
    { id: 'settings', label: 'Settings' },
];

export default function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
    return (
        <div className="w-full overflow-x-auto bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 no-scrollbar">
            <div className="flex min-w-max">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => onTabChange(tab.id)}
                        className={`
              py-3 px-4 text-sm font-medium border-b-2 transition-colors whitespace-nowrap
              ${activeTab === tab.id
                                ? 'border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400'
                                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
                            }
            `}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>
        </div>
    );
}
