import React from 'react';
import TabNavigation from '@/components/TabNavigation';
import CurrencySelector from '@/components/CurrencySelector';
import ThemeToggle from '@/components/ThemeToggle';
import UserMenu from '@/components/UserMenu';
import { Settings as SettingsIcon, Home } from 'lucide-react';

interface ControlBarProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
    onLogout?: () => void;
    currency: string;
    onCurrencyChange: (currency: string) => void;
    fxRate?: number;
    availableCurrencies?: string[];
    onSettingsClick: () => void;
    onUserClick: () => void;
    user?: any;
}

export default function ControlBar({
    activeTab,
    onTabChange,
    onLogout,
    currency,
    onCurrencyChange,
    fxRate,
    availableCurrencies,
    onSettingsClick,
    onUserClick,
    user
}: ControlBarProps) {
    return (
        <div className="w-full border-b border-border/50 bg-background/60 backdrop-blur-2xl z-[60] hidden md:block sticky top-0 shadow-sm dark:shadow-black/20">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:pr-8 flex items-center justify-between py-2">
                {/* Left: Navigation */}
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => onTabChange('performance')}
                        className="p-2 rounded-xl text-cyan-500 hover:bg-cyan-500/10 transition-all duration-300 group"
                        title="Dashboard"
                    >
                        <Home className="w-4 h-4 transition-transform duration-300 group-hover:scale-110" />
                    </button>
                    <div className="h-4 w-px bg-border/60 mx-1" />
                    <TabNavigation
                        activeTab={activeTab}
                        onTabChange={onTabChange}
                        side="bottom"
                        align="left"
                    />
                </div>

                {/* Right: Controls */}
                <div className="flex items-center gap-1.5">
                    <ThemeToggle />

                    <button
                        onClick={onSettingsClick}
                        className="p-2 rounded-xl text-indigo-500 hover:text-cyan-500 hover:bg-cyan-500/10 transition-all duration-300 group"
                        title="Project Settings"
                    >
                        <SettingsIcon className="w-4 h-4 transition-transform duration-300 group-hover:rotate-90" />
                    </button>

                    <div className="h-5 w-px bg-border/60 mx-1" />

                    <CurrencySelector
                        currentCurrency={currency}
                        onChange={onCurrencyChange}
                        fxRate={fxRate}
                        side="bottom"
                        availableCurrencies={availableCurrencies}
                    />

                    <div className="h-5 w-px bg-border/60 mx-1" />

                    <UserMenu
                        user={user}
                        onLogout={onLogout}
                        onUserClick={onUserClick}
                        align="right"
                    />
                </div>
            </div>
        </div>
    );
}
