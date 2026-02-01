import React from 'react';
import TabNavigation from '@/components/TabNavigation';
import CurrencySelector from '@/components/CurrencySelector';
import ThemeToggle from '@/components/ThemeToggle';
import { UserCircle, Settings as SettingsIcon, LogOut, Home } from 'lucide-react';

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
        <div className="w-full border-b border-border bg-background/40 backdrop-blur-md z-[60] hidden md:block sticky top-0">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:pr-8 flex items-center justify-between py-2">
                {/* Left: Navigation */}
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => onTabChange('performance')}
                        className="p-2 rounded-xl text-cyan-500 hover:bg-accent/10 transition-all duration-300 group"
                        title="Dashboard"
                    >
                        <Home className="w-5 h-5 transition-transform duration-300 group-hover:scale-110" />
                    </button>
                    <div className="h-4 w-px bg-border mx-1" />
                    <TabNavigation
                        activeTab={activeTab}
                        onTabChange={onTabChange}
                        side="bottom"
                        align="left"
                    />
                </div>

                {/* Right: Controls */}
                <div className="flex items-center gap-2">
                    <ThemeToggle />

                    <button
                        onClick={onSettingsClick}
                        className="p-2 rounded-xl text-cyan-500 hover:bg-accent/10 transition-all duration-300 group"
                        title="Settings"
                    >
                        <SettingsIcon className="w-5 h-5 transition-transform duration-300 group-hover:rotate-90" />
                    </button>

                    <div className="h-6 w-px bg-border mx-1" />

                    <CurrencySelector
                        currentCurrency={currency}
                        onChange={onCurrencyChange}
                        fxRate={fxRate}
                        side="bottom"
                        availableCurrencies={availableCurrencies}
                    />

                    <div className="h-6 w-px bg-border mx-1" />

                    <button
                        onClick={onUserClick}
                        className="p-2 rounded-xl text-cyan-500 hover:bg-accent/10 transition-all duration-300 group"
                        title="User Settings"
                    >
                        <UserCircle className="w-5 h-5 transition-transform duration-300 group-hover:scale-110" />
                    </button>

                    <button
                        onClick={() => user && onLogout?.()}
                        className="p-2 rounded-xl text-cyan-500 hover:bg-accent/10 transition-all duration-300 group"
                        title="Log Out"
                    >
                        <LogOut className="w-5 h-5 transition-transform duration-300 group-hover:scale-110" />
                    </button>
                </div>
            </div>
        </div>
    );
}
