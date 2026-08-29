"use client";

import React from 'react';
import ScreenerView from '@/components/ScreenerView';

const ScreenerPage = () => {
    return (
        <div className="min-h-screen bg-background p-4 md:p-8">
            {/* Standalone route: no PageHeader here at any width, and
                ScreenerView's own heading is md:hidden - so md+ needs this. */}
            <h1 className="hidden md:block mb-6 text-2xl font-bold tracking-tight text-foreground">
                Screener
            </h1>
            <ScreenerView currency="USD" />
        </div>
    );
};

export default ScreenerPage;
