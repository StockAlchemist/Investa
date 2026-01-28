"use client";

import React from 'react';
import ScreenerView from '@/components/ScreenerView';

const ScreenerPage = () => {
    return (
        <div className="min-h-screen bg-background p-4 md:p-8">
            <ScreenerView currency="USD" />
        </div>
    );
};

export default ScreenerPage;
