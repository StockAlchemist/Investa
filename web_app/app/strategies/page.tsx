"use client";

import React from 'react';
import StrategiesView from '@/components/StrategiesView';

const StrategiesPage = () => {
    return (
        <div className="min-h-screen bg-background p-4 md:p-8">
            <StrategiesView currency="USD" />
        </div>
    );
};

export default StrategiesPage;
