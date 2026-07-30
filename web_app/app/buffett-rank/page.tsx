"use client";

import React from 'react';
import BuffettRankView from '@/components/BuffettRankView';

const BuffettRankPage = () => {
    return (
        <div className="min-h-screen bg-background p-4 md:p-8">
            <BuffettRankView currency="USD" />
        </div>
    );
};

export default BuffettRankPage;
