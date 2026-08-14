'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';

const AIChat = dynamic(() => import('@/components/AIChat'), {
    ssr: false,
    loading: () => null,
});

export default function LazyAIChat() {
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        if ('requestIdleCallback' in window) {
            const id = (window as any).requestIdleCallback(() => setMounted(true), { timeout: 4000 });
            return () => (window as any).cancelIdleCallback(id);
        }
        const timer = setTimeout(() => setMounted(true), 2500);
        return () => clearTimeout(timer);
    }, []);

    if (!mounted) return null;
    return <AIChat />;
}
