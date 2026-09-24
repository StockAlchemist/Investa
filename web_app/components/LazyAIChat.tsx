'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useAuth } from '@/context/AuthContext';

const AIChat = dynamic(() => import('@/components/AIChat'), {
    ssr: false,
    loading: () => null,
});

export default function LazyAIChat() {
    const [mounted, setMounted] = useState(false);
    const { user } = useAuth();

    useEffect(() => {
        if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
            const id = window.requestIdleCallback(() => setMounted(true), { timeout: 4000 });
            return () => {
                window.cancelIdleCallback(id);
            };
        }
        const timer = setTimeout(() => setMounted(true), 2500);
        return () => clearTimeout(timer);
    }, []);

    if (!mounted) return null;
    // Keyed by user: the chat lives in the root layout, above the login screen,
    // so without a remount its messages would carry into the next session.
    return <AIChat key={user?.id ?? 'signed-out'} />;
}
