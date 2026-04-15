'use client';

import dynamic from 'next/dynamic';

const AIChat = dynamic(() => import('@/components/AIChat'), {
    ssr: false,
    loading: () => null, // No loading state — chat bubble appears when ready
});

export default function LazyAIChat() {
    return <AIChat />;
}
