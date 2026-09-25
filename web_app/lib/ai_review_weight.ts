"use client";

import { useCallback, useSyncExternalStore } from 'react';
import { AI_REVIEW_WEIGHT_PRESETS, DEFAULT_AI_REVIEW_WEIGHT } from '@/lib/api';

/**
 * The AI-review weight the reader chose for the ranking and the strategies.
 *
 * One preference shared by both screens, so the Rankings list and a strategy's
 * top 20 are always built from the same blend. Kept per browser rather than on
 * the server: it is a viewing choice, and the backend takes it per request.
 */
const STORAGE_KEY = 'investa.aiReviewWeight';
const CHANGE_EVENT = 'investa:ai-review-weight';

function read(): number {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw === null) return DEFAULT_AI_REVIEW_WEIGHT;
        const value = Number(raw);
        // Only a preset is honoured, so a stale or hand-edited entry cannot
        // leave the picker with nothing selected.
        return (AI_REVIEW_WEIGHT_PRESETS as readonly number[]).includes(value)
            ? value
            : DEFAULT_AI_REVIEW_WEIGHT;
    } catch {
        return DEFAULT_AI_REVIEW_WEIGHT;
    }
}

function subscribe(onChange: () => void): () => void {
    window.addEventListener(CHANGE_EVENT, onChange);
    window.addEventListener('storage', onChange);
    return () => {
        window.removeEventListener(CHANGE_EVENT, onChange);
        window.removeEventListener('storage', onChange);
    };
}

export function useAiReviewWeight(): [number, (weight: number) => void] {
    const weight = useSyncExternalStore(subscribe, read, () => DEFAULT_AI_REVIEW_WEIGHT);
    const setWeight = useCallback((next: number) => {
        try {
            localStorage.setItem(STORAGE_KEY, String(next));
        } catch {}
        window.dispatchEvent(new Event(CHANGE_EVENT));
    }, []);
    return [weight, setWeight];
}

/** `0.2` → `20%`; `0` reads as "Off" because it switches the review out entirely. */
export function formatAiReviewWeight(weight: number): string {
    return weight <= 0 ? 'Off' : `${Math.round(weight * 100)}%`;
}
