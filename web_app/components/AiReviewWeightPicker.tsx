"use client";

import React from 'react';
import { AI_REVIEW_WEIGHT_PRESETS } from '@/lib/api';
import { formatAiReviewWeight, useAiReviewWeight } from '@/lib/ai_review_weight';

/**
 * How much of the final ranking score the AI review carries.
 *
 * The same control on Rankings and Strategies, writing one shared preference,
 * so the two screens cannot disagree about which blend they show.
 */
export default function AiReviewWeightPicker() {
    const [weight, setWeight] = useAiReviewWeight();
    return (
        <div className="flex items-center gap-2">
            <span
                className="text-xs font-medium text-muted-foreground whitespace-nowrap"
                title="Share of the final score given to the AI review of moat, financial strength, predictability and growth"
            >
                AI review
            </span>
            <div role="radiogroup" aria-label="AI review weight" className="segmented h-8">
                {AI_REVIEW_WEIGHT_PRESETS.map((preset) => (
                    <button
                        key={preset}
                        type="button"
                        role="radio"
                        aria-checked={weight === preset}
                        onClick={() => setWeight(preset)}
                        className="text-xs whitespace-nowrap tabular-nums"
                    >
                        {formatAiReviewWeight(preset)}
                    </button>
                ))}
            </div>
        </div>
    );
}
