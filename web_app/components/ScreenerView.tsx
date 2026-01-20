"use client";

import React, { useState, useEffect } from 'react';
import ScreenerInput from './ScreenerInput';
import ScreenerResults from './ScreenerResults';
import { Telescope } from 'lucide-react';
import { runScreener, fetchScreenerReview } from '@/lib/api';

interface ScreenerViewProps {
    currency: string;
}

const ScreenerView: React.FC<ScreenerViewProps> = ({ currency }) => {
    const [results, setResults] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [reviewingSymbol, setReviewingSymbol] = useState<string | null>(null);
    const [isInitialized, setIsInitialized] = useState(false);

    useEffect(() => {
        const handleUpdate = (e: any) => {
            const { symbol, analysis } = e.detail;

            // Calculate average AI score live
            let aiScore = null;
            if (analysis && analysis.scorecard) {
                const vals = Object.values(analysis.scorecard).filter(v => typeof v === 'number') as number[];
                if (vals.length > 0) {
                    aiScore = vals.reduce((a, b) => a + b, 0) / vals.length;
                }
            }

            setResults(prev => prev.map(item =>
                item.symbol === symbol ? { ...item, has_ai_review: true, ai_score: aiScore } : item
            ));
        };

        window.addEventListener('stock-analysis-updated', handleUpdate as EventListener);
        return () => window.removeEventListener('stock-analysis-updated', handleUpdate as EventListener);
    }, []);

    const handleRunScreener = async (universeType: string, universeId: string | null, manualSymbols: string[]) => {
        setIsLoading(true);
        try {
            const data = await runScreener({
                universe_type: universeType,
                universe_id: universeId,
                manual_symbols: manualSymbols
            });
            setResults(data);
        } catch (e) {
            console.error("Screening error", e);
        } finally {
            setIsLoading(false);
        }
    };

    const handleReview = async (symbol: string, force: boolean = false) => {
        setReviewingSymbol(symbol);
        try {
            const data = await fetchScreenerReview(symbol, force);
            if (data) {
                // Dispatch event so all components (and our own listener) update live
                window.dispatchEvent(new CustomEvent('stock-analysis-updated', {
                    detail: { symbol, analysis: data }
                }));
            }
            return data;
        } catch (e) {
            console.error("Review error", e);
        } finally {
            setReviewingSymbol(null);
        }
        return null;
    };

    return (
        <div className="space-y-6 animate-in fade-in duration-500">
            <header className="space-y-2">
                <h2 className="text-2xl font-bold leading-none tracking-tight bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent w-fit">
                    Market Screener
                </h2>
                <p className="text-muted-foreground text-sm font-medium max-w-2xl leading-relaxed">
                    Identify high-probability investment opportunities using quantitative <span className="text-cyan-500/80">intrinsic value models</span> and <span className="text-blue-500/80">AI-powered fundamental audits</span>.
                </p>
            </header>

            <div className="space-y-6">
                <ScreenerInput onRunScreener={handleRunScreener} isLoading={isLoading} />

                <ScreenerResults
                    results={results}
                    onReview={handleReview}
                    reviewingSymbol={reviewingSymbol}
                    currency={currency}
                />
            </div>
        </div>
    );
};

export default ScreenerView;
