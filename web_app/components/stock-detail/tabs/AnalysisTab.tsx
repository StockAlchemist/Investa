import React, { useState, useEffect } from 'react';
import { useTheme } from 'next-themes';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
    Sparkles,
    Shield,
    Zap,
    Target,
    Activity as LucideActivity,
    RotateCcw,
    Loader2,
    Info,
    TrendingUp,
    Calendar
} from 'lucide-react';
import { fetchStockAnalysis } from '../../../lib/api';
import { Badge } from '../../ui/badge';
import { Skeleton } from '../../ui/skeleton';
import { cn } from '../../../lib/utils';
import { formatCalendarDate } from '@/lib/market_time';

interface AnalysisTabProps {
    symbol: string;
    isOpen: boolean;
}

export const AnalysisTab: React.FC<AnalysisTabProps> = ({ symbol, isOpen }) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- AI analysis payload
    const [analysis, setAnalysis] = useState<any>(null);
    const [analysisLoading, setAnalysisLoading] = useState(false);
    const [analysisError, setAnalysisError] = useState<string | null>(null);

    const { resolvedTheme } = useTheme();
    const isDarkMode = resolvedTheme === 'dark';

    useEffect(() => {
        setAnalysis(null);
        setAnalysisError(null);
    }, [symbol]);

    useEffect(() => {
        if (isOpen && symbol && !analysis && !analysisLoading && !analysisError) {
            const getAnalysis = async () => {
                setAnalysisLoading(true);
                try {
                    setAnalysisError(null);
                    const data = await fetchStockAnalysis(symbol);
                    if (data && data.error) {
                        setAnalysisError(data.error);
                    } else {
                        setAnalysis(data);
                    }
                } catch (err) {
                    console.error("Analysis fetch error:", err);
                    setAnalysisError(err instanceof Error ? err.message : "Failed to load AI analysis.");
                } finally {
                    setAnalysisLoading(false);
                }
            };
            getAnalysis();
        }
    }, [isOpen, symbol, analysis, analysisLoading, analysisError]);

    const handleRegenerateAnalysis = async () => {
        setAnalysisLoading(true);
        setAnalysisError(null);
        try {
            const data = await fetchStockAnalysis(symbol, true);
            if (data && data.error) {
                setAnalysisError(data.error);
            } else {
                setAnalysis(data);
                window.dispatchEvent(new CustomEvent('stock-analysis-updated', {
                    detail: { symbol, analysis: data }
                }));
            }
        } catch (err) {
            console.error("Analysis regeneration error:", err);
            setAnalysisError(err instanceof Error ? err.message : "Failed to regenerate AI analysis.");
        } finally {
            setAnalysisLoading(false);
        }
    };

    if (analysisLoading) {
        return (
            <div className="space-y-6 animate-in fade-in duration-500">
                <Skeleton className="h-32 w-full rounded-2xl" />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Skeleton className="h-48 rounded-2xl" />
                    <Skeleton className="h-48 rounded-2xl" />
                    <Skeleton className="h-48 rounded-2xl" />
                    <Skeleton className="h-48 rounded-2xl" />
                </div>
            </div>
        );
    }

    if (analysisError) {
        const isRateLimit = analysisError.includes('429') || analysisError.toLowerCase().includes('too many requests');
        let displayError = analysisError.includes('Failed to resolve')
            ? 'Network connection issue. Please check your internet or DNS settings.'
            : analysisError.length > 200 ? analysisError.substring(0, 200) + '...' : analysisError;

        if (isRateLimit) {
            displayError = "Gemini API rate limit reached. The AI model is currently busy. Please wait a minute and try again.";
        }

        return (
            <div className="flex flex-col items-center justify-center py-20 text-center animate-in fade-in duration-500">
                <div className="w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center mb-4">
                    <Info className="w-8 h-8 text-destructive" />
                </div>
                <h3 className="text-xl font-bold mb-2">{isRateLimit ? 'Rate Limit Reached' : 'Analysis Failed'}</h3>
                <p className="text-muted-foreground max-w-md">{displayError}</p>
                <button
                    onClick={() => {
                        setAnalysis(null);
                        setAnalysisError(null);
                    }}
                    className="mt-6 px-6 py-2 bg-secondary hover:bg-muted rounded-full transition-colors font-medium cursor-pointer"
                >
                    Try Again
                </button>
            </div>
        );
    }

    if (!analysis) {
        return (
            <div className="flex flex-col items-center justify-center py-20 text-center">
                <Sparkles className="w-12 h-12 text-purple-500/20 mb-4" />
                <p className="text-muted-foreground">No analysis data available.</p>
            </div>
        );
    }

    const topics = [
        { id: 'moat', title: 'Moat & Edge', icon: Shield, color: 'text-blue-500', bg: 'bg-blue-500/10', content: analysis?.analysis?.moat, score: analysis?.scorecard?.moat },
        { id: 'strength', title: 'Financial Strength', icon: Zap, color: 'text-amber-500', bg: 'bg-amber-500/10', content: analysis?.analysis?.financial_strength, score: analysis?.scorecard?.financial_strength },
        { id: 'predictability', title: 'Predictability', icon: Target, color: 'text-emerald-500', bg: 'bg-emerald-500/10', content: analysis?.analysis?.predictability, score: analysis?.scorecard?.predictability },
        { id: 'growth', title: 'Growth Perspective', icon: LucideActivity, color: 'text-purple-500', bg: 'bg-purple-500/10', content: analysis?.analysis?.growth_perspective, score: analysis?.scorecard?.growth }
    ];

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            {/* Scorecard Header */}
            <div className={cn(
                "p-6 rounded-[2rem] overflow-hidden relative",
                isDarkMode ? "bg-slate-900/50" : "bg-white"
            )}>
                <div className="flex items-center gap-4 relative z-10">
                    <div className="w-12 h-12 rounded-2xl bg-purple-500 flex items-center justify-center shrink-0">
                        <Sparkles className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex flex-col">
                        <div className="flex items-center gap-3">
                            <h3 className="text-xl font-bold">AI Fundamental Review</h3>
                            <button
                                onClick={handleRegenerateAnalysis}
                                disabled={analysisLoading}
                                className="flex items-center gap-1.5 text-[10px] font-bold text-purple-600 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300 transition-colors uppercase tracking-wider mt-0.5 cursor-pointer"
                                title="Regenerate AI Analysis"
                            >
                                {analysisLoading ? (
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                ) : (
                                    <RotateCcw className="w-3 h-3" />
                                )}
                                Regenerate
                            </button>
                        </div>
                        <div className="text-sm text-muted-foreground leading-relaxed mt-1 markdown-content bg-transparent p-0 border-none shadow-none">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis.summary || ""}</ReactMarkdown>
                        </div>
                    </div>
                </div>
                <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/5 rounded-full blur-3xl -mr-32 -mt-32" />
            </div>

            {/* Score Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {topics.map(t => (
                    <div key={t.id} className={cn(
                        "p-4 rounded-3xl flex flex-col items-center justify-center gap-2",
                        isDarkMode ? "bg-slate-900/10" : "bg-zinc-50/50"
                    )}>
                        <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">{t.id}</span>
                        <div className={cn("text-3xl font-black", t.color)}>{t.score}<span className="text-sm opacity-50 font-normal">/10</span></div>
                    </div>
                ))}
            </div>

            {/* Narrative Details */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {topics.map(t => (
                    <div key={t.id} className={cn(
                        "p-6 rounded-[2rem] transition-all group",
                        isDarkMode ? "bg-slate-900/30 hover:bg-slate-900/50" : "bg-white"
                    )}>
                        <div className="flex items-center gap-3 mb-4">
                            <div className={cn("p-2.5 rounded-xl", t.bg)}>
                                <t.icon className={cn("w-5 h-5", t.color)} />
                            </div>
                            <h4 className="font-bold text-lg">{t.title}</h4>
                        </div>
                        <div className="text-sm leading-relaxed text-muted-foreground group-hover:text-foreground transition-colors markdown-content bg-transparent p-0 border-none shadow-none">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{t.content || ""}</ReactMarkdown>
                        </div>
                    </div>
                ))}
            </div>

            {/* Market Sentiment & Catalysts */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Sentiment Gauge */}
                <div className={cn(
                    "p-6 rounded-[2rem] transition-all",
                    isDarkMode ? "bg-slate-900/30" : "bg-white"
                )}>
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-3">
                            <div className="p-2.5 rounded-xl bg-indigo-500/10 text-indigo-500">
                                <TrendingUp className="w-5 h-5" />
                            </div>
                            <h4 className="font-bold text-lg">Market Sentiment</h4>
                        </div>
                        {typeof analysis.sentiment === 'number' && (
                            <Badge className={cn(
                                "border-none px-3 py-1",
                                analysis.sentiment >= 70 ? "bg-emerald-500/20 text-emerald-500" :
                                analysis.sentiment >= 40 ? "bg-amber-500/20 text-amber-500" :
                                "bg-rose-500/20 text-rose-500"
                            )}>
                                {analysis.sentiment >= 70 ? 'Bullish' : analysis.sentiment >= 40 ? 'Neutral' : 'Bearish'}
                            </Badge>
                        )}
                    </div>

                    {typeof analysis.sentiment === 'number' ? (
                        <div className="flex flex-col items-center py-4">
                            <div className="relative w-full h-4 bg-muted rounded-full overflow-hidden mb-4">
                                <div 
                                    className={cn(
                                        "h-full rounded-full transition-all duration-1000 ease-out",
                                        analysis.sentiment >= 70 ? "bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]" :
                                        analysis.sentiment >= 40 ? "bg-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.5)]" :
                                        "bg-rose-500 shadow-[0_0_15px_rgba(244,63,94,0.5)]"
                                    )}
                                    style={{ width: `${analysis.sentiment}%` }}
                                />
                            </div>
                            <div className="flex justify-between w-full text-[10px] font-bold text-muted-foreground uppercase tracking-wider px-1">
                                <span>Extreme Fear</span>
                                <span className="text-foreground text-lg">{analysis.sentiment.toFixed(0)}%</span>
                                <span>Extreme Greed</span>
                            </div>
                            <p className="text-xs text-muted-foreground text-center mt-6 leading-relaxed">
                                Current market vibe based on news flow, analyst ratings, and social trends.
                            </p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center py-10 text-muted-foreground opacity-30 italic">
                            <div className="w-16 h-16 bg-muted rounded-2xl flex items-center justify-center text-muted-foreground mb-4">
                                <LucideActivity className="w-8 h-8 mb-2" />
                            </div>
                            <p className="text-xs">Sentiment data pending...</p>
                        </div>
                    )}
                </div>

                {/* Catalyst Timeline */}
                <div className={cn(
                    "p-6 rounded-[2rem] transition-all",
                    isDarkMode ? "bg-slate-900/30" : "bg-white"
                )}>
                    <div className="flex items-center gap-3 mb-6">
                        <div className="p-2.5 rounded-xl bg-amber-500/10 text-amber-500">
                            <Calendar className="w-5 h-5" />
                        </div>
                        <h4 className="font-bold text-lg">Upcoming Catalysts</h4>
                    </div>
                    
                    <div className="space-y-4">
                        {analysis.catalysts && analysis.catalysts.length > 0 ? (
                            analysis.catalysts.map((c: { date?: string; event?: string; impact?: string }, i: number) => (
                                <div key={i} className="flex gap-4 group">
                                    <div className="flex flex-col items-center">
                                        <div className={cn(
                                            "w-2 h-2 rounded-full mt-1.5 shrink-0",
                                            c.impact === 'High' ? "bg-rose-500" :
                                            c.impact === 'Medium' ? "bg-amber-500" : "bg-blue-500"
                                        )} />
                                        {i < analysis.catalysts.length - 1 && (
                                            <div className="w-px h-full bg-border/40 my-1" />
                                        )}
                                    </div>
                                    <div className="flex-1 pb-4">
                                        <div className="flex items-center justify-between mb-1">
                                            <p className="text-sm font-bold text-foreground transition-colors group-hover:text-indigo-500">{c.event}</p>
                                            <Badge variant="outline" className="text-[9px] font-bold uppercase border-muted-foreground/20 text-muted-foreground">
                                                {c.impact}
                                            </Badge>
                                        </div>
                                        <p className="text-[11px] font-medium text-muted-foreground">{formatCalendarDate(c.date)}</p>
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="flex flex-col items-center justify-center py-10 text-muted-foreground opacity-30 italic">
                                <Info className="w-8 h-8 mb-2" />
                                <p className="text-xs">No catalysts detected in latest run.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="text-center pb-8">
                <p className="text-[10px] text-muted-foreground uppercase tracking-[0.2em] font-medium opacity-50">
                    Generated by Google Gemini 3 Flash
                </p>
            </div>
        </div>
    );
};
