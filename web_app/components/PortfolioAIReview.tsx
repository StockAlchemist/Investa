'use client';

import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchPortfolioAIReview } from '@/lib/api';
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshCw, Gem, Scale, Target, AlertTriangle, Lightbulb, Clock } from 'lucide-react';
import { MetricCard } from '@/components/MetricCard';
import PortfolioOptimization from '@/components/PortfolioOptimization';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface PortfolioAIReviewProps {
    currency: string;
    accounts: string[];
}

// 1-10 score → tailwind color token used for text + bar fill.
function scoreTone(score: number): { text: string; bar: string } {
    if (!score) return { text: 'text-muted-foreground', bar: 'bg-muted-foreground/40' };
    if (score >= 8) return { text: 'text-emerald-500', bar: 'bg-emerald-500' };
    if (score >= 5) return { text: 'text-yellow-500', bar: 'bg-yellow-500' };
    return { text: 'text-rose-500', bar: 'bg-rose-500' };
}

function letterGrade(avg: number): string {
    if (avg >= 8.5) return 'A';
    if (avg >= 7) return 'B';
    if (avg >= 5.5) return 'C';
    if (avg >= 4) return 'D';
    return 'F';
}

function timeAgo(iso?: string): string | null {
    if (!iso) return null;
    const t = new Date(iso).getTime();
    if (isNaN(t)) return null;
    const diffMin = Math.floor((Date.now() - t) / 60000);
    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    const h = Math.floor(diffMin / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ScoreCard({ title, score, icon: Icon, onClick }: { title: string; score: number; icon: any; onClick: () => void }) {
    const tone = scoreTone(score);
    const filled = Math.round(score);
    return (
        <button
            onClick={onClick}
            className="metric-card p-4 text-left hover:bg-accent/5 transition-colors"
        >
            <div className="flex items-center justify-between mb-2.5">
                <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground/80 font-semibold">
                    <Icon className="w-3 h-3" />
                    <span>{title}</span>
                </div>
                <span className={cn('text-lg font-bold tabular-nums', tone.text)}>
                    {score || '-'}<span className="text-[10px] text-muted-foreground font-medium">/10</span>
                </span>
            </div>
            <div className="flex gap-0.5">
                {Array.from({ length: 10 }).map((_, i) => (
                    <div
                        key={i}
                        className={cn('h-1.5 flex-1 rounded-full transition-colors', i < filled ? tone.bar : 'bg-muted')}
                    />
                ))}
            </div>
        </button>
    );
}

export default function PortfolioAIReview({ currency, accounts }: PortfolioAIReviewProps) {
    const [selectedMetric, setSelectedMetric] = React.useState<{ title: string; score: number; content: string } | null>(null);

    const queryClient = useQueryClient();
    const { data, isLoading, isError, error } = useQuery({
        queryKey: ['portfolioAIReview', currency, accounts],
        queryFn: ({ signal }) => fetchPortfolioAIReview(currency, accounts, false, signal),
        staleTime: 24 * 60 * 60 * 1000, // Cache for 24 hours
        retry: false,
    });

    const refreshMutation = useMutation({
        mutationFn: () => fetchPortfolioAIReview(currency, accounts, true),
        onSuccess: (newData) => {
            queryClient.setQueryData(['portfolioAIReview', currency, accounts], newData);
        },
    });

    const handleRefresh = () => {
        refreshMutation.mutate();
    };

    if (isLoading) {
        return (
            <div className="space-y-6 animate-pulse p-1">
                <div className="flex justify-between items-center mb-6">
                    <div className="h-8 w-48 bg-muted rounded-xl"></div>
                    <div className="h-9 w-32 bg-muted rounded-xl"></div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {[1, 2, 3, 4].map(i => (
                        <div key={i} className="h-32 bg-muted rounded-2xl"></div>
                    ))}
                </div>
                <div className="h-64 bg-muted rounded-2xl"></div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="h-48 bg-muted rounded-2xl"></div>
                    <div className="h-48 bg-muted rounded-2xl"></div>
                </div>
            </div>
        );
    }

    if (isError) {
        return (
            <div className="flex flex-col items-center justify-center p-12 text-center space-y-4 rounded-xl bg-card/50">
                <AlertTriangle className="w-10 h-10 text-rose-500 mb-2" />
                <h3 className="text-lg font-semibold">Unable to generate analysis</h3>
                <p className="text-muted-foreground max-w-md">
                    {(error as Error).message || "An unexpected error occurred while analyzing your portfolio."}
                </p>
                <Button onClick={handleRefresh} variant="outline" className="mt-4 gap-2">
                    <RefreshCw className="w-4 h-4" /> Try Again
                </Button>
            </div>
        );
    }

    // If we have data but also a warning (e.g. RateLimit fallback), show a banner
    let warningBanner = null;
    if (data?.warning === "RateLimit") {
        warningBanner = (
            <div className="mb-6 p-4 rounded-xl bg-yellow-500/10 flex items-start gap-3 fade-in animate-in duration-500">
                <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5 shrink-0" />
                <div className="flex-1">
                    <h3 className="font-semibold text-yellow-500 text-sm">AI Service Busy</h3>
                    <p className="text-muted-foreground text-xs mt-1">
                        {data.message || "Showing cached analysis. Please try refreshing again later."}
                    </p>
                </div>
                <Button
                    onClick={handleRefresh}
                    variant="ghost"
                    size="sm"
                    className="text-yellow-500 hover:text-yellow-600 hover:bg-yellow-500/10 h-8 px-2"
                    disabled={refreshMutation.isPending}
                >
                    <RefreshCw className={`w-3 h-3 mr-2 ${refreshMutation.isPending ? 'animate-spin' : ''}`} />
                    Retry
                </Button>
            </div>
        );
    } else if (data?.error === "RateLimit") {
        return (
            <div className="flex flex-col items-center justify-center p-8 text-center space-y-4 rounded-xl bg-yellow-500/10 fade-in animate-in duration-500">
                <AlertTriangle className="w-10 h-10 text-yellow-500 mb-2" />
                <h3 className="text-lg font-semibold text-yellow-500">AI Service Busy</h3>
                <p className="text-muted-foreground max-w-md text-sm">
                    {data.message || "We're experiencing high demand. Please try again in a moment."}
                </p>
                <Button
                    onClick={handleRefresh}
                    variant="outline"
                    className="mt-4 gap-2 border-none hover:bg-yellow-500/10 shadow-none"
                    disabled={refreshMutation.isPending}
                >
                    <RefreshCw className={`w-4 h-4 ${refreshMutation.isPending ? 'animate-spin' : ''}`} />
                    {refreshMutation.isPending ? 'Retrying...' : 'Try Again'}
                </Button>
            </div>
        );
    }

    if (!data || (data.error && !data.warning)) {
        return (
            <div className="flex flex-col items-center justify-center p-12 text-center space-y-4 rounded-xl bg-card/50">
                <AlertTriangle className="w-10 h-10 text-rose-500 mb-2" />
                <h3 className="text-lg font-semibold">Unable to generate analysis</h3>
                <p className="text-muted-foreground max-w-md">
                    {data?.message || data?.error || "An unexpected error occurred while analyzing your portfolio."}
                </p>
                <Button onClick={handleRefresh} variant="outline" className="mt-4 gap-2">
                    <RefreshCw className={`w-4 h-4 ${refreshMutation.isPending ? 'animate-spin' : ''}`} />
                    {refreshMutation.isPending ? 'Retrying...' : 'Try Again'}
                </Button>
            </div>
        );
    }

    const { scorecard, analysis, summary, recommendations } = data;

    // Composite grade: average the three scored dimensions (ignore zeros/missing).
    const scoreDims = [scorecard?.business_quality, scorecard?.value_discipline, scorecard?.thesis_integrity]
        .filter((v): v is number => typeof v === 'number' && v > 0);
    const overallAvg = scoreDims.length > 0 ? scoreDims.reduce((a, b) => a + b, 0) / scoreDims.length : null;
    const overallGrade = overallAvg != null ? letterGrade(overallAvg) : null;
    const overallTone = overallAvg != null ? scoreTone(overallAvg) : null;
    const generatedAgo = timeAgo(data?.generated_at);

    // Helper to determine color based on score (1-10)
    const getScoreColor = (score: number) => {
        if (!score) return "text-muted-foreground";
        if (score >= 8) return "text-emerald-500";
        if (score >= 5) return "text-yellow-500";
        return "text-rose-500";
    };

    // Helper to determine theme color based on score
    const getScoreTheme = (score: number) => {
        if (!score) return "gray-500";
        if (score >= 8) return "emerald-500";
        if (score >= 5) return "yellow-500";
        return "rose-500";
    };

    const getModalClasses = (score: number) => {
        if (!score) return "bg-gray-50/95 dark:bg-gray-950/90";
        if (score >= 8) return "bg-emerald-50/95 dark:bg-emerald-950/90";
        if (score >= 5) return "bg-yellow-50/95 dark:bg-yellow-950/90";
        return "bg-rose-50/95 dark:bg-rose-950/90";
    };

    return (
        <div className="space-y-6 animate-in fade-in duration-500">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <header className="space-y-2">
                    <h2 className="text-2xl font-bold leading-none tracking-tight bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent w-fit">
                        Portfolio AI Review
                    </h2>
                    <p className="text-muted-foreground text-sm font-medium max-w-2xl leading-relaxed">
                        AI-driven insights and recommendations for your portfolio.
                    </p>
                    {generatedAgo && (
                        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground/70">
                            <Clock className="w-3 h-3" />
                            <span>Generated {generatedAgo}</span>
                        </div>
                    )}
                </header>
                <div className="flex items-center gap-4">
                    {overallAvg != null && overallTone && (
                        <div className="flex items-center gap-3 px-4 py-2 rounded-2xl bg-card border border-border/60">
                            <div className={cn('text-3xl font-black leading-none', overallTone.text)}>
                                {overallGrade}
                            </div>
                            <div className="leading-tight">
                                <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">Overall</div>
                                <div className={cn('text-sm font-bold tabular-nums', overallTone.text)}>
                                    {overallAvg.toFixed(1)}<span className="text-[10px] text-muted-foreground font-medium">/10</span>
                                </div>
                            </div>
                        </div>
                    )}
                    <Button
                        onClick={handleRefresh}
                        variant="outline"
                        size="sm"
                        disabled={refreshMutation.isPending}
                        className="gap-2 bg-card hover:bg-accent"
                    >
                        <RefreshCw className={`w-4 h-4 ${refreshMutation.isPending ? 'animate-spin' : ''}`} />
                        {refreshMutation.isPending ? 'Analyzing...' : 'Refresh Analysis'}
                    </Button>
                </div>
            </div>

            {/* Scorecard */}
            {warningBanner}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <ScoreCard
                    title="Business Quality"
                    score={scorecard?.business_quality || 0}
                    icon={Gem}
                    onClick={() => setSelectedMetric({
                        title: "Business Quality",
                        score: scorecard?.business_quality || 0,
                        content: analysis?.business_quality || "No analysis available."
                    })}
                />
                <ScoreCard
                    title="Value Discipline"
                    score={scorecard?.value_discipline || 0}
                    icon={Scale}
                    onClick={() => setSelectedMetric({
                        title: "Value Discipline",
                        score: scorecard?.value_discipline || 0,
                        content: analysis?.value_discipline || "No analysis available."
                    })}
                />
                <ScoreCard
                    title="Thesis Integrity"
                    score={scorecard?.thesis_integrity || 0}
                    icon={Target}
                    onClick={() => setSelectedMetric({
                        title: "Thesis Integrity",
                        score: scorecard?.thesis_integrity || 0,
                        content: analysis?.thesis_integrity || "No analysis available."
                    })}
                />
                <MetricCard
                    title="Key Actions"
                    value={recommendations ? recommendations.length : 0}
                    subValue={null}
                    isCurrency={false}
                    icon={Lightbulb}
                    colorClass="text-rose-500"
                />
            </div>

            {/* Executive Summary */}
            <div className="metric-card card-shine p-6 relative overflow-hidden group">
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-rose-500 opacity-80" />
                <h3 className="section-label text-rose-600 dark:text-rose-400 mb-4 relative z-10">
                    Executive Summary
                </h3>
                <div className="prose dark:prose-invert max-w-none prose-sm opacity-90">
                    <div className="markdown-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{summary || ""}</ReactMarkdown>
                    </div>
                </div>
            </div>

            {/* Optimization Hub */}
            {data?.optimizations && data.optimizations.length > 0 && (
                <PortfolioOptimization optimizations={data.optimizations} />
            )}

            {/* Detailed Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {analysis?.business_quality && (
                    <div className="metric-card p-6 relative overflow-hidden group">
                        <div className="absolute top-0 left-0 right-0 h-[2px] bg-rose-500 opacity-80" />
                        <h3 className="section-label mb-4 relative z-10">Business Quality</h3>
                        <div className="prose dark:prose-invert max-w-none text-sm text-muted-foreground opacity-90 relative z-10">
                            <div className="markdown-content">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis.business_quality || ""}</ReactMarkdown>
                            </div>
                        </div>
                    </div>
                )}
                {analysis?.value_discipline && (
                    <div className="metric-card p-6 relative overflow-hidden group">
                        <div className="absolute top-0 left-0 right-0 h-[2px] bg-rose-500 opacity-80" />
                        <h3 className="section-label mb-4 relative z-10">Value Discipline</h3>
                        <div className="prose dark:prose-invert max-w-none text-sm text-muted-foreground opacity-90 relative z-10">
                            <div className="markdown-content">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis.value_discipline || ""}</ReactMarkdown>
                            </div>
                        </div>
                    </div>
                )}
                {analysis?.thesis_integrity && (
                    <div className="metric-card p-6 relative overflow-hidden group">
                        <div className="absolute top-0 left-0 right-0 h-[2px] bg-rose-500 opacity-80" />
                        <h3 className="section-label mb-4 relative z-10">Thesis Integrity</h3>
                        <div className="prose dark:prose-invert max-w-none text-sm text-muted-foreground opacity-90 relative z-10">
                            <div className="markdown-content">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis.thesis_integrity || ""}</ReactMarkdown>
                            </div>
                        </div>
                    </div>
                )}

                {/* Actionable Recommendations */}
                <div className="metric-card card-shine lg:col-span-2 p-6 relative overflow-hidden group">
                    <div className="absolute top-0 left-0 right-0 h-[2px] bg-rose-500 opacity-80" />
                    <h3 className="section-label text-rose-600 dark:text-rose-400 flex items-center gap-2 mb-4 relative z-10">
                        <Lightbulb className="w-4 h-4" />
                        Actionable Recommendations
                    </h3>
                    <div className="prose dark:prose-invert max-w-none text-sm opacity-90 relative z-10">
                        {analysis?.actionable_recommendations ? (
                            <div className="markdown-content">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis.actionable_recommendations || ""}</ReactMarkdown>
                            </div>
                        ) : (
                            <ul className="list-disc pl-5">
                                {recommendations?.map((rec: string, i: number) => (
                                    <li key={i}>{rec}</li>
                                ))}
                            </ul>
                        )}
                    </div>
                </div>
            </div>

            {/* Simple Modal Overlay */}
            {selectedMetric && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 animate-in fade-in duration-200" onClick={() => setSelectedMetric(null)}>
                    <div
                        className={`w-full max-w-lg m-4 p-8 rounded-[2rem] ${getModalClasses(selectedMetric.score)} animate-in zoom-in-95 duration-200`}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="pb-2">
                            <div className="flex justify-between items-start">
                                <div>
                                    <p className="text-sm font-medium text-muted-foreground uppercase tracking-widest mb-1">Score Explanation</p>
                                    <div className="flex items-center gap-3">
                                        <h3 className={`text-xl font-bold ${getScoreColor(selectedMetric.score).replace('text-', 'text-')}`}>
                                            {selectedMetric.title}
                                        </h3>
                                        <div className={`text-lg font-bold px-2 py-0.5 rounded-md bg-muted ${getScoreColor(selectedMetric.score)}`}>
                                            {selectedMetric.score}/10
                                        </div>
                                    </div>
                                </div>
                                <Button variant="ghost" size="sm" className="h-8 w-8 p-0 rounded-full hover:bg-muted" onClick={() => setSelectedMetric(null)}>
                                    <span className="sr-only">Close</span>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-x w-4 h-4"><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg>
                                </Button>
                            </div>
                        </div>
                        <div className="pt-4">
                            <div className="prose dark:prose-invert max-w-none text-sm leading-relaxed text-muted-foreground">
                                <div className="markdown-content">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{selectedMetric.content || ""}</ReactMarkdown>
                                </div>
                            </div>
                            <div className="mt-6 pt-4 flex justify-end">
                                <Button variant="secondary" size="sm" onClick={() => setSelectedMetric(null)}>Close</Button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="text-[10px] text-muted-foreground/50 text-center mt-8 uppercase tracking-widest font-medium">
                AI-Generated Analysis • Not Financial Advice
            </div>
        </div>
    );
}
