'use client';

import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchPortfolioAIReview } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshCw, ShieldCheck, TrendingUp, AlertTriangle, Lightbulb, PieChart } from 'lucide-react';
import { MetricCard } from '@/components/MetricCard';

interface PortfolioAIReviewProps {
    currency: string;
    accounts: string[];
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
            <div className="flex flex-col items-center justify-center p-12 text-center space-y-4 rounded-xl border border-dashed border-border bg-card/50">
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
            <div className="mb-6 p-4 rounded-xl border border-yellow-500/30 bg-yellow-500/10 flex items-start gap-3 fade-in animate-in duration-500">
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
            <div className="flex flex-col items-center justify-center p-8 text-center space-y-4 rounded-xl border border-yellow-500/30 bg-yellow-500/10 fade-in animate-in duration-500">
                <AlertTriangle className="w-10 h-10 text-yellow-500 mb-2" />
                <h3 className="text-lg font-semibold text-yellow-500">AI Service Busy</h3>
                <p className="text-muted-foreground max-w-md text-sm">
                    {data.message || "We're experiencing high demand. Please try again in a moment."}
                </p>
                <Button
                    onClick={handleRefresh}
                    variant="outline"
                    className="mt-4 gap-2 border-yellow-500/50 hover:bg-yellow-500/10"
                    disabled={refreshMutation.isPending}
                >
                    <RefreshCw className={`w-4 h-4 ${refreshMutation.isPending ? 'animate-spin' : ''}`} />
                    {refreshMutation.isPending ? 'Retrying...' : 'Try Again'}
                </Button>
            </div>
        );
    }

    if (!data || (data.error && !data.warning)) {
        // Generic Error Fallback if data.error exists but isn't RateLimit (or is RateLimit without cache)
        if (data?.error) {
            return (
                <div className="flex flex-col items-center justify-center p-12 text-center space-y-4 rounded-xl border border-dashed border-border bg-card/50">
                    <AlertTriangle className="w-10 h-10 text-rose-500 mb-2" />
                    <h3 className="text-lg font-semibold">Unable to generate analysis</h3>
                    <p className="text-muted-foreground max-w-md">
                        {data.message || data.error || "An unexpected error occurred."}
                    </p>
                    <Button onClick={handleRefresh} variant="outline" className="mt-4 gap-2">
                        <RefreshCw className="w-4 h-4" /> Try Again
                    </Button>
                </div>
            );
        }
        return null;
    }

    const { scorecard, analysis, summary, recommendations } = data;

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
        if (!score) return "border-gray-200 bg-gray-50/95 dark:border-gray-800 dark:bg-gray-950/90";
        if (score >= 8) return "border-emerald-200 bg-emerald-50/95 dark:border-emerald-900/50 dark:bg-emerald-950/90";
        if (score >= 5) return "border-yellow-200 bg-yellow-50/95 dark:border-yellow-900/50 dark:bg-yellow-950/90";
        return "border-rose-200 bg-rose-50/95 dark:border-rose-900/50 dark:bg-rose-950/90";
    };

    return (
        <div className="space-y-6 animate-in fade-in duration-500">
            <div className="flex items-center justify-between">
                <header className="space-y-2">
                    <h2 className="text-2xl font-bold leading-none tracking-tight bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent w-fit">
                        Portfolio AI Review
                    </h2>
                    <p className="text-muted-foreground text-sm font-medium max-w-2xl leading-relaxed">
                        AI-driven insights and recommendations for your portfolio.
                    </p>
                </header>
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

            {/* Scorecard */}
            {warningBanner}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <MetricCard
                    title="Diversification"
                    value={scorecard?.diversification || '-'}
                    subValue={null}
                    isCurrency={false}
                    icon={PieChart}
                    colorClass={getScoreColor(scorecard?.diversification)}
                    onClick={() => setSelectedMetric({
                        title: "Diversification Analysis",
                        score: scorecard?.diversification || 0,
                        content: analysis?.diversification || "No analysis available."
                    })}
                />
                <MetricCard
                    title="Risk Profile"
                    value={scorecard?.risk_profile || '-'}
                    subValue={null}
                    isCurrency={false}
                    icon={ShieldCheck}
                    colorClass={getScoreColor(scorecard?.risk_profile)}
                    onClick={() => setSelectedMetric({
                        title: "Risk Assessment",
                        score: scorecard?.risk_profile || 0,
                        content: analysis?.risk_profile || "No analysis available."
                    })}
                />
                <MetricCard
                    title="Performance"
                    value={scorecard?.performance || '-'}
                    subValue={null}
                    isCurrency={false}
                    icon={TrendingUp}
                    colorClass={getScoreColor(scorecard?.performance)}
                    onClick={() => setSelectedMetric({
                        title: "Performance Review",
                        score: scorecard?.performance || 0,
                        content: analysis?.performance || "No analysis available."
                    })}
                />
                <MetricCard
                    title="Key Actions"
                    value={recommendations ? recommendations.length : 0}
                    subValue={null}
                    isCurrency={false}
                    icon={Lightbulb}
                    colorClass="text-cyan-500"
                />
            </div>

            {/* Executive Summary */}
            <Card className="border-purple-500/20 bg-purple-500/5 overflow-hidden">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-purple-600 dark:text-purple-400">
                        Executive Summary
                    </CardTitle>
                </CardHeader>
                <CardContent className="prose dark:prose-invert max-w-none prose-sm">
                    <p className="whitespace-pre-wrap">{summary}</p>
                </CardContent>
            </Card>

            {/* Detailed Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {analysis?.diversification && (
                    <Card>
                        <CardHeader><CardTitle className="text-base">Diversification Analysis</CardTitle></CardHeader>
                        <CardContent className="prose dark:prose-invert max-w-none text-sm text-muted-foreground">
                            <p className="whitespace-pre-wrap">{analysis.diversification}</p>
                        </CardContent>
                    </Card>
                )}
                {analysis?.risk_profile && (
                    <Card>
                        <CardHeader><CardTitle className="text-base">Risk Assessment</CardTitle></CardHeader>
                        <CardContent className="prose dark:prose-invert max-w-none text-sm text-muted-foreground">
                            <p className="whitespace-pre-wrap">{analysis.risk_profile}</p>
                        </CardContent>
                    </Card>
                )}
                {analysis?.performance && (
                    <Card>
                        <CardHeader><CardTitle className="text-base">Performance Review</CardTitle></CardHeader>
                        <CardContent className="prose dark:prose-invert max-w-none text-sm text-muted-foreground">
                            <p className="whitespace-pre-wrap">{analysis.performance}</p>
                        </CardContent>
                    </Card>
                )}

                {/* Actionable Recommendations */}
                <Card className="border-cyan-500/20 bg-cyan-500/5 lg:col-span-2">
                    <CardHeader><CardTitle className="text-cyan-600 dark:text-cyan-400 flex items-center gap-2">
                        <Lightbulb className="w-5 h-5" />
                        Actionable Recommendations
                    </CardTitle></CardHeader>
                    <CardContent className="prose dark:prose-invert max-w-none text-sm">
                        {analysis?.actionable_recommendations ? (
                            <p className="whitespace-pre-wrap">{analysis.actionable_recommendations}</p>
                        ) : (
                            <ul className="list-disc pl-5">
                                {recommendations?.map((rec: string, i: number) => (
                                    <li key={i}>{rec}</li>
                                ))}
                            </ul>
                        )}
                    </CardContent>
                </Card>
            </div>

            {/* Simple Modal Overlay */}
            {selectedMetric && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={() => setSelectedMetric(null)}>
                    <Card
                        className={`w-full max-w-lg m-4 shadow-2xl border ${getModalClasses(selectedMetric.score)} ring-1 ring-border animate-in zoom-in-95 duration-200`}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <CardHeader className="pb-2">
                            <div className="flex justify-between items-start">
                                <div>
                                    <p className="text-sm font-medium text-muted-foreground uppercase tracking-widest mb-1">Score Explanation</p>
                                    <div className="flex items-center gap-3">
                                        <CardTitle className={`text-xl ${getScoreColor(selectedMetric.score).replace('text-', 'text-')}`}>
                                            {selectedMetric.title}
                                        </CardTitle>
                                        <div className={`text-lg font-bold px-2 py-0.5 rounded-md bg-muted/50 border border-border/50 ${getScoreColor(selectedMetric.score)}`}>
                                            {selectedMetric.score}/10
                                        </div>
                                    </div>
                                </div>
                                <Button variant="ghost" size="sm" className="h-8 w-8 p-0 rounded-full hover:bg-muted" onClick={() => setSelectedMetric(null)}>
                                    <span className="sr-only">Close</span>
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-x w-4 h-4"><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg>
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent className="pt-4">
                            <div className="prose dark:prose-invert max-w-none text-sm leading-relaxed text-muted-foreground">
                                <p className="whitespace-pre-wrap">{selectedMetric.content}</p>
                            </div>
                            <div className="mt-6 pt-4 border-t border-border/50 flex justify-end">
                                <Button variant="secondary" size="sm" onClick={() => setSelectedMetric(null)}>Close</Button>
                            </div>
                        </CardContent>
                    </Card>
                </div>
            )}

            <div className="text-[10px] text-muted-foreground/50 text-center mt-8 uppercase tracking-widest font-medium">
                AI-Generated Analysis • Not Financial Advice
            </div>
        </div>
    );
}
