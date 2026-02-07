'use client';

import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchPortfolioAIReview } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshCw, ShieldCheck, TrendingUp, AlertTriangle, Lightbulb, Sparkles, PieChart } from 'lucide-react';
import { MetricCard } from '@/components/MetricCard';

interface PortfolioAIReviewProps {
    currency: string;
    accounts: string[];
}

export default function PortfolioAIReview({ currency, accounts }: PortfolioAIReviewProps) {
    const queryClient = useQueryClient();

    const { data, isLoading, isError, error } = useQuery({
        queryKey: ['portfolioAIReview', currency, accounts],
        queryFn: ({ signal }) => fetchPortfolioAIReview(currency, accounts, false, signal),
        staleTime: 24 * 60 * 60 * 1000, // Cache for 24 hours
        retry: 1
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

    if (!data) return null;

    const { scorecard, analysis, summary, recommendations } = data;

    // Helper to determine color based on score (1-10)
    const getScoreColor = (score: number) => {
        if (!score) return "text-muted-foreground";
        if (score >= 8) return "text-emerald-500";
        if (score >= 5) return "text-yellow-500";
        return "text-rose-500";
    };

    return (
        <div className="space-y-6 animate-in fade-in duration-500">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
                        <Sparkles className="w-6 h-6 text-purple-500" />
                        Portfolio AI Review
                    </h2>
                    <p className="text-sm text-muted-foreground mt-1">
                        AI-driven insights and recommendations for your portfolio.
                    </p>
                </div>
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
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <MetricCard
                    title="Diversification"
                    value={scorecard?.diversification || '-'}
                    subValue={null}
                    isCurrency={false}
                    icon={PieChart}
                    colorClass={getScoreColor(scorecard?.diversification)}
                />
                <MetricCard
                    title="Risk Profile"
                    value={scorecard?.risk_profile || '-'}
                    subValue={null}
                    isCurrency={false}
                    icon={ShieldCheck}
                    colorClass={getScoreColor(scorecard?.risk_profile)}
                />
                <MetricCard
                    title="Performance"
                    value={scorecard?.performance || '-'}
                    subValue={null}
                    isCurrency={false}
                    icon={TrendingUp}
                    colorClass={getScoreColor(scorecard?.performance)}
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

            <div className="text-[10px] text-muted-foreground/50 text-center mt-8 uppercase tracking-widest font-medium">
                AI-Generated Analysis • Not Financial Advice
            </div>
        </div>
    );
}
