'use client';

import React, { useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Activity, Percent, ArrowDownRight, Zap, X, Info, PieChart, ShieldCheck, TrendingUp, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Skeleton } from "@/components/ui/skeleton";
import { createPortal } from 'react-dom';
import { PortfolioHealth } from '../lib/api';

interface RiskMetricsProps {
    metrics: {
        'Max Drawdown'?: number;
        'Volatility (Ann.)'?: number;
        'Sharpe Ratio'?: number;
        'Sortino Ratio'?: number;
    };
    portfolioHealth: PortfolioHealth | null;
    isLoading: boolean;
    isRefreshing?: boolean;
}

interface MetricItemProps {
    label: string;
    value: string | number | React.ReactNode;
    icon: any;
    description?: string;
    colorClass?: string;
    onClick: () => void;
    className?: string; // Allow custom styling for the card
    valueClassName?: string;
}

const MetricItem = ({ label, value, icon: Icon, description, colorClass, onClick }: MetricItemProps) => (
    <div
        onClick={onClick}
        className="flex flex-col p-3 bg-secondary/30 rounded-xl hover:bg-cyan-500/5 cursor-pointer transition-all duration-300 group/item h-full"
    >
        <div className="flex items-center justify-center gap-2 text-muted-foreground mb-2">
            <Icon className="w-3.5 h-3.5 group-hover/item:text-cyan-500 transition-colors" />
            <span className="text-[10px] font-bold uppercase tracking-widest">{label}</span>
        </div>

        <div className={cn("flex-1 flex items-center justify-center text-3xl font-bold font-mono tracking-tighter", colorClass || "text-foreground")}>
            {value}
        </div>

        {description && (
            <div className="flex items-center justify-between mt-1 pt-2 border-t border-border/10">
                <p className="text-[9px] text-muted-foreground opacity-60 font-medium group-hover/item:opacity-80 transition-opacity truncate w-full text-center">
                    {description}
                </p>
            </div>
        )}
    </div>
);

const ScoreRing = ({ score }: { score: number }) => {
    const radius = 48;
    const stroke = 6;
    const normalizedScore = Math.max(0, Math.min(100, score));
    const circumference = radius * 2 * Math.PI;
    const strokeDashoffset = circumference - (normalizedScore / 100) * circumference;

    let colorClass = "text-emerald-500";
    if (score < 40) colorClass = "text-red-500";
    else if (score < 60) colorClass = "text-yellow-500";
    else if (score < 80) colorClass = "text-cyan-500";

    return (
        <div className="relative flex items-center justify-center w-32 h-32">
            <svg
                className="transform -rotate-90 w-32 h-32"
                viewBox="0 0 110 110"
            >
                <circle
                    className=""
                    style={{ color: 'var(--ring-track)' }}
                    strokeWidth={stroke}
                    stroke="currentColor"
                    fill="transparent"
                    r={radius}
                    cx="55"
                    cy="55"
                />
                <circle
                    className={cn(colorClass, "transition-all duration-1000 ease-out")}
                    strokeWidth={stroke}
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    strokeLinecap="round"
                    stroke="currentColor"
                    fill="transparent"
                    r={radius}
                    cx="55"
                    cy="55"
                />
            </svg>
            <div className="absolute flex flex-col items-center justify-center">
                <span className={cn("text-3xl font-bold", colorClass)}>{score}</span>
            </div>
        </div>
    );
};

const RISK_EXPLANATIONS: Record<string, { title: string, description: string, interpretation: string, formula?: string }> = {
    'Sharpe Ratio': {
        title: 'Sharpe Ratio',
        description: 'Measures the performance of an investment compared to a risk-free asset, after adjusting for its risk.',
        interpretation: 'A higher Sharpe ratio indicates better risk-adjusted performance. Generally, a ratio > 1 is considered good. (< 1: Suboptimal, 1 - 2: Good, 2 - 3: Very Good, > 3: Excellent)',
        formula: '(Rp - Rf) / σp'
    },
    'Sortino Ratio': {
        title: 'Sortino Ratio',
        description: 'A variation of the Sharpe ratio that differentiates harmful volatility from total overall volatility by using the asset\'s standard deviation of negative portfolio returns.',
        interpretation: 'Like the Sharpe ratio, a higher result is better. It gives a more realistic view of downside risk for investors who don\'t mind upside volatility. (< 1: Bad, 1 - 2: Adequate, > 2: Great, > 3: Excellent)',
        formula: '(Rp - Rf) / σd'
    },
    'Volatility': {
        title: 'Annualized Volatility',
        description: 'A statistical measure of the dispersion of returns for a given security or market index. In this context, it represents the annualized standard deviation.',
        interpretation: 'Higher volatility means the price can change dramatically in a short time period in either direction. Lower volatility indicates steadier price action. (< 10%: Low Risk, 10-20%: Moderate, 20-30%: High, > 30%: Speculative)',
        formula: 'std_dev(returns) * √252'
    },
    'Max Drawdown': {
        title: 'Maximum Drawdown',
        description: 'The maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained.',
        interpretation: 'It is an indicator of downside risk over a specified time period. A lower (closer to 0%) drawdown suggests better capital preservation capabilities. (0-10%: Excellent, 10-20%: Good, 20-30%: Fair, > 30%: Concerning)',
        formula: '(Trough Value - Peak Value) / Peak Value'
    }
};

// ...

export default function RiskMetrics({ metrics, portfolioHealth, isLoading, isRefreshing = false }: RiskMetricsProps) {
    const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
    const [isHealthModalOpen, setIsHealthModalOpen] = useState(false);

    // Close modal on escape key
    React.useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                setSelectedMetric(null);
                setIsHealthModalOpen(false);
            }
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, []);

    if (isLoading) {
        return (
            <Card className="h-full bg-card">
                <CardContent className="p-6">
                    <div className="flex justify-between items-center mb-6">
                        <Skeleton className="h-4 w-32" />
                        <Skeleton className="h-8 w-8 rounded-lg" />
                    </div>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                        {[1, 2, 3, 4].map((i) => (
                            <div key={i} className="h-24 rounded-xl bg-background/50 p-4 space-y-2">
                                <Skeleton className="h-3 w-16" />
                                <Skeleton className="h-6 w-12" />
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>
        );
    }

    const formatPercent = (val: number | undefined) => {
        if (val === undefined) return 'N/A';
        return `${(val * 100).toFixed(2)}%`;
    };

    const formatNumber = (val: number | undefined) => {
        if (val === undefined) return 'N/A';
        return val.toFixed(2);
    };

    const items = [
        {
            key: 'Sharpe Ratio',
            label: 'Sharpe Ratio',
            value: formatNumber(metrics['Sharpe Ratio']),
            description: 'Risk-adjusted return',
            color: metrics['Sharpe Ratio'] && metrics['Sharpe Ratio'] > 1 ? 'text-emerald-500' : 'text-muted-foreground',
            icon: Activity,
            onClick: () => setSelectedMetric('Sharpe Ratio')
        },
        {
            key: 'Sortino Ratio',
            label: 'Sortino Ratio',
            value: formatNumber(metrics['Sortino Ratio']),
            description: 'Downside risk-adjusted',
            color: metrics['Sortino Ratio'] && metrics['Sortino Ratio'] > 1 ? 'text-emerald-500' : 'text-muted-foreground',
            icon: Zap,
            onClick: () => setSelectedMetric('Sortino Ratio')
        },
        {
            key: 'Volatility',
            label: 'Volatility',
            value: formatPercent(metrics['Volatility (Ann.)']),
            description: 'Annualized std dev',
            color: 'text-muted-foreground',
            icon: Percent,
            onClick: () => setSelectedMetric('Volatility')
        },
        {
            key: 'Max Drawdown',
            label: 'Max Drawdown',
            value: formatPercent(metrics['Max Drawdown']),
            description: 'Peak to trough decline',
            color: 'text-red-500',
            icon: ArrowDownRight,
            onClick: () => setSelectedMetric('Max Drawdown')
        },
    ];

    const activeExplanation = selectedMetric ? RISK_EXPLANATIONS[selectedMetric] : null;

    return (
        <React.Fragment>
            <Card className="h-full hover:bg-accent/5 transition-all duration-300 hover:shadow-md group relative overflow-hidden">
                <CardContent className="h-full p-4 flex flex-col gap-4">
                    <div className="flex justify-between items-start">
                        <div className="flex items-center gap-2">
                            <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Risk Analytics</h3>
                            {isRefreshing && !isLoading && (
                                <Loader2 className="w-3 h-3 animate-spin text-cyan-500 opacity-70" />
                            )}
                        </div>
                        <div className="absolute top-3 right-3 p-1.5 rounded-lg bg-secondary/50 text-muted-foreground group-hover:text-cyan-500 group-hover:bg-cyan-500/10 transition-colors">
                            <Activity className="w-4 h-4" />
                        </div>
                    </div>

                    <div className="flex flex-col lg:flex-row gap-4 h-full flex-1">
                        {/* Left Side: Portfolio Health */}
                        {portfolioHealth && (
                            <div
                                className="lg:w-[40%] flex flex-row items-center p-2 rounded-xl bg-secondary/5 transition-colors cursor-pointer group/health relative overflow-hidden"
                                onClick={() => setIsHealthModalOpen(true)}
                            >
                                <div className="absolute top-2 right-2 opacity-0 group-hover/health:opacity-100 transition-opacity">
                                    <Info className="w-3.5 h-3.5 text-muted-foreground" />
                                </div>

                                {/* Ring Section */}
                                <div className="flex flex-col items-center justify-center p-4 min-w-[130px] gap-2">
                                    <ScoreRing score={portfolioHealth.overall_score} />
                                    <div className={cn(
                                        "text-base font-bold tracking-tight",
                                        portfolioHealth.overall_score >= 80 ? "text-cyan-500" :
                                            portfolioHealth.overall_score >= 60 ? "text-emerald-500" :
                                                portfolioHealth.overall_score >= 40 ? "text-yellow-500" : "text-red-500"
                                    )}>
                                        {portfolioHealth.rating}
                                    </div>
                                </div>

                                {/* Breakdown Section */}
                                <div className="flex flex-col justify-center gap-3 w-full pr-2 border-l border-white/5 pl-4 py-2">
                                    <div className="flex justify-between items-center group/row">
                                        <div className="flex items-center gap-2 text-muted-foreground">
                                            <PieChart className="w-4 h-4" />
                                            <span className="text-xs font-bold uppercase tracking-wider">Diversification</span>
                                        </div>
                                        <span className={cn(
                                            "font-mono text-base font-bold",
                                            portfolioHealth.components.diversification.score >= 60 ? "text-emerald-500" : "text-yellow-500"
                                        )}>
                                            {portfolioHealth.components.diversification.score}
                                        </span>
                                    </div>
                                    <div className="flex justify-between items-center group/row">
                                        <div className="flex items-center gap-2 text-muted-foreground">
                                            <TrendingUp className="w-4 h-4" />
                                            <span className="text-xs font-bold uppercase tracking-wider">Efficiency</span>
                                        </div>
                                        <span className={cn(
                                            "font-mono text-base font-bold",
                                            portfolioHealth.components.efficiency.score >= 60 ? "text-emerald-500" : "text-yellow-500"
                                        )}>
                                            {portfolioHealth.components.efficiency.score}
                                        </span>
                                    </div>
                                    <div className="flex justify-between items-center group/row">
                                        <div className="flex items-center gap-2 text-muted-foreground">
                                            <ShieldCheck className="w-4 h-4" />
                                            <span className="text-xs font-bold uppercase tracking-wider">Stability</span>
                                        </div>
                                        <span className={cn(
                                            "font-mono text-base font-bold",
                                            portfolioHealth.components.stability.score >= 60 ? "text-emerald-500" : "text-yellow-500"
                                        )}>
                                            {portfolioHealth.components.stability.score}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Right Side: Metrics Grid */}
                        <div className="flex-1 grid grid-cols-2 gap-3">
                            {items.map((item) => (
                                <MetricItem
                                    key={item.key}
                                    label={item.label}
                                    value={item.value}
                                    icon={item.icon}
                                    description={item.description}
                                    colorClass={item.color}
                                    onClick={item.onClick}
                                />
                            ))}
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Metric Explanation Modal */}
            {selectedMetric && activeExplanation && typeof document !== 'undefined' && createPortal(
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 isolate">
                    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={() => setSelectedMetric(null)} />
                    <div
                        style={{ backgroundColor: 'var(--menu-solid)' }}
                        className="relative w-full max-w-md border-border/50 rounded-2xl shadow-2xl p-6 animate-in zoom-in-95 duration-200"
                    >
                        <button
                            onClick={() => setSelectedMetric(null)}
                            className="absolute top-4 right-4 p-2 rounded-full hover:bg-secondary/50 text-muted-foreground transition-colors"
                        >
                            <X className="w-4 h-4" />
                        </button>

                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-3 rounded-xl bg-cyan-500/10 text-cyan-500">
                                <Info className="w-6 h-6" />
                            </div>
                            <div>
                                <h3 className="text-lg font-bold text-foreground">{activeExplanation.title}</h3>
                                <p className="text-xs text-muted-foreground font-mono">Metric Explanation</p>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-2">
                                <h4 className="text-xs font-bold uppercase tracking-widest text-muted-foreground">What is it?</h4>
                                <p className="text-sm text-foreground/90 leading-relaxed">
                                    {activeExplanation.description}
                                </p>
                            </div>

                            <div className="h-px bg-border/50" />

                            <div className="space-y-2">
                                <h4 className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Interpretation</h4>
                                <p className="text-sm text-foreground/90 leading-relaxed">
                                    {activeExplanation.interpretation}
                                </p>
                            </div>

                            {activeExplanation.formula && (
                                <div className="pt-2">
                                    <div className="bg-secondary/30 rounded-lg p-3 border border-border/50">
                                        <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Approx. Formula</div>
                                        <div className="font-mono text-xs text-cyan-600 dark:text-cyan-400">
                                            {activeExplanation.formula}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>,
                document.body
            )}

            {/* Health Analysis Modal */}
            {isHealthModalOpen && portfolioHealth && typeof document !== 'undefined' && createPortal(
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 isolate">
                    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={() => setIsHealthModalOpen(false)} />
                    <div
                        style={{ backgroundColor: 'var(--menu-solid)' }}
                        className="relative w-full max-w-lg border-border/50 rounded-2xl shadow-2xl p-6 animate-in zoom-in-95 duration-200 max-h-[90vh] overflow-y-auto"
                    >
                        <button
                            onClick={() => setIsHealthModalOpen(false)}
                            className="absolute top-4 right-4 p-2 rounded-full hover:bg-secondary/50 text-muted-foreground transition-colors"
                        >
                            <X className="w-4 h-4" />
                        </button>

                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-3 rounded-xl bg-cyan-500/10 text-cyan-500">
                                <Activity className="w-6 h-6" />
                            </div>
                            <div>
                                <h3 className="text-lg font-bold text-foreground">Portfolio Health Analysis</h3>
                                <p className="text-xs text-muted-foreground font-mono">Scoring Methodology</p>
                            </div>
                        </div>

                        <div className="space-y-6">
                            <div className="p-4 bg-secondary/20 rounded-xl border border-border/50 space-y-4">
                                <div>
                                    <h4 className="text-sm font-bold text-foreground mb-2">Overall Score</h4>
                                    <p className="text-xs text-muted-foreground mb-3">
                                        The overall score is a weighted average of three key pillars:
                                    </p>
                                    <div className="grid grid-cols-3 gap-2 text-center text-xs">
                                        <div className="p-2 bg-background/50 rounded-lg border border-border/50">
                                            <div className="font-bold text-emerald-500">40%</div>
                                            <div className="text-[10px] text-muted-foreground uppercase tracking-wide mt-1">Diversification</div>
                                        </div>
                                        <div className="p-2 bg-background/50 rounded-lg border border-border/50">
                                            <div className="font-bold text-emerald-500">40%</div>
                                            <div className="text-[10px] text-muted-foreground uppercase tracking-wide mt-1">Efficiency</div>
                                        </div>
                                        <div className="p-2 bg-background/50 rounded-lg border border-border/50">
                                            <div className="font-bold text-cyan-500">20%</div>
                                            <div className="text-[10px] text-muted-foreground uppercase tracking-wide mt-1">Stability</div>
                                        </div>
                                    </div>
                                </div>

                                <div className="border-t border-border/50 pt-3">
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2">Score Legend</p>
                                    <div className="flex items-center justify-between text-[10px]">
                                        <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-red-500" /><span>0-39 Critical</span></div>
                                        <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-yellow-500" /><span>40-59 Fair</span></div>
                                        <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-emerald-500" /><span>60-79 Good</span></div>
                                        <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-cyan-500" /><span>80-100 Excellent</span></div>
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-5">
                                <div className="space-y-2 pb-4 border-b border-border/30 last:border-0 last:pb-0">
                                    <h4 className="text-sm font-bold flex items-center gap-2">
                                        <PieChart className="w-4 h-4 text-emerald-500" />
                                        Diversification (HHI)
                                        <span className={cn(
                                            "ml-auto text-xs font-mono px-2 py-0.5 rounded",
                                            portfolioHealth.components.diversification.score >= 60 ? "bg-emerald-500/10 text-emerald-500" : "bg-yellow-500/10 text-yellow-500"
                                        )}>
                                            {portfolioHealth.components.diversification.metric}
                                        </span>
                                    </h4>
                                    <div className="text-xs text-muted-foreground leading-relaxed">
                                        Score: <strong>{portfolioHealth.components.diversification.score}</strong>/100
                                    </div>
                                    <p className="text-xs text-muted-foreground leading-relaxed">
                                        Measured using the <strong>Herfindahl-Hirschman Index (HHI)</strong>. Lower is better, indicating less concentration.
                                    </p>
                                    <div className="mt-2 text-[10px] text-muted-foreground/70 bg-secondary/30 p-2 rounded-lg border border-border/50">
                                        <span className="font-bold opacity-100">Range:</span> Score 80-100: &lt;0.12 (Excellent) • Score 60-79: 0.12-0.35 (Moderate) • Score &lt;60: &gt;0.35 (Concentrated)
                                    </div>
                                </div>

                                <div className="space-y-2 pb-4 border-b border-border/30 last:border-0 last:pb-0">
                                    <h4 className="text-sm font-bold flex items-center gap-2">
                                        <Activity className="w-4 h-4 text-emerald-500" />
                                        Efficiency (Sharpe Cost)
                                        <span className={cn(
                                            "ml-auto text-xs font-mono px-2 py-0.5 rounded",
                                            portfolioHealth.components.efficiency.score >= 60 ? "bg-emerald-500/10 text-emerald-500" : "bg-yellow-500/10 text-yellow-500"
                                        )}>
                                            {portfolioHealth.components.efficiency.metric}
                                        </span>
                                    </h4>
                                    <div className="text-xs text-muted-foreground leading-relaxed">
                                        Score: <strong>{portfolioHealth.components.efficiency.score}</strong>/100
                                    </div>
                                    <p className="text-xs text-muted-foreground leading-relaxed">
                                        Return generated per unit of risk. Higher is better.
                                    </p>
                                    <div className="mt-2 text-[10px] text-muted-foreground/70 bg-secondary/30 p-2 rounded-lg border border-border/50">
                                        <span className="font-bold opacity-100">Range:</span> Score 80-100: &gt;1.0 (Excellent) • Score 30-79: 0-1.0 (Fair) • Score &lt;30: &lt;0 (Poor)
                                    </div>
                                </div>

                                <div className="space-y-2 pb-4 border-b border-border/30 last:border-0 last:pb-0">
                                    <h4 className="text-sm font-bold flex items-center gap-2">
                                        <ShieldCheck className="w-4 h-4 text-cyan-500" />
                                        Stability (Volatility)
                                        <span className={cn(
                                            "ml-auto text-xs font-mono px-2 py-0.5 rounded",
                                            portfolioHealth.components.stability.score >= 60 ? "bg-emerald-500/10 text-emerald-500" : "bg-yellow-500/10 text-yellow-500"
                                        )}>
                                            {portfolioHealth.components.stability.metric}
                                        </span>
                                    </h4>
                                    <div className="text-xs text-muted-foreground leading-relaxed">
                                        Score: <strong>{portfolioHealth.components.stability.score}</strong>/100
                                    </div>
                                    <p className="text-xs text-muted-foreground leading-relaxed">
                                        Annualized volatility. Lower suggests steadier growth.
                                    </p>
                                    <div className="mt-2 text-[10px] text-muted-foreground/70 bg-secondary/30 p-2 rounded-lg border border-border/50">
                                        <span className="font-bold opacity-100">Range:</span> Score 80-100: 5-25% (Ideal) • Score 40-79: &lt;5% or 25-35% • Score &lt;40: &gt;35% (High)
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>,
                document.body
            )}
        </React.Fragment>
    );
}
