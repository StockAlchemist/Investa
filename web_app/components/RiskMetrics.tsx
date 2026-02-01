'use client';

import React, { useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Activity, Percent, ArrowDownRight, Zap, X, Info } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Skeleton } from "@/components/ui/skeleton";
import { createPortal } from 'react-dom';

interface RiskMetricsProps {
    metrics: {
        'Max Drawdown'?: number;
        'Volatility (Ann.)'?: number;
        'Sharpe Ratio'?: number;
        'Sortino Ratio'?: number;
    };
    isLoading: boolean;
}

interface MetricItemProps {
    label: string;
    value: string | number;
    icon: any;
    description?: string;
    colorClass?: string;
    onClick: () => void;
}

const MetricItem = ({ label, value, icon: Icon, description, colorClass, onClick }: MetricItemProps) => (
    <div
        onClick={onClick}
        className="flex flex-col p-4 bg-background/50 rounded-xl border border-border hover:border-cyan-500/50 hover:bg-cyan-500/5 hover:shadow-sm cursor-pointer transition-all duration-300 group/item h-full justify-between"
    >
        <div>
            <div className="flex items-center gap-2 text-muted-foreground mb-3">
                <div className="p-1.5 rounded-md bg-secondary/50 text-muted-foreground group-hover/item:text-cyan-500 group-hover/item:bg-cyan-500/10 transition-colors">
                    <Icon className="w-3.5 h-3.5" />
                </div>
                <span className="text-[10px] font-bold uppercase tracking-widest">{label}</span>
            </div>
            <div className={cn("text-2xl font-bold font-mono tracking-tighter", colorClass || "text-foreground")}>
                {value}
            </div>
        </div>
        {description && (
            <div className="flex items-center justify-between mt-2">
                <p className="text-[10px] text-muted-foreground opacity-60 font-medium group-hover/item:opacity-80 transition-opacity">
                    {description}
                </p>
                <Info className="w-3 h-3 text-cyan-500 opacity-0 group-hover/item:opacity-100 transition-opacity -translate-x-2 group-hover/item:translate-x-0 duration-300" />
            </div>
        )}
    </div>
);

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

export default function RiskMetrics({ metrics, isLoading }: RiskMetricsProps) {
    const [selectedMetric, setSelectedMetric] = useState<string | null>(null);

    // Close modal on escape key
    React.useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setSelectedMetric(null);
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, []);

    if (isLoading) {
        return (
            <Card className="h-full border-border bg-card">
                <CardContent className="p-6">
                    <div className="flex justify-between items-center mb-6">
                        <Skeleton className="h-4 w-32" />
                        <Skeleton className="h-8 w-8 rounded-lg" />
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {[1, 2, 3, 4].map((i) => (
                            <div key={i} className="h-24 rounded-xl border border-border/50 bg-background/50 p-4 space-y-2">
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
            icon: Activity
        },
        {
            key: 'Sortino Ratio',
            label: 'Sortino Ratio',
            value: formatNumber(metrics['Sortino Ratio']),
            description: 'Downside risk-adjusted',
            color: metrics['Sortino Ratio'] && metrics['Sortino Ratio'] > 1 ? 'text-emerald-500' : 'text-muted-foreground',
            icon: Zap
        },
        {
            key: 'Volatility',
            label: 'Volatility',
            value: formatPercent(metrics['Volatility (Ann.)']),
            description: 'Annualized std dev',
            color: 'text-muted-foreground',
            icon: Percent
        },
        {
            key: 'Max Drawdown',
            label: 'Max Drawdown',
            value: formatPercent(metrics['Max Drawdown']),
            description: 'Peak to trough decline',
            color: 'text-red-500',
            icon: ArrowDownRight
        },
    ];

    const activeExplanation = selectedMetric ? RISK_EXPLANATIONS[selectedMetric] : null;

    return (
        <React.Fragment>
            <Card className="h-full border-border hover:border-cyan-500/20 transition-all duration-300 hover:shadow-md group">
                <CardContent className="h-full p-4 sm:p-6 flex flex-col">
                    <div className="flex justify-between items-start mb-4">
                        <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Risk Analytics</h3>
                        <div className="p-2 rounded-lg bg-secondary/50 text-muted-foreground group-hover:text-cyan-500 group-hover:bg-cyan-500/10 transition-colors">
                            <Activity className="w-4 h-4" />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4 flex-1">
                        {items.map((item) => (
                            <MetricItem
                                key={item.key}
                                label={item.label}
                                value={item.value}
                                icon={item.icon}
                                description={item.description}
                                colorClass={item.color}
                                onClick={() => setSelectedMetric(item.key)}
                            />
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* Explanation Modal */}
            {selectedMetric && activeExplanation && typeof document !== 'undefined' && createPortal(
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 isolate">
                    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200" onClick={() => setSelectedMetric(null)} />
                    <div
                        style={{ backgroundColor: 'var(--menu-solid)' }}
                        className="relative w-full max-w-md border border-border/50 rounded-2xl shadow-2xl p-6 animate-in zoom-in-95 duration-200"
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
        </React.Fragment>
    );
}
