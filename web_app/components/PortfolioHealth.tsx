'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { PortfolioHealth } from '../lib/api';
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { Activity, PieChart, ShieldCheck } from 'lucide-react';

interface PortfolioHealthProps {
    data: PortfolioHealth | null;
    isLoading: boolean;
}

const ScoreRing = ({ score }: { score: number }) => {
    const radius = 40;
    const stroke = 8;
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
                viewBox="0 0 100 100"
            >
                <circle
                    className=""
                    style={{ color: 'var(--ring-track)' }}
                    strokeWidth={stroke}
                    stroke="currentColor"
                    fill="transparent"
                    r={radius}
                    cx="50"
                    cy="50"
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
                    cx="50"
                    cy="50"
                />
            </svg>
            <div className="absolute flex flex-col items-center">
                <span className={cn("text-3xl font-bold", colorClass)}>{score}</span>
                <span className="text-xs text-muted-foreground uppercase tracking-wider">Health</span>
            </div>
        </div>
    );
};

const HealthBar = ({ score, label, icon: Icon, value }: { score: number, label: string, icon: any, value: string | number }) => {
    let colorClass = "bg-emerald-500";
    if (score < 40) colorClass = "bg-red-500";
    else if (score < 60) colorClass = "bg-yellow-500";
    else if (score < 80) colorClass = "bg-cyan-500";

    return (
        <div className="flex flex-col gap-1 w-full">
            <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-2 text-muted-foreground">
                    <Icon className="w-4 h-4" />
                    <span>{label}</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs font-mono bg-secondary px-1.5 py-0.5 rounded text-foreground">{value}</span>
                    <span className={cn("font-bold text-foreground")}>{score}</span>
                </div>
            </div>
            <div className="h-2 w-full bg-secondary/50 rounded-full overflow-hidden">
                <div
                    className={cn("h-full rounded-full transition-all duration-1000", colorClass)}
                    style={{ width: `${score}%` }}
                />
            </div>
        </div>
    );
};

export function PortfolioHealthComponent({ data, isLoading }: PortfolioHealthProps) {
    if (isLoading) {
        return <Skeleton className="h-[250px] w-full rounded-xl" />;
    }

    if (!data) return null;

    return (
        <Card
            className="border border-border"
            style={{ backgroundColor: 'var(--menu-solid)' }}
        >
            <CardHeader className="pb-4">
                <CardTitle>Portfolio Health</CardTitle>
                <CardDescription>Comprehensive risk and diversification analysis</CardDescription>
            </CardHeader>
            <CardContent>
                <div className="flex flex-col md:flex-row gap-8 items-center md:items-start">
                    {/* Ring Chart Section */}
                    <div className="flex flex-col items-center gap-2 min-w-[150px]">
                        <ScoreRing score={data.overall_score ?? 0} />
                        <div className="text-center">
                            <div className="text-lg font-semibold text-foreground">{data.rating}</div>
                            <p className="text-xs text-muted-foreground">Overall Rating</p>
                        </div>
                    </div>

                    {/* Bars Section */}
                    <div className="flex-1 w-full space-y-5 pt-2">
                        {data.debug_error && (
                            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-md mb-4">
                                <p className="text-xs text-red-500 font-mono break-all font-bold">
                                    DEBUG ERROR: {data.debug_error}
                                </p>
                            </div>
                        )}
                        <HealthBar
                            score={data.components?.diversification?.score ?? 0}
                            label="Diversification"
                            icon={PieChart}
                            value={`HHI: ${data.components?.diversification?.metric ?? 'N/A'}`}
                        />
                        <HealthBar
                            score={data.components?.efficiency?.score ?? 0}
                            label="Efficiency (Sharpe)"
                            icon={Activity}
                            value={String(data.components?.efficiency?.metric ?? 'N/A')}
                        />
                        <HealthBar
                            score={data.components?.stability?.score ?? 0}
                            label="Stability (Low Vol)"
                            icon={ShieldCheck}
                            value={String(data.components?.stability?.metric ?? 'N/A')}
                        />
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
