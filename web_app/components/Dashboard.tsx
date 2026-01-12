import { useState, useEffect, useRef } from 'react';
import { PortfolioSummary, PerformanceData } from '../lib/api';
import { formatCurrency, cn } from '../lib/utils';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { LayoutDashboard } from 'lucide-react';

// Lazy component import logic handled by parent or standard import above
import RiskMetrics from './RiskMetrics';
import { SectorAttribution, TopContributors } from './AttributionChart';

interface DashboardProps {
    summary: PortfolioSummary;
    currency: string;
    history?: PerformanceData[];
    isLoading?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    riskMetrics?: any;
    riskMetricsLoading?: boolean;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    attributionData?: any;
    attributionLoading?: boolean;
}

interface MetricCardProps {
    title: string;
    value: string | number;
    subValue?: number;
    isCurrency?: boolean;
    colorClass?: string;
    valueClassName?: string;
    containerClassName?: string;
    subValueClassName?: string;
    currency?: string;
    isHero?: boolean;
    isPercent?: boolean;
    vertical?: boolean;
    sparklineData?: { value: number }[];
    isLoading?: boolean;
}

const MetricCard = ({
    title,
    value,
    subValue,
    isCurrency = true,
    colorClass = '',
    valueClassName = 'text-xl sm:text-2xl',
    containerClassName = '',
    subValueClassName = '',
    vertical = false,
    sparklineData,
    currency = 'USD',
    isLoading = false
}: MetricCardProps) => (
    <Card className={cn(
        "h-full transition-all duration-300 relative overflow-hidden group",
        "hover:bg-accent/5 transition-colors",
        containerClassName
    )}>
        <CardContent className="h-full flex flex-col justify-center p-4 sm:p-6 relative">
            <p className="text-sm font-medium text-muted-foreground relative z-10 uppercase tracking-wider text-[10px]">{title}</p>

            <div className="mt-2 flex items-center gap-2 sm:gap-3 flex-wrap relative z-10">
                {isLoading ? (
                    <Skeleton className="h-8 w-32" />
                ) : (
                    <h3 className={cn("font-bold tracking-tight", colorClass || "text-foreground", valueClassName)}>
                        {value !== null && value !== undefined ? (isCurrency && typeof value === 'number' ? formatCurrency(value, currency) : value) : '-'}
                    </h3>
                )}
                {isLoading ? (
                    <Skeleton className="h-5 w-12 rounded-full" />
                ) : subValue !== undefined && subValue !== null && (
                    <Badge variant={subValue >= 0 ? "success" : "destructive"} className={cn("text-[9px] sm:text-[11px] font-bold px-1.5 sm:px-2 py-0.5", subValueClassName)}>
                        {subValue > 0 ? '+' : ''}{subValue.toFixed(2)}%
                    </Badge>
                )}
            </div>

            {!isLoading && sparklineData && sparklineData.length > 1 && (
                <div className="absolute inset-0 z-0 pointer-events-none opacity-30 group-hover:opacity-50 transition-opacity">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={sparklineData}>
                            <Line
                                type="monotone"
                                dataKey="value"
                                stroke={subValue && subValue >= 0 ? "#10b981" : "#f43f5e"}
                                strokeWidth={2}
                                dot={false}
                                isAnimationActive={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
            {isLoading && (
                <div className="absolute bottom-0 left-0 right-0 h-10 px-6">
                    <Skeleton className="h-full w-full opacity-20" />
                </div>
            )}
        </CardContent>
    </Card>
);

const COMPLEX_METRIC_IDS = ['riskMetrics', 'sectorContribution', 'topContributors'];

const DEFAULT_ITEMS = [
    { id: 'portfolioValue', title: 'Total Portfolio Value', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'dayGL', title: "Day's Gain/Loss", colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'totalReturn', title: 'Total Return', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'unrealizedGL', title: 'Unrealized G/L', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'realizedGain', title: 'Realized Gain', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'ytdDividends', title: 'Total Dividends', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'annualTWR', title: 'Annual TWR', colSpan: '' },
    { id: 'cashBalance', title: 'Cash Balance', colSpan: '' },
    { id: 'fxGL', title: 'FX Gain/Loss', colSpan: '' },
    { id: 'fees', title: 'Fees', colSpan: '' },
    { id: 'riskMetrics', title: 'Risk Analytics', colSpan: 'col-span-1 md:col-span-2 lg:col-span-4' },
    { id: 'sectorContribution', title: 'Sector Contribution', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'topContributors', title: 'Top Contributors', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
];

export default function Dashboard({
    summary,
    currency,
    history = [],
    isLoading = false,
    riskMetrics = {},
    riskMetricsLoading = false,
    attributionData = null,
    attributionLoading = false
}: DashboardProps) {
    const [visibleItems, setVisibleItems] = useState<string[]>([]);
    const [isConfigOpen, setIsConfigOpen] = useState(false);
    const configRef = useRef<HTMLDivElement>(null);

    // Initialize visibility from localStorage
    useEffect(() => {
        const saved = localStorage.getItem('investa_dashboard_visible_items');
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                if (Array.isArray(parsed)) {
                    setVisibleItems(parsed);
                    return;
                }
            } catch (e) {
                console.error("Failed to parse saved dashboard visibility", e);
            }
        }
        // Default to all items
        setVisibleItems(DEFAULT_ITEMS.map(i => i.id));
    }, []);

    // Persist visibility to localStorage
    useEffect(() => {
        if (visibleItems.length > 0) {
            localStorage.setItem('investa_dashboard_visible_items', JSON.stringify(visibleItems));
        }
    }, [visibleItems]);

    // Close config dropdown when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (configRef.current && !configRef.current.contains(event.target as Node)) {
                setIsConfigOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    const toggleItem = (id: string) => {
        if (visibleItems.includes(id)) {
            // Don't allow hiding all items - keep at least 1
            if (visibleItems.length > 1) {
                setVisibleItems(visibleItems.filter(i => i !== id));
            }
        } else {
            // Add back and resort
            const newVisible = DEFAULT_ITEMS
                .filter(item => item.id === id || visibleItems.includes(item.id))
                .map(item => item.id);

            setVisibleItems(newVisible);
        }
    };

    const m = summary?.metrics;
    const am = summary?.account_metrics;

    if (!m && !isLoading) {
        return (
            <div className="flex items-center justify-center p-12">
                <div className="animate-pulse flex flex-col items-center">
                    <div className="h-4 w-32 bg-slate-200 dark:bg-slate-700 rounded mb-4"></div>
                    <div className="h-8 w-48 bg-slate-200 dark:bg-slate-700 rounded"></div>
                </div>
            </div>
        );
    }

    // Prepare data helpers
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const cashBalance = (am?.['Cash'] as any)?.['total_market_value_display'] || 0;
    const dayGL = (m?.day_change_display as number) || 0;
    const dayGLPct = (m?.day_change_percent as number) || 0;
    const unrealizedGL = (m?.unrealized_gain as number) || 0;
    const unrealizedGLPct = m ? ((m.unrealized_gain as number) / ((m.cost_basis_held as number) || 1)) * 100 : 0;
    const fxGL = (m?.fx_gain_loss_display as number) || 0;
    const fxGLPct = (m?.fx_gain_loss_pct as number) || 0;

    const dayGLColor = dayGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const unrealizedGLColor = unrealizedGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const fxGLColor = fxGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    const totalGain = (m?.total_gain as number) || 0;
    const realizedGain = (m?.realized_gain as number) || 0;

    const totalReturnColor = totalGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const realizedGainColor = realizedGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    // Render helper same as before
    const renderContent = (id: string) => {
        switch (id) {
            case 'portfolioValue':
                return <MetricCard
                    title="Total Portfolio Value"
                    value={m?.market_value ?? 0}
                    valueClassName="text-3xl sm:text-5xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'dayGL':
                return <MetricCard
                    title="Day's Gain/Loss"
                    value={dayGL}
                    subValue={dayGLPct}
                    colorClass={dayGLColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", dayGLPct >= 0 ? "bg-emerald-600 text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    sparklineData={history.map(d => ({ value: d.twr }))}
                    isLoading={isLoading}
                />;
            case 'totalReturn':
                return <MetricCard
                    title="Total Return"
                    value={totalGain}
                    subValue={m?.total_return_pct}
                    colorClass={totalReturnColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", (m?.total_return_pct || 0) >= 0 ? "bg-emerald-600 text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'annualTWR':
                return <MetricCard
                    title="Annual TWR"
                    value={m?.annualized_twr !== undefined && m?.annualized_twr !== null ? `${m.annualized_twr.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass={m?.annualized_twr && m.annualized_twr >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}
                    isLoading={isLoading}
                />;
            case 'unrealizedGL':
                return <MetricCard
                    title="Unrealized G/L"
                    value={unrealizedGL}
                    subValue={unrealizedGLPct}
                    colorClass={unrealizedGLColor}
                    valueClassName="text-2xl sm:text-3xl"
                    subValueClassName={cn("text-base sm:text-xl", unrealizedGLPct >= 0 ? "bg-emerald-600 text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'fxGL':
                return <MetricCard
                    title="FX Gain/Loss"
                    value={fxGL}
                    subValue={fxGLPct}
                    colorClass={fxGLColor}
                    subValueClassName={cn("text-base sm:text-xl", fxGLPct >= 0 ? "bg-emerald-600 text-white hover:bg-emerald-700 border-none" : "bg-rose-600 text-white hover:bg-rose-700 border-none")}
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'realizedGain':
                return <MetricCard
                    title="Realized Gain"
                    value={realizedGain}
                    colorClass={realizedGainColor}
                    valueClassName="text-2xl sm:text-3xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'cashBalance':
                return <MetricCard
                    title="Cash Balance"
                    value={cashBalance}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'ytdDividends':
                return <MetricCard
                    title="Total Dividends"
                    value={m?.dividends ?? 0}
                    valueClassName="text-2xl sm:text-3xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'fees':
                return <MetricCard
                    title="Fees"
                    value={m?.commissions ?? 0}
                    colorClass="text-rose-600 dark:text-rose-400"
                    currency={currency}
                    isLoading={isLoading}
                />;
            case 'riskMetrics':
                return <RiskMetrics metrics={riskMetrics} isLoading={riskMetricsLoading!} />;
            case 'sectorContribution':
                return <SectorAttribution data={attributionData} isLoading={attributionLoading!} currency={currency} />;
            case 'topContributors':
                return <TopContributors data={attributionData} isLoading={attributionLoading!} currency={currency} />;
            default:
                return null;
        }
    };

    const visibleScalarItems = DEFAULT_ITEMS.filter(item => visibleItems.includes(item.id) && !COMPLEX_METRIC_IDS.includes(item.id));
    const visibleComplexItems = DEFAULT_ITEMS.filter(item => visibleItems.includes(item.id) && COMPLEX_METRIC_IDS.includes(item.id));

    return (
        <div className="mb-10 space-y-6">
            <div className="flex justify-end" ref={configRef}>
                <div className="relative">
                    <button
                        onClick={() => setIsConfigOpen(!isConfigOpen)}
                        className={cn(
                            "flex flex-col items-center justify-center gap-1.5 p-3 rounded-2xl transition-all duration-300 group",
                            "bg-card hover:bg-accent/10",
                            "border border-border shadow-sm",
                            "font-semibold tracking-tight min-w-[80px]",
                            isConfigOpen ? "border-cyan-500/50 ring-2 ring-cyan-500/20" : "text-cyan-500",
                            "flex-row py-2 px-4 h-[44px]"
                        )}
                        title="Configure Dashboard"
                    >
                        <LayoutDashboard className="w-4 h-4 text-cyan-500 mr-2" />
                        <div className="flex flex-col items-center leading-none gap-0">
                            <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent truncate font-bold uppercase text-[14px]">
                                Layout
                            </span>
                        </div>
                    </button>

                    {isConfigOpen && (
                        <div
                            style={{ backgroundColor: 'var(--menu-solid)' }}
                            className="absolute right-0 top-full mt-2 min-w-[240px] w-max origin-top-right border border-border rounded-xl shadow-xl outline-none z-50 overflow-hidden"
                        >
                            <div className="py-1 max-h-[80vh] overflow-y-auto">
                                <div className="px-4 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider bg-muted/30">
                                    Dashboard Elements
                                </div>
                                {DEFAULT_ITEMS.map((item) => {
                                    const isVisible = visibleItems.includes(item.id);
                                    return (
                                        <button
                                            key={item.id}
                                            onClick={() => toggleItem(item.id)}
                                            className={cn(
                                                "group flex items-center justify-between w-full px-4 py-3 text-sm font-medium transition-colors last:border-0",
                                                isVisible
                                                    ? 'bg-[#0097b2] text-white shadow-sm'
                                                    : 'text-popover-foreground hover:bg-black/5 dark:hover:bg-white/5'
                                            )}
                                        >
                                            <span>{item.title}</span>
                                            {isVisible && (
                                                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                                </svg>
                                            )}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Scalar Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {visibleScalarItems.map((item) => (
                    <div key={item.id} className={item.colSpan}>
                        {renderContent(item.id)}
                    </div>
                ))}
            </div>

            {/* Complex/Tall Metrics Grid */}
            {visibleComplexItems.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {visibleComplexItems.map((item) => (
                        <div key={item.id} className={item.colSpan}>
                            {renderContent(item.id)}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
