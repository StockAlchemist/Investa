import React from 'react';
import { List, PieChart as PieChartIcon } from 'lucide-react';
import { useStockModal } from '@/context/StockModalContext';
import {
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell,
    Tooltip,
    Legend
} from 'recharts';

interface HoldingsTabProps {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- fundamentals payload
    fundamentals: any;
}

const COLORS = ['#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#f59e0b', '#10b981', '#6366f1'];

export const HoldingsTab: React.FC<HoldingsTabProps> = ({ fundamentals }) => {
    const { openStockDetail } = useStockModal();

    if (!fundamentals?.etf_data) return null;
    const { top_holdings, sector_weightings } = fundamentals.etf_data;

    const sectorData = Object.entries(sector_weightings || {}).map(([name, value]) => ({
        name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        value: (value as number) * 100
    })).sort((a, b) => b.value - a.value);

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Top Holdings Table */}
                <div className="bg-muted rounded-2xl p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <List className="w-5 h-5 text-indigo-500" />
                        Top Holdings
                    </h3>
                    <div className="overflow-hidden rounded-xl">
                        <table className="w-full text-sm">
                            <thead className="bg-secondary/50">
                                <tr>
                                    <th className="px-4 py-2 text-left font-medium text-muted-foreground">Symbol</th>
                                    <th className="px-4 py-2 text-right font-medium text-muted-foreground">% Assets</th>
                                </tr>
                            </thead>
                            <tbody>
                                {top_holdings?.map((h: { symbol: string; percent: number }, i: number) => (
                                    <tr
                                        key={i}
                                        onClick={() => openStockDetail(h.symbol)}
                                        className="hover:bg-accent/10 cursor-pointer transition-colors"
                                    >
                                        <td className="px-4 py-2 font-semibold text-indigo-600 dark:text-indigo-400 hover:underline">{h.symbol}</td>
                                        <td className="px-4 py-2 text-right tabular-nums">{(h.percent * 100).toFixed(2)}%</td>
                                    </tr>
                                ))}
                                {(!top_holdings || top_holdings.length === 0) && (
                                    <tr>
                                        <td colSpan={2} className="px-4 py-8 text-center text-muted-foreground">No holdings data available</td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>


                {/* Sector Allocation Chart */}
                <div className="bg-muted rounded-2xl p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <PieChartIcon className="w-5 h-5 text-indigo-500" />
                        Sector Allocation
                    </h3>
                    {sectorData.length > 0 ? (
                        <div className="h-[300px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={sectorData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={80}
                                        paddingAngle={2}
                                        dataKey="value"
                                    >
                                        {sectorData.map((_, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="rgba(0,0,0,0.1)" />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                        content={({ active, payload }) => {
                                            if (active && payload && payload.length) {
                                                return (
                                                    <div className="bg-background/95 backdrop-blur-xl p-3 rounded-xl border border-border/50 shadow-2xl">
                                                        <p className="font-medium text-foreground">{payload[0].name}</p>
                                                        <p className="text-sm text-muted-foreground">
                                                            {Number(payload[0].value).toFixed(2)}%
                                                        </p>
                                                    </div>
                                                );
                                            }
                                            return null;
                                        }}
                                    />
                                    <Legend
                                        layout="vertical"
                                        verticalAlign="middle"
                                        align="right"
                                        formatter={(value) => <span className="text-xs text-muted-foreground ml-1">{value}</span>}
                                    />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                            No sector data available
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
