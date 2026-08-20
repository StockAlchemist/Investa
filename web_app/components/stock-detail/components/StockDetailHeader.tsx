import React from 'react';
import { Badge } from '../../ui/badge';
import StockIcon from '../../StockIcon';
import { formatCurrency } from '../../../lib/utils';

interface StockDetailHeaderProps {
    symbol: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- fundamentals payload from API
    fundamentals: any;
    currency: string;
    fxRate: number;
    domain?: string;
    onClose?: () => void;
    onBack?: () => void;
}

export const StockDetailHeader: React.FC<StockDetailHeaderProps> = ({
    symbol,
    fundamentals,
    currency,
    fxRate,
    domain,
}) => {
    return (
        <div className="p-4 sm:p-6 pb-2 sm:pb-3 flex justify-between items-start gap-3">
            <div className="flex items-center gap-3 sm:gap-4 flex-1 min-w-0">
                <div className="w-10 h-10 sm:w-16 sm:h-16 rounded-xl sm:rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-lg sm:text-3xl font-bold text-white overflow-hidden flex-shrink-0">
                    <StockIcon symbol={symbol} size="100%" className="w-full h-full p-2 bg-white" domain={domain} />
                </div>
                <div className="flex-1 min-w-0 pr-2 sm:pr-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-1 sm:mb-2">
                        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                            <h2 className="text-lg sm:text-3xl font-black tracking-tight truncate shrink">{fundamentals?.shortName || symbol}</h2>
                            <Badge className="bg-secondary text-secondary-foreground border-none font-mono text-[9px] sm:text-xs shrink-0">{symbol}</Badge>
                        </div>
                        {fundamentals?.regularMarketPrice && (
                            <div className="flex items-baseline gap-1 text-indigo-600 dark:text-indigo-400">
                                <span className="text-xl sm:text-3xl font-black tracking-tight tabular-nums">
                                    {formatCurrency(fundamentals.regularMarketPrice * fxRate, currency)}
                                </span>
                            </div>
                        )}
                    </div>
                    <p className="text-muted-foreground flex items-center gap-1.5 sm:gap-2 text-[9px] sm:text-sm">
                        <span className="font-semibold text-indigo-500">{fundamentals?.sector}</span>
                        <span className="text-border">•</span>
                        <span className="truncate max-w-[120px] sm:max-w-none">{fundamentals?.industry}</span>
                    </p>
                </div>
            </div>
        </div>
    );
};

