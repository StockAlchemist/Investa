import React from 'react';
import { X } from 'lucide-react';
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
    onClose,
}) => {
    return (
        <div className="p-4 sm:p-6 pb-2 sm:pb-3 flex justify-between items-start gap-3">
            <div className="flex items-center gap-3 sm:gap-4 flex-1 min-w-0">
                <div className="w-10 h-10 sm:w-16 sm:h-16 rounded-full bg-card border border-border flex items-center justify-center text-lg sm:text-2xl font-semibold text-foreground overflow-hidden flex-shrink-0">
                    <StockIcon symbol={symbol} size="100%" className="w-full h-full p-2 bg-white" domain={domain} />
                </div>
                <div className="flex-1 min-w-0 pr-2 sm:pr-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-1 sm:mb-2">
                        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                            <h2 className="font-display text-2xl sm:text-[40px] sm:leading-[44px] font-normal tracking-[-0.01em] truncate shrink">{fundamentals?.shortName || symbol}</h2>
                            <Badge variant="secondary" className="text-xs shrink-0">{symbol}</Badge>
                        </div>
                        {fundamentals?.regularMarketPrice && (
                            <div className="flex items-baseline gap-1 text-foreground">
                                <span className="text-xl sm:text-[32px] sm:leading-[38px] font-semibold tracking-[-0.01em] tabular-nums">
                                    {formatCurrency(fundamentals.regularMarketPrice * fxRate, currency)}
                                </span>
                            </div>
                        )}
                    </div>
                    <p className="text-muted-foreground flex items-center gap-1.5 sm:gap-2 text-xs sm:text-sm">
                        <span className="font-medium text-ink-2">{fundamentals?.sector}</span>
                        <span className="text-border">•</span>
                        <span className="truncate max-w-[120px] sm:max-w-none">{fundamentals?.industry}</span>
                    </p>
                </div>
            </div>

            {onClose && (
                <button
                    onClick={onClose}
                    className="p-1 px-2 -mr-1 hover:bg-muted rounded-full transition-colors text-muted-foreground hover:text-foreground relative z-20 cursor-pointer"
                    aria-label="Close modal"
                >
                    <X className="w-5 h-5 sm:w-6 sm:h-6" />
                </button>
            )}
        </div>
    );
};

