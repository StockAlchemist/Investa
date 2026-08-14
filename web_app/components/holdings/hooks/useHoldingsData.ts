import { useState, useMemo, useCallback } from 'react';
import { useQueryClient, useMutation } from '@tanstack/react-query';
import { Holding, Lot, updateHoldingTags } from '../../../lib/api';
import { SortConfig, GroupingOption } from '../types';
import {
    COLUMN_DEFINITIONS,
    CASH_DUST_THRESHOLD,
    INVESTMENT_TYPE_MAP,
    CURRENCY_MAP
} from '../constants';
import { isCashSymbol, normalizeMarketName } from '../holdingsUtils';

export function getHoldingValue(holding: Holding, header: string, currency: string): string | number | string[] | number[] | null {
    const prefix = COLUMN_DEFINITIONS[header];
    if (!prefix) return null;

    if (holding[prefix] !== undefined) return holding[prefix] as string | number | string[] | number[] | null;

    const keyWithCurrency = `${prefix} (${currency})`;
    if (holding[keyWithCurrency] !== undefined) return holding[keyWithCurrency] as string | number | string[] | number[] | null;

    const foundKey = Object.keys(holding).find(k => k.startsWith(prefix));
    if (foundKey) {
        return holding[foundKey] as string | number | string[] | number[] | null;
    }
    return null;
}

interface UseHoldingsDataOptions {
    holdings: Holding[];
    currency: string;
    visibleColumns: string[];
    sortConfig: SortConfig;
}

export function useHoldingsData({
    holdings,
    currency,
    visibleColumns,
    sortConfig,
}: UseHoldingsDataOptions) {
    const queryClient = useQueryClient();

    const [searchQuery, setSearchQuery] = useState("");
    const [selectedAccounts, setSelectedAccounts] = useState<Set<string>>(new Set());
    const [groupBy, setGroupBy] = useState<GroupingOption>(null);
    const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
    const [visibleRows, setVisibleRows] = useState(10);

    const getValue = useCallback((holding: Holding, header: string) => {
        return getHoldingValue(holding, header, currency);
    }, [currency]);

    // Tags editing state
    const [editingTags, setEditingTags] = useState<{ symbol: string, account: string, currentTags: string } | null>(null);
    const [tagsInput, setTagsInput] = useState("");

    const updateTagsMutation = useMutation({
        mutationFn: ({ account, symbol, tags }: { account: string, symbol: string, tags: string }) => updateHoldingTags(account, symbol, tags),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['holdings'] });
            queryClient.invalidateQueries({ queryKey: ['summary'] });
            setEditingTags(null);
        },
    });

    const handleEditTags = (symbol: string, account: string, currentTags: string[]) => {
        setEditingTags({ symbol, account, currentTags: currentTags.join(", ") });
        setTagsInput(currentTags.join(", "));
    };

    const handleSaveTags = () => {
        if (editingTags) {
            updateTagsMutation.mutate({
                account: editingTags.account,
                symbol: editingTags.symbol,
                tags: tagsInput
            });
        }
    };

    const getLotValue = useCallback((lot: Lot, header: string, holdingPrice?: number) => {
        if (header === 'Quantity') return lot.Quantity;
        if (header === 'Cost Basis' || header === 'Total Buy Cost') return lot['Cost Basis'];
        if (header === 'Mkt Val') {
            if (lot['Market Value']) return lot['Market Value'];
            if (holdingPrice && lot.Quantity) return holdingPrice * lot.Quantity;
            return null;
        }
        if (header === 'Unreal. G/L' || header === 'Total G/L') {
            if (lot['Unreal. Gain']) return lot['Unreal. Gain'];
            const mktVal = lot['Market Value'] || (holdingPrice ? holdingPrice * lot.Quantity : 0);
            if (mktVal && lot['Cost Basis']) return mktVal - lot['Cost Basis'];
            return null;
        }
        if (header === 'Unreal. G/L %' || header === 'Total Ret %') {
            if (lot['Unreal. Gain %']) return lot['Unreal. Gain %'];
            const mktVal = lot['Market Value'] || (holdingPrice ? holdingPrice * lot.Quantity : 0);
            if (mktVal && lot['Cost Basis']) return ((mktVal - lot['Cost Basis']) / lot['Cost Basis']) * 100;
            return null;
        }
        if ((header === 'Price' || header === 'Avg Cost') && lot.Quantity) return lot['Cost Basis'] / lot.Quantity;

        if (header === 'Symbol') return `Lot: ${lot.Date}`;
        if (header === 'Account' && !visibleColumns.includes('Symbol')) return `Lot: ${lot.Date}`;

        return null;
    }, [visibleColumns]);

    const getExpansionKey = (holding: Holding) => {
        return visibleColumns.includes('Account') ? `${holding.Symbol}-${holding.Account}` : holding.Symbol;
    };

    const toggleAccount = (account: string) => {
        setSelectedAccounts(prev => {
            const next = new Set(prev);
            if (next.has(account)) next.delete(account);
            else next.add(account);
            return next;
        });
    };

    const uniqueAccounts = useMemo(() => Array.from(new Set(holdings.map(h => h.Account).filter(Boolean) as string[])).sort(), [holdings]);

    const filteredHoldings = useMemo(() => {
        if (!holdings) return [];
        return holdings.filter(h => {
            const matchesSearch = h.Symbol.toLowerCase().includes(searchQuery.toLowerCase());
            const matchesAccount = selectedAccounts.size === 0 || (h.Account && selectedAccounts.has(h.Account));
            if (isCashSymbol(h.Symbol)) {
                const mktVal = getValue(h, 'Mkt Val');
                const amount = typeof mktVal === 'number' ? mktVal : 0;
                if (Math.abs(amount) <= CASH_DUST_THRESHOLD) return false;
            }
            return matchesSearch && matchesAccount;
        });
    }, [holdings, searchQuery, selectedAccounts, getValue]);

    const aggregatedHoldings = useMemo(() => {
        if (visibleColumns.includes('Account')) {
            return filteredHoldings;
        }

        const grouped = new Map<string, Holding>();

        const getRawVal = (h: Holding, key: string): number => {
            const val = getValue(h, key);
            return typeof val === 'number' ? val : 0;
        };

        filteredHoldings.forEach(h => {
            if (!grouped.has(h.Symbol)) {
                grouped.set(h.Symbol, { ...h, lots: h.lots ? [...h.lots] : [] });
            } else {
                const current = grouped.get(h.Symbol)!;

                if (h.lots) {
                    current.lots = (current.lots || []).concat(h.lots);
                }

                const keysToSum = [
                    "Quantity", "Mkt Val", "Cost Basis", "Day Chg",
                    "Unreal. G/L", "Real. G/L", "Divs", "Fees",
                    "Total G/L", "Total Buy Cost", "Est. Income",
                    "Contribution %", "FX G/L", "% of Total"
                ];

                keysToSum.forEach(header => {
                    const def = COLUMN_DEFINITIONS[header];
                    if (def) {
                        const valA = getRawVal(current, header);
                        const valB = getRawVal(h, header);
                        (current as Record<string, number>)[def] = valA + valB;
                    }
                });

                const currentTags = Array.isArray(getValue(current, "Tags")) ? getValue(current, "Tags") as string[] : [];
                const newTags = Array.isArray(getValue(h, "Tags")) ? getValue(h, "Tags") as string[] : [];
                const mergedTags = Array.from(new Set([...currentTags, ...newTags]));
                (current as Record<string, string[]>)["Tags"] = mergedTags;
            }
        });

        return Array.from(grouped.values()).map(h => {
            const qty = getRawVal(h, 'Quantity');
            const mktVal = getRawVal(h, 'Mkt Val');
            const costBasis = getRawVal(h, 'Cost Basis');
            const dayChg = getRawVal(h, 'Day Chg');
            const unrealGl = getRawVal(h, 'Unreal. G/L');
            const estIncome = getRawVal(h, 'Est. Income');
            const totalGl = getRawVal(h, 'Total G/L');

            if (qty !== 0) {
                (h as Record<string, number>)['Price'] = mktVal / qty;
                (h as Record<string, number>)['Avg Cost'] = costBasis / qty;
            }

            if (mktVal - dayChg !== 0) {
                (h as Record<string, number>)['Day Change %'] = (dayChg / (mktVal - dayChg)) * 100;
            } else {
                (h as Record<string, number>)['Day Change %'] = 0;
            }

            const EPSILON = 0.0001;
            const totalBuyCost = getRawVal(h, 'Total Buy Cost');
            const denominator = (Math.abs(totalBuyCost) > EPSILON) ? totalBuyCost : costBasis;
            const hasDenominator = Math.abs(denominator) > EPSILON;

            if (hasDenominator) {
                (h as Record<string, number>)['Unreal. Gain %'] = (unrealGl / denominator) * 100;
                (h as Record<string, number>)['Div. Yield (Cost) %'] = (estIncome / denominator) * 100;
                (h as Record<string, number>)['Total Return %'] = (totalGl / denominator) * 100;
            } else {
                (h as Record<string, number>)['Unreal. Gain %'] = unrealGl > EPSILON ? Infinity : 0;
                (h as Record<string, number>)['Div. Yield (Cost) %'] = estIncome > EPSILON ? Infinity : 0;
                (h as Record<string, number>)['Total Return %'] = totalGl > EPSILON ? Infinity : 0;
            }

            if (mktVal !== 0) {
                (h as Record<string, number>)['Div. Yield (Current) %'] = (estIncome / mktVal) * 100;
            }

            if ((h as Record<string, number | null | undefined>)['Aggregate IRR (%)'] != null) {
                (h as Record<string, number | null | undefined>)['IRR (%)'] = (h as Record<string, number | null | undefined>)['Aggregate IRR (%)'];
            }

            return h;
        });
    }, [filteredHoldings, visibleColumns, getValue]);

    const groupedHoldings = useMemo(() => {
        if (!groupBy) return null;

        const groups = new Map<string, {
            key: string;
            holdings: Holding[];
            aggregates: Record<string, number>;
        }>();

        aggregatedHoldings.forEach(h => {
            let groupKey = 'Other';
            if (groupBy === 'Market') {
                const rawExchange = (h as Record<string, string>)['fullExchangeName'] || (h as Record<string, string>)['exchange'] || (h as Record<string, string>)['Market'] || 'Unknown';
                groupKey = normalizeMarketName(rawExchange);
            } else if (groupBy === 'quoteType') {
                const rawType = (h as Record<string, string>)['quoteType'] || 'Other';
                groupKey = INVESTMENT_TYPE_MAP[rawType] || rawType;
            } else if (groupBy === 'Country') {
                groupKey = (h as Record<string, string>)['geography'] || (h as Record<string, string>)['Country'] || 'Unknown';
            } else if (groupBy === 'Currency') {
                const rawCurrency = (h as Record<string, string>)['Local Currency'] || 'Unknown';
                groupKey = CURRENCY_MAP[rawCurrency] || rawCurrency;
            } else {
                const val = getValue(h, groupBy);
                groupKey = val ? String(val) : 'Other';
            }

            if (!groups.has(groupKey)) {
                groups.set(groupKey, {
                    key: groupKey,
                    holdings: [],
                    aggregates: {
                        'Mkt Val': 0,
                        'Day Chg': 0,
                        'Cost Basis': 0,
                        'Unreal. G/L': 0,
                        'Real. G/L': 0,
                        'Divs': 0,
                        'Fees': 0,
                        'Total G/L': 0,
                        'Total Buy Cost': 0,
                    }
                });
            }

            const group = groups.get(groupKey)!;
            group.holdings.push(h);

            const getNum = (key: string) => {
                const val = getValue(h, key);
                return typeof val === 'number' ? val : 0;
            };

            group.aggregates['Mkt Val'] += getNum('Mkt Val');
            group.aggregates['Day Chg'] += getNum('Day Chg');
            group.aggregates['Cost Basis'] += getNum('Cost Basis');
            group.aggregates['Unreal. G/L'] += getNum('Unreal. G/L');
            group.aggregates['Real. G/L'] += getNum('Real. G/L');
            group.aggregates['Divs'] += getNum('Divs');
            group.aggregates['Fees'] += getNum('Fees');
            group.aggregates['Total G/L'] += getNum('Total G/L');
            group.aggregates['Total Buy Cost'] += getNum('Total Buy Cost');
        });

        return Array.from(groups.values()).map(g => {
            if (g.aggregates['Mkt Val'] !== 0 && g.aggregates['Mkt Val'] - g.aggregates['Day Chg'] !== 0) {
                g.aggregates['Day Chg %'] = (g.aggregates['Day Chg'] / (g.aggregates['Mkt Val'] - g.aggregates['Day Chg'])) * 100;
            }
            const costDenominator = (Math.abs(g.aggregates['Total Buy Cost']) > 0.0001)
                ? g.aggregates['Total Buy Cost']
                : g.aggregates['Cost Basis'];

            if (Math.abs(costDenominator) > 0.0001) {
                g.aggregates['Unreal. G/L %'] = (g.aggregates['Unreal. G/L'] / costDenominator) * 100;
                g.aggregates['Total Ret %'] = (g.aggregates['Total G/L'] / costDenominator) * 100;
            }

            g.holdings.sort((a, b) => {
                const valA = getValue(a, sortConfig.key);
                const valB = getValue(b, sortConfig.key);

                if (valA === null || valA === undefined) return 1;
                if (valB === null || valB === undefined) return -1;

                if (typeof valA === 'number' && typeof valB === 'number') {
                    return sortConfig.direction === 'asc' ? valA - valB : valB - valA;
                }
                return sortConfig.direction === 'asc'
                    ? String(valA).localeCompare(String(valB))
                    : String(valB).localeCompare(String(valA));
            });

            return g;
        }).sort((a, b) => {
            if (sortConfig.key in a.aggregates && sortConfig.key in b.aggregates) {
                const valA = a.aggregates[sortConfig.key];
                const valB = b.aggregates[sortConfig.key];
                return sortConfig.direction === 'asc' ? valA - valB : valB - valA;
            }
            return b.aggregates['Mkt Val'] - a.aggregates['Mkt Val'];
        });
    }, [groupBy, aggregatedHoldings, getValue, sortConfig]);

    const toggleGroup = (groupKey: string) => {
        setExpandedGroups(prev => {
            const next = new Set(prev);
            if (next.has(groupKey)) next.delete(groupKey);
            else next.add(groupKey);
            return next;
        });
    };

    const handleSetGroupBy = (option: GroupingOption) => {
        setGroupBy(option);
        if (option) {
            setExpandedGroups(new Set());
        }
    };

    const sortedHoldings = useMemo(() => {
        if (groupBy) return aggregatedHoldings;
        return [...aggregatedHoldings].sort((a, b) => {
            const valA = getValue(a, sortConfig.key);
            const valB = getValue(b, sortConfig.key);

            if (valA === null || valA === undefined) return 1;
            if (valB === null || valB === undefined) return -1;

            if (typeof valA === 'number' && typeof valB === 'number') {
                return sortConfig.direction === 'asc' ? valA - valB : valB - valA;
            }
            return sortConfig.direction === 'asc'
                ? String(valA).localeCompare(String(valB))
                : String(valB).localeCompare(String(valA));
        });
    }, [aggregatedHoldings, sortConfig, groupBy, getValue]);

    const handleShowMore = () => {
        setVisibleRows(prev => prev + 20);
    };

    const handleShowAll = () => {
        setVisibleRows(sortedHoldings.length);
    };

    return {
        searchQuery,
        setSearchQuery,
        selectedAccounts,
        setSelectedAccounts,
        toggleAccount,
        uniqueAccounts,
        groupBy,
        setGroupBy,
        handleSetGroupBy,
        expandedGroups,
        toggleGroup,
        getValue,
        getLotValue,
        getExpansionKey,
        filteredHoldings,
        aggregatedHoldings,
        groupedHoldings,
        sortedHoldings,
        visibleRows,
        handleShowMore,
        handleShowAll,
        editingTags,
        setEditingTags,
        tagsInput,
        setTagsInput,
        handleEditTags,
        handleSaveTags,
        updateTagsMutation,
    };
}
