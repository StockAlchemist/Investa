import { Holding } from '../../lib/api';

export interface HoldingsTableProps {
    holdings: Holding[];
    currency: string;
    isLoading?: boolean;
}

export type SortDirection = 'asc' | 'desc';

export interface SortConfig {
    key: string;
    direction: SortDirection;
}

export type GroupingOption = 'Market' | 'Currency' | 'Sector' | 'Industry' | 'quoteType' | 'Country' | null;

export interface ColumnGroupDef {
    label: string;
    cols: string[];
}
