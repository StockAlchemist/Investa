import { Settings as SettingsType, Holding } from '../../lib/api';

export type Tab = 'accounts' | 'symbols' | 'overrides' | 'advanced' | 'account';

export interface SettingsProps {
    settings: SettingsType | null;
    holdings: Holding[];
    availableAccounts: string[];
    initialTab?: Tab;
    benchmarks: string[];
    onBenchmarksChange: (benchmarks: string[]) => void;
}

export interface TabDefinition {
    id: Tab;
    label: string;
    /** Tooltip on the category rail. Settings no longer draws a banner that repeats it. */
    description: string;
    icon: React.ElementType;
}
