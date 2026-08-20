import {
    LayoutDashboard,
    TrendingUp,
    FileText,
    BarChart3,
    DollarSign,
    PieChart as PieChartIcon,
    Sparkles,
    Newspaper,
    Wallet,
} from 'lucide-react';
import { TabType } from '../types';
import { cn } from '../../../lib/utils';

interface StockDetailTabsProps {
    activeTab: TabType;
    setActiveTab: (tab: TabType) => void;
    isEtf: boolean;
}

function TabButton({ active, onClick, icon: Icon, label }: {
    active: boolean;
    onClick: () => void;
    icon: React.ElementType;
    label: React.ReactNode;
}) {
    return (
        <button
            onClick={onClick}
            className={cn(
                "py-4 px-4 flex items-center gap-2 text-sm font-medium transition-all relative border-b-2 outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/20 cursor-pointer",
                active ? "text-indigo-600 dark:text-indigo-400 border-indigo-500" : "text-muted-foreground hover:text-foreground border-transparent"
            )}
        >
            <Icon className="w-5 h-5 sm:w-4 sm:h-4" />
            <span className="whitespace-nowrap hidden sm:inline">{label}</span>
        </button>
    );
}

export const StockDetailTabs: React.FC<StockDetailTabsProps> = ({
    activeTab,
    setActiveTab,
    isEtf,
}) => {
    return (
        <div className="px-4 sm:px-6 flex justify-around sm:justify-start gap-2 sm:gap-6 overflow-x-auto no-scrollbar">
            <TabButton
                active={activeTab === 'overview'}
                onClick={() => setActiveTab('overview')}
                icon={LayoutDashboard}
                label="Overview"
            />
            <TabButton
                active={activeTab === 'position'}
                onClick={() => setActiveTab('position')}
                icon={Wallet}
                label="Position & Lots"
            />
            <TabButton
                active={activeTab === 'chart'}
                onClick={() => setActiveTab('chart')}
                icon={TrendingUp}
                label="Chart"
            />
            <TabButton
                active={activeTab === 'analysis'}
                onClick={() => setActiveTab('analysis')}
                icon={Sparkles}
                label="Analysis"
            />
            {!isEtf && (
                <>
                    <TabButton
                        active={activeTab === 'financials'}
                        onClick={() => setActiveTab('financials')}
                        icon={FileText}
                        label="Financials"
                    />
                    <TabButton
                        active={activeTab === 'ratios'}
                        onClick={() => setActiveTab('ratios')}
                        icon={BarChart3}
                        label="Ratios & Trends"
                    />
                </>
            )}
            <TabButton
                active={activeTab === 'valuation'}
                onClick={() => setActiveTab('valuation')}
                icon={DollarSign}
                label="Valuation"
            />
            {isEtf && (
                <TabButton
                    active={activeTab === 'holdings'}
                    onClick={() => setActiveTab('holdings')}
                    icon={PieChartIcon}
                    label="Holdings"
                />
            )}
            <TabButton
                active={activeTab === 'news'}
                onClick={() => setActiveTab('news')}
                icon={Newspaper}
                label="News"
            />
        </div>
    );
};
