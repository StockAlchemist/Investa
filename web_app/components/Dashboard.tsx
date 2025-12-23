import React, { useState, useEffect } from 'react';
import { PortfolioSummary } from '../lib/api';
import { formatCurrency } from '../lib/utils';
import {
    DndContext,
    closestCenter,
    KeyboardSensor,
    PointerSensor,
    useSensor,
    useSensors,
    DragEndEvent
} from '@dnd-kit/core';
import {
    arrayMove,
    SortableContext,
    sortableKeyboardCoordinates,
    rectSortingStrategy,
    useSortable
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

interface DashboardProps {
    summary: PortfolioSummary;
    currency: string;
}

const MetricCard = ({
    title,
    value,
    subValue,
    isCurrency = true,
    isPercent = false,
    colorClass = '',
    vertical = false,
    valueClassName = 'text-2xl',
    containerClassName = '',
    isHero = false,
    subValueClassName = '',
    currency = 'USD'
}: any) => (
    <div className={`
        relative overflow-hidden rounded-2xl p-5 transition-all duration-300 h-full
        ${isHero
            ? 'bg-gradient-to-br from-white to-slate-50 dark:from-slate-800 dark:to-slate-900 shadow-md hover:shadow-lg border border-slate-200 dark:border-slate-700'
            : 'bg-white dark:bg-slate-800 shadow-sm hover:shadow border border-slate-100 dark:border-slate-700'
        }
        ${containerClassName}
    `}>
        {isHero && (
            <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-gradient-to-br from-slate-100 to-transparent dark:from-slate-700 rounded-full opacity-50 blur-2xl pointer-events-none"></div>
        )}

        <p className="text-sm font-medium text-slate-500 dark:text-slate-400 relative z-10">{title}</p>

        <div className={`mt-2 flex items-baseline gap-2 flex-wrap relative z-10`}>
            <h3 className={`font-extrabold tracking-tight ${valueClassName} ${colorClass || 'text-slate-900 dark:text-white'}`}>
                {value !== null && value !== undefined ? (isCurrency ? formatCurrency(value, currency) : value) : '-'}
            </h3>
            {subValue && (
                <span className={`
                    ${subValueClassName || valueClassName} font-semibold
                    ${subValue >= 0
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : 'text-rose-600 dark:text-rose-400'
                    }
                `}>
                    ({subValue > 0 ? '+' : ''}{subValue.toFixed(2)}%)
                </span>
            )}
        </div>
    </div>
);

// Sortable Item Wrapper
function SortableMetricItem({ id, colSpan, children, isCustomizing }: any) {
    const {
        attributes,
        listeners,
        setNodeRef,
        transform,
        transition,
    } = useSortable({ id });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
    };

    return (
        <div
            ref={setNodeRef}
            style={style}
            className={`${colSpan} ${isCustomizing ? 'cursor-grab active:cursor-grabbing hover:ring-2 hover:ring-blue-500 rounded-2xl' : ''}`}
            {...(isCustomizing ? { ...attributes, ...listeners } : {})}
        >
            {/* Overlay to intercept clicks when customizing */}
            {isCustomizing && <div className="absolute inset-0 z-20 bg-transparent" />}
            {children}
        </div>
    );
}

const DEFAULT_ITEMS = [
    { id: 'portfolioValue', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'dayGL', colSpan: 'col-span-1 md:col-span-2 lg:col-span-2' },
    { id: 'totalReturn', colSpan: '' },
    { id: 'annualTWR', colSpan: '' },
    { id: 'unrealizedGL', colSpan: '' },
    { id: 'unrealizedGLPct', colSpan: '' },
    { id: 'realizedGain', colSpan: '' },
    { id: 'cashBalance', colSpan: '' },
    { id: 'ytdDividends', colSpan: '' },
    { id: 'fees', colSpan: '' },
];

export default function Dashboard({ summary, currency }: DashboardProps) {
    const m = summary?.metrics;
    const am = summary?.account_metrics;

    // State
    const [isCustomizing, setIsCustomizing] = useState(false);
    const [items, setItems] = useState(DEFAULT_ITEMS);
    const [mounted, setMounted] = useState(false);

    // Load saved order on mount
    useEffect(() => {
        setMounted(true);
        const savedOrder = localStorage.getItem('dashboard-order');
        if (savedOrder) {
            try {
                const parsedOrder = JSON.parse(savedOrder);
                // Validate parsedOrder contains valid IDs
                if (Array.isArray(parsedOrder) && parsedOrder.length === DEFAULT_ITEMS.length) {
                    setItems(parsedOrder);
                }
            } catch (e) {
                console.error("Failed to parse saved dashboard order", e);
            }
        }
    }, []);

    // Save order when changed
    useEffect(() => {
        if (mounted) {
            localStorage.setItem('dashboard-order', JSON.stringify(items));
        }
    }, [items, mounted]);

    const sensors = useSensors(
        useSensor(PointerSensor),
        useSensor(KeyboardSensor, {
            coordinateGetter: sortableKeyboardCoordinates,
        })
    );

    const handleDragEnd = (event: DragEndEvent) => {
        const { active, over } = event;

        if (over && active.id !== over.id) {
            setItems((items) => {
                const oldIndex = items.findIndex((item) => item.id === active.id);
                const newIndex = items.findIndex((item) => item.id === over.id);

                return arrayMove(items, oldIndex, newIndex);
            });
        }
    };

    if (!m) {
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
    const cashBalance = am?.['Cash']?.['total_market_value_display'] || 0;
    const dayGL = m.day_change_display || 0;
    const dayGLPct = m.day_change_percent || 0;
    const unrealizedGL = m.unrealized_gain || 0;
    const unrealizedGLPct = (m.unrealized_gain / (m.cost_basis_held || 1)) * 100;

    const dayGLColor = dayGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const unrealizedGLColor = unrealizedGL >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    const totalGain = m.total_gain || 0;
    const realizedGain = m.realized_gain || 0;

    const totalReturnColor = totalGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';
    const realizedGainColor = realizedGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400';

    // Render helper
    const renderContent = (id: string) => {
        switch (id) {
            case 'portfolioValue':
                return <MetricCard
                    title="Total Portfolio Value"
                    value={m.market_value}
                    valueClassName="text-4xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                />;
            case 'dayGL':
                return <MetricCard
                    title="Day's Gain/Loss"
                    value={dayGL}
                    subValue={dayGLPct}
                    colorClass={dayGLColor}
                    valueClassName="text-4xl"
                    subValueClassName="text-2xl"
                    containerClassName="h-full flex flex-col justify-center"
                    isHero={true}
                    currency={currency}
                />;
            case 'totalReturn':
                return <MetricCard
                    title="Total Return"
                    value={totalGain}
                    colorClass={totalReturnColor}
                    currency={currency}
                />;
            case 'annualTWR':
                return <MetricCard
                    title="Annual TWR"
                    value={m.annualized_twr !== undefined && m.annualized_twr !== null ? `${m.annualized_twr.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass={m.annualized_twr && m.annualized_twr >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}
                />;
            case 'unrealizedGL':
                return <MetricCard
                    title="Unrealized G/L"
                    value={unrealizedGL}
                    colorClass={unrealizedGLColor}
                    currency={currency}
                />;
            case 'unrealizedGLPct':
                return <MetricCard
                    title="Unrealized G/L %"
                    value={unrealizedGLPct !== undefined && unrealizedGLPct !== null ? `${unrealizedGLPct.toFixed(2)}%` : '-'}
                    isCurrency={false}
                    colorClass={unrealizedGLColor}
                />;
            case 'realizedGain':
                return <MetricCard
                    title="Realized Gain"
                    value={realizedGain}
                    colorClass={realizedGainColor}
                    currency={currency}
                />;
            case 'cashBalance':
                return <MetricCard
                    title="Cash Balance"
                    value={cashBalance}
                    currency={currency}
                />;
            case 'ytdDividends':
                return <MetricCard
                    title="YTD Dividends"
                    value={m.dividends}
                    currency={currency}
                />;
            case 'fees':
                return <MetricCard
                    title="Fees"
                    value={m.commissions}
                    colorClass="text-rose-600 dark:text-rose-400"
                    currency={currency}
                />;
            default:
                return null;
        }
    };

    return (
        <div className="mb-10">
            <div className="flex justify-end mb-2">
                <button
                    onClick={() => setIsCustomizing(!isCustomizing)}
                    className={`text-xs font-medium px-3 py-1 rounded transition-colors ${isCustomizing
                            ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                            : 'text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800'
                        }`}
                >
                    {isCustomizing ? 'Done Customizing' : 'Customize Dashboard'}
                </button>
            </div>

            <DndContext
                sensors={sensors}
                collisionDetection={closestCenter}
                onDragEnd={handleDragEnd}
            >
                <SortableContext
                    items={items.map(i => i.id)}
                    strategy={rectSortingStrategy}
                >
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {items.map((item) => (
                            <SortableMetricItem
                                key={item.id}
                                id={item.id}
                                colSpan={item.colSpan}
                                isCustomizing={isCustomizing}
                            >
                                {renderContent(item.id)}
                            </SortableMetricItem>
                        ))}
                    </div>
                </SortableContext>
            </DndContext>
        </div>
    );
}
