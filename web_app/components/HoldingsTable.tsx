"use client";

import React from 'react';
import { useStockModal } from '@/context/StockModalContext';
import { HoldingsTableProps } from './holdings/types';
import { useHoldingsState } from './holdings/hooks/useHoldingsState';
import { useHoldingsData } from './holdings/hooks/useHoldingsData';
import { HoldingsToolbar } from './holdings/HoldingsToolbar';
import { HoldingsDesktopTable } from './holdings/HoldingsDesktopTable';
import { HoldingsMobileCards } from './holdings/HoldingsMobileCards';
import { HoldingsTagModal } from './holdings/HoldingsTagModal';

export type { HoldingsTableProps };

export default function HoldingsTable({ holdings, currency, isLoading = false }: HoldingsTableProps) {
    const { openStockDetail } = useStockModal();

    const {
        visibleColumns,
        setVisibleColumns,
        expandedLots,
        setExpandedLots,
        expandedCards,
        setExpandedCards,
        sortConfig,
        handleSort,
        toggleColumn,
        toggleLotExpansion,
        toggleCardExpansion,
        isColumnMenuOpen,
        setIsColumnMenuOpen,
        columnMenuRef,
        isAccountMenuOpen,
        setIsAccountMenuOpen,
        accountMenuRef,
        isGroupByMenuOpen,
        setIsGroupByMenuOpen,
        groupByMenuRef,
        draggedColumn,
        handleDragStart,
        handleDragOver,
        handleDrop,
        mobileViewMode,
        setMobileViewMode,
    } = useHoldingsState();

    const {
        searchQuery,
        setSearchQuery,
        selectedAccounts,
        setSelectedAccounts,
        toggleAccount,
        uniqueAccounts,
        groupBy,
        handleSetGroupBy,
        expandedGroups,
        toggleGroup,
        getValue,
        getLotValue,
        getExpansionKey,
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
    } = useHoldingsData({
        holdings,
        currency,
        visibleColumns,
        sortConfig,
    });

    const visibleHoldings = sortedHoldings.slice(0, visibleRows);

    const toggleAllLots = () => {
        if (expandedLots.size > 0) {
            setExpandedLots(new Set());
        } else {
            setExpandedLots(new Set(aggregatedHoldings.filter(h => h.lots?.length).map(getExpansionKey)));
        }
    };

    const toggleAllCards = () => {
        if (expandedCards.size > 0) {
            setExpandedCards(new Set());
        } else {
            setExpandedCards(new Set(aggregatedHoldings.map(getExpansionKey)));
        }
    };

    return (
        <>
            <div className="metric-card card-shine scrollbar-thin scrollbar-thumb-zinc-700/50 scrollbar-track-transparent transition-all duration-300 relative overflow-hidden">

                <HoldingsToolbar
                    totalItemsCount={holdings.length}
                    aggregatedCount={aggregatedHoldings.length}
                    groupedCount={groupedHoldings?.length || 0}
                    groupBy={groupBy}
                    searchQuery={searchQuery}
                    setSearchQuery={setSearchQuery}
                    isGroupByMenuOpen={isGroupByMenuOpen}
                    setIsGroupByMenuOpen={setIsGroupByMenuOpen}
                    groupByMenuRef={groupByMenuRef}
                    handleSetGroupBy={handleSetGroupBy}
                    isAccountMenuOpen={isAccountMenuOpen}
                    setIsAccountMenuOpen={setIsAccountMenuOpen}
                    accountMenuRef={accountMenuRef}
                    selectedAccounts={selectedAccounts}
                    setSelectedAccounts={setSelectedAccounts}
                    uniqueAccounts={uniqueAccounts}
                    toggleAccount={toggleAccount}
                    isColumnMenuOpen={isColumnMenuOpen}
                    setIsColumnMenuOpen={setIsColumnMenuOpen}
                    columnMenuRef={columnMenuRef}
                    visibleColumns={visibleColumns}
                    setVisibleColumns={setVisibleColumns}
                    toggleColumn={toggleColumn}
                    mobileViewMode={mobileViewMode}
                    setMobileViewMode={setMobileViewMode}
                    expandedCards={expandedCards}
                    toggleAllCards={toggleAllCards}
                    expandedLots={expandedLots}
                    toggleAllLots={toggleAllLots}
                    holdings={holdings}
                />

                <HoldingsDesktopTable
                    mobileViewMode={mobileViewMode}
                    visibleColumns={visibleColumns}
                    isLoading={isLoading}
                    onMoveColumn={(header, delta) => setVisibleColumns(previous => {
                        const next = [...previous]; const index = next.indexOf(header); const target = index + delta;
                        if (index < 0 || target < 0 || target >= next.length) return previous;
                        [next[index], next[target]] = [next[target], next[index]];
                        return next;
                    })}
                    draggedColumn={draggedColumn}
                    handleDragStart={handleDragStart}
                    handleDragOver={handleDragOver}
                    handleDrop={handleDrop}
                    handleSort={handleSort}
                    sortConfig={sortConfig}
                    groupBy={groupBy}
                    groupedHoldings={groupedHoldings}
                    visibleHoldings={visibleHoldings}
                    expandedGroups={expandedGroups}
                    toggleGroup={toggleGroup}
                    expandedLots={expandedLots}
                    toggleLotExpansion={toggleLotExpansion}
                    getExpansionKey={getExpansionKey}
                    getValue={getValue}
                    getLotValue={getLotValue}
                    currency={currency}
                    openStockDetail={openStockDetail}
                    editingTags={editingTags}
                    setEditingTags={setEditingTags}
                    tagsInput={tagsInput}
                    setTagsInput={setTagsInput}
                    handleEditTags={handleEditTags}
                    handleSaveTags={handleSaveTags}
                />

                <HoldingsMobileCards
                    mobileViewMode={mobileViewMode}
                    visibleHoldings={visibleHoldings}
                    currency={currency}
                    openStockDetail={openStockDetail}
                    expandedCards={expandedCards}
                    toggleCardExpansion={toggleCardExpansion}
                    expandedLots={expandedLots}
                    toggleLotExpansion={toggleLotExpansion}
                    getExpansionKey={getExpansionKey}
                    getValue={getValue}
                    getLotValue={getLotValue}
                />

                {!groupBy && visibleRows < sortedHoldings.length && (
                    <div className="flex justify-center gap-4 p-4">
                        <button
                            onClick={handleShowMore}
                            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary-hover transition-colors text-sm font-medium"
                        >
                            Show More
                        </button>
                        <button
                            onClick={handleShowAll}
                            className="px-4 py-2 bg-card text-foreground rounded-md hover:bg-secondary transition-colors text-sm font-medium"
                        >
                            Show All
                        </button>
                    </div>
                )}
            </div>

            <HoldingsTagModal
                editingTags={editingTags}
                setEditingTags={setEditingTags}
                tagsInput={tagsInput}
                setTagsInput={setTagsInput}
                handleSaveTags={handleSaveTags}
                isPending={updateTagsMutation.isPending}
            />
        </>
    );
}
