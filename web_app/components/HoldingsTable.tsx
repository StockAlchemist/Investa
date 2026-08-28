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
        expandedCards,
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
            expandedLots.clear();
        } else {
            aggregatedHoldings.forEach(h => {
                if (h.lots && h.lots.length > 0) {
                    expandedLots.add(getExpansionKey(h));
                }
            });
        }
    };

    const toggleAllCards = () => {
        if (expandedCards.size > 0) {
            expandedCards.clear();
        } else {
            aggregatedHoldings.forEach(h => {
                expandedCards.add(getExpansionKey(h));
            });
        }
    };

    return (
        <>
            <div className="metric-card card-shine mt-6 scrollbar-thin scrollbar-thumb-zinc-700/50 scrollbar-track-transparent transition-all duration-300 relative overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-indigo-500 opacity-80" />

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
                            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-[#0086a0] transition-colors text-sm font-medium"
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
