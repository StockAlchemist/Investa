'use client';

/**
 * The phone's tab bar — the web twin of `MainView.phoneShell`, whose `TabView`
 * shows the first four sections and folds the rest behind iOS's own "More" tab.
 *
 * The four are `AppSection.allCases.prefix(4)`: Dashboard, Portfolio,
 * Performance, Transactions — the same four, in the same order, as
 * `PRIMARY_NAV` here. Everything past them lives on More, exactly as it does on
 * the device.
 *
 * This replaced a three-item bar (Home / Markets / Settings) whose "Home" stood
 * for nine different tabs at once, so the bar could not say which one you were
 * on.
 */

import { Ellipsis } from 'lucide-react';
import { cn } from '@/lib/utils';
import { PRIMARY_NAV } from '@/lib/navigation';

/** The tabs iOS keeps in the bar itself; the rest go to More. */
export const PHONE_TABS = PRIMARY_NAV.slice(0, 4);
const PHONE_TAB_IDS = new Set(PHONE_TABS.map(t => t.id));

/** True when `tab` lives behind the More tab rather than in the bar. */
export function isMoreTab(tab: string): boolean {
  return !PHONE_TAB_IDS.has(tab);
}

export function MobileTabBar({ activeTab, onTabChange, onMoreClick, moreActive }: {
  activeTab: string;
  onTabChange: (tab: string) => void;
  onMoreClick: () => void;
  /** True while the More screen itself is showing, or a section it holds. */
  moreActive: boolean;
}) {
  const item = (
    key: string,
    label: string,
    Icon: React.ComponentType<{ className?: string }>,
    active: boolean,
    onClick: () => void,
  ) => (
    <button
      key={key}
      type="button"
      onClick={onClick}
      aria-label={label}
      aria-current={active ? 'page' : undefined}
      className={cn(
        'flex flex-1 flex-col items-center justify-center gap-0.5 pt-1.5 pb-1 transition-colors',
        active ? 'text-primary' : 'text-muted-foreground',
      )}
    >
      <Icon className="h-[22px] w-[22px]" />
      {/* One line, in full: the tab bar is data text, and a truncated label no
          longer names its destination. */}
      <span className="text-[10px] font-medium leading-tight whitespace-nowrap">{label}</span>
    </button>
  );

  return (
    <nav className="ios-bar fixed inset-x-0 bottom-0 z-50 border-t border-border/60 pb-safe md:hidden">
      <div className="flex items-stretch">
        {PHONE_TABS.map(({ id, label, icon }) =>
          item(id, label, icon, !moreActive && activeTab === id, () => onTabChange(id)))}
        {item('__more', 'More', Ellipsis, moreActive, onMoreClick)}
      </div>
    </nav>
  );
}
