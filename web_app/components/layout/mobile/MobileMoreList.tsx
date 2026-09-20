'use client';

/**
 * The More screen — where the phone keeps every section the tab bar has no room
 * for, as iOS's own "More" tab does in `MainView.phoneShell`.
 *
 * Drawn as grouped cards of touch-height rows rather than the system's grey
 * grouped list, for the same reason `SettingsHubIOS` is: same `.card-standard`
 * chrome, same section labels and same accent as every other tab, so More is
 * not the one screen drawn in a foreign style.
 */

import { ChevronRight, LogOut } from 'lucide-react';
import { cn } from '@/lib/utils';
import { PRIMARY_NAV, SECONDARY_NAV, SETTINGS_NAV, type NavItem } from '@/lib/navigation';

/** The sections the tab bar itself carries — excluded from this list. */
const IN_TAB_BAR = 4;

function Row({ item, active, onSelect, last }: {
  item: NavItem;
  active: boolean;
  onSelect: () => void;
  last: boolean;
}) {
  const Icon = item.icon;
  return (
    <>
      <button
        type="button"
        onClick={onSelect}
        aria-current={active ? 'page' : undefined}
        className="flex min-h-[44px] w-full items-center gap-3 px-3.5 text-left transition-colors active:bg-foreground/[0.06]"
      >
        <Icon className={cn('h-[22px] w-[22px] shrink-0', active ? 'text-primary' : 'text-muted-foreground')} />
        <span className={cn(
          'flex-1 truncate text-[15px]',
          active ? 'font-semibold text-primary' : 'text-foreground',
        )}>
          {item.label}
        </span>
        <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground/60" />
      </button>
      {/* Hairline inset past the glyph, as `SettingsRowDivider` is. */}
      {!last && <div className="ml-[48px] h-px bg-border/60" />}
    </>
  );
}

function Group({ title, items, activeTab, onTabChange }: {
  title: string;
  items: readonly NavItem[];
  activeTab: string;
  onTabChange: (tab: string) => void;
}) {
  if (items.length === 0) return null;
  return (
    <section className="space-y-2">
      <p className="section-label px-1">{title}</p>
      <div className="card-standard overflow-hidden">
        {items.map((item, i) => (
          <Row
            key={item.id}
            item={item}
            active={activeTab === item.id}
            onSelect={() => onTabChange(item.id)}
            last={i === items.length - 1}
          />
        ))}
      </div>
    </section>
  );
}

export function MobileMoreList({ activeTab, onTabChange, user, onLogout }: {
  activeTab: string;
  onTabChange: (tab: string) => void;
  user: { username: string; alias?: string } | null;
  onLogout: () => void;
}) {
  return (
    <div className="space-y-5 pb-4">
      <Group title="Portfolio" items={PRIMARY_NAV.slice(IN_TAB_BAR)} activeTab={activeTab} onTabChange={onTabChange} />
      <Group title="Research" items={SECONDARY_NAV} activeTab={activeTab} onTabChange={onTabChange} />
      <Group title="App" items={[SETTINGS_NAV]} activeTab={activeTab} onTabChange={onTabChange} />

      {/* Who is signed in, with sign-out beside it — the sidebar footer's own
          anatomy, which the phone otherwise has nowhere to put. */}
      <div className="card-standard flex min-h-[44px] items-center gap-3 px-3.5">
        <span className="flex h-7 w-7 shrink-0 select-none items-center justify-center rounded-full bg-primary/15 text-xs font-bold text-primary">
          {(user?.alias || user?.username)?.[0]?.toUpperCase() ?? 'U'}
        </span>
        <span className="min-w-0 flex-1 truncate text-[15px] font-medium text-foreground">
          {user?.alias || user?.username}
        </span>
        <button
          type="button"
          onClick={onLogout}
          aria-label="Sign out"
          className="flex h-8 w-8 items-center justify-center rounded-control text-muted-foreground transition-colors active:bg-down/10 active:text-down"
        >
          <LogOut className="h-[18px] w-[18px]" />
        </button>
      </div>
    </div>
  );
}
