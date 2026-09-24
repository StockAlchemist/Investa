'use client';

/* eslint-disable @next/next/no-img-element -- small static brand logos; next/image optimization adds no value and would require extra config */

import { cn } from '@/lib/utils';
import { ChevronLeft, LogOut, Search } from 'lucide-react';
import { PRIMARY_NAV, SECONDARY_NAV, SETTINGS_NAV } from '@/lib/navigation';

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  user: { username: string; alias?: string } | null;
  onLogout: () => void;
  collapsed: boolean;
  onToggle: () => void;
  onUserClick: () => void;
  dayChangePct?: number;
  /** Opens the ⌘K command palette — the sidebar's search field is its trigger. */
  onSearchClick?: () => void;
}

function NavItem({ id, label, icon: Icon, activeTab, onTabChange, collapsed }: {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  activeTab: string;
  onTabChange: (tab: string) => void;
  collapsed: boolean;
}) {
  const active = activeTab === id;
  return (
    <button
      onClick={() => onTabChange(id)}
      title={collapsed ? label : undefined}
      aria-label={label}
      aria-current={active ? 'page' : undefined}
      className={cn(
        // Ledger: the active destination is raised onto a white card with a
        // hairline rule, in ink — not tinted in the accent. Cobalt is kept for
        // things you act on; where you *are* is not an action.
        'group/item relative flex items-center w-full rounded-control text-sm transition-colors duration-150',
        collapsed ? 'h-9 justify-center' : 'h-9 px-2.5 gap-2.5',
        active
          ? 'bg-card text-foreground font-semibold shadow-[0_0_0_1px_hsl(var(--border)),0_1px_2px_rgb(22_23_27/0.06)] dark:bg-muted dark:shadow-none'
          : 'text-ink-2 font-medium hover:bg-muted hover:text-foreground',
      )}
    >
      <div className="w-5 flex items-center justify-center shrink-0">
        <Icon className="w-[18px] h-[18px]" />
      </div>
      {!collapsed && <span className="truncate">{label}</span>}
      {collapsed && (
        <span className="pointer-events-none absolute left-full ml-2 z-50 px-2 py-1 rounded-md bg-popover border border-border text-xs font-medium shadow-lg whitespace-nowrap opacity-0 group-hover/item:opacity-100 transition-opacity duration-150 delay-200">
          {label}
        </span>
      )}
    </button>
  );
}

export function Sidebar({
  activeTab, onTabChange, user, onLogout, collapsed, onToggle, onUserClick, dayChangePct, onSearchClick,
}: SidebarProps) {
  return (
    <aside
      className={cn(
        'relative hidden md:flex flex-col h-screen border-r border-border bg-background shrink-0 transition-[width] duration-300 ease-in-out overflow-visible',
        collapsed ? 'w-14' : 'w-[232px]',
      )}
    >
      {/* Logo */}
      <div
        className={cn(
          'flex items-center h-16 shrink-0',
          collapsed ? 'flex-col justify-center gap-0.5' : 'px-4 gap-2.5',
        )}
      >
        <img src="/logo-sm.webp"      alt="Investa" width={28} height={28} className="w-7 h-7 rounded-lg shrink-0 dark:hidden" />
        <img src="/logo-dark-sm.webp" alt="Investa" width={28} height={28} className="w-7 h-7 rounded-lg shrink-0 hidden dark:block" />
        {!collapsed && (
          <span className="font-display text-[26px] leading-7 text-foreground">Investa</span>
        )}
        {collapsed && dayChangePct !== undefined && (
          <span className={cn(
            'text-[9px] font-bold tabular-nums leading-none',
            dayChangePct >= 0 ? 'text-up' : 'text-down',
          )}>
            {dayChangePct >= 0 ? '+' : ''}{dayChangePct.toFixed(1)}%
          </span>
        )}
      </div>

      {/* Search — the command palette's trigger, where the eye looks first. */}
      {onSearchClick && (
        <div className={cn('shrink-0 pb-2', collapsed ? 'px-2' : 'px-3')}>
          <button
            type="button"
            onClick={onSearchClick}
            aria-label="Search or jump to"
            title={collapsed ? 'Search (⌘K)' : undefined}
            className={cn(
              'flex items-center w-full h-9 rounded-control border border-border bg-card text-muted-foreground text-[13px] hover:border-input hover:text-foreground transition-colors',
              collapsed ? 'justify-center' : 'px-2.5 gap-2',
            )}
          >
            <Search className="w-[15px] h-[15px] shrink-0" />
            {!collapsed && (
              <>
                <span className="flex-1 text-left">Search or jump to</span>
                <kbd className="font-sans text-[11px] font-medium bg-muted rounded px-1.5 py-0.5">⌘K</kbd>
              </>
            )}
          </button>
        </div>
      )}

      {/* Navigation — two named groups, the same two the macOS sidebar shows. */}
      <nav aria-label="Main" className="flex-1 overflow-y-auto overflow-x-hidden px-2 py-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {!collapsed && <div className="section-label px-2.5 pt-1 pb-1.5">Portfolio</div>}
        <div className="space-y-0.5">
          {PRIMARY_NAV.map(item => <NavItem key={item.id} {...item} activeTab={activeTab} onTabChange={onTabChange} collapsed={collapsed} />)}
        </div>
        {collapsed
          ? <div className="my-3 mx-2 border-t border-border" />
          : <div className="section-label px-2.5 pt-5 pb-1.5">Research</div>}
        <div className="space-y-0.5">
          {SECONDARY_NAV.map(item => <NavItem key={item.id} {...item} activeTab={activeTab} onTabChange={onTabChange} collapsed={collapsed} />)}
        </div>
      </nav>

      {/* Bottom utilities */}
      <div className="px-2 py-2 space-y-0.5 shrink-0">
        <NavItem id={SETTINGS_NAV.id} label={SETTINGS_NAV.label} icon={SETTINGS_NAV.icon} activeTab={activeTab} onTabChange={onTabChange} collapsed={collapsed} />

        {/* User row */}
        <div
          onClick={onUserClick}
          className={cn(
            'flex items-center h-9 rounded-control hover:bg-muted transition-colors duration-150 cursor-pointer',
            collapsed ? 'justify-center' : 'px-3 gap-2.5',
          )}
        >
          <div className="w-6 h-6 rounded-full bg-foreground text-background text-[11px] font-semibold flex items-center justify-center shrink-0 select-none">
            {(user?.alias || user?.username)?.[0]?.toUpperCase() ?? 'U'}
          </div>
          {!collapsed && (
            <>
              <span className="text-sm font-medium text-foreground truncate flex-1 min-w-0">
                {user?.alias || user?.username}
              </span>
              <button
                onClick={e => { e.stopPropagation(); onLogout(); }}
                title="Sign out"
                aria-label="Sign out"
                className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
              >
                <LogOut className="w-3.5 h-3.5" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Collapse / expand toggle */}
      <button
        onClick={onToggle}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        className="absolute -right-3 top-[76px] z-20 w-6 h-6 rounded-full bg-card border border-border shadow-sm flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted transition-all"
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        <ChevronLeft
          className={cn('w-3 h-3 transition-transform duration-300', collapsed && 'rotate-180')}
        />
      </button>
    </aside>
  );
}
