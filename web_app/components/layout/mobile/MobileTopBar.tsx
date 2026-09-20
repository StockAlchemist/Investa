'use client';

/* eslint-disable @next/next/no-img-element -- small static brand logos; next/image optimization adds no value and would require extra config */

/**
 * The phone's navigation bar — the web twin of the `.toolbar` in
 * `MainView.phoneTabContent`: the app icon at the leading edge, the wordmark
 * beside it once there is room for it, and the index strip trailing.
 *
 * The native bar names the app, never the tab — the tab bar below already names
 * the tab, and `GlobalControlBar.showsSectionTitle` is false at compact width
 * for the same reason. The web header behaves the same way below `md`.
 */

import { IndexStrip } from './IndexStrip';

export function MobileTopBar({ indices, onIndexClick }: {
  indices?: Record<string, unknown>;
  onIndexClick?: () => void;
}) {
  return (
    <header className="ios-bar sticky top-0 z-40 shrink-0 pt-safe md:hidden">
      <div className="flex h-11 items-center gap-2 px-3">
        <img src="/logo-sm.webp"      alt="Investa" width={34} height={34} className="h-[34px] w-[34px] shrink-0 rounded-lg dark:hidden" />
        <img src="/logo-dark-sm.webp" alt="Investa" width={34} height={34} className="hidden h-[34px] w-[34px] shrink-0 rounded-lg dark:block" />
        {/* The native title appears once the shell is wider than 450pt. */}
        <span className="hidden min-[450px]:inline text-lg font-bold text-foreground">Investa</span>

        <div className="min-w-0 flex-1" />

        {indices && Object.keys(indices).length > 0 && (
          <IndexStrip
            indices={indices as Parameters<typeof IndexStrip>[0]['indices']}
            onClick={onIndexClick}
          />
        )}
      </div>
    </header>
  );
}
