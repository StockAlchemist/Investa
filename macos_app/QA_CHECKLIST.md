# Investa native app — on-device QA checklist

Manual checks for the paths best verified by typing/interacting on a real device
or a hands-on simulator session (⌘R from Xcode). These cover the flows that can't
be fully exercised by automated/headless verification (keyboard input, save
round-trips), plus quick smoke checks of recent work.

**Prereqs:** backend running, signed in, an account with holdings (so the
Portfolio/tags screens have data).

## 1. AI chat (send + persistence)

- [ ] Tap the floating ✨ button (bottom-right) → panel opens with the welcome message.
- [ ] Type a question (e.g. "What are my 3 largest holdings?") → **Send** → a coherent reply appears; "Thinking…" shows while waiting.
- [ ] Send a follow-up that relies on context (e.g. "Why those?") → reply uses prior turns (header shows **MEMORY ACTIVE**).
- [ ] Quit and relaunch the app → reopen chat → **history is still there** (persisted).
- [ ] Tap the **trash** icon → conversation clears.
- [ ] macOS + iPhone: panel fits, scrolls, input usable.

## 2. Holding tag editing (save round-trip)

- [ ] **macOS / iPad** (Portfolio → holdings table, scroll right to **Tags**): tap a Tags cell → editor opens for the right symbol → type `Core, Tech` → **Save** → cell shows `CORE`, `TECH` after refresh.
- [ ] **iPhone** (Portfolio → holding card → **TAG — ✎**): same flow; tag chips appear on the card.
- [ ] Edit again → clear the field → **Save** → tags return to `—`.
- [ ] If the symbol is held in **multiple accounts**, confirm the tags apply across all of them.

## 3. Transaction form (autocomplete / validation / submit)

- [ ] Transactions → **+** → type a partial **symbol** → autocomplete dropdown → pick one.
- [ ] Change **Type** (Dividend, Split, Transfer, …) → fields enable/disable correctly; no fields wiped when **editing** an existing transaction.
- [ ] Enter qty/price → **Total** auto-updates; override it manually → it sticks.
- [ ] Submit invalid input (empty symbol, negative price) → inline validation message.
- [ ] Save a real transaction → appears with the correct signed total; delete it.

## 4. Symbol search & PDF import (quick smoke)

- [ ] Search box: type a ticker → results dropdown → tap → Stock Detail opens.
- [ ] Transactions → Import → **Choose PDF / Image…** → file panel opens (no crash) → pick a statement → edit a review row → import → rows land with correct signs.

## Notes

- AI chat **send** and tag-edit **save** are the two most worth a human pass — the
  rest were verified via UI render + API contract.
- Date pickers should show **Gregorian** years (not the Buddhist era) even under a
  Thai locale.
