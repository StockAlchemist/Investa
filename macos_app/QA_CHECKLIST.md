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

## 4. Valuation tab (live re-blend + refusals)

- [ ] Stock Detail → **Valuation** on a large cap (e.g. `AAPL`): blended value, margin of
      safety, and a **Confidence** bar all render; the weight chips under it sum to 100%.
- [ ] "How this blend was built" names the profile and lists any held-out model with its
      reason (a sub-60%-payout dividend payer should show **DDM held out** plus a
      dividend-only floor).
- [ ] A bank (`JPM`) blends **D-NI**; a REIT (`SPG`) blends **D-CFO** and the distribution.
      Neither should show a DCF weight. A REIT whose DDM was refused (payout over 150% of
      depreciation-charged net income, e.g. `O`) correctly blends D-CFO alone, at lower
      confidence — that is not a bug.
- [ ] Open a model card → edit a parameter (drag growth or discount rate) → **that model's
      value and the blended headline both move**, with the default shown alongside; the
      headline never exceeds 5x or falls below 0.1x spot. **Reset All to Defaults** restores.
- [ ] A Mean P/E card shows a **median traded** multiple with its year span and a "Usually
      Traded At" range — never today's P/E (that would make the fair value equal the price).
- [ ] A company with no dividend, no positive EPS and no free cash flow shows **Not valued**
      with a reason, not a number.
- [ ] **iPhone**: scroll the whole Valuation tab top to bottom on a stock with all twelve
      model cards expanded (**All Methods**) — no blank frames, no crash. This tab has the
      deepest view tree in the app.

## 5. Symbol search & PDF import (quick smoke)

- [ ] Search box: type a ticker → results dropdown → tap → Stock Detail opens.
- [ ] Transactions → Import → **Choose PDF / Image…** → file panel opens (no crash) → pick a statement → edit a review row → import → rows land with correct signs.

## Notes

- AI chat **send** and tag-edit **save** are the two most worth a human pass — the
  rest were verified via UI render + API contract.
- Date pickers should show **Gregorian** years (not the Buddhist era) even under a
  Thai locale.
