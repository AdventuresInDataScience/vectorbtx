# Section G domain audit — holding costs vs broker practice

**Why this exists.** The plan's Section G was written from a summarised domain review that
collapsed two different products into one rule. The error was structural rather than
cosmetic: it stated a margin share-dealing convention and applied it to spread bets and
CFDs, which do not work that way. Because cost errors are invisible in an equity curve —
the backtest still runs, still looks plausible — the whole of Section G is re-checked here
against primary broker documentation before any more is built on it.

**Method.** Every finding is checked against a broker's own published costs schedule,
preferring IG (the reference UK retail broker for all three products) and corroborated
where possible. Findings are graded:

| Grade | Meaning |
|---|---|
| **STRUCTURAL** | The model is wrong. Cannot be fixed by changing a default. |
| **DEFAULT** | The mechanism is right, the shipped default is wrong. |
| **GAP** | A real cost that is not modelled at all. |
| **CONFIRMED** | The plan is right. Recorded so it is not re-litigated. |

**Status.** G1 is fixed. G2–G5 are not yet implemented — that is the next work item.

---

## Summary

| # | Finding | Grade | Impact if unfixed |
|---|---|---|---|
| G1 | Financing basis is per-mode, not universally "borrowed portion" | **STRUCTURAL** | *Fixed.* Understated derivative financing by `1 − 1/leverage` |
| G2 | FX spread bets and CFDs are charged carry **and** financing | **STRUCTURAL** | Double-charges every FX derivative position |
| G3 | Triple day is Friday for shares/indices, Wednesday only for FX | **DEFAULT** | Mis-times the weekend roll for non-FX |
| G4 | Day count is 365 for GBP markets, 360 elsewhere | **DEFAULT** | ~1.4% overstatement on every GBP position |
| G5 | FX admin fee defaults to zero | **DEFAULT** | Omits the broker's entire FX funding margin |
| G6 | Short share borrow charge not modelled | **GAP** | Understates cost of short share positions |
| G7 | Futures and forwards carry no overnight funding | **GAP** | Only matters if dated products are added |
| G8 | Holidays are zeroed; brokers still fund them | **DEFAULT** | Small, systematic understatement |
| G9 | Markup is adverse in both directions | **CONFIRMED** | — |
| G10 | Seven funding days per week | **CONFIRMED** | — |
| G11 | 22:00 UK cut-off, inert on daily bars | **CONFIRMED** | — |

---

## G1 — Financing basis is per-mode — STRUCTURAL — *fixed*

**The plan said.** Financing applies only to the borrowed portion; a cash-covered
position finances nothing. Stated three times (§1.5, §2.2, §G) and applied explicitly to
a 5× leveraged CFD in §F's worked example. The only exception it allows is carry.

**Broker practice.** For CFDs and spread bets the charge is on the **full notional**.
<cite index="6-1">The broker is effectively lending the money to control a position larger than the margin deposit, and the interest is calculated on the position's full notional value rather than on the margin.</cite> IG's own formula is contracts × value per contract × price × rate ÷ divisor — position value throughout, with no reference to account cash.

**Why the plan's rule isn't merely a simplification.** It is the correct rule for a
different product. Margin share dealing lends you the shortfall between the purchase and
your cash, so interest accrues on the debit balance. A spread bet or CFD involves no
purchase and no loan of that kind: the broker funds the position in full and holds margin
as collateral. The two coincide only at infinite leverage.

**Consequence.** Financing was understated by `1 − 1/leverage` — four fifths of the true
cost at 5× — and fell to **zero** for an unlevered derivative position that a broker
charges in full.

**Resolution.** `FINANCING_BASIS` in `instruments/defaults.py`: `equity → "borrowed"`,
`spreadbet`/`cfd`/`fx_spot` → `"notional"`. Consumed by `build_charges`.

---

## G2 — FX derivatives are charged twice — STRUCTURAL — *not yet fixed*

**The defect.** The default cost stacks are:

```
cfd_fx:        commission + spread + slippage + carry + financing
spreadbet_fx:  spread + slippage + carry + financing
```

Both carry **and** financing. The plan treats them as separate costs that legitimately
coexist, on the reasoning that carry is about the currency pair while financing is about
leverage.

**Broker practice.** For FX there is only one overnight charge, and it *is* the tom-next
swap. <cite index="17-1">For forex and spot metals trades the overnight funding is the tom-next rate plus a small admin fee.</cite> The tom-next rate already embeds both currencies' interest rates — that is what makes it a *differential*. There is no second, leverage-driven charge layered on top.

**Why this got worse with G1.** Previously `financing` on an FX derivative was computed on
the borrowed portion, which for a long could never exceed cash, so it was ~zero and the
double-charge was invisible. Now that financing correctly uses full notional, the FX
stacks charge tom-next **plus** a full-notional interest charge. G1 turned a latent bug
into an active one.

**Fix.** Remove `financing` from `("cfd", "fx")` and `("spreadbet", "fx")`. FX-underlying
derivatives get carry only. Add a validation rule rejecting `carry` and `financing`
together in one `active_costs` set, so the combination cannot be reintroduced by hand.

**Severity note.** This is the single largest remaining number error in the cost model.
At a 5% benchmark it roughly doubles the holding cost of every FX spread bet and FX CFD.

---

## G3 — Triple day is Friday for shares and indices — DEFAULT — *not yet fixed*

**The plan said.** Triple day defaults to Wednesday for everything in the UK, configurable
per instrument.

**Broker practice.** Two different conventions, for two different reasons:

- **FX and spot metals — Wednesday.** Spot FX settles T+2, so a position rolled over
  Wednesday night settles the following Monday and is charged three days' carry. IG
  applies three days' funding to forex positions held through Wednesday's cut-off.
- **Shares, indices, ETFs — Friday.** These are not settlement-driven; the charge simply
  covers the calendar weekend. <cite index="13-1">Positions held through the Friday cut-off are adjusted for three days' worth of funding to cover the weekend.</cite> Corroborated independently: <cite index="11-1">for indices and shares, some brokers charge all three weekend days on Friday night.</cite>

**Consequence.** Total weekly funding is unaffected (both conventions bill seven days —
see G10), so a long backtest's aggregate cost is roughly right. What is wrong is
*timing*: a share position opened Thursday and closed Friday is charged 1× under the
Wednesday convention and 3× under the correct Friday one. For short-holding-period
strategies — the ones most sensitive to cost — this is a material per-trade error.

**Fix.** Make `triple_day` default per **mode and asset class** rather than globally:
`fx` → `WED`; everything else → `FRI`. The mechanism already exists and is per-instrument;
only the default is wrong.

---

## G4 — Day count is 365 for GBP markets — DEFAULT — *not yet fixed*

**The plan said.** ACT/360 for all currencies, justified as simpler and as erring toward
overstating cost.

**Broker practice.** The divisor follows the **denomination of the market**, not a house
convention. IG uses a 365-day divisor for the FTSE 100 and other GBP-denominated markets
(also SGD and ZAR), and 360 for everything else.

**Consequence.** Every GBP-denominated position is overstated by 365/360 − 1 ≈ **1.4%** of
its financing cost. Small per night, systematic over a multi-year backtest, and precisely
the kind of quiet bias that the plan's own §M note about explicit day-count conversion was
written to prevent. The plan's "overstating is the safe direction" reasoning also inverts
for received credits, which it acknowledges elsewhere.

**Fix.** Replace the module-level `DAY_COUNT = 360.0` with a currency lookup —
`{GBP, SGD, ZAR} → 365`, default 360 — keyed on the instrument's `quote_currency`.
`annual_to_period` already takes `day_count` as an argument, so only the call sites and a
per-currency default table change.

---

## G5 — FX admin fee defaults to zero — DEFAULT — *not yet fixed*

**The plan said.** Carry uses the tom-next rate. `carry_markup` defaults to `0.0` in
`build_charges`.

**Broker practice.** Brokers apply an administration fee to the tom-next rate — IG quotes
1% on forex and spot metals in its Swiss schedule and 0.5% on the conversion rate
elsewhere. The plan itself sets a 3% markup for interbank-rate financing but leaves the FX
equivalent at zero, which is an inconsistency rather than a considered choice.

**Consequence.** Every FX carry number is missing the broker's entire margin on it. Unlike
G2 this understates cost, so it flatters carry strategies specifically.

**Fix.** Default `carry_markup` to 0.01 in `settings["vectorbtx"]`, adverse in both
directions exactly as the financing markup already is. This is a one-line default change;
the adverse-application machinery in `tom_next_differential` is already correct.

---

## G6 — Short share borrow charge — GAP — *document, do not build*

Shorting a share CFD incurs a **borrow charge** separate from funding, reflecting the cost
of locating the stock, and it varies by stock and can change at short notice. IG bills it
as a daily cash adjustment including an administration fee.

**Assessment.** Correctly out of scope for v1: the rate is instrument-specific,
time-varying, and not available from any free data source, so modelling it would mean
inventing numbers. **But it must be documented as a known understatement of short-side
cost**, alongside the plan's existing limitations list. Short-share-CFD strategies will
look better here than in reality, and that is exactly the direction users should be warned
about.

---

## G7 — Futures and forwards have no overnight funding — GAP — *scope note*

Dated products do not attract overnight funding; the funding cost is built into their
wider spread instead. vectorbtX has no dated-contract mode, so nothing is wrong today.
Recording it because "add a futures mode" is an obvious future request, and whoever does it
must know that the mode's default cost stack contains **no financing and no carry** — the
opposite of every existing mode.

---

## G8 — Holidays are zeroed — DEFAULT — *low severity*

`day_multipliers` sets a holiday's multiplier to zero. Brokers fund any position open at
the cut-off, and a market holiday does not stop interest accruing on the money. The
plan's own §2.2 says weekends and holidays carry no *separate* charge — which is true, but
was implemented as no charge at all rather than as "the weekend is covered by the triple
day."

**Consequence.** Roughly 8 days a year unfunded per instrument, ~2% understatement of
annual financing. Small, but one-directional.

**Fix.** Treat holidays as ordinary funded days by default, and reserve the zero
multiplier for genuine non-trading bars. Keep the `holidays` argument for users who want
the current behaviour.

---

## G9 — The adverse markup is right — CONFIRMED

The plan requires `benchmark + markup` financing a long and `benchmark − markup` on a
short. IG expresses the same thing from the other side — the admin fee with the benchmark
added for longs and deducted for shorts — which is algebraically identical: paying
`markup − benchmark` is receiving `benchmark − markup`. <cite index="16-1">For a short position the calculation is the same except the benchmark rate is deducted from the admin fee.</cite>

This also confirms the plan's warning that a naive addition credits shorts too
generously, **and** the corollary already implemented in `financing_charge`: when the
markup exceeds the benchmark, a short is *charged* rather than credited. In a low-rate
period shorts pay on both sides. The implementation handles this and has a test for it.

---

## G10 — Seven funding days per week — CONFIRMED

Both conventions bill seven days across a week: FX as 1+1+3+1+1 (Wednesday triple), shares
as 1+1+1+1+3 (Friday triple). The plan's warning against additionally charging calendar
days — which would bill nine days a week — is correct and correctly implemented.
`trading_day_count` has a test asserting exactly seven.

---

## G11 — The 22:00 UK cut-off is right, and inert on daily bars — CONFIRMED

IG's daily cut-off is 10pm UK time, varying for some international markets (US shares roll
at 8pm New York, Monday to Thursday). The plan's 22:00 default is correct, as is its
insistence that the setting is **inert on daily bars** — one bar per day means the cut-off
cannot change which bar a charge lands on. Both are implemented and documented.

---

## What this changes about the plan

Section G's arithmetic modules are sound: `financing_charge`, `carry_charge`,
`annual_to_period` and `day_multipliers` all compute what they claim to, and their tests
are real. **The defects are in the inputs those functions are given** — which basis, which
day, which divisor, which stack. That is the reassuring version of this audit: the
machinery is fine and the wiring is wrong, so G2–G5 are configuration and lookup-table
changes rather than a rewrite.

The unreassuring version: none of these would have been caught by any test in the suite,
because every test encodes the plan's assumptions. The D7 worked examples were meant to be
the defence against exactly this, and they were derived from the same faulty summary.
**Any new fixture for G2–G5 must be worked from broker documentation, not from the plan.**

## Recommended order

1. **G2** — the only one that is currently double-counting. Structural, and made active by
   the G1 fix.
2. **G4, G3** — one lookup table each; G4 touches every GBP backtest.
3. **G5, G8** — default changes.
4. **G6, G7** — documentation only, into the known-limitations list.

Then re-derive the Section G worked examples from IG's published formulae and commit them
as fixtures, replacing the plan-derived ones.

---

# Appendix — spot check of Sections E and I

The same summarisation damaged Section G, so E (per-trade costs) and I (multi-currency)
were checked for the same class of error: a real-world convention flattened into a single
rule. Both are in better shape than G. One genuine gap, one confirmation each.

## E1 — UK stamp duty is not modelled — GAP — *material*

**Finding.** The plan does not mention stamp duty anywhere. UK share **purchases** attract
0.5% Stamp Duty Reserve Tax; spread bets and CFDs do not.

**Why this matters more than an ordinary missing cost.** It is not a rounding item — it is
0.5% of notional on every equity buy, comparable to several months of financing. And it is
one of the *primary economic reasons* UK retail traders use spread bets at all. Omitting it
makes equity mode look systematically cheaper than reality and, worse, biases the
**cross-mode comparison** that Section H exists to make correct: a user comparing "same
trade as a spread bet vs as shares" gets an answer tilted toward shares by 50bp per entry.

**Assessment.** Cheap to model and it fits the existing machinery exactly — a
buy-side-only percentage on `fees`, per instrument. The obstacle is that vectorbt's `fees`
field is symmetric (applied to buys and sells alike), so a buy-only charge needs either a
side-aware series or half the rate applied to both sides as an approximation. The former
is exact and is a level-1 change.

**Related, smaller:** the PTM levy (£1 on UK trades above £10,000) is also absent. Trivial
in size; worth a `commission_flat` note rather than its own field.

## E2 — Spread is charged half per side — CONFIRMED

`spec.spread_points` is the **full** bid/ask gap and half is applied on each leg, so a
round trip pays the whole spread. Verified numerically: `spread_points=2.0` at price 100
with `point_size=1.0` resolves to `slippage=0.01`, i.e. one point per side. vectorbt
applies `slippage` adversely in both directions (buy at `price x (1 + s)`, sell at
`price x (1 - s)`), so the direction is right without further work.

## I1 — The FX markup is applied per leg, not per trade — CONFIRMED

`fx_markup` (default 0.5%, matching IG's published conversion admin fee) is carried on the
converted frame and folded into `slippage`, so it is charged adversely on entry **and**
exit — about 1% round trip. That is the correct reading: a spread on the exchange rate
behaves exactly like a spread on price. Verified: `fx_markup=0.005` resolves to
`slippage=0.005` per leg.

## I2 — Section I's four gotchas are correctly implemented — CONFIRMED

One rate per bar across all of OHLC (with `check_ohlc_ordering` enforcing the `H >= C >= L`
invariant afterwards), percentage costs left unconverted while fixed costs route through
`cost_currency`, reciprocals derived from one canonical direction rather than fetched, and
an explicit fill policy that warns rather than silently forward-filling. These are the
items most likely to produce phantom P&L and they were built as specified.

## Conclusion

The damage was concentrated in G. E and I describe conventions that are genuinely uniform
across products, so there was less for a summary to flatten. **E1 (stamp duty) is the one
item worth building**, and it should be treated as a Section H correctness issue — it
distorts cross-mode equivalence — rather than as a missing line item in Section E.
