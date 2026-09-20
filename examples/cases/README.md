# Case matrix A–L

Twelve end-to-end backtests covering the combinations that actually differ: instrument mode
× single/portfolio × shared/separate cash × shared/separate strategy. Each file is
standalone and runnable.

Every case passes all four price series by keyword — `close=`, `open=`, `high=`, `low=` —
which is required. Orders fill at the next bar's open, stops are checked intrabar, and
`pf.x.intrabar_drawdown()` measures the worst moment against the lows rather than the closes.

Each file is a real script — `vbt.YFData.download(...)`, `vbt.x.FREDData.download_rates(...)`,
`vbt.MA.run(...)` — so it can be copied straight out and run against live data:

```bash
export FRED_API_KEY=...                              # free from fred.stlouisfed.org
python examples/cases/C_spreadbet_fx_single.py
```

To run them offline with no network and no API key, use the harness. It substitutes **only**
Yahoo and FRED — deterministic synthetic prices and the static rate table — and uses the real
library for everything else:

```bash
python examples/cases/_harness.py                    # all twelve
python examples/cases/_harness.py C_spreadbet_fx_single.py
```

| | Case | What it isolates |
|---|---|---|
| A | CFD, single | Lots × contract size; financing on **full notional**; Friday roll |
| B | Spread bet on a **share** | £/pt with no pair underneath → financing, Friday roll, 5× tier |
| C | Spread bet on **FX** | Same product, FX underlying → **carry**, Wednesday roll, 30× tier |
| D | Equity, unleveraged | The baseline: whole shares, no loan, no financing |
| E | Equity, leveraged | Funded margin; financing on the **debit balance**, not the notional |
| F | Equities, shared cash, **separate** strategies | Per-column signals need no special API |
| G | Equities, shared cash, **one** strategy | The least code: signal on the whole frame |
| H | Equities, **separate** cash pools | `cash_sharing=False` → per-column `init_cash` and leverage |
| I | FX spread bets, shared cash | Each pair declares its own currencies, for carry |
| J | One-country portfolio, 6-monthly rebalance | `weights` + `schedule`; no signals |
| K | Multi-country, equal weight, yearly | J plus currencies — **both** close and open converted |
| L | Mixed FX + index + share spread bets, pooled | One rule, four economics, all from `asset_class` |

## What B, C and L show together

The same product — a spread bet — behaves differently depending on what sits underneath it,
and none of it is stated per column. Set `asset_class` and the rest follows:

| | leverage cap | overnight cost | weekend roll |
|---|---|---|---|
| `fx` | 30× | carry (tom-next) | Wednesday (spot settles T+2) |
| `index` | 10× | financing | Friday (calendar weekend) |
| `stocks` | 5× | financing | Friday |

A **mixed-asset spread bet portfolio** — FX pairs alongside indices and shares in one pooled
account — is L exactly. One `mode="spreadbet"`, one entry rule, and four sets of economics
that follow from `asset_class` and the quote currency without being stated:

```
             pip   lev  roll  overnight
EURUSD=X  0.0001    30   WED  ['carry']
USDJPY=X    0.01    30   WED  ['carry']
UKX            1    10   FRI  ['financing']
VOD.L       0.01     5   FRI  ['financing']
```

Note the JPY pair's pip differs from the other FX leg, and neither is stated. Pooled cash
across mixed leverage tiers funds the pot at the highest tier in the group and warns; each
column's own `max_size` still caps it individually.

## Two things these cases exposed

Writing them found two live bugs, both now fixed and covered by tests:

- **Rebalanced weights were a fraction of buying power, not equity.** A 60/40 book on
  equity mode's *default* 2× leverage was silently running at 120/80. J and K are the cases
  that caught it.
- **Stops could not fire intrabar.** `high` and `low` were optional, so a stop was compared
  only against the close and fired a bar late or never, while drawdown missed every intrabar
  excursion. Both are now required.
- **A spread bet sized below the minimum stake traded nothing.** Correct behaviour, and
  correctly warned — but B was originally written with a 300-point stop on a £20,000
  account, which is £0.67/pt against a £1/pt minimum. The comment in B now explains it,
  because it is an easy trap on a low-priced share with a wide stop.
