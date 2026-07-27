# CS Oracle Team Intelligence Engine 8.0

A clean, no-odds CS2 prediction website using PandaScore match/results data, calibrated temporal XGBoost models, exact-score prediction, Oracle Elo and clickable team intelligence profiles.

## What Version 8 fixes

- Fixes `StreamlitDuplicateElementId` by assigning a deterministic unique key to every dynamic Plotly chart.
- Adds explicit unique keys to dynamic controls and downloads.
- Keeps charts stable when the same team appears in Match Laboratory, Team Intelligence and leading-team expanders.
- Canonicalises team history by PandaScore team ID, so renamed organisations keep one continuous Elo/form history.
- Applies the same canonical names to optional map and player uploads.

## Clickable team profiles

Open **Oracle Elo & hot form** and click any team name. The profile shows:

- World rank, Oracle/Core/Fast Elo and peak Elo
- 30-day momentum, heat score, last-10 form, opposition strength, streak and rust
- Full Oracle/Core/Fast Elo history chart
- Rating DNA: Core, Fast, LAN, Online, BO1, BO3 and BO5 ratings
- Recent five-match form and Elo-movement chart
- BO1/BO3/BO5 and LAN/online performance for the last year
- Recent Elo calculation ledger
- Biggest Elo gains and most damaging losses
- Upcoming fixtures
- Current PandaScore roster when available

The ranking includes filters for active teams, all teams, super-hot teams and fastest-rising teams, plus team search and selectable row count.

## Daily prediction controls

- Today is the default.
- One click switches to Tomorrow, Next 3 days, All loaded or a chosen date.
- Times use the selected timezone; default is Europe/Bucharest.
- Sort by best signal, highest win probability or start time.

## Accuracy improvements

- Stable team-ID canonicalisation prevents Elo and form being split when an organisation changes display name.
- Fast, balanced and long-memory XGBoost experts remain evaluated chronologically.
- Model stacking, probability calibration, exact-score modelling and the untouched newest-period accuracy proof remain active.
- The app still refuses to claim 70% unless the held-out historical test genuinely proves it for the selected subset.

## Upload

Replace these files in your existing GitHub repository:

- `app.py`
- `requirements.txt`

Commit the files, then open Streamlit and choose **Manage app → Reboot app**.

The header must show:

`CS Oracle · Team Intelligence Engine 8.0.0-team-intelligence`

## Streamlit secret

```toml
PANDASCORE_TOKEN = "your-private-token"
```

No bookmaker or odds API is used.
