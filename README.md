# CS Oracle Today Intelligence Engine 7.0

A clean, no-odds CS2 prediction app using PandaScore match/results data plus optional lawful player and map CSV uploads.

## New daily controls

- **Today** is the default view.
- One click switches to **Tomorrow**, **Next 3 days**, **All loaded**, or **Choose date**.
- Dates are calculated in the selected local timezone; the default is `Europe/Bucharest`.
- Every selected date gets its own ranked prediction board.
- Sort by best overall signal, highest win chance, earliest start, or latest start.
- Download exactly the currently shown ranked list.

## Accuracy Engine 7 upgrades

- Context-aware stack using model agreement, sample depth, event tier, BO format, LAN/online context, and Elo separation.
- Candidate XGBoost models are evaluated across multiple chronological slices with a stability penalty.
- Current-event form is learned separately from overall form.
- Favourite-conversion rate, underdog upset rate, close-match conversion, high-tier form, and performance consistency are added as pre-match features.
- Core Elo, Fast Elo, format Elo, LAN/online Elo, opponent-adjusted performance, workload, rest, rust, and score-margin history remain active.
- Adaptive probability calibration and an untouched newest-period accuracy test remain mandatory.
- The separate exact-score head remains aligned to the calibrated winner probability.
- Elo history graphs, form pulse, hot-team ranking, and detailed Elo ledger remain included.

## Upload

Replace these two files in the existing GitHub repository:

- `app.py`
- `requirements.txt`

Commit the changes and reboot the Streamlit app.

The header must show:

`CS Oracle · Today Intelligence Engine 7.0.0-today-intelligence`

## Streamlit secret

```toml
PANDASCORE_TOKEN = "your-private-token"
```

No odds API is used.

## Accuracy honesty

This version is designed to improve robustness with the available free data, but it cannot guarantee 70% across every match. The app only shows an Elite 70% label when the newest untouched chronological test genuinely proves it on a meaningful sample.
