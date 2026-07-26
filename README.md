# CS Oracle Pro

A Streamlit CS2 prediction dashboard built around **one learned model**: an XGBoost map-win probability model. It then simulates the map veto and complete BO1/BO3/BO5 series.

## What it shows

- Daily upcoming CS2 matches from PandaScore
- Complete daily prediction board
- Predicted winner and exact series score
- Most likely veto and maps
- Map-by-map win probabilities
- Current five-player lineup table
- 90-day rating, ADR, KAST, K/D and opening strength when supplied by the data provider or CSV
- Overall and map-specific Elo
- Top team Elo ranking
- Time-split backtest: accuracy, Brier score, log loss and ROC AUC
- Timestamped prediction receipt with SHA-256 fingerprint

## Data policy

HLTV's current Terms of Service prohibit data mining and web scraping. This project therefore does **not** scrape HLTV automatically.

Use one of these lawful routes:

1. PandaScore API for fixtures and, on an eligible plan, detailed historical CS2 games/player statistics.
2. GRID Open Access after approval.
3. Your own permitted CSV data using the included templates.

The `rating` field in the CSV can contain an HLTV rating only when you obtained and entered that data lawfully yourself. The app does not claim that every provider's `rating` is identical to HLTV Rating.

## Easiest Streamlit deployment

Upload these files and folders directly to the root of your GitHub repository:

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- the three CSV templates (optional but recommended)

Do not upload only the ZIP.

In Streamlit Community Cloud use:

- Repository: `your-github-name/cs-oracle-pro`
- Branch: `main`
- Main file path: `app.py`

## Add your PandaScore token safely

In Streamlit, open **Manage app → Settings → Secrets** and enter:

```toml
PANDASCORE_TOKEN = "your-private-token"
```

Never commit your real token to GitHub.

## Detailed model data

The model requires at least 160 completed map rows. You have two choices:

### A. Build from PandaScore

Open **Setup & data** and press **Build map database from API**. This requires a PandaScore plan that grants access to `GET /csgo/matches/{id}/games`.

### B. Upload CSV files

Use:

- `map_history_template.csv`
- `player_form_template.csv`
- `veto_history_template.csv`

The app refuses to invent missing historical maps or player statistics. If a field is missing, confidence is lowered.

## Model design

For every historical map, features are created before updating the ratings, preventing future leakage. The row is then mirrored to remove Team A bias. Recent maps receive exponential time weighting.

The single XGBoost model uses:

- Overall Elo difference
- Map Elo difference
- 90-day map and overall form
- Recent round margin
- Map experience
- Rest and seven-day workload
- Head-to-head
- Lineup rating, ADR, KAST, K/D and opening rating
- Star ceiling and weakest-player floor
- Lineup sample size and completeness
- Event tier, LAN context, BO format and pick ownership

The veto simulator uses recent pick, ban, play and map-win tendencies. It runs both possible veto orders and plays the chosen maps until one side wins the series.

## Accuracy claim

This is a serious architecture, but no honest model can be called the world's most accurate before a large locked live record exists. Save the prediction receipts and evaluate at least 100 pre-match predictions. Optimise for Brier score and log loss, not only winner hit rate.
