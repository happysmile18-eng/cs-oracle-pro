# CS Oracle Market Edge 5.0

This version adds live bookmaker odds, a Top Bankers sorter and an independent value-edge finder to Precision Engine 4.

## Upload

Replace these files in your existing GitHub repository:

- `app.py`
- `requirements.txt`

Commit them, then use **Streamlit → Manage app → Reboot app**. The header must show **5.0.0-market-edge**.

## Streamlit Secrets

```toml
PANDASCORE_TOKEN = "your-private-pandascore-token"
ODDS_API_KEY = "your-private-odds-api-key"
```

Do not put either key in GitHub. Regenerate keys that have appeared in screenshots or chat.

## What is new

- Odds-API.io event matching with strict team-name and start-time checks.
- Batch odds requests and caching.
- Best match-winner odds and bookmaker.
- Bookmaker margin removal and robust no-vig consensus.
- Separate independent Oracle, market probability and market-assisted final forecast.
- Top Bankers: minimum odds defaults to 1.40, then sorted by estimated winning chance.
- Value Edge: independent Oracle probability compared with no-vig market probability.
- Edge, expected value, Oracle fair odds and value grade.
- Exact-score distribution rescaled to the final winner forecast without changing the independent value calculation.
- Existing Core/Fast/Oracle Elo graphs, hot-form table, chronological accuracy proof and prediction receipts remain.

## Important

The free Odds-API.io plan is limited to two selected bookmakers. The app does not claim that a Banker is guaranteed or that a positive estimated EV will win. Historical market-assisted accuracy is not shown until historical odds are available.
