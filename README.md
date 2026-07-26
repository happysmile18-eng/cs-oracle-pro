# CS Oracle Pure Prediction Engine 6.0

This is the clean no-odds version. It uses only the PandaScore fixture/results feed plus optional lawful CSV uploads.

## Accuracy upgrades

- Three temporal XGBoost winner models with different memory speeds.
- Chronological stacking with Oracle Elo and opponent-adjusted form.
- Calibration method competition: identity, Platt, beta, and isotonic.
- Learned confidence shrink to reduce overconfidence.
- Untouched newest-period test and selective 70% gate.
- Separate exact-score head.
- Core, Fast and Oracle Elo history.
- Recent form pulse graph and hot-form ranking under each team.
- No bookmaker API, odds widgets, value finder, or rate-limit dependency.

## Upload

Replace `app.py` and `requirements.txt` in the existing GitHub repository. Commit, then reboot the Streamlit app.

The header must show:

`CS Oracle · Pure Prediction Engine 6.0.0-pure-prediction`

## Secret

Only this is needed:

```toml
PANDASCORE_TOKEN = "your-private-token"
```

Delete `ODDS_API_KEY` from Streamlit Secrets if desired; this app never reads it.

## Accuracy

The app cannot guarantee 70% across every CS2 match. It will only label a selective subset as 70%-proven when the newest untouched backtest actually reaches the target with at least 20 matches.
