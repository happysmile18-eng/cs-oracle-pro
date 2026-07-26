# CS Oracle Accuracy Engine 3.0

Upload `app.py` and `requirements.txt` directly to the root of your existing GitHub repository, replacing the old files. Keep Streamlit's main file path as `app.py`.

Your Streamlit secret remains:

```toml
PANDASCORE_TOKEN = "your-private-token"
```

Then open **Manage app → Reboot app**. The first run can take one or two minutes because the app loads history, rebuilds multi-speed Elo, trains the exact-score XGBoost model, calibrates it and calculates the chronological test metrics.

## Main improvements

- Direct multiclass exact-score prediction for BO1/BO3/BO5
- Stable Core Elo + Fast Elo + LAN/online Elo + format Elo
- Full Elo history, detailed update ledger and formulas
- Opponent-adjusted form, score margins, sweep rates, fatigue, rust and workload
- Symmetric prediction averaging to eliminate team-order bias
- Chronological train/calibration/test evaluation
- Confidence-coverage accuracy table and probability calibration table
- Optional lawful player and map CSV overlays

The free PandaScore plan does not automatically expose complete player ADR/rating and map-veto history. The model works without them and clearly states this limitation.
