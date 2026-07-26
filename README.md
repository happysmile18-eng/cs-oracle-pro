# CS Oracle Precision Engine 4.0

This version is designed to maximise **honest selective winner accuracy** from the data available on PandaScore Free.

## Main upgrades

- Dedicated binary XGBoost winner model.
- Separate conditional exact-score model.
- Three winner-model configurations tested chronologically; the best is selected automatically.
- Platt probability calibration.
- Learned precision gate that can refuse weak predictions.
- The app labels a match **Elite 70% proven** only when the newest untouched historical test reaches at least 70% on 20+ selected matches.
- Core, Fast and Oracle Elo history charts.
- Hot-form leaderboard and form labels.
- Elo graph expanders under leading teams.

## Upload to GitHub

Replace the existing files in your repository with:

- `app.py`
- `requirements.txt`

Keep the main file path in Streamlit as `app.py`.

After committing the files, open Streamlit and use **Manage app → Reboot app**.

The first build may take several minutes because the app trains multiple candidate models, calibration and the precision gate.

## Streamlit secret

```toml
PANDASCORE_TOKEN = "your-new-private-token"
```

Do not publish the token in GitHub or screenshots.

## Accuracy promise

The app does not guarantee 70% across all matches. It attempts to achieve it only on a selective subset and shows the untouched backtest result. If the target is not proven, the Elite label remains disabled.
