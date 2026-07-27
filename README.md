# CS Oracle — Drift-Aware Precision Engine V9

A Streamlit CS2 prediction and team-strength application built for the free PandaScore fixture/results feed.

## What changed in V9

- Adaptive Elo: current-form weight increases only when inactivity, uncertainty, Core/Fast divergence or probable structural change makes old strength less trustworthy.
- Glicko-style rating deviation: inactive and lightly observed teams carry more uncertainty; active teams become more certain.
- Probable roster-reset protection: repeated results far from expectation and long inactivity reduce reliance on stale Core Elo without inventing player data.
- Dynamic event strength: official/heuristic tournament tier is blended with the actual pre-match strength of the field.
- Nine temporal specialists: ultra-recent, fast, balanced, long-memory, BO1, BO3, LAN, elite-event and lower-tier models compete chronologically.
- Recent-form relevance is selected from evidence, not hard-coded: the validation period decides which memory speeds survive into the final stack.
- A separate reliability gate compares logistic and shallow boosted selectors and keeps the one that performs best on later validation data.
- Winner training no longer inherits exact-score class-balancing weights; score balancing is reserved for the separate exact-score model.
- Team pages now show rating certainty, adaptive Fast-Elo weight and estimated reset risk alongside the existing Elo history, form pulse, ledger, roster and fixtures.
- Daily predictions still support Today, Tomorrow, Next 3 days, custom date, Strong + Elite and Elite-only filters.

## Accuracy policy

The engine targets better than 60% on well-covered professional matches and 70% on a selective subset, but it never forces or fabricates those numbers. The `Accuracy proof` tab uses a strict chronological split and enables the Elite label only when the newest untouched test data genuinely reaches the target on a meaningful sample.

No model can guarantee 70% across every CS2 fixture with series-level free data. BO1s, roster changes and low-tier matches contain substantial uncertainty.

## GitHub / Streamlit installation

Upload these files directly to the repository root:

- `app.py`
- `requirements.txt`

Your Streamlit secret should contain only:

```toml
PANDASCORE_TOKEN = "your-private-token"
```

Then use **Manage app → Reboot app**. The header should show:

`CS Oracle · Drift-Aware Precision Engine 9.0.0-drift-aware-precision`

## Data boundary

PandaScore Free supplies fixtures, results, teams and basic rosters. It does not automatically supply complete 90-day ADR, KAST, player ratings, map veto frequencies or round telemetry. Optional lawful CSV uploads remain available for player and map context.
