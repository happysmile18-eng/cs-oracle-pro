from __future__ import annotations

import hashlib
import io
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

APP_VERSION = "2.0.0-free"
API_ROOT = "https://api.pandascore.co"
ACTIVE_POOL = ["Ancient", "Anubis", "Cache", "Dust 2", "Inferno", "Mirage", "Nuke"]
DEFAULT_HISTORY_DAYS = 480
DEFAULT_HISTORY_PAGES = 14
MIN_MODEL_MATCHES = 90

st.set_page_config(page_title="CS Oracle Pro", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{--red:#ff5364;--ink:#f5f7fb;--muted:#929bad;--panel:#0c1017;--line:#222a37;--green:#61dda2;--gold:#ffd166}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,sans-serif}
.block-container{max-width:1240px;padding-top:1.1rem;padding-bottom:3rem}
#MainMenu,footer{visibility:hidden}[data-testid="stHeader"]{background:rgba(6,8,12,.72);backdrop-filter:blur(12px)}
.hero{padding:1.45rem;border:1px solid var(--line);border-radius:25px;background:radial-gradient(circle at 88% 10%,rgba(255,83,100,.22),transparent 35%),linear-gradient(145deg,#131925,#080b11);margin-bottom:1rem}
.eyebrow{color:#ff8290;font-size:.72rem;font-weight:850;letter-spacing:.18em;text-transform:uppercase}.hero h1{font-size:clamp(2.25rem,7vw,4.9rem);line-height:.93;margin:.4rem 0 .55rem;letter-spacing:-.055em}.hero p{color:var(--muted);max-width:900px;margin:0}
.card{padding:1.05rem 1.1rem;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,#111620,#090c12);height:100%}.metric{font-size:2rem;font-weight:900;letter-spacing:-.04em}.muted{color:var(--muted)}
.callout{padding:1rem 1.1rem;border-radius:16px;border:1px solid #293344;background:#0e141d}.good{color:var(--green)}.warn{color:var(--gold)}.bad{color:#ff7d89}
.match-card{padding:1rem;border:1px solid var(--line);border-radius:18px;background:#0c1118;margin:.55rem 0}.winner{font-size:1.45rem;font-weight:900;letter-spacing:-.03em}.pill{display:inline-block;padding:.26rem .55rem;border:1px solid #30394a;border-radius:999px;font-size:.75rem;color:#d1d7e3;margin-right:.3rem}
.rank-row{display:grid;grid-template-columns:52px minmax(150px,1.25fr) 1fr 1fr 1fr;gap:.75rem;align-items:center;padding:.8rem .9rem;border-bottom:1px solid #1d2430}.rank-row:last-child{border-bottom:0}.rank-n{font-weight:900;font-size:1.2rem;color:#ff7d89}.team{font-weight:850}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:16px;overflow:hidden}.stButton>button,.stDownloadButton>button{border-radius:13px;font-weight:800;min-height:43px}
@media(max-width:700px){.rank-row{grid-template-columns:40px 1fr 1fr}.hide-mobile{display:none}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_dt(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return ts if not pd.isna(ts) else pd.Timestamp.now(tz="UTC")


def normalise_team(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "natus vincere": "NAVI",
        "team vitality": "Vitality",
        "g2 esports": "G2",
        "faze clan": "FaZe",
    }
    return aliases.get(text.lower(), text)


def normalise_map(value: Any) -> str:
    text = str(value or "").replace("de_", "").replace("_", " ").strip().lower()
    aliases = {
        "dust2": "Dust 2", "dust ii": "Dust 2", "dust 2": "Dust 2",
        "ancient": "Ancient", "anubis": "Anubis", "cache": "Cache",
        "inferno": "Inferno", "mirage": "Mirage", "nuke": "Nuke",
        "overpass": "Overpass", "train": "Train", "vertigo": "Vertigo",
    }
    return aliases.get(text, text.title())


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def logistic_elo(diff: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def logit(p: float) -> float:
    p = min(max(p, 1e-5), 1 - 1e-5)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def token_from_secrets() -> str:
    try:
        return str(st.secrets.get("PANDASCORE_TOKEN", "")).strip()
    except Exception:
        return ""


class APIError(RuntimeError):
    pass


@dataclass
class PandaClient:
    token: str
    timeout: int = 30

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.token:
            raise APIError("PandaScore token missing.")
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        url = f"{API_ROOT}{path}"
        for attempt in range(4):
            try:
                r = requests.get(url, headers=headers, params=params or {}, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == 3:
                    raise APIError(f"Connection failed: {exc}") from exc
                time.sleep(1.2 * (attempt + 1))
                continue
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise APIError(f"PandaScore returned {r.status_code}. Check the token and plan access.")
            if r.status_code >= 400:
                raise APIError(f"PandaScore returned {r.status_code}: {r.text[:180]}")
            return r.json()
        raise APIError("PandaScore request failed after retries.")

    def paged(self, path: str, params: dict[str, Any] | None = None, max_pages: int = 10) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            p = dict(params or {})
            p.update({"page": page, "per_page": 100})
            batch = self._get(path, p)
            if not isinstance(batch, list):
                break
            rows.extend(batch)
            if len(batch) < 100:
                break
        return rows

    def past_matches(self, days: int, pages: int) -> list[dict[str, Any]]:
        end = datetime.now(timezone.utc).date().isoformat()
        start = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        params = {"range[begin_at]": f"{start},{end}", "sort": "-begin_at"}
        return self.paged("/csgo/matches/past", params, max_pages=pages)

    def upcoming_matches(self, days: int = 3) -> list[dict[str, Any]]:
        start = datetime.now(timezone.utc).date().isoformat()
        end = (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()
        params = {"range[begin_at]": f"{start},{end}", "sort": "begin_at"}
        return self.paged("/csgo/matches/upcoming", params, max_pages=4)

    def team(self, team_id: int | str) -> dict[str, Any]:
        obj = self._get(f"/teams/{team_id}")
        return obj if isinstance(obj, dict) else {}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_past(token: str, days: int, pages: int) -> list[dict[str, Any]]:
    return PandaClient(token).past_matches(days, pages)


@st.cache_data(ttl=900, show_spinner=False)
def cached_upcoming(token: str, days: int) -> list[dict[str, Any]]:
    return PandaClient(token).upcoming_matches(days)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cached_team(token: str, team_id: int | str) -> dict[str, Any]:
    return PandaClient(token).team(team_id)


def extract_teams(match: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    opponents = match.get("opponents") or []
    teams: list[dict[str, Any]] = []
    for op in opponents:
        obj = op.get("opponent") or {}
        if obj.get("id") and obj.get("name"):
            teams.append(obj)
    if len(teams) < 2:
        return None
    return teams[0], teams[1]


def tier_value(match: dict[str, Any]) -> float:
    tournament = match.get("tournament") or {}
    serie = match.get("serie") or {}
    raw = str(tournament.get("tier") or serie.get("tier") or "").lower().strip()
    mapping = {"s": 1.0, "a": 0.82, "b": 0.62, "c": 0.42, "d": 0.25}
    if raw in mapping:
        return mapping[raw]
    name = " ".join([
        str((match.get("league") or {}).get("name") or ""),
        str(serie.get("full_name") or serie.get("name") or ""),
        str(tournament.get("name") or ""),
    ]).lower()
    if any(k in name for k in ["major", "iem", "blast premier", "esl pro league", "pgl"]):
        return 0.95
    if any(k in name for k in ["challenger", "masters", "qualifier", "academy"]):
        return 0.55
    return 0.45


def is_lan(match: dict[str, Any]) -> int:
    tournament = match.get("tournament") or {}
    return int(bool(match.get("is_lan") or tournament.get("is_lan") or match.get("location") or tournament.get("location")))


def result_scores(match: dict[str, Any], team_a_id: int, team_b_id: int) -> tuple[int, int]:
    score_by_id: dict[int, int] = {}
    for item in match.get("results") or []:
        tid = safe_int(item.get("team_id"), -1)
        if tid >= 0:
            score_by_id[tid] = safe_int(item.get("score"), 0)
    return score_by_id.get(team_a_id, 0), score_by_id.get(team_b_id, 0)


def extract_game_winners(match: dict[str, Any]) -> list[int]:
    output: list[tuple[int, int]] = []
    for game in match.get("games") or []:
        winner = game.get("winner") or {}
        winner_id = safe_int(winner.get("id") or game.get("winner_id"), -1)
        if winner_id >= 0:
            output.append((safe_int(game.get("position"), len(output) + 1), winner_id))
    output.sort(key=lambda x: x[0])
    return [winner_id for _, winner_id in output]


def extract_map_rows(match: dict[str, Any]) -> list[dict[str, Any]]:
    pair = extract_teams(match)
    if pair is None:
        return []
    ta, tb = pair
    rows: list[dict[str, Any]] = []
    for game in match.get("games") or []:
        map_obj = game.get("map") or {}
        map_name = normalise_map(map_obj.get("name") or game.get("map_name") or "")
        if not map_name or map_name == "":
            continue
        winner_obj = game.get("winner") or {}
        winner_id = safe_int(winner_obj.get("id") or game.get("winner_id"), -1)
        if winner_id not in (safe_int(ta.get("id"), -2), safe_int(tb.get("id"), -3)):
            continue
        rows.append({
            "date": parse_dt(game.get("begin_at") or match.get("begin_at")),
            "team_a": normalise_team(ta.get("name")),
            "team_b": normalise_team(tb.get("name")),
            "map": map_name,
            "winner": normalise_team(ta.get("name")) if winner_id == safe_int(ta.get("id")) else normalise_team(tb.get("name")),
            "position": safe_int(game.get("position"), 1),
        })
    return rows


def match_row(match: dict[str, Any], completed_only: bool) -> dict[str, Any] | None:
    pair = extract_teams(match)
    if pair is None:
        return None
    ta, tb = pair
    ta_id, tb_id = safe_int(ta.get("id"), -1), safe_int(tb.get("id"), -1)
    if ta_id < 0 or tb_id < 0:
        return None
    winner_id = safe_int(match.get("winner_id"), -1)
    if completed_only and winner_id not in (ta_id, tb_id):
        return None
    sa, sb = result_scores(match, ta_id, tb_id)
    if completed_only and sa == sb == 0:
        games = extract_game_winners(match)
        sa = sum(1 for w in games if w == ta_id)
        sb = sum(1 for w in games if w == tb_id)
    if completed_only and sa == sb:
        return None
    tournament = match.get("tournament") or {}
    serie = match.get("serie") or {}
    league = match.get("league") or {}
    bo = safe_int(match.get("number_of_games"), 3)
    if bo not in (1, 2, 3, 5):
        bo = max(1, min(5, bo or 3))
    return {
        "match_id": match.get("id"),
        "date": parse_dt(match.get("begin_at") or match.get("scheduled_at")),
        "team_a": normalise_team(ta.get("name")),
        "team_b": normalise_team(tb.get("name")),
        "team_a_id": ta_id,
        "team_b_id": tb_id,
        "team_a_image": ta.get("image_url"),
        "team_b_image": tb.get("image_url"),
        "winner_a": int(winner_id == ta_id) if winner_id in (ta_id, tb_id) else None,
        "score_a": sa,
        "score_b": sb,
        "best_of": bo,
        "lan": is_lan(match),
        "tier": tier_value(match),
        "event": tournament.get("name") or serie.get("full_name") or league.get("name") or "Unknown event",
        "status": match.get("status") or "not_started",
        "raw": match,
    }


def history_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [row for m in matches if (row := match_row(m, completed_only=True)) is not None]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("match_id", keep="last").reset_index(drop=True)


def upcoming_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [row for m in matches if (row := match_row(m, completed_only=False)) is not None]
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("date").drop_duplicates("match_id", keep="last")
    return out[out["date"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)].reset_index(drop=True)


@dataclass
class TeamState:
    elo: float = 1500.0
    matches: int = 0
    last_date: pd.Timestamp | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=80))
    opponent_elo_ewma: float = 1500.0
    margin_ewma: float = 0.0
    streak: int = 0


FEATURES = [
    "elo_diff", "winrate_30_diff", "winrate_90_diff", "form10_diff",
    "opp_strength_diff", "margin_diff", "experience_diff", "rest_diff",
    "workload14_diff", "h2h_diff", "bo3_form_diff", "lan_form_diff",
    "streak_diff", "best_of", "lan", "tier",
]


def record_values(state: TeamState, when: pd.Timestamp, days: int | None = None, last_n: int | None = None, context: str | None = None) -> list[float]:
    items = list(state.recent)
    if days is not None:
        cutoff = when - pd.Timedelta(days=days)
        items = [x for x in items if x[0] >= cutoff]
    if context == "bo3":
        items = [x for x in items if x[4] >= 3]
    elif context == "lan":
        items = [x for x in items if x[5] == 1]
    if last_n is not None:
        items = items[-last_n:]
    return [float(x[1]) for x in items]


def win_rate(values: list[float], prior_games: float = 4.0) -> float:
    return (sum(values) + 0.5 * prior_games) / (len(values) + prior_games)


def workload(state: TeamState, when: pd.Timestamp, days: int = 14) -> int:
    cutoff = when - pd.Timedelta(days=days)
    return sum(1 for x in state.recent if x[0] >= cutoff)


def rest_days(state: TeamState, when: pd.Timestamp) -> float:
    if state.last_date is None:
        return 14.0
    return float(min(30.0, max(0.0, (when - state.last_date).total_seconds() / 86400.0)))


def build_features(
    states: dict[str, TeamState],
    h2h: dict[tuple[str, str], deque],
    when: pd.Timestamp,
    team_a: str,
    team_b: str,
    best_of: int,
    lan: int,
    tier: float,
) -> dict[str, float]:
    a, b = states[team_a], states[team_b]
    h = list(h2h[(team_a, team_b)])
    h_reverse = list(h2h[(team_b, team_a)])
    h_a = sum(h[-8:])
    h_b = sum(h_reverse[-8:])
    h_total = len(h[-8:]) + len(h_reverse[-8:])
    h2h_edge = (h_a - h_b) / max(3.0, h_total)
    return {
        "elo_diff": a.elo - b.elo,
        "winrate_30_diff": win_rate(record_values(a, when, days=30)) - win_rate(record_values(b, when, days=30)),
        "winrate_90_diff": win_rate(record_values(a, when, days=90)) - win_rate(record_values(b, when, days=90)),
        "form10_diff": win_rate(record_values(a, when, last_n=10)) - win_rate(record_values(b, when, last_n=10)),
        "opp_strength_diff": a.opponent_elo_ewma - b.opponent_elo_ewma,
        "margin_diff": a.margin_ewma - b.margin_ewma,
        "experience_diff": math.log1p(a.matches) - math.log1p(b.matches),
        "rest_diff": rest_days(a, when) - rest_days(b, when),
        "workload14_diff": float(workload(a, when) - workload(b, when)),
        "h2h_diff": h2h_edge,
        "bo3_form_diff": win_rate(record_values(a, when, days=180, context="bo3")) - win_rate(record_values(b, when, days=180, context="bo3")),
        "lan_form_diff": win_rate(record_values(a, when, days=240, context="lan")) - win_rate(record_values(b, when, days=240, context="lan")),
        "streak_diff": float(a.streak - b.streak),
        "best_of": float(best_of),
        "lan": float(lan),
        "tier": float(tier),
    }


def mirror_features(row: dict[str, float]) -> dict[str, float]:
    mirrored = dict(row)
    for key in FEATURES:
        if key.endswith("_diff"):
            mirrored[key] = -mirrored[key]
    return mirrored


def update_state(
    states: dict[str, TeamState],
    h2h: dict[tuple[str, str], deque],
    when: pd.Timestamp,
    team_a: str,
    team_b: str,
    a_win: int,
    score_a: int,
    score_b: int,
    best_of: int,
    lan: int,
    tier: float,
) -> None:
    a, b = states[team_a], states[team_b]
    pre_a, pre_b = a.elo, b.elo
    expected_a = logistic_elo(pre_a - pre_b)
    needed = max(1, math.ceil(best_of / 2))
    margin = (score_a - score_b) / needed
    k = 30.0 + 12.0 * tier + 4.0 * min(1.5, abs(margin))
    delta = k * (a_win - expected_a)
    a.elo += delta
    b.elo -= delta
    alpha = 0.12
    a.opponent_elo_ewma = (1 - alpha) * a.opponent_elo_ewma + alpha * pre_b
    b.opponent_elo_ewma = (1 - alpha) * b.opponent_elo_ewma + alpha * pre_a
    a.margin_ewma = 0.84 * a.margin_ewma + 0.16 * margin
    b.margin_ewma = 0.84 * b.margin_ewma - 0.16 * margin
    a.matches += 1
    b.matches += 1
    a.last_date = when
    b.last_date = when
    a_result = float(a_win)
    b_result = float(1 - a_win)
    a.recent.append((when, a_result, pre_b, margin, best_of, lan))
    b.recent.append((when, b_result, pre_a, -margin, best_of, lan))
    a.streak = (a.streak + 1) if a_win else min(-1, a.streak - 1) if a.streak < 0 else -1
    b.streak = (b.streak + 1) if not a_win else min(-1, b.streak - 1) if b.streak < 0 else -1
    h2h[(team_a, team_b)].append(a_win)
    h2h[(team_b, team_a)].append(1 - a_win)


@dataclass
class ModelBundle:
    model: xgb.XGBClassifier | None
    states: dict[str, TeamState]
    h2h: dict[tuple[str, str], deque]
    training_rows: int
    metrics: dict[str, float]
    feature_gain: dict[str, float]
    newest_date: pd.Timestamp | None


def make_training_data(history: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, TeamState], dict[tuple[str, str], deque]]:
    states: dict[str, TeamState] = defaultdict(TeamState)
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=20))
    x_rows: list[dict[str, float]] = []
    y_rows: list[int] = []
    dates: list[pd.Timestamp] = []
    newest = history["date"].max()
    for r in history.sort_values("date").itertuples(index=False):
        features = build_features(states, h2h, r.date, r.team_a, r.team_b, int(r.best_of), int(r.lan), float(r.tier))
        age_days = max(0.0, (newest - r.date).total_seconds() / 86400.0)
        weight = (0.5 ** (age_days / 210.0)) * (0.72 + 0.55 * float(r.tier))
        x_rows.append(features)
        y_rows.append(int(r.winner_a))
        dates.append(r.date)
        x_rows.append(mirror_features(features))
        y_rows.append(1 - int(r.winner_a))
        dates.append(r.date)
        update_state(states, h2h, r.date, r.team_a, r.team_b, int(r.winner_a), int(r.score_a), int(r.score_b), int(r.best_of), int(r.lan), float(r.tier))
    xdf = pd.DataFrame(x_rows, columns=FEATURES)
    y = np.asarray(y_rows, dtype=int)
    weights = []
    for d in dates:
        age_days = max(0.0, (newest - d).total_seconds() / 86400.0)
        weights.append(0.5 ** (age_days / 210.0))
    return xdf, y, np.asarray(weights, dtype=float), states, h2h


def new_model(seed: int = 27) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=420,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=5.0,
        subsample=0.88,
        colsample_bytree=0.9,
        reg_alpha=0.25,
        reg_lambda=2.8,
        gamma=0.03,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=2,
        tree_method="hist",
    )


@st.cache_resource(show_spinner=False)
def train_bundle_cached(history_json: str) -> ModelBundle:
    history = pd.read_json(io.StringIO(history_json), orient="split")
    history["date"] = pd.to_datetime(history["date"], utc=True)
    xdf, y, weights, states, h2h = make_training_data(history)
    metrics: dict[str, float] = {}
    feature_gain: dict[str, float] = {}
    model: xgb.XGBClassifier | None = None
    if len(history) >= MIN_MODEL_MATCHES and len(np.unique(y)) == 2:
        unique_dates = sorted(history["date"].unique())
        cutoff = unique_dates[max(1, int(len(unique_dates) * 0.80)) - 1]
        mirrored_dates = np.repeat(history.sort_values("date")["date"].to_numpy(), 2)
        train_mask = mirrored_dates <= cutoff
        test_mask = mirrored_dates > cutoff
        if train_mask.sum() >= 100 and test_mask.sum() >= 30:
            eval_model = new_model(26)
            eval_model.fit(xdf.loc[train_mask, FEATURES], y[train_mask], sample_weight=weights[train_mask])
            p = eval_model.predict_proba(xdf.loc[test_mask, FEATURES])[:, 1]
            yt = y[test_mask]
            metrics = {
                "accuracy": float(accuracy_score(yt, p >= 0.5)),
                "brier": float(brier_score_loss(yt, p)),
                "logloss": float(log_loss(yt, p, labels=[0, 1])),
                "auc": float(roc_auc_score(yt, p)) if len(np.unique(yt)) == 2 else float("nan"),
                "test_rows": int(test_mask.sum()),
            }
        model = new_model(27)
        model.fit(xdf[FEATURES], y, sample_weight=weights)
        raw_gain = model.get_booster().get_score(importance_type="gain")
        total_gain = sum(raw_gain.values()) or 1.0
        feature_gain = {k: float(raw_gain.get(k, 0.0) / total_gain) for k in FEATURES}
    return ModelBundle(model, states, h2h, len(xdf), metrics, feature_gain, history["date"].max() if not history.empty else None)


def current_feature_row(bundle: ModelBundle, match: dict[str, Any]) -> dict[str, float]:
    return build_features(
        bundle.states,
        bundle.h2h,
        match["date"],
        match["team_a"],
        match["team_b"],
        int(match["best_of"]),
        int(match["lan"]),
        float(match["tier"]),
    )


def core_probability(bundle: ModelBundle, match: dict[str, Any]) -> tuple[float, dict[str, float]]:
    f = current_feature_row(bundle, match)
    if bundle.model is None:
        return logistic_elo(f["elo_diff"]), f
    x1 = pd.DataFrame([f], columns=FEATURES)
    x2 = pd.DataFrame([mirror_features(f)], columns=FEATURES)
    p1 = float(bundle.model.predict_proba(x1)[:, 1][0])
    p2 = 1.0 - float(bundle.model.predict_proba(x2)[:, 1][0])
    return 0.5 * (p1 + p2), f


def solve_map_probability(series_p: float, best_of: int) -> float:
    if best_of <= 1:
        return series_p
    wins_needed = best_of // 2 + 1

    def series_prob(q: float) -> float:
        total = 0.0
        for losses in range(wins_needed):
            total += math.comb(wins_needed - 1 + losses, losses) * (q ** wins_needed) * ((1 - q) ** losses)
        return total

    lo, hi = 1e-4, 1 - 1e-4
    for _ in range(60):
        mid = (lo + hi) / 2
        if series_prob(mid) < series_p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def exact_scores(map_p: float, best_of: int) -> dict[str, float]:
    wins_needed = best_of // 2 + 1
    out: dict[str, float] = {}
    for losses in range(wins_needed):
        prob_a = math.comb(wins_needed - 1 + losses, losses) * (map_p ** wins_needed) * ((1 - map_p) ** losses)
        prob_b = math.comb(wins_needed - 1 + losses, losses) * ((1 - map_p) ** wins_needed) * (map_p ** losses)
        out[f"{wins_needed}-{losses}"] = prob_a
        out[f"{losses}-{wins_needed}"] = prob_b
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def confidence_score(bundle: ModelBundle, match: dict[str, Any], p: float) -> tuple[int, str]:
    a = bundle.states[match["team_a"]]
    b = bundle.states[match["team_b"]]
    sample = min(a.matches, b.matches)
    sample_quality = min(1.0, sample / 24.0)
    recency = 1.0
    if bundle.newest_date is not None:
        age = max(0.0, (pd.Timestamp.now(tz="UTC") - bundle.newest_date).total_seconds() / 86400.0)
        recency = max(0.55, 1.0 - age / 180.0)
    certainty = min(1.0, abs(p - 0.5) / 0.24)
    score = round(100 * (0.42 * sample_quality + 0.28 * recency + 0.30 * certainty))
    label = "High" if score >= 76 else "Medium" if score >= 58 else "Cautious"
    return score, label


def feature_reasons(features: dict[str, float], p: float) -> tuple[list[str], list[str]]:
    a: list[tuple[float, str]] = []
    b: list[tuple[float, str]] = []
    mapping = {
        "elo_diff": lambda v: f"Oracle Elo edge: {abs(v):.0f} points",
        "winrate_30_diff": lambda v: f"30-day form edge: {abs(v) * 100:.1f} percentage points",
        "winrate_90_diff": lambda v: f"90-day form edge: {abs(v) * 100:.1f} percentage points",
        "form10_diff": lambda v: f"Last-10 form edge: {abs(v) * 100:.1f} percentage points",
        "opp_strength_diff": lambda v: f"Stronger recent opposition: {abs(v):.0f} Elo",
        "margin_diff": lambda v: f"Better recent series margin: {abs(v):.2f}",
        "rest_diff": lambda v: f"Rest advantage: {abs(v):.1f} days",
        "workload14_diff": lambda v: f"Schedule-load difference: {abs(v):.0f} matches",
        "h2h_diff": lambda v: f"Recent head-to-head edge: {abs(v):.2f}",
        "bo3_form_diff": lambda v: f"BO3 form edge: {abs(v) * 100:.1f} percentage points",
        "lan_form_diff": lambda v: f"LAN form edge: {abs(v) * 100:.1f} percentage points",
        "streak_diff": lambda v: f"Current streak edge: {abs(v):.0f}",
    }
    scales = {
        "elo_diff": 1 / 140, "winrate_30_diff": 5.0, "winrate_90_diff": 4.0,
        "form10_diff": 4.6, "opp_strength_diff": 1 / 170, "margin_diff": 1.5,
        "rest_diff": 0.08, "workload14_diff": -0.12, "h2h_diff": 2.0,
        "bo3_form_diff": 3.2, "lan_form_diff": 2.4, "streak_diff": 0.10,
    }
    for key, fn in mapping.items():
        value = features.get(key, 0.0)
        signed = value * scales.get(key, 1.0)
        if abs(signed) < 0.08:
            continue
        item = (abs(signed), fn(value))
        (a if signed > 0 else b).append(item)
    a_text = [x[1] for x in sorted(a, reverse=True)[:4]]
    b_text = [x[1] for x in sorted(b, reverse=True)[:4]]
    if not a_text:
        a_text = ["No large single statistical edge; prediction comes from the combined profile."]
    if not b_text:
        b_text = ["No large single statistical edge; upset route depends on execution and veto."]
    return a_text, b_text


def map_history_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for m in matches:
        rows.extend(extract_map_rows(m))
    return pd.DataFrame(rows)


def clean_uploaded_map_data(uploaded: Any) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
    frame = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
    required = {"date", "team_a", "team_b", "map", "winner"}
    if not required.issubset(frame.columns):
        raise ValueError("Map CSV needs: date, team_a, team_b, map, winner")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    for c in ["team_a", "team_b", "winner"]:
        frame[c] = frame[c].map(normalise_team)
    frame["map"] = frame["map"].map(normalise_map)
    return frame.dropna(subset=["date", "team_a", "team_b", "map", "winner"])


def map_stats(map_history: pd.DataFrame, team: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    if map_history.empty:
        return pd.DataFrame()
    subset = map_history[map_history["date"] <= cutoff].copy()
    rows = []
    for m in ACTIVE_POOL:
        relevant = subset[((subset["team_a"] == team) | (subset["team_b"] == team)) & (subset["map"] == m)]
        wins = int((relevant["winner"] == team).sum())
        n = len(relevant)
        shrunk = (wins + 3.5) / (n + 7.0)
        rows.append({"map": m, "maps": n, "wins": wins, "win_rate": shrunk})
    return pd.DataFrame(rows)


def projected_veto(map_history: pd.DataFrame, team_a: str, team_b: str, when: pd.Timestamp, best_of: int, base_map_p: float) -> tuple[list[dict[str, Any]], str]:
    sa = map_stats(map_history, team_a, when)
    sb = map_stats(map_history, team_b, when)
    if sa.empty or sb.empty or (sa["maps"].sum() + sb["maps"].sum()) < 24:
        needed = best_of if best_of in (1, 3, 5) else 3
        maps = [{"map": f"Map {i+1} (identity unavailable)", "picker": "Unknown", "team_a_p": base_map_p} for i in range(needed)]
        return maps, "Free fixture data does not expose enough historical map identities. Match and score predictions remain active; exact veto names need map-history data."
    a = sa.set_index("map")
    b = sb.set_index("map")
    remaining = ACTIVE_POOL.copy()
    a_weak = min(remaining, key=lambda m: (a.loc[m, "win_rate"], -a.loc[m, "maps"]))
    remaining.remove(a_weak)
    b_weak = min(remaining, key=lambda m: (b.loc[m, "win_rate"], -b.loc[m, "maps"]))
    remaining.remove(b_weak)
    a_pick = max(remaining, key=lambda m: (a.loc[m, "win_rate"] - b.loc[m, "win_rate"], a.loc[m, "maps"]))
    remaining.remove(a_pick)
    b_pick = max(remaining, key=lambda m: (b.loc[m, "win_rate"] - a.loc[m, "win_rate"], b.loc[m, "maps"]))
    remaining.remove(b_pick)
    sequence: list[tuple[str, str]] = [(a_pick, team_a), (b_pick, team_b)]
    if best_of >= 3:
        a_ban2 = min(remaining, key=lambda m: a.loc[m, "win_rate"])
        remaining.remove(a_ban2)
        b_ban2 = min(remaining, key=lambda m: b.loc[m, "win_rate"])
        remaining.remove(b_ban2)
        if remaining:
            sequence.append((remaining[0], "Decider"))
    if best_of == 1:
        sequence = [(max(remaining + [a_pick, b_pick], key=lambda m: min(a.loc[m, "win_rate"], b.loc[m, "win_rate"])), "Decider")]
    if best_of == 5:
        all_ranked = sorted(ACTIVE_POOL, key=lambda m: abs(a.loc[m, "win_rate"] - b.loc[m, "win_rate"]), reverse=True)
        sequence = [(m, team_a if a.loc[m, "win_rate"] >= b.loc[m, "win_rate"] else team_b) for m in all_ranked[:5]]
    output = []
    for map_name, picker in sequence[:best_of]:
        edge = float(a.loc[map_name, "win_rate"] - b.loc[map_name, "win_rate"])
        map_p = sigmoid(logit(base_map_p) + 1.55 * edge)
        output.append({"map": map_name, "picker": picker, "team_a_p": map_p})
    return output, "Map projection uses uploaded or embedded map history with Bayesian shrinkage; it is a projection, not a confirmed veto."


def player_overlay_frame(uploaded: Any) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
    frame = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
    required = {"team", "player", "rating", "adr"}
    if not required.issubset(frame.columns):
        raise ValueError("Player CSV needs: team, player, rating, adr")
    frame = frame.copy()
    frame["team"] = frame["team"].map(normalise_team)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["adr"] = pd.to_numeric(frame["adr"], errors="coerce")
    return frame.dropna(subset=["team", "player", "rating", "adr"])


def player_adjustment(player_data: pd.DataFrame, team_a: str, team_b: str) -> tuple[float, str]:
    if player_data.empty:
        return 0.0, "No automatic ADR/rating adjustment: PandaScore Free does not include post-game player statistics."
    a = player_data[player_data["team"] == team_a].sort_values("rating", ascending=False).head(5)
    b = player_data[player_data["team"] == team_b].sort_values("rating", ascending=False).head(5)
    if len(a) < 4 or len(b) < 4:
        return 0.0, "Player data incomplete, so no player overlay was applied."
    rating_edge = float(a["rating"].mean() - b["rating"].mean())
    adr_edge = float(a["adr"].mean() - b["adr"].mean())
    adjustment = max(-0.32, min(0.32, 2.2 * rating_edge + 0.012 * adr_edge))
    return adjustment, f"Lineup overlay: rating edge {rating_edge:+.3f}, ADR edge {adr_edge:+.1f}."


def team_players(team_obj: dict[str, Any]) -> pd.DataFrame:
    players = team_obj.get("players") or []
    rows = []
    for p in players:
        rows.append({
            "Player": p.get("name") or p.get("first_name") or p.get("slug") or "Unknown",
            "Nationality": p.get("nationality") or "—",
            "Age": p.get("age") or "—",
        })
    return pd.DataFrame(rows)


def elo_table(bundle: ModelBundle) -> pd.DataFrame:
    rows = []
    for team, s in bundle.states.items():
        if s.matches < 3:
            continue
        wr90 = win_rate(record_values(s, pd.Timestamp.now(tz="UTC"), days=90))
        rows.append({"Team": team, "Oracle Elo": round(s.elo), "Matches": s.matches, "90-day form": wr90, "Last played": s.last_date})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["Oracle Elo", "Matches"], ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out


def render_elo_list(frame: pd.DataFrame, limit: int = 40) -> None:
    if frame.empty:
        return
    html = '<div class="card">'
    for r in frame.head(limit).itertuples(index=False):
        html += f"""
        <div class="rank-row">
          <div class="rank-n">#{int(r.Rank)}</div>
          <div class="team">{r.Team}</div>
          <div><b>{int(getattr(r, '_2', r[2]))}</b><br><span class="muted">Oracle Elo</span></div>
          <div class="hide-mobile"><b>{pct(float(getattr(r, '_4', r[4])))}</b><br><span class="muted">90-day form</span></div>
          <div class="hide-mobile"><span class="pill">{int(getattr(r, '_3', r[3]))} matches</span></div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">CS Oracle Pro · {APP_VERSION}</div>
      <h1>Predictions that work<br>on the free API.</h1>
      <p>The core model trains automatically from PandaScore's free historical match results. It combines dynamic Elo, 30/90-day form, opponent quality, series margin, BO3/LAN records, rest, workload and head-to-head. Map and player overlays activate when lawful detailed data is supplied—but daily winner and score predictions never stay locked.</p>
    </div>
    """, unsafe_allow_html=True)


def prediction_for_match(bundle: ModelBundle, match: dict[str, Any], player_data: pd.DataFrame) -> dict[str, Any]:
    core_p, features = core_probability(bundle, match)
    p_adj_logit, overlay_note = player_adjustment(player_data, match["team_a"], match["team_b"])
    final_p = sigmoid(logit(core_p) + p_adj_logit)
    map_p = solve_map_probability(final_p, int(match["best_of"]))
    scores = exact_scores(map_p, int(match["best_of"]))
    best_score = max(scores, key=scores.get)
    a_score, b_score = [int(x) for x in best_score.split("-")]
    winner = match["team_a"] if a_score > b_score else match["team_b"]
    winner_p = final_p if winner == match["team_a"] else 1 - final_p
    confidence, confidence_label = confidence_score(bundle, match, final_p)
    reasons_a, reasons_b = feature_reasons(features, final_p)
    return {
        "core_p": core_p, "final_p": final_p, "map_p": map_p, "scores": scores,
        "score": best_score, "winner": winner, "winner_p": winner_p,
        "confidence": confidence, "confidence_label": confidence_label,
        "features": features, "reasons_a": reasons_a, "reasons_b": reasons_b,
        "overlay_note": overlay_note,
    }


hero()

with st.sidebar:
    st.header("Data and controls")
    token = st.text_input("PandaScore token", value=token_from_secrets(), type="password")
    history_days = st.slider("History window", 240, 720, DEFAULT_HISTORY_DAYS, 30)
    history_pages = st.slider("Maximum history pages", 6, 20, DEFAULT_HISTORY_PAGES, 1)
    uploaded_maps = st.file_uploader("Optional map history CSV", type=["csv"], help="Columns: date, team_a, team_b, map, winner")
    uploaded_players = st.file_uploader("Optional 3-month player CSV", type=["csv"], help="Columns: team, player, rating, adr")
    st.caption("The free API provides fixtures, final results and rosters. Detailed map and player statistics are optional overlays.")

try:
    user_map_history = clean_uploaded_map_data(uploaded_maps)
    player_data = player_overlay_frame(uploaded_players)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if not token:
    st.info('Add PANDASCORE_TOKEN in Streamlit Secrets: PANDASCORE_TOKEN = "your-token"')
    st.stop()

try:
    with st.spinner("Loading free historical match results and today's fixtures…"):
        raw_past = cached_past(token, history_days, history_pages)
        raw_upcoming = cached_upcoming(token, 3)
except APIError as exc:
    st.error(str(exc))
    st.stop()

history = history_frame(raw_past)
upcoming = upcoming_frame(raw_upcoming)
embedded_maps = map_history_frame(raw_past)
map_history = user_map_history if not user_map_history.empty else embedded_maps

if history.empty:
    st.error("No completed CS2 history was returned. Recheck the token and try Refresh data.")
    st.stop()

with st.spinner("Training the one XGBoost match model…"):
    bundle = train_bundle_cached(history.to_json(orient="split", date_format="iso"))

prediction_tab, match_tab, elo_tab, data_tab = st.tabs(["Daily predictions", "Deep match", "Oracle Elo", "Data & accuracy"])

with prediction_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="card"><div class="eyebrow">History</div><div class="metric">{len(history):,}</div><div class="muted">completed matches</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="eyebrow">Teams</div><div class="metric">{len(bundle.states):,}</div><div class="muted">dynamic Elo profiles</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="eyebrow">Upcoming</div><div class="metric">{len(upcoming):,}</div><div class="muted">next three days</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card"><div class="eyebrow">Map rows</div><div class="metric">{len(map_history):,}</div><div class="muted">optional veto intelligence</div></div>', unsafe_allow_html=True)

    if bundle.model is None:
        st.warning(f"Only {len(history)} usable completed matches were found. Predictions use dynamic Elo until at least {MIN_MODEL_MATCHES} matches are available.")
    else:
        st.success("The free-data XGBoost model is trained. Predictions are active—no paid historical plan required.")

    if upcoming.empty:
        st.info("No upcoming CS2 matches were returned for the next three days.")
    else:
        st.markdown("### Ranked daily prediction board")
        board = []
        for r in upcoming.head(30).to_dict("records"):
            pred = prediction_for_match(bundle, r, player_data)
            board.append({
                "Start": r["date"], "Match": f'{r["team_a"]} vs {r["team_b"]}', "Event": r["event"],
                "BO": r["best_of"], "Prediction": pred["winner"], "Score": pred["score"],
                "Win probability": 100.0 * pred["winner_p"], "Confidence": pred["confidence"],
            })
        board_df = pd.DataFrame(board).sort_values(["Confidence", "Win probability"], ascending=False)
        st.dataframe(
            board_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Start": st.column_config.DatetimeColumn(format="DD MMM HH:mm"),
                "Win probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=100.0),
                "Confidence": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100),
            },
        )
        st.download_button("Download today's predictions", board_df.to_csv(index=False).encode(), "cs_oracle_daily_predictions.csv", "text/csv", use_container_width=True)

with match_tab:
    if upcoming.empty:
        st.info("No match available to analyse.")
    else:
        labels = [f'{r.date.strftime("%d %b %H:%M UTC")} · {r.team_a} vs {r.team_b} · BO{r.best_of}' for r in upcoming.itertuples(index=False)]
        selected_label = st.selectbox("Choose match", labels)
        selected = upcoming.iloc[labels.index(selected_label)].to_dict()
        pred = prediction_for_match(bundle, selected, player_data)

        left, right = st.columns([1.25, 1])
        with left:
            st.markdown(f"### {selected['team_a']} vs {selected['team_b']}")
            st.caption(f"{selected['event']} · BO{selected['best_of']} · {selected['date'].strftime('%d %b %Y %H:%M UTC')}")
            st.markdown(f"""
            <div class="match-card">
              <div class="eyebrow">Oracle call</div>
              <div class="winner">{pred['winner']} {pred['score']}</div>
              <div style="margin-top:.4rem"><span class="pill">{pct(pred['winner_p'])} win chance</span><span class="pill">{pred['confidence_label']} confidence · {pred['confidence']}/100</span></div>
            </div>
            """, unsafe_allow_html=True)
            p_a = pred["final_p"]
            fig = go.Figure(go.Bar(
                x=[selected["team_a"], selected["team_b"]],
                y=[p_a, 1 - p_a],
                text=[pct(p_a), pct(1 - p_a)],
                textposition="auto",
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### Exact score distribution")
            score_df = pd.DataFrame([{"Score": k, "Probability": 100.0 * v} for k, v in pred["scores"].items()])
            st.dataframe(score_df, hide_index=True, use_container_width=True, column_config={"Probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=100.0)})
            st.caption(f"Core model before optional lineup overlay: {pct(pred['core_p'])} for {selected['team_a']}.")
            st.caption(pred["overlay_note"])

        st.markdown("### Why the model ranks it this way")
        a_col, b_col = st.columns(2)
        with a_col:
            st.markdown(f"#### {selected['team_a']}")
            for reason in pred["reasons_a"]:
                st.write(f"• {reason}")
        with b_col:
            st.markdown(f"#### {selected['team_b']}")
            for reason in pred["reasons_b"]:
                st.write(f"• {reason}")

        st.markdown("### Projected maps and veto")
        projected, map_note = projected_veto(map_history, selected["team_a"], selected["team_b"], selected["date"], int(selected["best_of"]), pred["map_p"])
        map_df = pd.DataFrame([
            {"Map": m["map"], "Picker": m["picker"], selected["team_a"] + " win chance": 100.0 * m["team_a_p"]}
            for m in projected
        ])
        if not map_df.empty:
            st.dataframe(map_df, hide_index=True, use_container_width=True, column_config={selected["team_a"] + " win chance": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=100.0)})
        st.caption(map_note)

        st.markdown("### Current lineups")
        team_cols = st.columns(2)
        for col, team_name, team_id in [
            (team_cols[0], selected["team_a"], selected["team_a_id"]),
            (team_cols[1], selected["team_b"], selected["team_b_id"]),
        ]:
            with col:
                st.markdown(f"#### {team_name}")
                try:
                    detail = cached_team(token, team_id)
                    players = team_players(detail)
                    if players.empty:
                        st.caption("Current roster not returned by the fixture feed.")
                    else:
                        st.dataframe(players, hide_index=True, use_container_width=True)
                except APIError:
                    st.caption("Roster request unavailable.")

        receipt = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "app_version": APP_VERSION,
            "match_id": selected["match_id"],
            "match": f"{selected['team_a']} vs {selected['team_b']}",
            "prediction": pred["winner"],
            "score": pred["score"],
            "probabilities": {selected["team_a"]: pred["final_p"], selected["team_b"]: 1 - pred["final_p"]},
            "history_matches": len(history),
            "model": "XGBoost binary match model with dynamic pre-match state features",
        }
        canonical = json.dumps(receipt, sort_keys=True).encode()
        receipt["sha256"] = hashlib.sha256(canonical).hexdigest()
        st.download_button("Download timestamped prediction receipt", json.dumps(receipt, indent=2).encode(), f"prediction_{selected['match_id']}.json", "application/json", use_container_width=True)

with elo_tab:
    st.markdown("### Oracle Elo ranking")
    st.caption("Ratings update chronologically after each completed series. Recent opponent quality, margins and event tier affect the update size.")
    rankings = elo_table(bundle)
    render_elo_list(rankings)
    if not rankings.empty:
        st.download_button("Download Elo ranking", rankings.to_csv(index=False).encode(), "cs_oracle_elo.csv", "text/csv", use_container_width=True)

with data_tab:
    st.markdown("### What changed in this fixed version")
    st.markdown("""
    - **No prediction lock:** the core model trains from free completed-match results, not paid map endpoints.
    - **One learned model:** XGBoost predicts the series winner; Elo and form are input features, not separate prediction models.
    - **Chronological features:** every training row is calculated before that match result updates the team state.
    - **Exact score engine:** the model's series probability is converted into an implied per-map probability and mathematically valid BO1/BO3/BO5 score distribution.
    - **Optional depth:** lawful map/player CSV data improves veto and lineup analysis, but is never required for basic predictions.
    """)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training matches", f"{len(history):,}")
    m2.metric("Newest result", history["date"].max().strftime("%d %b %Y"))
    m3.metric("XGBoost rows", f"{bundle.training_rows:,}")
    m4.metric("Map rows", f"{len(map_history):,}")

    if bundle.metrics:
        st.markdown("### Time-split backtest")
        a, b, c, d = st.columns(4)
        a.metric("Winner accuracy", f"{bundle.metrics['accuracy'] * 100:.1f}%")
        b.metric("Brier score", f"{bundle.metrics['brier']:.3f}")
        c.metric("Log loss", f"{bundle.metrics['logloss']:.3f}")
        d.metric("ROC AUC", f"{bundle.metrics['auc']:.3f}")
        st.caption(f"The newest {bundle.metrics['test_rows']:,} mirrored rows were held out chronologically. This is more honest than random train/test mixing, but it is still not a guarantee of future accuracy.")
    else:
        st.info("A full time-split backtest needs more history. Predictions still use dynamic Elo until enough completed matches are available.")

    if bundle.feature_gain:
        gain_df = pd.DataFrame([{"Feature": k, "Share of XGBoost gain": 100.0 * v} for k, v in bundle.feature_gain.items()]).sort_values("Share of XGBoost gain", ascending=False)
        st.markdown("### Model feature importance")
        st.dataframe(gain_df, hide_index=True, use_container_width=True, column_config={"Share of XGBoost gain": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=100.0)})

    st.markdown("### Optional data templates")
    map_template = "date,team_a,team_b,map,winner\n2026-07-01,Team A,Team B,Mirage,Team A\n"
    player_template = "team,player,rating,adr\nTeam A,Player1,1.12,78.4\n"
    x, y = st.columns(2)
    x.download_button("Download map template", map_template.encode(), "map_history_template.csv", "text/csv", use_container_width=True)
    y.download_button("Download player template", player_template.encode(), "player_form_template.csv", "text/csv", use_container_width=True)

st.caption("CS Oracle Pro provides probabilistic estimates, not certainty. Regenerate any API token that has been shared publicly.")
