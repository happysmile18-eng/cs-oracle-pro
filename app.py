from __future__ import annotations

import hashlib
import io
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

APP_VERSION = "7.0.0-today-intelligence"
TARGET_PRECISION = 0.70
API_ROOT = "https://api.pandascore.co"
DEFAULT_HISTORY_DAYS = 820
DEFAULT_HISTORY_PAGES = 28
MIN_MODEL_MATCHES = 160
ACTIVE_POOL = ["Ancient", "Anubis", "Cache", "Dust 2", "Inferno", "Mirage", "Nuke"]
SCORE_CLASSES = ["1-0", "0-1", "2-0", "2-1", "0-2", "1-2", "3-0", "3-1", "3-2", "0-3", "1-3", "2-3"]

st.set_page_config(page_title="CS Oracle Pure Prediction", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{--red:#ff596b;--red2:#ff8793;--ink:#f6f7fb;--muted:#929bac;--panel:#0c1017;--line:#232b38;--green:#61dda2;--gold:#ffd166;--blue:#80bfff}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,sans-serif}
.block-container{max-width:1260px;padding-top:1rem;padding-bottom:3.5rem}
#MainMenu,footer{visibility:hidden}[data-testid="stHeader"]{background:rgba(6,8,12,.76);backdrop-filter:blur(12px)}
.hero{padding:1.45rem;border:1px solid var(--line);border-radius:25px;background:radial-gradient(circle at 88% 7%,rgba(255,89,107,.24),transparent 36%),linear-gradient(145deg,#141a26,#080b11);margin-bottom:1rem}
.eyebrow{color:var(--red2);font-size:.72rem;font-weight:850;letter-spacing:.18em;text-transform:uppercase}.hero h1{font-size:clamp(2.15rem,7vw,4.8rem);line-height:.94;margin:.4rem 0 .55rem;letter-spacing:-.055em}.hero p{color:var(--muted);max-width:920px;margin:0}
.card{padding:1.05rem 1.1rem;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,#111620,#090c12);height:100%}.metric{font-size:2rem;font-weight:900;letter-spacing:-.04em}.muted{color:var(--muted)}
.callout{padding:1rem 1.1rem;border-radius:16px;border:1px solid #293344;background:#0e141d}.good{color:var(--green)}.warn{color:var(--gold)}.bad{color:#ff7d89}
.match-card{padding:1rem;border:1px solid var(--line);border-radius:18px;background:#0c1118;margin:.55rem 0}.winner{font-size:1.55rem;font-weight:900;letter-spacing:-.03em}.pill{display:inline-block;padding:.26rem .55rem;border:1px solid #30394a;border-radius:999px;font-size:.75rem;color:#d1d7e3;margin:.25rem .3rem 0 0}.hot{color:#ff9f43}.superhot{color:#ff5d6c;font-weight:900}.cold{color:#80bfff}.precision{padding:.45rem .65rem;border-radius:12px;background:#121a25;border:1px solid #293447;font-weight:850}
.rank-row{display:grid;grid-template-columns:52px minmax(150px,1.25fr) .85fr .85fr .85fr .85fr;gap:.7rem;align-items:center;padding:.78rem .9rem;border-bottom:1px solid #1d2430}.rank-row:last-child{border-bottom:0}.rank-n{font-weight:900;font-size:1.2rem;color:#ff7d89}.team{font-weight:850}
.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;padding:.85rem;border:1px solid #293344;border-radius:14px;background:#080b10;overflow:auto}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:16px;overflow:hidden}.stButton>button,.stDownloadButton>button{border-radius:13px;font-weight:800;min-height:43px}
@media(max-width:760px){.rank-row{grid-template-columns:38px 1fr 1fr}.hide-mobile{display:none}.block-container{padding-left:.7rem;padding-right:.7rem}}
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


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def local_now(timezone_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(timezone_name))
    except Exception:
        return datetime.now(timezone.utc)


def normalise_event(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return text[:180]


def normalise_team(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "natus vincere": "NAVI", "team vitality": "Vitality", "g2 esports": "G2",
        "faze clan": "FaZe", "mousesports": "MOUZ", "team liquid": "Liquid",
        "ninjas in pyjamas": "NIP", "virtus.pro": "Virtus.pro",
    }
    return aliases.get(text.lower(), text)


def normalise_map(value: Any) -> str:
    text = str(value or "").replace("de_", "").replace("_", " ").strip().lower()
    aliases = {
        "dust2": "Dust 2", "dust ii": "Dust 2", "dust 2": "Dust 2",
        "ancient": "Ancient", "anubis": "Anubis", "cache": "Cache", "inferno": "Inferno",
        "mirage": "Mirage", "nuke": "Nuke", "overpass": "Overpass", "train": "Train", "vertigo": "Vertigo",
    }
    return aliases.get(text, text.title())


def logistic_elo(diff: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def logit(p: float) -> float:
    p = min(max(float(p), 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def softmax(log_values: np.ndarray) -> np.ndarray:
    z = log_values - np.max(log_values, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


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
        last_error = "Unknown API error"
        for attempt in range(5):
            try:
                response = requests.get(url, headers=headers, params=params or {}, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"Connection failed: {exc}"
                time.sleep(1.0 + 1.1 * attempt)
                continue
            if response.status_code == 429:
                last_error = "PandaScore rate limit reached"
                time.sleep(1.5 + 1.5 * attempt)
                continue
            if response.status_code in (401, 403):
                raise APIError(f"PandaScore returned {response.status_code}. Check the token and plan access.")
            if response.status_code >= 400:
                raise APIError(f"PandaScore returned {response.status_code}: {response.text[:180]}")
            return response.json()
        raise APIError(last_error)

    def paged(self, path: str, params: dict[str, Any] | None = None, max_pages: int = 10) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            request_params = dict(params or {})
            request_params.update({"page": page, "per_page": 100})
            batch = self._get(path, request_params)
            if not isinstance(batch, list):
                break
            rows.extend(batch)
            if len(batch) < 100:
                break
        return rows

    def past_matches(self, days: int, pages: int) -> list[dict[str, Any]]:
        end = datetime.now(timezone.utc).date().isoformat()
        start = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        return self.paged("/csgo/matches/past", {"range[begin_at]": f"{start},{end}", "sort": "-begin_at"}, pages)

    def upcoming_matches(self, days: int = 7) -> list[dict[str, Any]]:
        start = datetime.now(timezone.utc).date().isoformat()
        end = (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()
        return self.paged("/csgo/matches/upcoming", {"range[begin_at]": f"{start},{end}", "sort": "begin_at"}, 5)

    def team(self, team_id: int | str) -> dict[str, Any]:
        obj = self._get(f"/teams/{team_id}")
        return obj if isinstance(obj, dict) else {}


@st.cache_data(ttl=6 * 3600, max_entries=8, show_spinner=False)
def cached_past(token: str, days: int, pages: int) -> list[dict[str, Any]]:
    return PandaClient(token).past_matches(days, pages)


@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def cached_upcoming(token: str, days: int) -> list[dict[str, Any]]:
    return PandaClient(token).upcoming_matches(days)


@st.cache_data(ttl=12 * 3600, max_entries=120, show_spinner=False)
def cached_team(token: str, team_id: int | str) -> dict[str, Any]:
    return PandaClient(token).team(team_id)


def extract_teams(match: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    teams: list[dict[str, Any]] = []
    for opponent in match.get("opponents") or []:
        obj = opponent.get("opponent") or {}
        if obj.get("id") and obj.get("name"):
            teams.append(obj)
    return (teams[0], teams[1]) if len(teams) >= 2 else None


def tier_value(match: dict[str, Any]) -> float:
    tournament = match.get("tournament") or {}
    serie = match.get("serie") or {}
    raw = str(tournament.get("tier") or serie.get("tier") or "").lower().strip()
    mapping = {"s": 1.0, "a": 0.84, "b": 0.66, "c": 0.47, "d": 0.30}
    if raw in mapping:
        return mapping[raw]
    name = " ".join([
        str((match.get("league") or {}).get("name") or ""), str(serie.get("full_name") or serie.get("name") or ""),
        str(tournament.get("name") or ""),
    ]).lower()
    if any(k in name for k in ["major", "iem", "blast premier", "esl pro league", "pgl", "esports world cup"]):
        return 0.96
    if any(k in name for k in ["challenger", "masters", "cct", "qualifier"]):
        return 0.58
    if any(k in name for k in ["academy", "open qualifier", "regional"]):
        return 0.40
    return 0.48


def is_lan(match: dict[str, Any]) -> int:
    tournament = match.get("tournament") or {}
    return int(bool(match.get("is_lan") or tournament.get("is_lan") or match.get("location") or tournament.get("location")))


def result_scores(match: dict[str, Any], team_a_id: int, team_b_id: int) -> tuple[int, int]:
    scores: dict[int, int] = {}
    for item in match.get("results") or []:
        team_id = safe_int(item.get("team_id"), -1)
        if team_id >= 0:
            scores[team_id] = safe_int(item.get("score"), 0)
    return scores.get(team_a_id, 0), scores.get(team_b_id, 0)


def extract_game_winners(match: dict[str, Any]) -> list[int]:
    output: list[tuple[int, int]] = []
    for game in match.get("games") or []:
        winner = game.get("winner") or {}
        winner_id = safe_int(winner.get("id") or game.get("winner_id"), -1)
        if winner_id >= 0:
            output.append((safe_int(game.get("position"), len(output) + 1), winner_id))
    return [winner for _, winner in sorted(output)]


def exact_score_class(score_a: int, score_b: int, best_of: int) -> str | None:
    value = f"{score_a}-{score_b}"
    if best_of == 1 and value in {"1-0", "0-1"}:
        return value
    if best_of == 3 and value in {"2-0", "2-1", "0-2", "1-2"}:
        return value
    if best_of == 5 and value in {"3-0", "3-1", "3-2", "0-3", "1-3", "2-3"}:
        return value
    return None


def mirror_score(score: str) -> str:
    left, right = score.split("-")
    return f"{right}-{left}"


def score_is_a_win(score: str) -> bool:
    left, right = score.split("-")
    return int(left) > int(right)


def valid_scores(best_of: int) -> list[str]:
    if best_of == 1:
        return ["1-0", "0-1"]
    if best_of == 5:
        return ["3-0", "3-1", "3-2", "0-3", "1-3", "2-3"]
    return ["2-0", "2-1", "0-2", "1-2"]


def match_row(match: dict[str, Any], completed_only: bool) -> dict[str, Any] | None:
    pair = extract_teams(match)
    if pair is None:
        return None
    team_a, team_b = pair
    team_a_id, team_b_id = safe_int(team_a.get("id"), -1), safe_int(team_b.get("id"), -1)
    if team_a_id < 0 or team_b_id < 0:
        return None
    winner_id = safe_int(match.get("winner_id"), -1)
    if completed_only and winner_id not in (team_a_id, team_b_id):
        return None
    score_a, score_b = result_scores(match, team_a_id, team_b_id)
    if completed_only and score_a == score_b == 0:
        winners = extract_game_winners(match)
        score_a = sum(w == team_a_id for w in winners)
        score_b = sum(w == team_b_id for w in winners)
    best_of = safe_int(match.get("number_of_games"), 3)
    if best_of not in (1, 3, 5):
        best_of = 3
    if completed_only and exact_score_class(score_a, score_b, best_of) is None:
        return None
    tournament = match.get("tournament") or {}
    serie = match.get("serie") or {}
    league = match.get("league") or {}
    return {
        "match_id": match.get("id"), "date": parse_dt(match.get("begin_at") or match.get("scheduled_at")),
        "team_a": normalise_team(team_a.get("name")), "team_b": normalise_team(team_b.get("name")),
        "team_a_id": team_a_id, "team_b_id": team_b_id,
        "team_a_image": team_a.get("image_url"), "team_b_image": team_b.get("image_url"),
        "winner_a": int(winner_id == team_a_id) if winner_id in (team_a_id, team_b_id) else None,
        "score_a": score_a, "score_b": score_b, "score_class": exact_score_class(score_a, score_b, best_of),
        "best_of": best_of, "lan": is_lan(match), "tier": tier_value(match),
        "event": tournament.get("name") or serie.get("full_name") or league.get("name") or "Unknown event",
        "status": match.get("status") or "not_started", "raw": match,
    }


def history_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [row for match in matches if (row := match_row(match, True)) is not None]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("match_id", keep="last").reset_index(drop=True)


def upcoming_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [row for match in matches if (row := match_row(match, False)) is not None]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values("date").drop_duplicates("match_id", keep="last")
    return frame[frame["date"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)].reset_index(drop=True)


@dataclass
class TeamState:
    core_elo: float = 1500.0
    fast_elo: float = 1500.0
    lan_elo: float = 1500.0
    online_elo: float = 1500.0
    bo1_elo: float = 1500.0
    bo3_elo: float = 1500.0
    bo5_elo: float = 1500.0
    matches: int = 0
    last_date: pd.Timestamp | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=140))
    rating_points: deque = field(default_factory=lambda: deque(maxlen=180))
    opponent_elo_ewma: float = 1500.0
    margin_ewma: float = 0.0
    performance_ewma: float = 0.0
    streak: int = 0
    peak_oracle: float = 1500.0


SIGNED_FEATURES = {
    "oracle_elo_diff", "core_elo_diff", "fast_elo_diff", "context_elo_diff", "format_elo_diff",
    "elo_momentum_30_diff", "elo_momentum_90_diff", "winrate_14_diff", "winrate_30_diff",
    "winrate_60_diff", "winrate_90_diff", "form5_diff", "form10_diff", "form20_diff",
    "performance_30_diff", "performance_90_diff", "opp_strength_diff", "margin_30_diff",
    "margin_90_diff", "sweep_win_diff", "sweep_loss_diff", "experience_diff", "rest_diff",
    "rust_diff", "fatigue_diff", "workload7_diff", "workload14_diff", "workload30_diff",
    "same_day_diff", "h2h_diff", "bo_form_diff", "lan_form_diff", "streak_diff",
    "event_form_diff", "event_matches_diff", "favorite_conversion_diff", "underdog_upset_diff",
    "close_match_diff", "high_tier_form_diff", "consistency_diff",
}

FEATURES = [
    "oracle_elo_diff", "core_elo_diff", "fast_elo_diff", "context_elo_diff", "format_elo_diff",
    "elo_momentum_30_diff", "elo_momentum_90_diff", "winrate_14_diff", "winrate_30_diff",
    "winrate_60_diff", "winrate_90_diff", "form5_diff", "form10_diff", "form20_diff",
    "performance_30_diff", "performance_90_diff", "opp_strength_diff", "margin_30_diff",
    "margin_90_diff", "sweep_win_diff", "sweep_loss_diff", "experience_diff", "rest_diff",
    "rest_abs", "rust_diff", "fatigue_diff", "workload7_diff", "workload14_diff", "workload30_diff",
    "same_day_diff", "h2h_diff", "bo_form_diff", "lan_form_diff", "streak_diff",
    "event_form_diff", "event_matches_diff", "favorite_conversion_diff", "underdog_upset_diff",
    "close_match_diff", "high_tier_form_diff", "consistency_diff", "sample_min",
    "best_of", "lan", "tier",
]


def regress_rating(rating: float, last_date: pd.Timestamp | None, when: pd.Timestamp, half_life_days: float) -> float:
    if last_date is None:
        return 1500.0
    idle_days = max(0.0, (when - last_date).total_seconds() / 86400.0)
    retention = 0.5 ** (idle_days / half_life_days)
    return 1500.0 + (rating - 1500.0) * retention


def effective_components(state: TeamState, when: pd.Timestamp, best_of: int, lan: int) -> dict[str, float]:
    core = regress_rating(state.core_elo, state.last_date, when, 520.0)
    fast = regress_rating(state.fast_elo, state.last_date, when, 125.0)
    context_raw = state.lan_elo if lan else state.online_elo
    context = regress_rating(context_raw, state.last_date, when, 420.0)
    format_raw = state.bo1_elo if best_of == 1 else state.bo5_elo if best_of == 5 else state.bo3_elo
    format_elo = regress_rating(format_raw, state.last_date, when, 440.0)
    oracle = 0.55 * core + 0.25 * fast + 0.12 * context + 0.08 * format_elo
    return {"core": core, "fast": fast, "context": context, "format": format_elo, "oracle": oracle}


def ranking_components(state: TeamState, when: pd.Timestamp) -> dict[str, float]:
    core = regress_rating(state.core_elo, state.last_date, when, 520.0)
    fast = regress_rating(state.fast_elo, state.last_date, when, 125.0)
    oracle = 0.66 * core + 0.34 * fast
    return {"core": core, "fast": fast, "oracle": oracle}


def recent_records(state: TeamState, when: pd.Timestamp, days: int | None = None, last_n: int | None = None, best_of: int | None = None, lan: int | None = None) -> list[dict[str, Any]]:
    records = list(state.recent)
    if days is not None:
        cutoff = when - pd.Timedelta(days=days)
        records = [record for record in records if record["date"] >= cutoff]
    if best_of is not None:
        records = [record for record in records if record["best_of"] == best_of]
    if lan is not None:
        records = [record for record in records if record["lan"] == lan]
    if last_n is not None:
        records = records[-last_n:]
    return records


def weighted_mean(records: Iterable[dict[str, Any]], key: str, when: pd.Timestamp, half_life: float, prior: float, prior_weight: float) -> float:
    numerator = prior * prior_weight
    denominator = prior_weight
    for record in records:
        age = max(0.0, (when - record["date"]).total_seconds() / 86400.0)
        weight = 0.5 ** (age / half_life)
        numerator += weight * safe_float(record.get(key), prior)
        denominator += weight
    return numerator / max(denominator, 1e-9)


def win_rate(state: TeamState, when: pd.Timestamp, days: int | None = None, last_n: int | None = None, best_of: int | None = None, lan: int | None = None) -> float:
    records = recent_records(state, when, days, last_n, best_of, lan)
    half_life = min(75.0, max(18.0, (days or 90) / 2.0))
    return weighted_mean(records, "result", when, half_life, 0.5, 4.5)


def performance_rate(state: TeamState, when: pd.Timestamp, days: int) -> float:
    records = recent_records(state, when, days=days)
    return weighted_mean(records, "performance", when, max(18.0, days / 2.0), 0.0, 5.0)


def margin_rate(state: TeamState, when: pd.Timestamp, days: int) -> float:
    records = recent_records(state, when, days=days)
    return weighted_mean(records, "margin", when, max(18.0, days / 2.0), 0.0, 5.0)


def sweep_rate(state: TeamState, when: pd.Timestamp, win: bool) -> float:
    records = recent_records(state, when, days=180, best_of=3)
    key = "sweep_win" if win else "sweep_loss"
    return weighted_mean(records, key, when, 80.0, 0.22, 6.0)


def workload(state: TeamState, when: pd.Timestamp, days: int) -> int:
    cutoff = when - pd.Timedelta(days=days)
    return sum(record["date"] >= cutoff for record in state.recent)


def same_day_matches(state: TeamState, when: pd.Timestamp) -> int:
    return sum(record["date"].date() == when.date() for record in state.recent)


def rest_days(state: TeamState, when: pd.Timestamp) -> float:
    if state.last_date is None:
        return 28.0
    return float(min(60.0, max(0.0, (when - state.last_date).total_seconds() / 86400.0)))


def rust_index(state: TeamState, when: pd.Timestamp) -> float:
    rest = rest_days(state, when)
    return max(0.0, rest - 18.0) / 18.0


def fatigue_index(state: TeamState, when: pd.Timestamp) -> float:
    return 0.24 * max(0, workload(state, when, 7) - 3) + 0.10 * max(0, workload(state, when, 14) - 6) + 0.8 * same_day_matches(state, when)


def rating_momentum(state: TeamState, when: pd.Timestamp, days: int) -> float:
    points = [point for point in state.rating_points if point["date"] <= when]
    if not points:
        return 0.0
    current = points[-1]["oracle"]
    cutoff = when - pd.Timedelta(days=days)
    previous = next((point["oracle"] for point in reversed(points) if point["date"] <= cutoff), points[0]["oracle"])
    return float(current - previous)


def decayed_h2h(h2h: dict[tuple[str, str], deque], team_a: str, team_b: str, when: pd.Timestamp) -> float:
    records = list(h2h[(team_a, team_b)])
    numerator, denominator = 0.0, 3.0
    for record in records:
        age = max(0.0, (when - record["date"]).total_seconds() / 86400.0)
        weight = 0.5 ** (age / 120.0)
        numerator += weight * (2.0 * record["result"] - 1.0)
        denominator += weight
    return numerator / denominator


def filtered_rate(state: TeamState, when: pd.Timestamp, predicate: Any, prior: float = 0.5, prior_weight: float = 4.0, half_life: float = 75.0) -> tuple[float, int]:
    records = [record for record in state.recent if predicate(record)]
    return weighted_mean(records, "result", when, half_life, prior, prior_weight), len(records)


def event_form(state: TeamState, when: pd.Timestamp, event: str) -> tuple[float, int]:
    event_key = normalise_event(event)
    if not event_key or event_key == "unknown event":
        return 0.5, 0
    cutoff = when - pd.Timedelta(days=50)
    records = [record for record in state.recent if record.get("event_key") == event_key and record["date"] >= cutoff]
    return weighted_mean(records, "result", when, 18.0, 0.5, 2.8), len(records)


def favorite_conversion(state: TeamState, when: pd.Timestamp) -> float:
    rate, _ = filtered_rate(state, when, lambda r: r.get("expected", 0.5) >= 0.60, prior_weight=5.5, half_life=95.0)
    return rate


def underdog_upset_rate(state: TeamState, when: pd.Timestamp) -> float:
    rate, _ = filtered_rate(state, when, lambda r: r.get("expected", 0.5) <= 0.40, prior=0.28, prior_weight=6.0, half_life=95.0)
    return rate


def close_match_rate(state: TeamState, when: pd.Timestamp) -> float:
    rate, _ = filtered_rate(state, when, lambda r: 0.42 <= r.get("expected", 0.5) <= 0.58, prior_weight=5.0, half_life=90.0)
    return rate


def high_tier_rate(state: TeamState, when: pd.Timestamp) -> float:
    rate, _ = filtered_rate(state, when, lambda r: r.get("tier", 0.0) >= 0.72, prior_weight=6.5, half_life=120.0)
    return rate


def consistency_score(state: TeamState, when: pd.Timestamp) -> float:
    records = recent_records(state, when, days=90, last_n=18)
    if len(records) < 3:
        return 0.0
    values = np.asarray([safe_float(record.get("performance"), 0.0) for record in records], dtype=float)
    # Higher is better: stable performance near or above expectation.
    return float(np.clip(values.mean() - 0.60 * values.std(ddof=0), -1.0, 1.0))


def build_features(states: dict[str, TeamState], h2h: dict[tuple[str, str], deque], when: pd.Timestamp, team_a: str, team_b: str, best_of: int, lan: int, tier: float, event: str = "") -> dict[str, float]:
    a, b = states[team_a], states[team_b]
    ca, cb = effective_components(a, when, best_of, lan), effective_components(b, when, best_of, lan)
    rest_a, rest_b = rest_days(a, when), rest_days(b, when)
    event_a, event_n_a = event_form(a, when, event)
    event_b, event_n_b = event_form(b, when, event)
    return {
        "oracle_elo_diff": ca["oracle"] - cb["oracle"], "core_elo_diff": ca["core"] - cb["core"],
        "fast_elo_diff": ca["fast"] - cb["fast"], "context_elo_diff": ca["context"] - cb["context"],
        "format_elo_diff": ca["format"] - cb["format"],
        "elo_momentum_30_diff": rating_momentum(a, when, 30) - rating_momentum(b, when, 30),
        "elo_momentum_90_diff": rating_momentum(a, when, 90) - rating_momentum(b, when, 90),
        "winrate_14_diff": win_rate(a, when, days=14) - win_rate(b, when, days=14),
        "winrate_30_diff": win_rate(a, when, days=30) - win_rate(b, when, days=30),
        "winrate_60_diff": win_rate(a, when, days=60) - win_rate(b, when, days=60),
        "winrate_90_diff": win_rate(a, when, days=90) - win_rate(b, when, days=90),
        "form5_diff": win_rate(a, when, last_n=5) - win_rate(b, when, last_n=5),
        "form10_diff": win_rate(a, when, last_n=10) - win_rate(b, when, last_n=10),
        "form20_diff": win_rate(a, when, last_n=20) - win_rate(b, when, last_n=20),
        "performance_30_diff": performance_rate(a, when, 30) - performance_rate(b, when, 30),
        "performance_90_diff": performance_rate(a, when, 90) - performance_rate(b, when, 90),
        "opp_strength_diff": a.opponent_elo_ewma - b.opponent_elo_ewma,
        "margin_30_diff": margin_rate(a, when, 30) - margin_rate(b, when, 30),
        "margin_90_diff": margin_rate(a, when, 90) - margin_rate(b, when, 90),
        "sweep_win_diff": sweep_rate(a, when, True) - sweep_rate(b, when, True),
        "sweep_loss_diff": sweep_rate(a, when, False) - sweep_rate(b, when, False),
        "experience_diff": math.log1p(a.matches) - math.log1p(b.matches),
        "rest_diff": rest_a - rest_b, "rest_abs": abs(rest_a - rest_b),
        "rust_diff": rust_index(a, when) - rust_index(b, when), "fatigue_diff": fatigue_index(a, when) - fatigue_index(b, when),
        "workload7_diff": float(workload(a, when, 7) - workload(b, when, 7)),
        "workload14_diff": float(workload(a, when, 14) - workload(b, when, 14)),
        "workload30_diff": float(workload(a, when, 30) - workload(b, when, 30)),
        "same_day_diff": float(same_day_matches(a, when) - same_day_matches(b, when)),
        "h2h_diff": decayed_h2h(h2h, team_a, team_b, when),
        "bo_form_diff": win_rate(a, when, days=180, best_of=best_of) - win_rate(b, when, days=180, best_of=best_of),
        "lan_form_diff": win_rate(a, when, days=240, lan=lan) - win_rate(b, when, days=240, lan=lan),
        "streak_diff": float(a.streak - b.streak),
        "event_form_diff": event_a - event_b,
        "event_matches_diff": float(math.log1p(event_n_a) - math.log1p(event_n_b)),
        "favorite_conversion_diff": favorite_conversion(a, when) - favorite_conversion(b, when),
        "underdog_upset_diff": underdog_upset_rate(a, when) - underdog_upset_rate(b, when),
        "close_match_diff": close_match_rate(a, when) - close_match_rate(b, when),
        "high_tier_form_diff": high_tier_rate(a, when) - high_tier_rate(b, when),
        "consistency_diff": consistency_score(a, when) - consistency_score(b, when),
        "sample_min": float(min(a.matches, b.matches)),
        "best_of": float(best_of), "lan": float(lan), "tier": float(tier),
    }


def mirror_features(features: dict[str, float]) -> dict[str, float]:
    mirrored = dict(features)
    for key in SIGNED_FEATURES:
        mirrored[key] = -mirrored.get(key, 0.0)
    return mirrored


def update_ratings(states: dict[str, TeamState], h2h: dict[tuple[str, str], deque], ledger: list[dict[str, Any]], when: pd.Timestamp, team_a: str, team_b: str, a_win: int, score_a: int, score_b: int, best_of: int, lan: int, tier: float, event: str) -> None:
    a, b = states[team_a], states[team_b]
    a_comp, b_comp = effective_components(a, when, best_of, lan), effective_components(b, when, best_of, lan)
    expected_a = logistic_elo(a_comp["oracle"] - b_comp["oracle"])
    needed = 1 if best_of == 1 else best_of // 2 + 1
    map_share_a = score_a / max(1.0, score_a + score_b)
    signed_margin = (score_a - score_b) / needed
    margin_multiplier = 1.0 + 0.20 * math.log1p(abs(score_a - score_b)) + (0.07 if score_b == 0 or score_a == 0 else 0.0)
    tier_multiplier = 0.82 + 0.43 * tier
    experience_multiplier = 1.32 - 0.45 * min(1.0, min(a.matches, b.matches) / 70.0)
    base_k = 27.0 * tier_multiplier * experience_multiplier * margin_multiplier
    core_delta = base_k * (a_win - expected_a)
    fast_delta = (49.0 * tier_multiplier * margin_multiplier) * (a_win - expected_a)
    context_delta = (34.0 * tier_multiplier * margin_multiplier) * (a_win - expected_a)
    format_delta = (31.0 * tier_multiplier * margin_multiplier) * (a_win - expected_a)

    # First regress stored ratings to the match date, then apply the new evidence.
    a.core_elo, b.core_elo = a_comp["core"] + core_delta, b_comp["core"] - core_delta
    a.fast_elo, b.fast_elo = a_comp["fast"] + fast_delta, b_comp["fast"] - fast_delta
    if lan:
        a.lan_elo, b.lan_elo = a_comp["context"] + context_delta, b_comp["context"] - context_delta
    else:
        a.online_elo, b.online_elo = a_comp["context"] + context_delta, b_comp["context"] - context_delta
    if best_of == 1:
        a.bo1_elo, b.bo1_elo = a_comp["format"] + format_delta, b_comp["format"] - format_delta
    elif best_of == 5:
        a.bo5_elo, b.bo5_elo = a_comp["format"] + format_delta, b_comp["format"] - format_delta
    else:
        a.bo3_elo, b.bo3_elo = a_comp["format"] + format_delta, b_comp["format"] - format_delta

    performance_a = 0.76 * (a_win - expected_a) + 0.24 * (map_share_a - 0.5)
    alpha = 0.13
    a.opponent_elo_ewma = (1 - alpha) * a.opponent_elo_ewma + alpha * b_comp["oracle"]
    b.opponent_elo_ewma = (1 - alpha) * b.opponent_elo_ewma + alpha * a_comp["oracle"]
    a.margin_ewma = 0.84 * a.margin_ewma + 0.16 * signed_margin
    b.margin_ewma = 0.84 * b.margin_ewma - 0.16 * signed_margin
    a.performance_ewma = 0.84 * a.performance_ewma + 0.16 * performance_a
    b.performance_ewma = 0.84 * b.performance_ewma - 0.16 * performance_a

    record_a = {"date": when, "result": float(a_win), "expected": expected_a, "performance": performance_a,
                "opponent_elo": b_comp["oracle"], "margin": signed_margin, "best_of": best_of, "lan": lan,
                "tier": tier, "event_key": normalise_event(event), "sweep_win": float(a_win == 1 and score_b == 0 and best_of >= 3),
                "sweep_loss": float(a_win == 0 and score_a == 0 and best_of >= 3)}
    record_b = {"date": when, "result": float(1 - a_win), "expected": 1 - expected_a, "performance": -performance_a,
                "opponent_elo": a_comp["oracle"], "margin": -signed_margin, "best_of": best_of, "lan": lan,
                "tier": tier, "event_key": normalise_event(event), "sweep_win": float(a_win == 0 and score_a == 0 and best_of >= 3),
                "sweep_loss": float(a_win == 1 and score_b == 0 and best_of >= 3)}
    a.recent.append(record_a); b.recent.append(record_b)
    a.matches += 1; b.matches += 1
    a.last_date = when; b.last_date = when
    a.streak = a.streak + 1 if a_win and a.streak >= 0 else 1 if a_win else a.streak - 1 if a.streak <= 0 else -1
    b.streak = b.streak + 1 if not a_win and b.streak >= 0 else 1 if not a_win else b.streak - 1 if b.streak <= 0 else -1
    h2h[(team_a, team_b)].append({"date": when, "result": int(a_win)})
    h2h[(team_b, team_a)].append({"date": when, "result": int(1 - a_win)})

    after_a = effective_components(a, when, best_of, lan)["oracle"]
    after_b = effective_components(b, when, best_of, lan)["oracle"]
    rank_a = ranking_components(a, when)["oracle"]
    rank_b = ranking_components(b, when)["oracle"]
    a.peak_oracle = max(a.peak_oracle, rank_a); b.peak_oracle = max(b.peak_oracle, rank_b)
    a.rating_points.append({"date": when, "oracle": rank_a, "core": a.core_elo, "fast": a.fast_elo})
    b.rating_points.append({"date": when, "oracle": rank_b, "core": b.core_elo, "fast": b.fast_elo})

    common = {"date": when, "event": event, "best_of": best_of, "lan": lan, "tier": tier,
              "expected_a": expected_a, "k": base_k, "margin_multiplier": margin_multiplier}
    ledger.append({**common, "team": team_a, "opponent": team_b, "result": "W" if a_win else "L",
                   "score": f"{score_a}-{score_b}", "expected": expected_a, "before": a_comp["oracle"], "after": after_a, "delta": after_a - a_comp["oracle"]})
    ledger.append({**common, "team": team_b, "opponent": team_a, "result": "L" if a_win else "W",
                   "score": f"{score_b}-{score_a}", "expected": 1 - expected_a, "before": b_comp["oracle"], "after": after_b, "delta": after_b - b_comp["oracle"]})


@dataclass
class TrainingData:
    x: pd.DataFrame
    labels: list[str]
    dates: np.ndarray
    weights: np.ndarray
    original: np.ndarray
    states: dict[str, TeamState]
    h2h: dict[tuple[str, str], deque]
    ledger: pd.DataFrame


def make_training_data(history: pd.DataFrame) -> TrainingData:
    states: dict[str, TeamState] = defaultdict(TeamState)
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=24))
    ledger: list[dict[str, Any]] = []
    x_rows: list[dict[str, float]] = []
    labels: list[str] = []
    dates: list[pd.Timestamp] = []
    originals: list[bool] = []
    base_weights: list[float] = []
    newest = history["date"].max()

    for row in history.sort_values("date").itertuples(index=False):
        features = build_features(states, h2h, row.date, row.team_a, row.team_b, int(row.best_of), int(row.lan), float(row.tier), str(row.event))
        # No future-cutoff weighting here. Each model fit applies its own temporal
        # half-life relative to the last date it is allowed to see.
        quality_weight = 0.70 + 0.62 * float(row.tier)
        base_weight = quality_weight
        x_rows.append(features); labels.append(str(row.score_class)); dates.append(row.date); originals.append(True); base_weights.append(base_weight)
        x_rows.append(mirror_features(features)); labels.append(mirror_score(str(row.score_class))); dates.append(row.date); originals.append(False); base_weights.append(base_weight)
        update_ratings(states, h2h, ledger, row.date, row.team_a, row.team_b, int(row.winner_a), int(row.score_a), int(row.score_b), int(row.best_of), int(row.lan), float(row.tier), str(row.event))

    counts = pd.Series(labels).value_counts().to_dict()
    class_weights = {label: min(2.4, math.sqrt(max(counts.values()) / max(1, count))) for label, count in counts.items()}
    weights = np.asarray([base_weights[i] * class_weights[labels[i]] for i in range(len(labels))], dtype=float)
    return TrainingData(pd.DataFrame(x_rows, columns=FEATURES), labels, np.asarray(dates, dtype="datetime64[ns]"), weights,
                        np.asarray(originals, dtype=bool), states, h2h, pd.DataFrame(ledger))


WINNER_MODEL_CONFIGS = [
    {"name": "Fast current-form depth-2", "half_life": 115.0, "n_estimators": 470, "max_depth": 2, "learning_rate": 0.030, "min_child_weight": 10.0, "subsample": 0.92, "colsample_bytree": 0.90, "reg_alpha": 0.42, "reg_lambda": 5.8, "gamma": 0.025},
    {"name": "Balanced temporal depth-2", "half_life": 205.0, "n_estimators": 520, "max_depth": 2, "learning_rate": 0.028, "min_child_weight": 9.0, "subsample": 0.93, "colsample_bytree": 0.90, "reg_alpha": 0.34, "reg_lambda": 5.2, "gamma": 0.020},
    {"name": "Hybrid depth-3", "half_life": 285.0, "n_estimators": 560, "max_depth": 3, "learning_rate": 0.025, "min_child_weight": 10.0, "subsample": 0.91, "colsample_bytree": 0.86, "reg_alpha": 0.46, "reg_lambda": 6.0, "gamma": 0.045},
    {"name": "Stable long-memory depth-3", "half_life": 430.0, "n_estimators": 600, "max_depth": 3, "learning_rate": 0.023, "min_child_weight": 12.0, "subsample": 0.90, "colsample_bytree": 0.84, "reg_alpha": 0.55, "reg_lambda": 6.8, "gamma": 0.060},
    {"name": "Highly regularised depth-4", "half_life": 250.0, "n_estimators": 620, "max_depth": 4, "learning_rate": 0.021, "min_child_weight": 15.0, "subsample": 0.88, "colsample_bytree": 0.80, "reg_alpha": 0.72, "reg_lambda": 7.5, "gamma": 0.085},
]

GATE_COLUMNS = [
    "model_confidence", "entropy", "elo_confidence", "model_elo_agree", "model_elo_gap",
    "symmetry_quality", "sample_quality", "tier", "abs_oracle_elo", "abs_fast_elo",
    "abs_momentum_30", "abs_form_30", "rest_abs", "best_of",
]


def new_winner_model(config: dict[str, Any], seed: int = 27) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **{key: value for key, value in config.items() if key not in {"name", "half_life"}},
        objective="binary:logistic", eval_metric="logloss", random_state=seed,
        n_jobs=2, tree_method="hist", max_bin=192,
    )


def new_score_model(num_classes: int, seed: int = 29, final: bool = False) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=570 if final else 340, max_depth=3, learning_rate=0.030 if final else 0.043,
        min_child_weight=8.5, subsample=0.90, colsample_bytree=0.85,
        reg_alpha=0.42, reg_lambda=4.8, gamma=0.045,
        objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss",
        random_state=seed, n_jobs=2, tree_method="hist", max_bin=192,
    )


def mirror_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([mirror_features(row) for row in frame.to_dict("records")], columns=FEATURES, index=frame.index)


def symmetric_binary_probs(model: xgb.XGBClassifier, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    original = model.predict_proba(frame[FEATURES])[:, 1]
    mirrored_raw = model.predict_proba(mirror_frame(frame)[FEATURES])[:, 1]
    mirrored_to_original = 1.0 - mirrored_raw
    gap = np.abs(original - mirrored_to_original)
    return np.clip(0.5 * (original + mirrored_to_original), 1e-6, 1 - 1e-6), gap


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    logs = np.log(np.clip(probs, 1e-9, 1.0)) / max(0.25, temperature)
    return softmax(logs)


def best_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    best_t, best_loss = 1.0, float("inf")
    for temperature in np.linspace(0.70, 1.75, 43):
        scaled = temperature_scale(probs, float(temperature))
        loss = float(log_loss(y, scaled, labels=list(range(probs.shape[1]))))
        if loss < best_loss:
            best_loss, best_t = loss, float(temperature)
    return best_t


def fit_platt(p: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    if len(p) < 40 or len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(C=0.70, solver="lbfgs", max_iter=1000)
    model.fit(np.asarray([logit(value) for value in p]).reshape(-1, 1), y)
    return model


def apply_platt(model: LogisticRegression | None, p: np.ndarray | float) -> np.ndarray:
    values = np.asarray(p, dtype=float).reshape(-1)
    if model is None:
        return np.clip(values, 1e-6, 1 - 1e-6)
    x = np.asarray([logit(value) for value in values]).reshape(-1, 1)
    return np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)


def symmetric_multiclass_probs(model: xgb.XGBClassifier, frame: pd.DataFrame, labels: list[str], temperature: float) -> np.ndarray:
    first = temperature_scale(model.predict_proba(frame[FEATURES]), temperature)
    second_raw = temperature_scale(model.predict_proba(mirror_frame(frame)[FEATURES]), temperature)
    index = {label: i for i, label in enumerate(labels)}
    second = np.zeros_like(second_raw)
    for source_i, label in enumerate(labels):
        target_i = index.get(mirror_score(label))
        if target_i is not None:
            second[:, target_i] += second_raw[:, source_i]
    return np.clip(0.5 * (first + second), 1e-12, 1.0)


def rescale_score_matrix(probs: np.ndarray, labels: list[str], winner_p: np.ndarray) -> np.ndarray:
    adjusted = probs.copy()
    win_idx = [i for i, label in enumerate(labels) if score_is_a_win(label)]
    lose_idx = [i for i, label in enumerate(labels) if not score_is_a_win(label)]
    for row_i, target in enumerate(np.asarray(winner_p).reshape(-1)):
        win_total = adjusted[row_i, win_idx].sum() if win_idx else 0.0
        lose_total = adjusted[row_i, lose_idx].sum() if lose_idx else 0.0
        if win_idx:
            adjusted[row_i, win_idx] *= target / max(win_total, 1e-12)
        if lose_idx:
            adjusted[row_i, lose_idx] *= (1.0 - target) / max(lose_total, 1e-12)
    return adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if mask.any():
            ece += mask.mean() * abs(float(y_true[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def calibration_table(y_true: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = lower + 0.1
        mask = (p >= lower) & (p < upper if upper < 1 else p <= upper)
        if mask.any():
            rows.append({"Probability band": f"{lower:.0%}–{upper:.0%}", "Predicted": 100 * p[mask].mean(),
                         "Actual": 100 * y_true[mask].mean(), "Matches": int(mask.sum())})
    return pd.DataFrame(rows)


def gate_frame(features: pd.DataFrame, p: np.ndarray, elo_p: np.ndarray, symmetry_gap: np.ndarray) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    elo_p = np.asarray(elo_p, dtype=float)
    entropy = -(p * np.log(np.clip(p, 1e-9, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-9, 1))) / math.log(2)
    return pd.DataFrame({
        "model_confidence": np.maximum(p, 1 - p),
        "entropy": entropy,
        "elo_confidence": np.maximum(elo_p, 1 - elo_p),
        "model_elo_agree": ((p >= 0.5) == (elo_p >= 0.5)).astype(float),
        "model_elo_gap": np.abs(p - elo_p),
        "symmetry_quality": 1.0 - np.clip(symmetry_gap / 0.22, 0.0, 1.0),
        "sample_quality": np.clip(features["sample_min"].to_numpy(dtype=float) / 35.0, 0.0, 1.0),
        "tier": features["tier"].to_numpy(dtype=float),
        "abs_oracle_elo": np.abs(features["oracle_elo_diff"].to_numpy(dtype=float)) / 250.0,
        "abs_fast_elo": np.abs(features["fast_elo_diff"].to_numpy(dtype=float)) / 300.0,
        "abs_momentum_30": np.abs(features["elo_momentum_30_diff"].to_numpy(dtype=float)) / 150.0,
        "abs_form_30": np.abs(features["winrate_30_diff"].to_numpy(dtype=float)) / 0.35,
        "rest_abs": np.clip(features["rest_abs"].to_numpy(dtype=float) / 30.0, 0.0, 2.0),
        "best_of": features["best_of"].to_numpy(dtype=float) / 5.0,
    }, index=features.index)[GATE_COLUMNS]


def choose_precision_threshold(scores: np.ndarray, correct: np.ndarray, target: float = TARGET_PRECISION) -> tuple[float, dict[str, Any]]:
    best: tuple[float, float, int] | None = None
    candidates = sorted(set(np.round(np.concatenate([np.linspace(0.30, 0.95, 131), np.quantile(scores, np.linspace(0.05, 0.95, 37))]), 3)))
    min_matches = max(18, int(round(len(scores) * 0.09)))
    for threshold in candidates:
        selected = scores >= threshold
        count = int(selected.sum())
        if count < min_matches:
            continue
        acc = float(correct[selected].mean())
        if acc >= target + 0.02:
            coverage = float(selected.mean())
            if best is None or coverage > best[1]:
                best = (threshold, coverage, count)
    if best is None:
        fallback = []
        for threshold in candidates:
            selected = scores >= threshold
            count = int(selected.sum())
            if count >= min_matches:
                acc = float(correct[selected].mean())
                fallback.append((acc, float(selected.mean()), threshold, count))
        if not fallback:
            return 0.90, {"accuracy": 0.0, "coverage": 0.0, "matches": 0, "target_met": False}
        acc, coverage, threshold, count = max(fallback, key=lambda item: (item[0], item[1]))
        return float(threshold), {"accuracy": acc, "coverage": coverage, "matches": count, "target_met": acc >= target}
    threshold, coverage, count = best
    selected = scores >= threshold
    acc = float(correct[selected].mean())
    return float(threshold), {"accuracy": acc, "coverage": coverage, "matches": count, "target_met": acc >= target}


@dataclass
class ProbabilityCalibrator:
    kind: str = "identity"
    model: Any = None
    shrink: float = 1.0


@dataclass
class WinnerEngine:
    models: list[xgb.XGBClassifier]
    names: list[str]
    stacker: LogisticRegression | None
    calibrator: ProbabilityCalibrator


@dataclass
class ModelBundle:
    winner_model: WinnerEngine | None
    score_model: xgb.XGBClassifier | None
    class_labels: list[str]
    score_temperature: float
    winner_calibrator: Any
    gate_model: Any
    gate_threshold: float
    gate_validation: dict[str, Any]
    states: dict[str, TeamState]
    h2h: dict[tuple[str, str], deque]
    ledger: pd.DataFrame
    training_rows: int
    metrics: dict[str, Any]
    feature_gain: dict[str, float]
    newest_date: pd.Timestamp | None
    winner_model_name: str


def temporal_weights(training: TrainingData, mask: np.ndarray, reference_date: np.datetime64 | pd.Timestamp, half_life: float) -> np.ndarray:
    dates = pd.to_datetime(training.dates[mask])
    reference = pd.Timestamp(reference_date)
    age_days = np.maximum(0.0, (reference - dates).total_seconds() / 86400.0)
    decay = np.power(0.5, age_days / max(45.0, float(half_life)))
    return np.asarray(training.weights[mask], dtype=float) * np.asarray(decay, dtype=float)


def fit_winner_candidate(training: TrainingData, mask: np.ndarray, config: dict[str, Any], reference_date: Any, seed: int) -> xgb.XGBClassifier:
    model = new_winner_model(config, seed)
    y = np.asarray([int(score_is_a_win(label)) for label in training.labels], dtype=int)
    model.fit(training.x.loc[mask, FEATURES], y[mask], sample_weight=temporal_weights(training, mask, reference_date, float(config["half_life"])))
    return model


def ensemble_base_probs(models: list[xgb.XGBClassifier], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not models:
        raise ValueError("Winner ensemble contains no models")
    probs, gaps = [], []
    for model in models:
        p, gap = symmetric_binary_probs(model, frame)
        probs.append(p); gaps.append(gap)
    matrix = np.column_stack(probs)
    gap_matrix = np.column_stack(gaps)
    return matrix, matrix.mean(axis=1), gap_matrix.mean(axis=1)


def heuristic_probabilities(frame: pd.DataFrame) -> np.ndarray:
    # Independent structured-data views that make the stack less brittle than a
    # single tree model. They contain no bookmaker information.
    score = (
        frame["oracle_elo_diff"].to_numpy(float) / 245.0
        + 1.35 * frame["winrate_30_diff"].to_numpy(float)
        + 0.75 * frame["winrate_90_diff"].to_numpy(float)
        + 0.95 * frame["performance_30_diff"].to_numpy(float)
        + 0.32 * frame["margin_30_diff"].to_numpy(float)
        + frame["elo_momentum_30_diff"].to_numpy(float) / 125.0
        + 0.30 * frame["h2h_diff"].to_numpy(float)
        + 0.62 * frame["event_form_diff"].to_numpy(float)
        + 0.30 * frame["high_tier_form_diff"].to_numpy(float)
        + 0.22 * frame["favorite_conversion_diff"].to_numpy(float)
        + 0.18 * frame["close_match_diff"].to_numpy(float)
        + 0.25 * frame["consistency_diff"].to_numpy(float)
        - 0.20 * frame["fatigue_diff"].to_numpy(float)
        - 0.16 * frame["rust_diff"].to_numpy(float)
    )
    return np.asarray([sigmoid(value) for value in score], dtype=float)


def stack_design(base_matrix: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    columns = [np.asarray([logit(value) for value in base_matrix[:, i]]) for i in range(base_matrix.shape[1])]
    elo = np.asarray([logistic_elo(value) for value in frame["oracle_elo_diff"].to_numpy(float)])
    fast = np.asarray([logistic_elo(0.72 * value + 0.28 * core) for value, core in zip(frame["fast_elo_diff"], frame["core_elo_diff"])])
    form = heuristic_probabilities(frame)
    base_mean = np.clip(base_matrix.mean(axis=1), 1e-6, 1 - 1e-6)
    base_spread = base_matrix.std(axis=1)
    sample_quality = np.clip(np.log1p(frame["sample_min"].to_numpy(float)) / np.log(81.0), 0.0, 1.0)
    tier = frame["tier"].to_numpy(float)
    bo1 = (frame["best_of"].to_numpy(float) == 1).astype(float)
    bo5 = (frame["best_of"].to_numpy(float) >= 5).astype(float)
    lan = frame["lan"].to_numpy(float)
    abs_elo = np.abs(frame["oracle_elo_diff"].to_numpy(float)) / 400.0
    event_edge = frame["event_form_diff"].to_numpy(float)
    consistency = frame["consistency_diff"].to_numpy(float)
    mean_logit = np.asarray([logit(value) for value in base_mean])
    columns.extend([
        np.asarray([logit(value) for value in elo]),
        np.asarray([logit(value) for value in fast]),
        np.asarray([logit(value) for value in form]),
        mean_logit, base_spread, sample_quality, tier, bo1, bo5, lan, abs_elo, event_edge, consistency,
        mean_logit * sample_quality, mean_logit * tier, mean_logit * (1.0 - np.minimum(base_spread * 5.0, 1.0)),
    ])
    return np.column_stack(columns)


def stacked_raw_probs(engine: WinnerEngine, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base, average, gaps = ensemble_base_probs(engine.models, frame)
    if engine.stacker is None:
        raw = average
    else:
        raw = engine.stacker.predict_proba(stack_design(base, frame))[:, 1]
    return np.clip(raw, 1e-6, 1 - 1e-6), gaps, base


def fit_probability_calibrator(kind: str, p: np.ndarray, y: np.ndarray) -> ProbabilityCalibrator:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    if kind == "identity":
        return ProbabilityCalibrator("identity", None, 1.0)
    if kind == "platt":
        model = LogisticRegression(C=0.55, solver="lbfgs", max_iter=1500)
        model.fit(np.asarray([logit(value) for value in p]).reshape(-1, 1), y)
        return ProbabilityCalibrator("Platt", model, 1.0)
    if kind == "beta":
        x = np.column_stack([np.log(p), -np.log1p(-p)])
        model = LogisticRegression(C=0.45, solver="lbfgs", max_iter=1500)
        model.fit(x, y)
        return ProbabilityCalibrator("Beta", model, 1.0)
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.015, y_max=0.985)
        model.fit(p, y)
        return ProbabilityCalibrator("Isotonic", model, 1.0)
    raise ValueError(f"Unknown calibration kind: {kind}")


def apply_probability_calibrator(calibrator: ProbabilityCalibrator, p: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(p, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
    if calibrator.kind == "identity" or calibrator.model is None:
        calibrated = values
    elif calibrator.kind == "Platt":
        x = np.asarray([logit(value) for value in values]).reshape(-1, 1)
        calibrated = calibrator.model.predict_proba(x)[:, 1]
    elif calibrator.kind == "Beta":
        x = np.column_stack([np.log(values), -np.log1p(-values)])
        calibrated = calibrator.model.predict_proba(x)[:, 1]
    elif calibrator.kind == "Isotonic":
        calibrated = calibrator.model.predict(values)
    else:
        calibrated = values
    calibrated = 0.5 + (np.asarray(calibrated, dtype=float) - 0.5) * float(calibrator.shrink)
    return np.clip(calibrated, 1e-6, 1 - 1e-6)


def select_calibrator(raw_fit: np.ndarray, y_fit: np.ndarray, raw_select: np.ndarray, y_select: np.ndarray) -> ProbabilityCalibrator:
    kinds = ["identity", "platt", "beta"]
    if len(raw_fit) >= 100 and len(np.unique(y_fit)) == 2:
        kinds.append("isotonic")
    best: tuple[float, ProbabilityCalibrator] | None = None
    for kind in kinds:
        try:
            base = fit_probability_calibrator(kind, raw_fit, y_fit)
        except Exception:
            continue
        for shrink in [0.72, 0.80, 0.88, 0.94, 1.00]:
            candidate = ProbabilityCalibrator(base.kind, base.model, shrink)
            pred = apply_probability_calibrator(candidate, raw_select)
            score = float(brier_score_loss(y_select, pred) + 0.10 * log_loss(y_select, pred, labels=[0, 1]) + 0.35 * expected_calibration_error(y_select, pred))
            if best is None or score < best[0]:
                best = (score, candidate)
    return best[1] if best is not None else ProbabilityCalibrator()


def engine_probs(engine: WinnerEngine, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw, gaps, base = stacked_raw_probs(engine, frame)
    return apply_probability_calibrator(engine.calibrator, raw), gaps, base


@st.cache_resource(show_spinner=False)
def train_bundle_cached(history_json: str) -> ModelBundle:
    history = pd.read_json(io.StringIO(history_json), orient="split")
    history["date"] = pd.to_datetime(history["date"], utc=True)
    training = make_training_data(history)
    labels_array = np.asarray(training.labels)
    winner_y_all = np.asarray([int(score_is_a_win(label)) for label in training.labels], dtype=int)
    all_score_labels = sorted(set(training.labels), key=lambda label: SCORE_CLASSES.index(label) if label in SCORE_CLASSES else 99)
    metrics: dict[str, Any] = {}
    feature_gain: dict[str, float] = {}
    final_engine: WinnerEngine | None = None
    final_score: xgb.XGBClassifier | None = None
    gate_model: Any = None
    gate_threshold = 0.90
    gate_validation: dict[str, Any] = {}
    score_temperature = 1.0
    selected_configs = WINNER_MODEL_CONFIGS[1:4]
    stacker: LogisticRegression | None = None
    calibrator = ProbabilityCalibrator()
    engine_name = "Dynamic Oracle Elo"

    if len(history) >= MIN_MODEL_MATCHES:
        unique_dates = np.sort(history["date"].dt.tz_localize(None).unique())
        train_cut = unique_dates[max(1, int(len(unique_dates) * 0.68)) - 1]
        calib_cut = unique_dates[max(2, int(len(unique_dates) * 0.85)) - 1]
        dates = training.dates
        train_mask = dates <= train_cut
        calib_mask = (dates > train_cut) & (dates <= calib_cut) & training.original
        test_mask = (dates > calib_cut) & training.original

        calib_indices = np.flatnonzero(calib_mask)
        if train_mask.sum() >= 300 and len(calib_indices) >= 95 and test_mask.sum() >= 80:
            # Three chronological calibration jobs: choose/stack, fit calibration,
            # and select calibration strength. The final test is never touched.
            first = max(40, int(len(calib_indices) * 0.46))
            second = max(first + 25, int(len(calib_indices) * 0.76))
            second = min(second, len(calib_indices) - 18)
            stack_idx = calib_indices[:first]
            cal_fit_idx = calib_indices[first:second]
            cal_select_idx = calib_indices[second:]

            candidate_rows = []
            candidate_models: list[tuple[dict[str, Any], xgb.XGBClassifier]] = []
            stack_frame = training.x.loc[stack_idx, FEATURES]
            stack_y = winner_y_all[stack_idx]
            chronological_folds = [fold for fold in np.array_split(np.arange(len(stack_idx)), 2) if len(fold) >= 12]
            for candidate_i, config in enumerate(WINNER_MODEL_CONFIGS):
                model = fit_winner_candidate(training, train_mask, config, train_cut, 41 + candidate_i)
                p, _ = symmetric_binary_probs(model, stack_frame)
                fold_scores = []
                for fold in chronological_folds:
                    fold_p = p[fold]; fold_y = stack_y[fold]
                    fold_scores.append(float(brier_score_loss(fold_y, fold_p) + 0.10 * log_loss(fold_y, fold_p, labels=[0, 1])))
                score = float(np.mean(fold_scores) + 0.22 * np.std(fold_scores)) if fold_scores else float(brier_score_loss(stack_y, p))
                candidate_rows.append((score, config, model))
            candidate_rows.sort(key=lambda item: item[0])
            selected_configs = [item[1] for item in candidate_rows[:3]]
            candidate_models = [(item[1], item[2]) for item in candidate_rows[:3]]
            selected_eval_models = [item[1] for item in candidate_models]

            base_stack, _, _ = ensemble_base_probs(selected_eval_models, stack_frame)
            stacker = LogisticRegression(C=0.28, solver="lbfgs", max_iter=1800)
            stacker.fit(stack_design(base_stack, stack_frame), stack_y)
            provisional = WinnerEngine(selected_eval_models, [cfg["name"] for cfg in selected_configs], stacker, ProbabilityCalibrator())

            fit_frame = training.x.loc[cal_fit_idx, FEATURES]
            fit_y = winner_y_all[cal_fit_idx]
            select_frame = training.x.loc[cal_select_idx, FEATURES]
            select_y = winner_y_all[cal_select_idx]
            fit_raw, _, _ = stacked_raw_probs(provisional, fit_frame)
            select_raw, _, _ = stacked_raw_probs(provisional, select_frame)
            calibrator = select_calibrator(fit_raw, fit_y, select_raw, select_y)
            eval_engine = WinnerEngine(selected_eval_models, [cfg["name"] for cfg in selected_configs], stacker, calibrator)
            engine_name = "Temporal stack: " + " + ".join(cfg["name"].split()[0] for cfg in selected_configs)

            # Gate is trained only on the calibration period, after the probability
            # engine has been fixed, and is evaluated only on untouched test data.
            calib_x = training.x.loc[calib_mask, FEATURES]
            calib_y = winner_y_all[calib_mask]
            calib_p, calib_gap, _ = engine_probs(eval_engine, calib_x)
            calib_elo_p = np.asarray([logistic_elo(value) for value in calib_x["oracle_elo_diff"]], dtype=float)
            calib_gate_x = gate_frame(calib_x, calib_p, calib_elo_p, calib_gap)
            calib_correct = ((calib_p >= 0.5).astype(int) == calib_y).astype(int)
            gate_split = max(45, int(len(calib_gate_x) * 0.68))
            gate_split = min(gate_split, len(calib_gate_x) - 20)
            if gate_split >= 35 and len(np.unique(calib_correct[:gate_split])) >= 2:
                gate_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.32, max_iter=1600, class_weight="balanced"))
                gate_model.fit(calib_gate_x.iloc[:gate_split], calib_correct[:gate_split])
                valid_scores = gate_model.predict_proba(calib_gate_x.iloc[gate_split:])[:, 1]
                gate_threshold, gate_validation = choose_precision_threshold(valid_scores, calib_correct[gate_split:])
                gate_model.fit(calib_gate_x, calib_correct)

            # Exact score head remains separate; winner totals are forced to match
            # the stronger calibrated winner engine.
            score_train_labels = sorted(set(labels_array[train_mask]), key=lambda label: SCORE_CLASSES.index(label))
            score_encoder = {label: i for i, label in enumerate(score_train_labels)}
            score_train_mask = train_mask & np.asarray([label in score_encoder for label in labels_array])
            eval_score_model = None
            if len(score_train_labels) >= 4 and score_train_mask.sum() >= 280:
                eval_score_model = new_score_model(len(score_train_labels), 61, final=False)
                score_y_train = np.asarray([score_encoder[label] for label in labels_array[score_train_mask]], dtype=int)
                eval_score_model.fit(training.x.loc[score_train_mask, FEATURES], score_y_train,
                                     sample_weight=temporal_weights(training, score_train_mask, train_cut, 235.0))
                score_calib_known = calib_mask & np.asarray([label in score_encoder for label in labels_array])
                if score_calib_known.sum() >= 50:
                    score_calib_x = training.x.loc[score_calib_known, FEATURES]
                    score_calib_probs = symmetric_multiclass_probs(eval_score_model, score_calib_x, score_train_labels, 1.0)
                    score_calib_y = np.asarray([score_encoder[label] for label in labels_array[score_calib_known]], dtype=int)
                    score_temperature = best_temperature(score_calib_probs, score_calib_y)

            test_x = training.x.loc[test_mask, FEATURES]
            test_y = winner_y_all[test_mask]
            test_p, test_gap, _ = engine_probs(eval_engine, test_x)
            test_pred = (test_p >= 0.5).astype(int)
            test_elo_p = np.asarray([logistic_elo(value) for value in test_x["oracle_elo_diff"]], dtype=float)
            test_gate_x = gate_frame(test_x, test_p, test_elo_p, test_gap)
            gate_scores = gate_model.predict_proba(test_gate_x)[:, 1] if gate_model is not None else np.maximum(test_p, 1 - test_p)
            elite = gate_scores >= gate_threshold

            exact_accuracy = float("nan")
            exact_logloss = float("nan")
            if eval_score_model is not None:
                test_score_known = test_mask & np.asarray([label in score_encoder for label in labels_array])
                if test_score_known.sum() >= 50:
                    test_score_x = training.x.loc[test_score_known, FEATURES]
                    score_probs = symmetric_multiclass_probs(eval_score_model, test_score_x, score_train_labels, score_temperature)
                    binary_for_score, _, _ = engine_probs(eval_engine, test_score_x)
                    score_probs = rescale_score_matrix(score_probs, score_train_labels, binary_for_score)
                    score_true = np.asarray([score_encoder[label] for label in labels_array[test_score_known]], dtype=int)
                    exact_accuracy = float(accuracy_score(score_true, score_probs.argmax(axis=1)))
                    exact_logloss = float(log_loss(score_true, score_probs, labels=list(range(len(score_train_labels)))))

            confidence_rows = []
            for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                selected = gate_scores >= threshold
                if selected.any():
                    confidence_rows.append({"Minimum precision score": f"{threshold:.0%}", "Coverage": 100 * selected.mean(),
                                            "Winner accuracy": 100 * accuracy_score(test_y[selected], test_pred[selected]), "Matches": int(selected.sum())})
            midpoint = max(1, len(test_y) // 2)
            metrics = {
                "winner_accuracy": float(accuracy_score(test_y, test_pred)),
                "recent_half_accuracy": float(accuracy_score(test_y[midpoint:], test_pred[midpoint:])),
                "exact_accuracy": exact_accuracy,
                "winner_brier": float(brier_score_loss(test_y, test_p)),
                "winner_logloss": float(log_loss(test_y, test_p, labels=[0, 1])),
                "winner_auc": float(roc_auc_score(test_y, test_p)) if len(np.unique(test_y)) == 2 else float("nan"),
                "multiclass_logloss": exact_logloss,
                "ece": expected_calibration_error(test_y, test_p),
                "elo_accuracy": float(accuracy_score(test_y, test_elo_p >= 0.5)),
                "elo_brier": float(brier_score_loss(test_y, test_elo_p)),
                "test_matches": int(len(test_y)),
                "score_temperature": score_temperature,
                "confidence_table": confidence_rows,
                "calibration_table": calibration_table(test_y, test_p).to_dict("records"),
                "precision_gate_accuracy": float(accuracy_score(test_y[elite], test_pred[elite])) if elite.any() else 0.0,
                "precision_gate_coverage": float(elite.mean()),
                "precision_gate_matches": int(elite.sum()),
                "precision_gate_target_met": bool(elite.sum() >= 20 and accuracy_score(test_y[elite], test_pred[elite]) >= TARGET_PRECISION) if elite.any() else False,
                "gate_threshold": gate_threshold,
                "gate_validation": gate_validation,
                "winner_model_name": engine_name,
                "calibrator": calibrator.kind,
                "probability_shrink": calibrator.shrink,
                "ensemble_members": [cfg["name"] for cfg in selected_configs],
            }

        # Refit selected temporal members on all permitted history. The stacker and
        # calibration mapping remain frozen from older chronological periods.
        final_models = []
        final_ref = training.dates.max()
        full_mask = np.ones(len(training.x), dtype=bool)
        for i, config in enumerate(selected_configs):
            final_models.append(fit_winner_candidate(training, full_mask, config, final_ref, 87 + i))
        final_engine = WinnerEngine(final_models, [cfg["name"] for cfg in selected_configs], stacker, calibrator)

        if len(all_score_labels) >= 4:
            score_encoder_all = {label: i for i, label in enumerate(all_score_labels)}
            score_y_all = np.asarray([score_encoder_all[label] for label in labels_array], dtype=int)
            final_score = new_score_model(len(all_score_labels), 99, final=True)
            final_score.fit(training.x[FEATURES], score_y_all,
                            sample_weight=temporal_weights(training, full_mask, final_ref, 235.0))

        gain_totals: dict[str, float] = defaultdict(float)
        for model in final_models:
            raw_gain = model.get_booster().get_score(importance_type="gain")
            for feature, value in raw_gain.items():
                gain_totals[feature] += float(value)
        total_gain = sum(gain_totals.values()) or 1.0
        feature_gain = {feature: gain_totals.get(feature, 0.0) / total_gain for feature in FEATURES}

    return ModelBundle(final_engine, final_score, all_score_labels, score_temperature, calibrator,
                       gate_model, gate_threshold, gate_validation, training.states, training.h2h,
                       training.ledger, len(training.x), metrics, feature_gain,
                       history["date"].max() if not history.empty else None, engine_name)


def winner_reliability(bundle: ModelBundle, features: dict[str, float]) -> dict[str, float]:
    frame = pd.DataFrame([features], columns=FEATURES)
    if bundle.winner_model is None:
        p = logistic_elo(features["oracle_elo_diff"])
        return {"p": p, "raw_p": p, "elo_p": p, "symmetry_gap": 0.0, "precision_score": max(p, 1 - p), "member_spread": 0.0}
    raw, gap, base = stacked_raw_probs(bundle.winner_model, frame)
    calibrated = apply_probability_calibrator(bundle.winner_model.calibrator, raw)
    elo_p = np.asarray([logistic_elo(features["oracle_elo_diff"])])
    gate_x = gate_frame(frame, calibrated, elo_p, gap)
    precision = float(bundle.gate_model.predict_proba(gate_x)[:, 1][0]) if bundle.gate_model is not None else float(max(calibrated[0], 1 - calibrated[0]))
    return {"p": float(calibrated[0]), "raw_p": float(raw[0]), "elo_p": float(elo_p[0]),
            "symmetry_gap": float(gap[0]), "precision_score": precision,
            "member_spread": float(np.std(base[0])) if base.size else 0.0}


def model_distribution(bundle: ModelBundle, match: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], str, dict[str, float]]:
    features = build_features(bundle.states, bundle.h2h, match["date"], match["team_a"], match["team_b"], int(match["best_of"]), int(match["lan"]), float(match["tier"]), str(match.get("event", "")))
    reliability = winner_reliability(bundle, features)
    p_a = reliability["p"]
    valid = valid_scores(int(match["best_of"]))
    if bundle.score_model is None or not bundle.class_labels:
        return analytic_score_distribution(p_a, int(match["best_of"])), features, "Pure calibrated temporal ensemble with analytic score fallback", reliability
    frame = pd.DataFrame([features], columns=FEATURES)
    matrix = symmetric_multiclass_probs(bundle.score_model, frame, bundle.class_labels, bundle.score_temperature)
    matrix = rescale_score_matrix(matrix, bundle.class_labels, np.asarray([p_a]))
    raw_dist = {label: float(prob) for label, prob in zip(bundle.class_labels, matrix[0])}
    combined = {score: raw_dist.get(score, 0.0) for score in valid}
    total = sum(combined.values())
    if total <= 1e-10:
        return analytic_score_distribution(p_a, int(match["best_of"])), features, "Pure calibrated temporal ensemble with format fallback", reliability
    return {score: value / total for score, value in combined.items()}, features, "Pure temporal ensemble + conditional exact-score XGBoost", reliability

def solve_map_probability(series_p: float, best_of: int) -> float:
    if best_of <= 1:
        return series_p
    needed = best_of // 2 + 1
    def series_prob(q: float) -> float:
        return sum(math.comb(needed - 1 + losses, losses) * q ** needed * (1 - q) ** losses for losses in range(needed))
    lo, hi = 1e-5, 1 - 1e-5
    for _ in range(60):
        mid = (lo + hi) / 2
        if series_prob(mid) < series_p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def analytic_score_distribution(series_p: float, best_of: int) -> dict[str, float]:
    q = solve_map_probability(series_p, best_of)
    needed = 1 if best_of == 1 else best_of // 2 + 1
    output: dict[str, float] = {}
    for losses in range(needed):
        output[f"{needed}-{losses}"] = math.comb(needed - 1 + losses, losses) * q ** needed * (1 - q) ** losses
        output[f"{losses}-{needed}"] = math.comb(needed - 1 + losses, losses) * (1 - q) ** needed * q ** losses
    total = sum(output.values())
    return {key: value / total for key, value in output.items()}


def clean_player_data(uploaded: Any) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
    frame = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
    required = {"team", "player", "rating", "adr"}
    if not required.issubset(frame.columns):
        raise ValueError("Player CSV needs team, player, rating and adr columns.")
    frame = frame.copy(); frame["team"] = frame["team"].map(normalise_team)
    for column in ["rating", "adr", "kast", "impact", "maps"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["team", "player", "rating", "adr"])


def player_overlay(player_data: pd.DataFrame, team_a: str, team_b: str) -> tuple[float, str]:
    if player_data.empty:
        return 0.0, "No player overlay applied: free PandaScore fixtures do not include 3-month ADR/rating history."
    a = player_data[player_data["team"] == team_a].sort_values("rating", ascending=False).head(5)
    b = player_data[player_data["team"] == team_b].sort_values("rating", ascending=False).head(5)
    if len(a) < 4 or len(b) < 4:
        return 0.0, "Player CSV is incomplete for one team; no overlay was applied."
    rating_edge = float(a["rating"].mean() - b["rating"].mean())
    adr_edge = float(a["adr"].mean() - b["adr"].mean())
    kast_edge = 0.0
    if "kast" in a and "kast" in b and a["kast"].notna().any() and b["kast"].notna().any():
        kast_edge = float(a["kast"].mean() - b["kast"].mean()) / 100.0
    adjustment = float(np.clip(1.75 * rating_edge + adr_edge / 70.0 + 0.9 * kast_edge, -0.30, 0.30))
    return adjustment, f"Player overlay: rating edge {rating_edge:+.3f}, ADR edge {adr_edge:+.1f}; logit adjustment {adjustment:+.3f}."


def apply_overlay(distribution: dict[str, float], adjustment: float) -> dict[str, float]:
    if abs(adjustment) < 1e-9:
        return distribution
    adjusted = {}
    for score, probability in distribution.items():
        adjusted[score] = probability * math.exp(adjustment if score_is_a_win(score) else -adjustment)
    total = sum(adjusted.values())
    return {score: probability / total for score, probability in adjusted.items()}


def prediction_for_match(bundle: ModelBundle, match: dict[str, Any], player_data: pd.DataFrame) -> dict[str, Any]:
    distribution, features, method, reliability = model_distribution(bundle, match)
    adjustment, overlay_note = player_overlay(player_data, match["team_a"], match["team_b"])
    distribution = apply_overlay(distribution, adjustment)
    p_a = sum(prob for score, prob in distribution.items() if score_is_a_win(score))
    a_is_pick = p_a >= 0.5
    eligible_scores = [score for score in distribution if score_is_a_win(score) == a_is_pick]
    predicted_score = max(eligible_scores, key=lambda score: distribution[score]) if eligible_scores else max(distribution, key=distribution.get)
    winner = match["team_a"] if a_is_pick else match["team_b"]
    winner_p = p_a if a_is_pick else 1 - p_a
    precision = float(reliability.get("precision_score", winner_p))
    if abs(adjustment) > 1e-9:
        precision *= 0.96
    proven_elite = bool(bundle.metrics.get("precision_gate_target_met", False)) if bundle.metrics else False
    threshold = float(bundle.gate_threshold)
    if proven_elite and precision >= threshold:
        label = "Elite 70% proven"
    elif precision >= max(0.66, threshold - 0.05):
        label = "Strong signal"
    elif precision >= 0.55:
        label = "Medium signal"
    else:
        label = "Cautious"
    reasons_a, reasons_b = feature_reasons(features)
    return {"distribution": dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True)), "features": features,
            "team_a_p": p_a, "winner": winner, "winner_p": winner_p, "score": predicted_score,
            "confidence": round(100 * precision), "confidence_label": label, "precision_score": precision,
            "elite_threshold": threshold, "elite_proven": proven_elite, "method": method,
            "overlay_note": overlay_note, "reasons_a": reasons_a, "reasons_b": reasons_b,
            "raw_model_p": reliability.get("raw_p", p_a), "elo_p": reliability.get("elo_p", p_a),
            "symmetry_gap": reliability.get("symmetry_gap", 0.0), "member_spread": reliability.get("member_spread", 0.0)}


def feature_reasons(features: dict[str, float]) -> tuple[list[str], list[str]]:
    descriptions = {
        "oracle_elo_diff": lambda v: f"Oracle Elo edge: {abs(v):.0f} points",
        "fast_elo_diff": lambda v: f"Fast-form Elo edge: {abs(v):.0f} points",
        "context_elo_diff": lambda v: f"LAN/online Elo edge: {abs(v):.0f} points",
        "format_elo_diff": lambda v: f"Format Elo edge: {abs(v):.0f} points",
        "elo_momentum_30_diff": lambda v: f"30-day Elo momentum edge: {abs(v):.0f}",
        "winrate_30_diff": lambda v: f"30-day weighted form edge: {abs(v) * 100:.1f} pp",
        "winrate_90_diff": lambda v: f"90-day weighted form edge: {abs(v) * 100:.1f} pp",
        "performance_30_diff": lambda v: f"Opponent-adjusted performance edge: {abs(v):.3f}",
        "opp_strength_diff": lambda v: f"Stronger recent schedule: {abs(v):.0f} Elo",
        "margin_30_diff": lambda v: f"Recent score-margin edge: {abs(v):.2f}",
        "sweep_win_diff": lambda v: f"Higher BO3 sweep profile: {abs(v) * 100:.1f} pp",
        "rust_diff": lambda v: f"Inactivity/rust difference: {abs(v):.2f}",
        "fatigue_diff": lambda v: f"Schedule-fatigue difference: {abs(v):.2f}",
        "h2h_diff": lambda v: f"Decayed head-to-head edge: {abs(v):.2f}",
        "bo_form_diff": lambda v: f"Same-format form edge: {abs(v) * 100:.1f} pp",
        "lan_form_diff": lambda v: f"Same-environment form edge: {abs(v) * 100:.1f} pp",
        "event_form_diff": lambda v: f"Current-event form edge: {abs(v) * 100:.1f} pp",
        "favorite_conversion_diff": lambda v: f"Favourite-conversion edge: {abs(v) * 100:.1f} pp",
        "underdog_upset_diff": lambda v: f"Underdog upset profile: {abs(v) * 100:.1f} pp",
        "close_match_diff": lambda v: f"Close-match conversion edge: {abs(v) * 100:.1f} pp",
        "high_tier_form_diff": lambda v: f"High-tier form edge: {abs(v) * 100:.1f} pp",
        "consistency_diff": lambda v: f"Performance consistency edge: {abs(v):.3f}",
    }
    scales = {"oracle_elo_diff": 1/120, "fast_elo_diff": 1/150, "context_elo_diff": 1/170,
              "format_elo_diff": 1/170, "elo_momentum_30_diff": 1/60, "winrate_30_diff": 5,
              "winrate_90_diff": 4, "performance_30_diff": 6, "opp_strength_diff": 1/180,
              "margin_30_diff": 1.6, "sweep_win_diff": 3, "rust_diff": -0.7,
              "fatigue_diff": -0.8, "h2h_diff": 2.5, "bo_form_diff": 3, "lan_form_diff": 2.5,
              "event_form_diff": 3.2, "favorite_conversion_diff": 2.0, "underdog_upset_diff": 1.7,
              "close_match_diff": 1.8, "high_tier_form_diff": 2.4, "consistency_diff": 2.2}
    a_items, b_items = [], []
    for key, describe in descriptions.items():
        value = features.get(key, 0.0)
        signed = value * scales.get(key, 1.0)
        if abs(signed) < 0.09:
            continue
        (a_items if signed > 0 else b_items).append((abs(signed), describe(value)))
    a = [item[1] for item in sorted(a_items, reverse=True)[:5]] or ["No single dominant edge; the call comes from the complete statistical profile."]
    b = [item[1] for item in sorted(b_items, reverse=True)[:5]] or ["No single dominant edge; the upset route depends on execution and veto."]
    return a, b


def extract_map_rows(match: dict[str, Any]) -> list[dict[str, Any]]:
    pair = extract_teams(match)
    if pair is None:
        return []
    a, b = pair
    rows = []
    for game in match.get("games") or []:
        map_obj = game.get("map") or {}
        map_name = normalise_map(map_obj.get("name") or game.get("map_name") or "")
        winner = game.get("winner") or {}
        winner_id = safe_int(winner.get("id") or game.get("winner_id"), -1)
        if map_name and winner_id in (safe_int(a.get("id"), -2), safe_int(b.get("id"), -3)):
            rows.append({"date": parse_dt(game.get("begin_at") or match.get("begin_at")),
                         "team_a": normalise_team(a.get("name")), "team_b": normalise_team(b.get("name")),
                         "map": map_name, "winner": normalise_team(a.get("name")) if winner_id == safe_int(a.get("id")) else normalise_team(b.get("name"))})
    return rows


def map_history_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in matches:
        rows.extend(extract_map_rows(match))
    return pd.DataFrame(rows)


def clean_map_data(uploaded: Any) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
    frame = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
    required = {"date", "team_a", "team_b", "map", "winner"}
    if not required.issubset(frame.columns):
        raise ValueError("Map CSV needs date, team_a, team_b, map and winner columns.")
    frame = frame.copy(); frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    for column in ["team_a", "team_b", "winner"]:
        frame[column] = frame[column].map(normalise_team)
    frame["map"] = frame["map"].map(normalise_map)
    return frame.dropna(subset=["date", "team_a", "team_b", "map", "winner"])


def map_stats(map_history: pd.DataFrame, team: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    if map_history.empty:
        return pd.DataFrame()
    subset = map_history[map_history["date"] <= cutoff]
    rows = []
    for map_name in ACTIVE_POOL:
        relevant = subset[((subset["team_a"] == team) | (subset["team_b"] == team)) & (subset["map"] == map_name)]
        wins = int((relevant["winner"] == team).sum()); count = len(relevant)
        rows.append({"map": map_name, "maps": count, "win_rate": (wins + 4.0) / (count + 8.0)})
    return pd.DataFrame(rows)


def projected_veto(map_history: pd.DataFrame, team_a: str, team_b: str, when: pd.Timestamp, best_of: int, series_p: float) -> tuple[pd.DataFrame, str]:
    stats_a, stats_b = map_stats(map_history, team_a, when), map_stats(map_history, team_b, when)
    if stats_a.empty or stats_b.empty or stats_a["maps"].sum() + stats_b["maps"].sum() < 30:
        return pd.DataFrame(), "Map identities are unavailable on the free fixture feed. Exact series score still comes directly from historical series-score learning."
    a, b = stats_a.set_index("map"), stats_b.set_index("map")
    remaining = [m for m in ACTIVE_POOL if m in a.index and m in b.index]
    a_ban = min(remaining, key=lambda m: (a.loc[m, "win_rate"], -a.loc[m, "maps"])); remaining.remove(a_ban)
    b_ban = min(remaining, key=lambda m: (b.loc[m, "win_rate"], -b.loc[m, "maps"])); remaining.remove(b_ban)
    a_pick = max(remaining, key=lambda m: a.loc[m, "win_rate"] - b.loc[m, "win_rate"]); remaining.remove(a_pick)
    b_pick = max(remaining, key=lambda m: b.loc[m, "win_rate"] - a.loc[m, "win_rate"]); remaining.remove(b_pick)
    sequence = [(a_pick, team_a), (b_pick, team_b)]
    if best_of >= 3 and remaining:
        sequence.append((max(remaining, key=lambda m: min(a.loc[m, "win_rate"], b.loc[m, "win_rate"])), "Decider"))
    if best_of == 1:
        sequence = [(max(remaining + [a_pick, b_pick], key=lambda m: min(a.loc[m, "win_rate"], b.loc[m, "win_rate"])), "Decider")]
    if best_of == 5:
        sequence = [(m, team_a if a.loc[m, "win_rate"] >= b.loc[m, "win_rate"] else team_b) for m in sorted(ACTIVE_POOL, key=lambda m: abs(a.loc[m, "win_rate"] - b.loc[m, "win_rate"]), reverse=True)[:5]]
    base_map_p = solve_map_probability(series_p, best_of)
    rows = []
    for map_name, picker in sequence[:best_of]:
        edge = float(a.loc[map_name, "win_rate"] - b.loc[map_name, "win_rate"])
        rows.append({"Map": map_name, "Projected picker": picker, f"{team_a} chance": 100 * sigmoid(logit(base_map_p) + 1.5 * edge),
                     f"{team_a} maps": int(a.loc[map_name, "maps"]), f"{team_b} maps": int(b.loc[map_name, "maps"])})
    return pd.DataFrame(rows), "Projected veto uses available map history with Bayesian shrinkage; it is not a confirmed veto."


def team_players(detail: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for player in detail.get("players") or []:
        rows.append({"Player": player.get("name") or player.get("slug") or "Unknown", "Nationality": player.get("nationality") or "—", "Age": player.get("age") or "—"})
    return pd.DataFrame(rows)


def hot_form_metrics(state: TeamState, when: pd.Timestamp) -> dict[str, Any]:
    momentum30 = rating_momentum(state, when, 30)
    momentum60 = rating_momentum(state, when, 60)
    wr10 = win_rate(state, when, last_n=10)
    wr30 = win_rate(state, when, days=30)
    performance30 = performance_rate(state, when, 30)
    opponent_quality = state.opponent_elo_ewma
    heat = 50.0
    heat += 22.0 * math.tanh(momentum30 / 75.0)
    heat += 9.0 * math.tanh(momentum60 / 130.0)
    heat += 11.0 * (2.0 * wr10 - 1.0)
    heat += 7.0 * math.tanh(performance30 * 4.2)
    heat += 3.0 * math.tanh((opponent_quality - 1500.0) / 160.0)
    if rest_days(state, when) > 24:
        heat -= min(8.0, (rest_days(state, when) - 24.0) / 3.0)
    heat = float(np.clip(heat, 0.0, 100.0))
    if heat >= 74:
        status = "🔥 Super hot"
    elif heat >= 62:
        status = "↑ Hot"
    elif heat >= 45:
        status = "→ Stable"
    elif heat >= 34:
        status = "↓ Cooling"
    else:
        status = "❄️ Cold"
    return {"heat": heat, "status": status, "momentum30": momentum30, "momentum60": momentum60,
            "wr10": wr10, "wr30": wr30, "performance30": performance30, "opponent_quality": opponent_quality}


def elo_table(bundle: ModelBundle) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for team, state in bundle.states.items():
        components = ranking_components(state, now)
        if state.matches < 3:
            continue
        form = hot_form_metrics(state, now)
        rows.append({"Team": team, "Oracle Elo": components["oracle"], "Core": components["core"], "Fast": components["fast"],
                     "30d change": form["momentum30"], "60d change": form["momentum60"], "Heat": form["heat"],
                     "Form": form["status"], "Last 10 win %": 100 * form["wr10"], "30d win %": 100 * form["wr30"],
                     "Recent opponent Elo": form["opponent_quality"], "Peak": state.peak_oracle, "Matches": state.matches,
                     "Last match": state.last_date, "Rust days": rest_days(state, now)})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(["Oracle Elo", "Matches"], ascending=False).reset_index(drop=True)
    frame.insert(0, "Rank", np.arange(1, len(frame) + 1))
    return frame


def render_elo_list(frame: pd.DataFrame, limit: int = 50) -> None:
    if frame.empty:
        st.info("No Elo ratings are available.")
        return
    html = '<div class="card" style="padding:0">'
    for _, row in frame.head(limit).iterrows():
        change = float(row["30d change"])
        direction = "+" if change >= 0 else ""
        form_class = "superhot" if float(row["Heat"]) >= 74 else "hot" if float(row["Heat"]) >= 62 else "cold" if float(row["Heat"]) < 34 else ""
        html += (
            '<div class="rank-row">'
            f'<div class="rank-n">{int(row["Rank"])}</div>'
            f'<div class="team">{row["Team"]}<br><span class="{form_class}" style="font-size:.75rem">{row["Form"]}</span></div>'
            f'<div>{float(row["Oracle Elo"]):.0f}</div>'
            f'<div class="hide-mobile">Core {float(row["Core"]):.0f}</div>'
            f'<div class="hide-mobile">Fast {float(row["Fast"]):.0f}</div>'
            f'<div>{direction}{change:.0f} / 30d</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def team_history_frame(bundle: ModelBundle, team: str) -> pd.DataFrame:
    if bundle.ledger.empty:
        return pd.DataFrame()
    return bundle.ledger[bundle.ledger["team"] == team].sort_values("date").reset_index(drop=True)


def team_rating_series(bundle: ModelBundle, team: str) -> pd.DataFrame:
    state = bundle.states.get(team)
    if state is None or not state.rating_points:
        return pd.DataFrame()
    frame = pd.DataFrame(list(state.rating_points)).copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def elo_chart(bundle: ModelBundle, team: str, height: int = 390, compact: bool = False) -> go.Figure:
    series = team_rating_series(bundle, team)
    fig = go.Figure()
    if not series.empty:
        fig.add_trace(go.Scatter(x=series["date"], y=series["oracle"], mode="lines+markers" if not compact else "lines", name="Oracle Elo", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=series["date"], y=series["core"], mode="lines", name="Core Elo", line=dict(width=1.7, dash="dot")))
        fig.add_trace(go.Scatter(x=series["date"], y=series["fast"], mode="lines", name="Fast-form Elo", line=dict(width=1.8, dash="dash")))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=34, b=8), yaxis_title="Elo", xaxis_title=None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0), hovermode="x unified")
    if compact:
        fig.update_layout(showlegend=False, height=220, margin=dict(l=4, r=4, t=12, b=4))
    return fig


def form_pulse_chart(bundle: ModelBundle, team: str, height: int = 260) -> go.Figure:
    history = team_history_frame(bundle, team).tail(18).copy()
    fig = go.Figure()
    if not history.empty:
        history["rolling_win"] = (history["result"] == "W").astype(float).rolling(5, min_periods=1).mean() * 100
        history["cumulative_elo"] = history["delta"].cumsum()
        fig.add_trace(go.Scatter(x=history["date"], y=history["rolling_win"], mode="lines+markers", name="5-match win form", line=dict(width=3), yaxis="y"))
        fig.add_trace(go.Bar(x=history["date"], y=history["delta"], name="Elo gained/lost", opacity=.42, yaxis="y2"))
        fig.update_layout(yaxis=dict(title="Win form", range=[0, 100], ticksuffix="%"),
                          yaxis2=dict(title="Elo move", overlaying="y", side="right", showgrid=False))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=34, b=8), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    return fig


def recent_results_text(bundle: ModelBundle, team: str, count: int = 10) -> str:
    history = team_history_frame(bundle, team).tail(count)
    if history.empty:
        return "No recent results"
    return " ".join("🟢 W" if result == "W" else "🔴 L" for result in history["result"].tolist())


# ----------------------------- App shell -----------------------------
st.markdown(f"""
<div class="hero"><div class="eyebrow">CS Oracle · Today Intelligence Engine {APP_VERSION}</div>
<h1>Daily CS2 prediction intelligence</h1>
<p>One-click Today filtering, local-time scheduling, context-aware temporal stacking, event-form intelligence, Oracle Elo and calibrated exact-score forecasts. The engine ranks only the matches you choose and never claims 70% unless untouched history proves it.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.success("Pure prediction mode: no odds or bookmaker API is used.")
    st.markdown("### Match clock")
    timezone_options = ["Europe/Bucharest", "UTC", "Europe/London", "Europe/Copenhagen", "America/New_York", "Asia/Singapore"]
    display_timezone = st.selectbox("Display timezone", timezone_options, index=0)
    st.caption(f"Today is {local_now(display_timezone).strftime('%A, %d %B %Y')} in {display_timezone}.")
    st.markdown("### Accuracy settings")
    history_days = st.slider("History window (days)", 420, 1100, DEFAULT_HISTORY_DAYS, 30)
    history_pages = st.slider("Maximum API pages", 12, 32, DEFAULT_HISTORY_PAGES, 1)
    st.caption("More history improves stability but takes longer on the first load. Recent matches receive much higher weight.")
    st.markdown("### Optional data depth")
    map_upload = st.file_uploader("Map history CSV", type=["csv"], help="Optional. Enables projected veto and map-specific context.")
    player_upload = st.file_uploader("3-month player form CSV", type=["csv"], help="Optional columns: team, player, rating, adr, kast, impact, maps.")
    st.caption("Predictions work from PandaScore Free data. Player and map uploads improve the parts the free feed does not contain.")

try:
    player_data = clean_player_data(player_upload)
except Exception as exc:
    st.sidebar.error(str(exc)); player_data = pd.DataFrame()
try:
    uploaded_maps = clean_map_data(map_upload)
except Exception as exc:
    st.sidebar.error(str(exc)); uploaded_maps = pd.DataFrame()

token = token_from_secrets()
if not token:
    st.error("PandaScore token missing.")
    st.code('PANDASCORE_TOKEN = "your-private-token"', language="toml")
    st.stop()

try:
    with st.spinner("Loading completed CS2 matches and rebuilding the Today Intelligence Engine…"):
        raw_history = cached_past(token, history_days, history_pages)
        raw_upcoming = cached_upcoming(token, 7)
except APIError as exc:
    st.error(str(exc)); st.stop()

history = history_frame(raw_history)
upcoming = upcoming_frame(raw_upcoming)
api_maps = map_history_frame(raw_history)
map_history = pd.concat([api_maps, uploaded_maps], ignore_index=True) if not uploaded_maps.empty else api_maps
if not map_history.empty:
    map_history = map_history.drop_duplicates(["date", "team_a", "team_b", "map", "winner"]).sort_values("date")

if len(history) < 35:
    st.error(f"Only {len(history)} usable completed matches were returned. Increase the history pages or check the API response.")
    st.stop()

history_json = history.to_json(orient="split", date_format="iso")
with st.spinner("Training temporal ensemble, calibration, exact-score head and Elo history…"):
    bundle = train_bundle_cached(history_json)

tabs = st.tabs(["Daily predictions", "Match laboratory", "Oracle Elo & hot form", "Accuracy proof", "Data & method"])
daily_tab, match_tab, elo_tab, accuracy_tab, method_tab = tabs

with daily_tab:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Completed matches", f"{len(history):,}")
    m2.metric("Teams rated", f"{len(bundle.states):,}")
    m3.metric("Winner model", bundle.winner_model_name)
    elite_acc = 100 * bundle.metrics.get("precision_gate_accuracy", 0.0) if bundle.metrics else 0.0
    elite_matches = bundle.metrics.get("precision_gate_matches", 0) if bundle.metrics else 0
    m4.metric("Elite gate proof", f"{elite_acc:.1f}%", delta=f"{elite_matches} untouched matches")
    if bundle.winner_model is None:
        st.warning(f"The learned model needs at least {MIN_MODEL_MATCHES} completed matches. Dynamic Oracle Elo is active meanwhile.")
    elif bundle.metrics.get("precision_gate_target_met", False):
        st.success("70% Precision Mode is historically proven on the newest untouched test period. Only matches crossing the learned gate receive the Elite label.")
    else:
        st.info("The engine is active, but it will not claim 70% unless the untouched test proves it. Strong and medium signals are still ranked honestly.")

    if upcoming.empty:
        st.info("No upcoming CS2 fixtures were returned for the next seven days.")
    else:
        local_dates = upcoming["date"].dt.tz_convert(display_timezone).dt.date
        today_local = local_now(display_timezone).date()
        scope = st.radio("Show matches", ["Today", "Tomorrow", "Next 3 days", "All loaded", "Choose date"], index=0, horizontal=True)
        chosen_date = None
        if scope == "Choose date":
            available_dates = sorted(set(local_dates.tolist()))
            default_date = today_local if today_local in available_dates else (available_dates[0] if available_dates else today_local)
            chosen_date = st.date_input("Match date", value=default_date, min_value=min(available_dates) if available_dates else today_local, max_value=max(available_dates) if available_dates else today_local)
        if scope == "Today":
            date_mask = local_dates == today_local
            scope_label = f"Today · {today_local.strftime('%d %b')}"
        elif scope == "Tomorrow":
            target = today_local + timedelta(days=1)
            date_mask = local_dates == target
            scope_label = f"Tomorrow · {target.strftime('%d %b')}"
        elif scope == "Next 3 days":
            target_end = today_local + timedelta(days=2)
            date_mask = (local_dates >= today_local) & (local_dates <= target_end)
            scope_label = f"{today_local.strftime('%d %b')}–{target_end.strftime('%d %b')}"
        elif scope == "Choose date":
            date_mask = local_dates == chosen_date
            scope_label = pd.Timestamp(chosen_date).strftime("%A · %d %b")
        else:
            date_mask = pd.Series(True, index=upcoming.index)
            scope_label = "All loaded fixtures"
        scoped_upcoming = upcoming.loc[date_mask].copy()
        st.caption(f"{scope_label} · times shown in {display_timezone} · {len(scoped_upcoming)} matches")

        if scoped_upcoming.empty:
            st.info(f"No CS2 matches are currently listed for {scope_label.lower()}.")
        else:
            rows = []
            for match in scoped_upcoming.head(70).to_dict("records"):
                prediction = prediction_for_match(bundle, match, player_data)
                local_start = match["date"].tz_convert(display_timezone)
                rows.append({"Start": local_start, "Match": f'{match["team_a"]} vs {match["team_b"]}', "Event": match["event"],
                             "BO": match["best_of"], "Prediction": prediction["winner"], "Exact score": prediction["score"],
                             "Win probability": 100 * prediction["winner_p"], "Precision score": 100 * prediction["precision_score"],
                             "Signal": prediction["confidence_label"]})
            board = pd.DataFrame(rows)
            f1, f2 = st.columns([1.2, 1])
            with f1:
                filter_mode = st.radio("Prediction filter", ["All matches", "Strong + Elite", "Elite only"], index=0, horizontal=True)
            with f2:
                sort_mode = st.selectbox("Sort ranking by", ["Best overall signal", "Highest win chance", "Earliest start", "Latest start"], index=0)
            if filter_mode == "Elite only":
                shown = board[board["Signal"] == "Elite 70% proven"].copy()
            elif filter_mode == "Strong + Elite":
                shown = board[board["Signal"].isin(["Strong signal", "Elite 70% proven"])].copy()
            else:
                shown = board.copy()
            if sort_mode == "Highest win chance":
                shown = shown.sort_values(["Win probability", "Precision score"], ascending=False)
            elif sort_mode == "Earliest start":
                shown = shown.sort_values(["Start", "Precision score"], ascending=[True, False])
            elif sort_mode == "Latest start":
                shown = shown.sort_values(["Start", "Precision score"], ascending=[False, False])
            else:
                shown = shown.sort_values(["Precision score", "Win probability"], ascending=False)
            shown.insert(0, "Rank", np.arange(1, len(shown) + 1))
            st.markdown(f"### Ranked board · {scope_label}")
            st.dataframe(shown, hide_index=True, use_container_width=True,
                         column_config={"Start": st.column_config.DatetimeColumn(format="DD MMM HH:mm"),
                                        "Win probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                        "Precision score": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
            filename_date = today_local.isoformat() if scope == "Today" else "selected"
            st.download_button("Download shown predictions", shown.to_csv(index=False).encode(), f"cs_oracle_{filename_date}.csv", "text/csv", use_container_width=True)

with match_tab:
    if upcoming.empty:
        st.info("No match is available to analyse.")
    else:
        labels = [f'{row.date.tz_convert(display_timezone).strftime("%d %b %H:%M")} · {row.team_a} vs {row.team_b} · BO{row.best_of}' for row in upcoming.itertuples(index=False)]
        selected_label = st.selectbox("Choose match", labels)
        selected = upcoming.iloc[labels.index(selected_label)].to_dict()
        prediction = prediction_for_match(bundle, selected, player_data)
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown(f"### {selected['team_a']} vs {selected['team_b']}")
            st.caption(f"{selected['event']} · BO{selected['best_of']} · {selected['date'].tz_convert(display_timezone).strftime('%d %b %Y %H:%M')} {display_timezone}")
            st.markdown(f'''<div class="match-card"><div class="eyebrow">Oracle call</div><div class="winner">{prediction['winner']} {prediction['score']}</div><div><span class="pill">{pct(prediction['winner_p'])} win chance</span><span class="pill">{prediction['confidence_label']} · {prediction['confidence']}/100</span><span class="pill">{prediction['method']}</span></div></div>''', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=[selected["team_a"], selected["team_b"]], y=[prediction["team_a_p"], 1 - prediction["team_a_p"]],
                                   text=[pct(prediction["team_a_p"]), pct(1 - prediction["team_a_p"])], textposition="auto"))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("#### Exact-score distribution")
            score_frame = pd.DataFrame([{"Score": score, "Probability": 100 * probability} for score, probability in prediction["distribution"].items()])
            st.dataframe(score_frame, hide_index=True, use_container_width=True,
                         column_config={"Probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
            st.caption(prediction["overlay_note"])
            st.markdown("#### Reliability checks")
            reliability_frame = pd.DataFrame([
                {"Check": "Calibrated model", "Value": 100 * prediction["team_a_p"]},
                {"Check": "Raw XGBoost", "Value": 100 * prediction["raw_model_p"]},
                {"Check": "Oracle Elo", "Value": 100 * prediction["elo_p"]},
                {"Check": "Order-symmetry gap", "Value": 100 * prediction["symmetry_gap"]},
                {"Check": "Ensemble disagreement", "Value": 100 * prediction.get("member_spread", 0.0)},
            ])
            st.dataframe(reliability_frame, hide_index=True, use_container_width=True)

        st.markdown("### Elo trajectories before this match")
        chart_cols = st.columns(2)
        for col, team in zip(chart_cols, [selected["team_a"], selected["team_b"]]):
            with col:
                form = hot_form_metrics(bundle.states[team], selected["date"])
                st.markdown(f"#### {team} · {form['status']}")
                st.plotly_chart(elo_chart(bundle, team, height=270, compact=True), use_container_width=True)
                st.caption(f"30d Elo {form['momentum30']:+.0f} · Last 10 {100 * form['wr10']:.0f}% · Opponent-adjusted form {form['performance30']:+.3f}")
                st.caption(recent_results_text(bundle, team))
                st.plotly_chart(form_pulse_chart(bundle, team, height=220), use_container_width=True)

        st.markdown("### Why the model ranks it this way")
        a_col, b_col = st.columns(2)
        with a_col:
            st.markdown(f"#### {selected['team_a']}")
            for reason in prediction["reasons_a"]: st.write(f"• {reason}")
        with b_col:
            st.markdown(f"#### {selected['team_b']}")
            for reason in prediction["reasons_b"]: st.write(f"• {reason}")

        st.markdown("### Current Elo components")
        component_rows = []
        for team in [selected["team_a"], selected["team_b"]]:
            state = bundle.states[team]; components = effective_components(state, selected["date"], int(selected["best_of"]), int(selected["lan"]))
            form = hot_form_metrics(state, selected["date"])
            component_rows.append({"Team": team, "Oracle match Elo": components["oracle"], "Core": components["core"], "Fast": components["fast"],
                                   "LAN/online": components["context"], "Format": components["format"], "Heat": form["heat"],
                                   "Form": form["status"], "Matches": state.matches, "Rest days": rest_days(state, selected["date"])})
        st.dataframe(pd.DataFrame(component_rows), hide_index=True, use_container_width=True,
                     column_config={"Heat": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100)})

        st.markdown("### Projected veto and maps")
        veto_frame, veto_note = projected_veto(map_history, selected["team_a"], selected["team_b"], selected["date"], int(selected["best_of"]), prediction["team_a_p"])
        if veto_frame.empty: st.info(veto_note)
        else:
            st.dataframe(veto_frame, hide_index=True, use_container_width=True); st.caption(veto_note)

        st.markdown("### Current lineups")
        cols = st.columns(2)
        for col, team, team_id in [(cols[0], selected["team_a"], selected["team_a_id"]), (cols[1], selected["team_b"], selected["team_b_id"])]:
            with col:
                st.markdown(f"#### {team}")
                try:
                    players = team_players(cached_team(token, team_id))
                    if players.empty: st.caption("Current roster was not returned by the team endpoint.")
                    else: st.dataframe(players, hide_index=True, use_container_width=True)
                except APIError: st.caption("Roster request unavailable.")

        receipt = {"generated_at": datetime.now(timezone.utc).isoformat(), "app_version": APP_VERSION, "match_id": selected["match_id"],
                   "match": f"{selected['team_a']} vs {selected['team_b']}", "prediction": prediction["winner"], "score": prediction["score"],
                   "probabilities": {selected["team_a"]: prediction["team_a_p"], selected["team_b"]: 1 - prediction["team_a_p"]},
                   "precision_score": prediction["precision_score"], "signal": prediction["confidence_label"],
                   "exact_score_distribution": prediction["distribution"], "history_matches": len(history),
                   "model": "Calibrated binary XGBoost winner head + conditional multiclass exact-score head + learned precision gate"}
        receipt["sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        st.download_button("Download locked prediction receipt", json.dumps(receipt, indent=2).encode(), f"prediction_{selected['match_id']}.json", "application/json", use_container_width=True)

with elo_tab:
    st.markdown("### Oracle Elo world ranking")
    st.caption("Public Elo blends 66% stable Core Elo and 34% Fast Elo. Match predictions also use LAN/online and BO-format ratings.")
    rankings = elo_table(bundle)
    render_elo_list(rankings)
    if not rankings.empty:
        st.download_button("Download full Elo table", rankings.to_csv(index=False).encode(), "cs_oracle_elo_v4.csv", "text/csv", use_container_width=True)

        st.markdown("### 🔥 Hottest teams right now")
        hot = rankings.sort_values(["Heat", "30d change"], ascending=False).head(15).copy()
        st.dataframe(hot[["Team", "Heat", "Form", "30d change", "60d change", "Last 10 win %", "Recent opponent Elo", "Oracle Elo"]],
                     hide_index=True, use_container_width=True,
                     column_config={"Heat": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
                                    "Last 10 win %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

        st.markdown("### Detailed team Elo history")
        selected_team = st.selectbox("Choose team", rankings["Team"].tolist())
        team_history = team_history_frame(bundle, selected_team)
        state = bundle.states[selected_team]; current = ranking_components(state, pd.Timestamp.now(tz="UTC")); form = hot_form_metrics(state, pd.Timestamp.now(tz="UTC"))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Oracle Elo", f"{current['oracle']:.0f}")
        c2.metric("Core Elo", f"{current['core']:.0f}")
        c3.metric("Fast Elo", f"{current['fast']:.0f}", delta=f"{form['momentum30']:+.0f} in 30d")
        c4.metric("Heat", f"{form['heat']:.0f}/100", delta=form["status"])
        c5.metric("Last 10", f"{100 * form['wr10']:.0f}%")
        st.plotly_chart(elo_chart(bundle, selected_team, height=450), use_container_width=True)
        st.markdown("#### Recent form pulse")
        st.caption(recent_results_text(bundle, selected_team, 12))
        st.plotly_chart(form_pulse_chart(bundle, selected_team, height=300), use_container_width=True)
        if not team_history.empty:
            display = team_history.tail(35).sort_values("date", ascending=False).copy()
            display["Expected win %"] = 100 * display["expected"]
            st.dataframe(display[["date", "opponent", "event", "result", "score", "before", "after", "delta", "k", "margin_multiplier", "Expected win %"]],
                         hide_index=True, use_container_width=True,
                         column_config={"date": st.column_config.DatetimeColumn(format="DD MMM YYYY HH:mm"),
                                        "Expected win %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

        st.markdown("### Elo graph under each leading team")
        card_count = st.slider("Number of team charts", 5, 15, 8, 1)
        for _, row in rankings.head(card_count).iterrows():
            with st.expander(f"#{int(row['Rank'])} {row['Team']} · {row['Form']} · Elo {row['Oracle Elo']:.0f}"):
                a, b, c, d = st.columns(4)
                a.metric("30d Elo", f"{row['30d change']:+.0f}")
                b.metric("60d Elo", f"{row['60d change']:+.0f}")
                c.metric("Heat", f"{row['Heat']:.0f}/100")
                d.metric("Last 10", f"{row['Last 10 win %']:.0f}%")
                st.plotly_chart(elo_chart(bundle, str(row["Team"]), compact=True), use_container_width=True)
                st.caption(recent_results_text(bundle, str(row["Team"])))
                st.plotly_chart(form_pulse_chart(bundle, str(row["Team"]), height=210), use_container_width=True)

        st.markdown("### How every Elo move is calculated")
        st.markdown('<div class="formula">Expected = 1 / (1 + 10 ^ (−Elo difference / 400))<br>K = 27 × event-tier multiplier × experience multiplier × score-margin multiplier<br>Core change = K × (actual result − expected result)<br>Fast Elo uses a larger K and regresses faster toward 1500<br>Oracle match Elo = 55% Core + 25% Fast + 12% LAN/Online + 8% BO format<br>Public ranking Elo = 66% Core + 34% Fast</div>', unsafe_allow_html=True)
        st.caption("Inactive ratings regress gradually toward 1500. The Hot Form score combines 30/60-day Elo momentum, last-10 results, opponent-adjusted performance, opponent quality and rust.")

with accuracy_tab:
    st.markdown("### Strict chronological accuracy proof")
    if not bundle.metrics:
        st.info("More history is needed for the train → calibrate → untouched-test evaluation.")
    else:
        metrics = bundle.metrics
        a, b, c, d = st.columns(4)
        a.metric("Winner accuracy", f"{100 * metrics['winner_accuracy']:.1f}%", delta=f"{100 * (metrics['winner_accuracy'] - metrics['elo_accuracy']):+.1f} pp vs Elo")
        exact_text = f"{100 * metrics['exact_accuracy']:.1f}%" if pd.notna(metrics.get("exact_accuracy")) else "—"
        b.metric("Exact-score accuracy", exact_text)
        c.metric("Winner Brier", f"{metrics['winner_brier']:.3f}", delta=f"{metrics['elo_brier'] - metrics['winner_brier']:+.3f} vs Elo", delta_color="normal")
        d.metric("Calibration error", f"{100 * metrics['ece']:.1f}%", delta=f"Recent half {100 * metrics.get('recent_half_accuracy', 0):.1f}% correct")
        st.caption(f"Oldest 70% train the models, the next 14% tune calibration and the precision gate, and the newest {metrics['test_matches']:,} matches stay untouched until final testing. Winner engine: {metrics['winner_model_name']} · calibration: {metrics.get('calibrator', '—')} · confidence shrink: {metrics.get('probability_shrink', 1.0):.2f}.")

        st.markdown("### 70% Precision Gate")
        gate_cols = st.columns(4)
        gate_cols[0].metric("Untouched accuracy", f"{100 * metrics['precision_gate_accuracy']:.1f}%")
        gate_cols[1].metric("Coverage", f"{100 * metrics['precision_gate_coverage']:.1f}%")
        gate_cols[2].metric("Matches", f"{metrics['precision_gate_matches']:,}")
        gate_cols[3].metric("Status", "PROVEN" if metrics["precision_gate_target_met"] else "NOT YET")
        if metrics["precision_gate_target_met"]:
            st.success("The Elite label exceeded 70% on the newest untouched period. This applies only to the gated subset, not every match.")
        else:
            st.warning("The newest untouched period did not reach 70%. The app therefore refuses to label any live match as a proven 70% signal.")

        st.markdown("### Accuracy by learned precision score")
        confidence_frame = pd.DataFrame(metrics.get("confidence_table", []))
        if not confidence_frame.empty:
            st.dataframe(confidence_frame, hide_index=True, use_container_width=True,
                         column_config={"Coverage": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                        "Winner accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
        st.markdown("### Probability calibration")
        calibration_frame = pd.DataFrame(metrics.get("calibration_table", []))
        if not calibration_frame.empty:
            st.dataframe(calibration_frame, hide_index=True, use_container_width=True,
                         column_config={"Predicted": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                        "Actual": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
    if bundle.feature_gain:
        st.markdown("### What the winner model actually uses")
        gain = pd.DataFrame([{"Feature": feature, "Share of model gain": 100 * share} for feature, share in bundle.feature_gain.items()]).sort_values("Share of model gain", ascending=False)
        st.dataframe(gain.head(25), hide_index=True, use_container_width=True,
                     column_config={"Share of model gain": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

with method_tab:
    st.markdown("### Why this is more accurate than the previous version")
    st.markdown("""
- **Robust temporal experts:** fast-form, balanced and long-memory XGBoost models are judged across multiple chronological validation slices, with a stability penalty for models that only work in one period.
- **Leak-resistant time weighting:** every historical fit calculates recency relative to its own cutoff, not a future date.
- **Context-aware stacking:** Oracle Elo, Fast Elo, opponent-adjusted form, model disagreement, sample depth, match tier, LAN/online context and BO format enter the final stack as separate evidence and interactions.
- **Adaptive calibration:** identity, Platt, beta and isotonic calibration compete on later data; the best mapping and confidence shrink are selected before the untouched test.
- **Winner-first architecture:** the final binary engine is optimised for the correct team, while a separate conditional score head estimates 2–0, 2–1 and other scores.
- **Learned precision gate:** reliability uses model confidence, ensemble agreement, Elo agreement, symmetry, sample depth, tier and schedule context.
- **No fake 70% promise:** Elite is enabled only when the newest untouched test actually reaches at least 70% on a meaningful sample.
- **Tournament and conversion intelligence:** current-event results, favourite conversion, underdog upset rate, close-match performance, high-tier form and performance consistency are learned from only pre-match history.
- **One-click match-day board:** Today, Tomorrow, Next 3 days or a chosen local date can be ranked independently.
- **Hot-form intelligence:** Core, Fast and Oracle Elo, recent Elo moves, rolling five-match form and recent results are visualised for every leading team.
""")
    st.markdown("### Why no LLM or neural network is the main predictor")
    st.info("An LLM can summarise roster news, but it cannot manufacture missing numerical evidence. Deep neural networks normally need rich map, round and player telemetry. For this structured free dataset, a calibrated temporal gradient-boosting ensemble is a safer and usually stronger choice.")
    st.markdown("### Honest data boundary")
    st.warning("PandaScore Free does not automatically provide complete 3-month player rating, ADR, KAST, map pick/ban frequency or per-map player performance. The engine uses all lawful series-level data available and accepts optional player/map CSV files. It does not scrape third-party sites.")
    st.markdown("### Optional CSV formats")
    player_template = "team,player,rating,adr,kast,impact,maps\nTeam A,Player1,1.14,79.2,72.8,1.18,38\n"
    map_template = "date,team_a,team_b,map,winner\n2026-07-01,Team A,Team B,Mirage,Team A\n"
    x, y = st.columns(2)
    x.download_button("Download player template", player_template.encode(), "player_form_template.csv", "text/csv", use_container_width=True)
    y.download_button("Download map template", map_template.encode(), "map_history_template.csv", "text/csv", use_container_width=True)

st.caption("CS Oracle Today Intelligence uses no bookmaker feed. It provides probabilistic estimates, not certainty. Never treat a prediction as guaranteed. Regenerate any API token that has been shared publicly.")
