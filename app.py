from __future__ import annotations

import hashlib
import io
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

APP_VERSION = "3.0.0-accuracy"
API_ROOT = "https://api.pandascore.co"
DEFAULT_HISTORY_DAYS = 820
DEFAULT_HISTORY_PAGES = 26
MIN_MODEL_MATCHES = 160
ACTIVE_POOL = ["Ancient", "Anubis", "Cache", "Dust 2", "Inferno", "Mirage", "Nuke"]
SCORE_CLASSES = ["1-0", "0-1", "2-0", "2-1", "0-2", "1-2", "3-0", "3-1", "3-2", "0-3", "1-3", "2-3"]

st.set_page_config(page_title="CS Oracle Accuracy", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

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
.match-card{padding:1rem;border:1px solid var(--line);border-radius:18px;background:#0c1118;margin:.55rem 0}.winner{font-size:1.55rem;font-weight:900;letter-spacing:-.03em}.pill{display:inline-block;padding:.26rem .55rem;border:1px solid #30394a;border-radius:999px;font-size:.75rem;color:#d1d7e3;margin:.25rem .3rem 0 0}
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

    def upcoming_matches(self, days: int = 4) -> list[dict[str, Any]]:
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
}

FEATURES = [
    "oracle_elo_diff", "core_elo_diff", "fast_elo_diff", "context_elo_diff", "format_elo_diff",
    "elo_momentum_30_diff", "elo_momentum_90_diff", "winrate_14_diff", "winrate_30_diff",
    "winrate_60_diff", "winrate_90_diff", "form5_diff", "form10_diff", "form20_diff",
    "performance_30_diff", "performance_90_diff", "opp_strength_diff", "margin_30_diff",
    "margin_90_diff", "sweep_win_diff", "sweep_loss_diff", "experience_diff", "rest_diff",
    "rest_abs", "rust_diff", "fatigue_diff", "workload7_diff", "workload14_diff", "workload30_diff",
    "same_day_diff", "h2h_diff", "bo_form_diff", "lan_form_diff", "streak_diff", "sample_min",
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


def build_features(states: dict[str, TeamState], h2h: dict[tuple[str, str], deque], when: pd.Timestamp, team_a: str, team_b: str, best_of: int, lan: int, tier: float) -> dict[str, float]:
    a, b = states[team_a], states[team_b]
    ca, cb = effective_components(a, when, best_of, lan), effective_components(b, when, best_of, lan)
    rest_a, rest_b = rest_days(a, when), rest_days(b, when)
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
        "streak_diff": float(a.streak - b.streak), "sample_min": float(min(a.matches, b.matches)),
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
                "tier": tier, "sweep_win": float(a_win == 1 and score_b == 0 and best_of >= 3),
                "sweep_loss": float(a_win == 0 and score_a == 0 and best_of >= 3)}
    record_b = {"date": when, "result": float(1 - a_win), "expected": 1 - expected_a, "performance": -performance_a,
                "opponent_elo": a_comp["oracle"], "margin": -signed_margin, "best_of": best_of, "lan": lan,
                "tier": tier, "sweep_win": float(a_win == 0 and score_a == 0 and best_of >= 3),
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
        features = build_features(states, h2h, row.date, row.team_a, row.team_b, int(row.best_of), int(row.lan), float(row.tier))
        age_days = max(0.0, (newest - row.date).total_seconds() / 86400.0)
        recency_weight = 0.5 ** (age_days / 190.0)
        quality_weight = 0.70 + 0.62 * float(row.tier)
        base_weight = recency_weight * quality_weight
        x_rows.append(features); labels.append(str(row.score_class)); dates.append(row.date); originals.append(True); base_weights.append(base_weight)
        x_rows.append(mirror_features(features)); labels.append(mirror_score(str(row.score_class))); dates.append(row.date); originals.append(False); base_weights.append(base_weight)
        update_ratings(states, h2h, ledger, row.date, row.team_a, row.team_b, int(row.winner_a), int(row.score_a), int(row.score_b), int(row.best_of), int(row.lan), float(row.tier), str(row.event))

    counts = pd.Series(labels).value_counts().to_dict()
    class_weights = {label: min(2.4, math.sqrt(max(counts.values()) / max(1, count))) for label, count in counts.items()}
    weights = np.asarray([base_weights[i] * class_weights[labels[i]] for i in range(len(labels))], dtype=float)
    return TrainingData(pd.DataFrame(x_rows, columns=FEATURES), labels, np.asarray(dates, dtype="datetime64[ns]"), weights,
                        np.asarray(originals, dtype=bool), states, h2h, pd.DataFrame(ledger))


def new_model(num_classes: int, seed: int = 27, evaluation: bool = False) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300 if evaluation else 560, max_depth=3, learning_rate=0.045 if evaluation else 0.032,
        min_child_weight=7.0, subsample=0.90, colsample_bytree=0.86, reg_alpha=0.34, reg_lambda=3.8,
        gamma=0.035, objective="multi:softprob", num_class=num_classes, eval_metric="mlogloss",
        random_state=seed, n_jobs=2, tree_method="hist", max_bin=192,
    )


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    logs = np.log(np.clip(probs, 1e-9, 1.0)) / max(0.25, temperature)
    return softmax(logs)


def best_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    best_t, best_loss = 1.0, float("inf")
    for temperature in np.linspace(0.65, 1.85, 49):
        scaled = temperature_scale(probs, float(temperature))
        loss = float(log_loss(y, scaled, labels=list(range(probs.shape[1]))))
        if loss < best_loss:
            best_loss, best_t = loss, float(temperature)
    return best_t


def winner_probability_from_matrix(probs: np.ndarray, class_labels: list[str]) -> np.ndarray:
    indices = [i for i, label in enumerate(class_labels) if score_is_a_win(label)]
    return probs[:, indices].sum(axis=1) if indices else np.full(len(probs), 0.5)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if not mask.any():
            continue
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


@dataclass
class ModelBundle:
    model: xgb.XGBClassifier | None
    class_labels: list[str]
    temperature: float
    states: dict[str, TeamState]
    h2h: dict[tuple[str, str], deque]
    ledger: pd.DataFrame
    training_rows: int
    metrics: dict[str, Any]
    feature_gain: dict[str, float]
    newest_date: pd.Timestamp | None


@st.cache_resource(show_spinner=False)
def train_bundle_cached(history_json: str) -> ModelBundle:
    history = pd.read_json(io.StringIO(history_json), orient="split")
    history["date"] = pd.to_datetime(history["date"], utc=True)
    training = make_training_data(history)
    all_labels = sorted(set(training.labels), key=lambda label: SCORE_CLASSES.index(label) if label in SCORE_CLASSES else 99)
    metrics: dict[str, Any] = {}
    feature_gain: dict[str, float] = {}
    final_model: xgb.XGBClassifier | None = None
    temperature = 1.0

    if len(history) >= MIN_MODEL_MATCHES and len(all_labels) >= 4:
        unique_dates = np.sort(history["date"].dt.tz_localize(None).unique())
        train_cut = unique_dates[max(1, int(len(unique_dates) * 0.70)) - 1]
        calib_cut = unique_dates[max(2, int(len(unique_dates) * 0.84)) - 1]
        dates = training.dates
        train_mask = dates <= train_cut
        calib_mask = (dates > train_cut) & (dates <= calib_cut) & training.original
        test_mask = (dates > calib_cut) & training.original

        train_classes = sorted(set(np.asarray(training.labels)[train_mask]), key=lambda label: SCORE_CLASSES.index(label))
        encoder = {label: i for i, label in enumerate(train_classes)}
        train_known = train_mask & np.asarray([label in encoder for label in training.labels])
        if train_known.sum() >= 260 and calib_mask.sum() >= 50 and test_mask.sum() >= 70:
            eval_model = new_model(len(train_classes), 26, evaluation=True)
            y_train = np.asarray([encoder[label] for label, keep in zip(training.labels, train_known) if keep], dtype=int)
            eval_model.fit(training.x.loc[train_known, FEATURES], y_train, sample_weight=training.weights[train_known])

            def predict_known(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                known_mask = mask & np.asarray([label in encoder for label in training.labels])
                probs = eval_model.predict_proba(training.x.loc[known_mask, FEATURES])
                actual = np.asarray([encoder[label] for label, keep in zip(training.labels, known_mask) if keep], dtype=int)
                return probs, actual, known_mask

            calib_probs, calib_y, _ = predict_known(calib_mask)
            if len(calib_y):
                temperature = best_temperature(calib_probs, calib_y)
            test_probs, test_y, known_test_mask = predict_known(test_mask)
            if len(test_y):
                test_probs = temperature_scale(test_probs, temperature)
                exact_pred = test_probs.argmax(axis=1)
                class_array = np.asarray(train_classes)
                true_scores = class_array[test_y]
                pred_scores = class_array[exact_pred]
                winner_true = np.asarray([score_is_a_win(score) for score in true_scores], dtype=int)
                winner_p = winner_probability_from_matrix(test_probs, train_classes)
                winner_pred = (winner_p >= 0.5).astype(int)
                test_feature_rows = training.x.loc[known_test_mask]
                elo_baseline_p = np.asarray([logistic_elo(value) for value in test_feature_rows["oracle_elo_diff"]], dtype=float)
                confidence = np.maximum(winner_p, 1 - winner_p)
                confidence_rows = []
                for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
                    selected = confidence >= threshold
                    if selected.any():
                        confidence_rows.append({"Minimum confidence": f"{threshold:.0%}", "Coverage": 100 * selected.mean(),
                                                "Winner accuracy": 100 * accuracy_score(winner_true[selected], winner_pred[selected]),
                                                "Matches": int(selected.sum())})
                metrics = {
                    "winner_accuracy": float(accuracy_score(winner_true, winner_pred)),
                    "exact_accuracy": float(accuracy_score(true_scores, pred_scores)),
                    "winner_brier": float(brier_score_loss(winner_true, winner_p)),
                    "multiclass_logloss": float(log_loss(test_y, test_probs, labels=list(range(len(train_classes))))),
                    "ece": expected_calibration_error(winner_true, winner_p),
                    "elo_accuracy": float(accuracy_score(winner_true, elo_baseline_p >= 0.5)),
                    "elo_brier": float(brier_score_loss(winner_true, elo_baseline_p)),
                    "test_matches": int(len(test_y)), "temperature": temperature,
                    "confidence_table": confidence_rows,
                    "calibration_table": calibration_table(winner_true, winner_p).to_dict("records"),
                }

        final_encoder = {label: i for i, label in enumerate(all_labels)}
        final_y = np.asarray([final_encoder[label] for label in training.labels], dtype=int)
        final_model = new_model(len(all_labels), 27, evaluation=False)
        final_model.fit(training.x[FEATURES], final_y, sample_weight=training.weights)
        raw_gain = final_model.get_booster().get_score(importance_type="gain")
        total_gain = sum(raw_gain.values()) or 1.0
        feature_gain = {feature: float(raw_gain.get(feature, 0.0) / total_gain) for feature in FEATURES}

    return ModelBundle(final_model, all_labels, temperature, training.states, training.h2h, training.ledger,
                       len(training.x), metrics, feature_gain, history["date"].max() if not history.empty else None)


def model_distribution(bundle: ModelBundle, match: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], str]:
    features = build_features(bundle.states, bundle.h2h, match["date"], match["team_a"], match["team_b"], int(match["best_of"]), int(match["lan"]), float(match["tier"]))
    valid = valid_scores(int(match["best_of"]))
    if bundle.model is None or not bundle.class_labels:
        p_a = logistic_elo(features["oracle_elo_diff"])
        return analytic_score_distribution(p_a, int(match["best_of"])), features, "Dynamic Oracle Elo fallback"

    original = pd.DataFrame([features], columns=FEATURES)
    mirrored = pd.DataFrame([mirror_features(features)], columns=FEATURES)
    p_original = temperature_scale(bundle.model.predict_proba(original), bundle.temperature)[0]
    p_mirror = temperature_scale(bundle.model.predict_proba(mirrored), bundle.temperature)[0]
    dist_1 = {label: float(prob) for label, prob in zip(bundle.class_labels, p_original)}
    dist_2 = {mirror_score(label): float(prob) for label, prob in zip(bundle.class_labels, p_mirror)}
    combined = {score: 0.5 * (dist_1.get(score, 0.0) + dist_2.get(score, 0.0)) for score in valid}
    total = sum(combined.values())
    if total <= 1e-8:
        generic_a = sum(prob for score, prob in dist_1.items() if score_is_a_win(score))
        return analytic_score_distribution(generic_a, int(match["best_of"])), features, "Exact-format fallback from model winner probability"
    return {score: prob / total for score, prob in combined.items()}, features, "Calibrated symmetric XGBoost exact-score model"


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
    distribution, features, method = model_distribution(bundle, match)
    adjustment, overlay_note = player_overlay(player_data, match["team_a"], match["team_b"])
    distribution = apply_overlay(distribution, adjustment)
    p_a = sum(prob for score, prob in distribution.items() if score_is_a_win(score))
    predicted_score = max(distribution, key=distribution.get)
    winner = match["team_a"] if score_is_a_win(predicted_score) else match["team_b"]
    winner_p = p_a if winner == match["team_a"] else 1 - p_a
    confidence, label = confidence_score(bundle, match, p_a)
    reasons_a, reasons_b = feature_reasons(features)
    return {"distribution": dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True)), "features": features,
            "team_a_p": p_a, "winner": winner, "winner_p": winner_p, "score": predicted_score,
            "confidence": confidence, "confidence_label": label, "method": method,
            "overlay_note": overlay_note, "reasons_a": reasons_a, "reasons_b": reasons_b}


def confidence_score(bundle: ModelBundle, match: dict[str, Any], p_a: float) -> tuple[int, str]:
    a, b = bundle.states[match["team_a"]], bundle.states[match["team_b"]]
    sample_quality = min(1.0, min(a.matches, b.matches) / 28.0)
    freshness = 1.0
    if bundle.newest_date is not None:
        age = max(0.0, (pd.Timestamp.now(tz="UTC") - bundle.newest_date).total_seconds() / 86400.0)
        freshness = max(0.5, 1.0 - age / 150.0)
    separation = min(1.0, abs(p_a - 0.5) / 0.22)
    model_quality = 0.65
    if bundle.metrics:
        model_quality = min(1.0, max(0.35, (bundle.metrics["winner_accuracy"] - 0.5) / 0.18))
    score = round(100 * (0.34 * sample_quality + 0.20 * freshness + 0.31 * separation + 0.15 * model_quality))
    label = "High" if score >= 78 else "Medium" if score >= 60 else "Cautious"
    return score, label


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
    }
    scales = {"oracle_elo_diff": 1/120, "fast_elo_diff": 1/150, "context_elo_diff": 1/170,
              "format_elo_diff": 1/170, "elo_momentum_30_diff": 1/60, "winrate_30_diff": 5,
              "winrate_90_diff": 4, "performance_30_diff": 6, "opp_strength_diff": 1/180,
              "margin_30_diff": 1.6, "sweep_win_diff": 3, "rust_diff": -0.7,
              "fatigue_diff": -0.8, "h2h_diff": 2.5, "bo_form_diff": 3, "lan_form_diff": 2.5}
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


def elo_table(bundle: ModelBundle) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for team, state in bundle.states.items():
        components = ranking_components(state, now)
        if state.matches < 3:
            continue
        momentum30 = rating_momentum(state, now, 30)
        rows.append({"Team": team, "Oracle Elo": components["oracle"], "Core": components["core"], "Fast": components["fast"],
                     "30d change": momentum30, "Peak": state.peak_oracle, "Matches": state.matches,
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
    for row in frame.head(limit).itertuples(index=False):
        direction = "+" if row._5 >= 0 else ""
        html += f'''<div class="rank-row"><div class="rank-n">{row.Rank}</div><div class="team">{row.Team}</div><div>{row._2:.0f}</div><div class="hide-mobile">Core {row.Core:.0f}</div><div class="hide-mobile">Fast {row.Fast:.0f}</div><div>{direction}{row._5:.0f} / 30d</div></div>'''
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def team_history_frame(bundle: ModelBundle, team: str) -> pd.DataFrame:
    if bundle.ledger.empty:
        return pd.DataFrame()
    return bundle.ledger[bundle.ledger["team"] == team].sort_values("date").reset_index(drop=True)


# ----------------------------- App shell -----------------------------
st.markdown(f"""
<div class="hero"><div class="eyebrow">CS Oracle · Accuracy Engine {APP_VERSION}</div>
<h1>Exact-score intelligence</h1>
<p>One calibrated XGBoost model learns the complete BO1/BO3/BO5 score outcome. Dynamic multi-speed Elo, opponent-adjusted form, score margins, format strength, LAN/online strength, fatigue, rust and schedule quality are features inside that single model.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Accuracy settings")
    history_days = st.slider("History window (days)", 420, 1100, DEFAULT_HISTORY_DAYS, 30)
    history_pages = st.slider("Maximum API pages", 12, 32, DEFAULT_HISTORY_PAGES, 1)
    st.caption("More history improves stability but takes longer on the first load. Recent results still receive much higher weight.")
    st.markdown("### Optional licensed/manual depth")
    map_upload = st.file_uploader("Map history CSV", type=["csv"], help="Optional. Enables projected veto and map-specific context.")
    player_upload = st.file_uploader("3-month player form CSV", type=["csv"], help="Optional columns: team, player, rating, adr, kast, impact, maps.")
    st.caption("The free PandaScore plan does not automatically expose full ADR/rating/map history. The core model works without these uploads.")

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
    with st.spinner("Loading completed CS2 matches and rebuilding the accuracy engine…"):
        raw_history = cached_past(token, history_days, history_pages)
        raw_upcoming = cached_upcoming(token, 4)
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
with st.spinner("Training exact-score model and calculating every Elo update…"):
    bundle = train_bundle_cached(history_json)

tabs = st.tabs(["Daily predictions", "Match laboratory", "Oracle Elo", "Accuracy proof", "Data & method"])
daily_tab, match_tab, elo_tab, accuracy_tab, method_tab = tabs

with daily_tab:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Completed matches", f"{len(history):,}")
    m2.metric("Teams rated", f"{len(bundle.states):,}")
    m3.metric("Exact-score rows", f"{bundle.training_rows:,}")
    m4.metric("Newest result", history["date"].max().strftime("%d %b %Y"))
    if bundle.model is None:
        st.warning(f"The exact-score model needs at least {MIN_MODEL_MATCHES} completed matches. Dynamic Oracle Elo is active meanwhile.")
    else:
        st.success("Accuracy Engine is active: exact score is learned directly rather than guessed from one generic map probability.")
    if upcoming.empty:
        st.info("No upcoming CS2 fixtures were returned for the next four days.")
    else:
        rows = []
        predictions: dict[Any, dict[str, Any]] = {}
        for match in upcoming.head(45).to_dict("records"):
            prediction = prediction_for_match(bundle, match, player_data); predictions[match["match_id"]] = prediction
            rows.append({"Start": match["date"], "Match": f'{match["team_a"]} vs {match["team_b"]}', "Event": match["event"],
                         "BO": match["best_of"], "Prediction": prediction["winner"], "Exact score": prediction["score"],
                         "Win probability": 100 * prediction["winner_p"], "Confidence": prediction["confidence"]})
        board = pd.DataFrame(rows).sort_values(["Confidence", "Win probability"], ascending=False)
        st.markdown("### Ranked daily board")
        st.dataframe(board, hide_index=True, use_container_width=True,
                     column_config={"Start": st.column_config.DatetimeColumn(format="DD MMM HH:mm"),
                                    "Win probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                    "Confidence": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100)})
        st.download_button("Download daily predictions", board.to_csv(index=False).encode(), "cs_oracle_accuracy_daily.csv", "text/csv", use_container_width=True)

with match_tab:
    if upcoming.empty:
        st.info("No match is available to analyse.")
    else:
        labels = [f'{row.date.strftime("%d %b %H:%M UTC")} · {row.team_a} vs {row.team_b} · BO{row.best_of}' for row in upcoming.itertuples(index=False)]
        selected_label = st.selectbox("Choose match", labels)
        selected = upcoming.iloc[labels.index(selected_label)].to_dict()
        prediction = prediction_for_match(bundle, selected, player_data)
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown(f"### {selected['team_a']} vs {selected['team_b']}")
            st.caption(f"{selected['event']} · BO{selected['best_of']} · {selected['date'].strftime('%d %b %Y %H:%M UTC')}")
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
            component_rows.append({"Team": team, "Oracle match Elo": components["oracle"], "Core": components["core"], "Fast": components["fast"],
                                   "LAN/online": components["context"], "Format": components["format"], "Matches": state.matches, "Rest days": rest_days(state, selected["date"])})
        st.dataframe(pd.DataFrame(component_rows), hide_index=True, use_container_width=True)

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
                   "exact_score_distribution": prediction["distribution"], "history_matches": len(history),
                   "model": "One calibrated symmetric XGBoost exact-series-score model; Oracle Elo is an input feature"}
        receipt["sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        st.download_button("Download locked prediction receipt", json.dumps(receipt, indent=2).encode(), f"prediction_{selected['match_id']}.json", "application/json", use_container_width=True)

with elo_tab:
    st.markdown("### Oracle Elo ranking")
    st.caption("The ranking blends a stable core rating with a fast rating that reacts sharply to current form. Match predictions additionally use LAN/online and BO1/BO3/BO5 context ratings.")
    rankings = elo_table(bundle)
    render_elo_list(rankings)
    if not rankings.empty:
        st.download_button("Download full Elo table", rankings.to_csv(index=False).encode(), "cs_oracle_elo_v3.csv", "text/csv", use_container_width=True)
        st.markdown("### Team Elo history")
        selected_team = st.selectbox("Choose team", rankings["Team"].tolist())
        team_history = team_history_frame(bundle, selected_team)
        state = bundle.states[selected_team]; current = ranking_components(state, pd.Timestamp.now(tz="UTC"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Oracle Elo", f"{current['oracle']:.0f}")
        c2.metric("Core Elo", f"{current['core']:.0f}")
        c3.metric("Fast Elo", f"{current['fast']:.0f}")
        c4.metric("Peak Elo", f"{state.peak_oracle:.0f}")
        if not team_history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=team_history["date"], y=team_history["after"], mode="lines+markers", name="Match-context Oracle Elo"))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=35, b=10), yaxis_title="Elo", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
            display = team_history.tail(30).sort_values("date", ascending=False).copy()
            display["Expected win %"] = 100 * display["expected"]
            st.dataframe(display[["date", "opponent", "event", "result", "score", "before", "after", "delta", "k", "margin_multiplier", "Expected win %"]],
                         hide_index=True, use_container_width=True,
                         column_config={"date": st.column_config.DatetimeColumn(format="DD MMM YYYY HH:mm"),
                                        "Expected win %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

        st.markdown("### How every Elo move is calculated")
        st.markdown('<div class="formula">Expected = 1 / (1 + 10 ^ (−Elo difference / 400))<br>K = 27 × event-tier multiplier × experience multiplier × score-margin multiplier<br>Core change = K × (actual result − expected result)<br>Oracle match Elo = 55% Core + 25% Fast + 12% LAN/Online + 8% BO format<br>Public ranking Elo = 66% Core + 34% Fast</div>', unsafe_allow_html=True)
        st.caption("Inactive ratings regress gradually toward 1500. The fast rating has a much shorter half-life, so old hot streaks disappear faster than long-term strength.")

with accuracy_tab:
    st.markdown("### Strict chronological accuracy proof")
    if not bundle.metrics:
        st.info("More history is needed for the train → calibrate → untouched-test evaluation.")
    else:
        metrics = bundle.metrics
        a, b, c, d = st.columns(4)
        a.metric("Winner accuracy", f"{100 * metrics['winner_accuracy']:.1f}%", delta=f"{100 * (metrics['winner_accuracy'] - metrics['elo_accuracy']):+.1f} pp vs Elo")
        b.metric("Exact-score accuracy", f"{100 * metrics['exact_accuracy']:.1f}%")
        c.metric("Winner Brier", f"{metrics['winner_brier']:.3f}", delta=f"{metrics['elo_brier'] - metrics['winner_brier']:+.3f} vs Elo", delta_color="normal")
        d.metric("Calibration error", f"{100 * metrics['ece']:.1f}%")
        st.caption(f"Training uses the oldest 70% of dates, calibration uses the next 14%, and the newest {metrics['test_matches']:,} matches remain untouched until final testing. Temperature: {metrics['temperature']:.2f}.")
        st.markdown("### Accuracy when the model is selective")
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
        st.markdown("### What the model actually uses")
        gain = pd.DataFrame([{"Feature": feature, "Share of model gain": 100 * share} for feature, share in bundle.feature_gain.items()]).sort_values("Share of model gain", ascending=False)
        st.dataframe(gain.head(25), hide_index=True, use_container_width=True,
                     column_config={"Share of model gain": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

with method_tab:
    st.markdown("### What is genuinely improved")
    st.markdown("""
- **Direct exact-score learning:** BO3 2–0, 2–1, 0–2 and 1–2 are separate learned outcomes. The site no longer invents the score from one identical map probability.
- **One prediction model:** a single multiclass XGBoost model produces the complete score distribution. Elo, form and schedule intelligence are inputs to it—not separate models voting together.
- **Four Elo speeds/contexts:** stable core, fast current form, LAN/online, and BO1/BO3/BO5 ratings.
- **Opponent-adjusted evidence:** a win against a strong team and a loss against a weak team move ratings differently through pre-match expectation.
- **Score-margin intelligence:** sweeps, close series and event tier alter Elo update size and become model features.
- **Rust and fatigue:** long inactivity, overloaded schedules and same-day matches are modelled separately rather than assuming more rest is always better.
- **Symmetry protection:** every prediction is averaged with the mathematically mirrored team order to prevent Team-A/Team-B ordering bias.
- **Calibration:** probabilities are temperature-calibrated on a later period before being tested on the newest untouched period.
""")
    st.markdown("### Honest data boundary")
    st.warning("With PandaScore Free, the app cannot automatically obtain complete 3-month player rating, ADR, KAST, map pick/ban frequency or per-map player performance. The new engine is the strongest version possible from series results, event context, schedules and current team data; optional lawful CSV uploads can add player and map depth.")
    st.markdown("### Optional CSV formats")
    player_template = "team,player,rating,adr,kast,impact,maps\nTeam A,Player1,1.14,79.2,72.8,1.18,38\n"
    map_template = "date,team_a,team_b,map,winner\n2026-07-01,Team A,Team B,Mirage,Team A\n"
    x, y = st.columns(2)
    x.download_button("Download player template", player_template.encode(), "player_form_template.csv", "text/csv", use_container_width=True)
    y.download_button("Download map template", map_template.encode(), "map_history_template.csv", "text/csv", use_container_width=True)

st.caption("CS Oracle provides probabilistic estimates, not certainty. Never treat a prediction as guaranteed. Regenerate any API token that has been shared publicly.")
