from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

APP_VERSION = "1.0.0"
API_ROOT = "https://api.pandascore.co"
ACTIVE_POOL_FALLBACK = ["Ancient", "Anubis", "Cache", "Dust 2", "Inferno", "Mirage", "Nuke"]
MIN_MAP_ROWS = 160
DEFAULT_SIMULATIONS = 20_000

st.set_page_config(
    page_title="CS Oracle Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
:root { --red:#ff4655; --ink:#f6f7fb; --muted:#8c95a7; --panel:#0d1118; --line:#202735; }
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
.block-container { max-width: 1240px; padding-top: 1.25rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stHeader"] { background:rgba(7,9,13,.72); backdrop-filter: blur(12px); }
.hero { padding: 1.4rem 1.5rem; border:1px solid var(--line); border-radius:24px;
 background:radial-gradient(circle at 85% 10%, rgba(255,70,85,.18), transparent 35%),
 linear-gradient(145deg,#121722,#080b10); margin-bottom:1rem; }
.eyebrow { color:#ff7b86; font-size:.72rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; }
.hero h1 { font-size:clamp(2.2rem,7vw,4.8rem); line-height:.92; margin:.4rem 0 .55rem; letter-spacing:-.055em; }
.hero p { color:var(--muted); max-width:850px; font-size:1rem; margin:0; }
.card { padding:1.05rem 1.1rem; border:1px solid var(--line); border-radius:20px; background:linear-gradient(145deg,#111620,#090c12); height:100%; }
.card h3 { margin:.15rem 0 .2rem; }
.metric-big { font-size:2rem; font-weight:850; letter-spacing:-.04em; }
.muted { color:var(--muted); }
.good { color:#6fe0a1; } .warn { color:#ffd166; } .bad { color:#ff7b86; }
.rank-row { display:grid; grid-template-columns:54px minmax(145px,1.2fr) 1fr 1fr 1fr; gap:.7rem;
 align-items:center; padding:.82rem .9rem; border-bottom:1px solid #1d2430; }
.rank-row:last-child { border-bottom:0; }
.rank-n { font-weight:900; font-size:1.25rem; color:#ff7b86; }
.team { font-weight:800; }
.pill { display:inline-block; padding:.28rem .55rem; border-radius:999px; border:1px solid #2b3444; color:#cfd5e2; font-size:.75rem; }
.callout { padding:1rem 1.1rem; border-radius:16px; border:1px solid #2a3342; background:#0d121a; }
.source-note { color:#7f899c; font-size:.78rem; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:16px; overflow:hidden; }
.stButton > button { border-radius:14px; font-weight:800; min-height:44px; }
.stDownloadButton > button { border-radius:14px; font-weight:800; }
@media(max-width:700px){ .rank-row{grid-template-columns:42px 1fr 1fr;}.hide-mobile{display:none;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------
# Generic helpers
# -----------------------------

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


def normalise_team(name: Any) -> str:
    text = str(name or "").strip()
    aliases = {
        "natus vincere": "NAVI",
        "navi": "NAVI",
        "team vitality": "Vitality",
        "g2 esports": "G2",
        "faze clan": "FaZe",
    }
    return aliases.get(text.lower(), text)


def normalise_map(name: Any) -> str:
    text = str(name or "").strip().replace("de_", "")
    key = text.lower().replace("_", " ").replace("-", " ")
    aliases = {
        "dust2": "Dust 2",
        "dust ii": "Dust 2",
        "ancient": "Ancient",
        "anubis": "Anubis",
        "cache": "Cache",
        "inferno": "Inferno",
        "mirage": "Mirage",
        "nuke": "Nuke",
        "overpass": "Overpass",
        "train": "Train",
        "vertigo": "Vertigo",
    }
    return aliases.get(key, text.title())


def parse_dt(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.now(tz="UTC")
    return ts


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def logistic(diff: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def softmax_choice(items: list[str], scores: list[float], temperature: float, rng: np.random.Generator) -> str:
    if len(items) == 1:
        return items[0]
    arr = np.asarray(scores, dtype=float)
    arr = (arr - np.max(arr)) / max(temperature, 1e-4)
    probs = np.exp(np.clip(arr, -30, 30))
    probs /= probs.sum()
    return str(rng.choice(items, p=probs))


def token_from_secrets() -> str:
    try:
        return str(st.secrets.get("PANDASCORE_TOKEN", "")).strip()
    except Exception:
        return ""


# -----------------------------
# PandaScore client
# -----------------------------

class APIError(RuntimeError):
    pass


@dataclass
class PandaScoreClient:
    token: str
    timeout: int = 25

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.token:
            raise APIError("A PandaScore token is required.")
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        url = f"{API_ROOT}{path}"
        for attempt in range(4):
            try:
                response = requests.get(url, headers=headers, params=params or {}, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == 3:
                    raise APIError(f"Connection failed: {exc}") from exc
                time.sleep(1.2 * (attempt + 1))
                continue
            if response.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code in (401, 403):
                raise APIError(
                    f"PandaScore returned {response.status_code}. Check the token and whether this endpoint is included in your plan."
                )
            if response.status_code >= 400:
                raise APIError(f"PandaScore returned {response.status_code}: {response.text[:180]}")
            return response.json()
        raise APIError("PandaScore request failed after retries.")

    def paged(self, path: str, params: dict[str, Any] | None = None, max_pages: int = 8) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        base = dict(params or {})
        for page in range(1, max_pages + 1):
            call_params = dict(base)
            call_params.update({"page": page, "per_page": 100})
            batch = self._get(path, call_params)
            if not isinstance(batch, list):
                break
            rows.extend(batch)
            if len(batch) < 100:
                break
        return rows

    def upcoming_matches(self, days: int = 2) -> list[dict[str, Any]]:
        start = datetime.now(timezone.utc).date().isoformat()
        end = (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()
        params = {"range[begin_at]": f"{start},{end}", "sort": "begin_at"}
        return self.paged("/csgo/matches/upcoming", params, max_pages=3)

    def past_matches(self, days: int = 180, max_pages: int = 6) -> list[dict[str, Any]]:
        end = datetime.now(timezone.utc).date().isoformat()
        start = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        params = {"range[begin_at]": f"{start},{end}", "sort": "-begin_at"}
        return self.paged("/csgo/matches/past", params, max_pages=max_pages)

    def match_games(self, match_id: int | str) -> list[dict[str, Any]]:
        result = self._get(f"/csgo/matches/{match_id}/games")
        return result if isinstance(result, list) else []

    def team(self, team_id: int | str) -> dict[str, Any]:
        result = self._get(f"/csgo/teams/{team_id}")
        return result if isinstance(result, dict) else {}

    def player_stats(self, player_id: int | str, from_date: str, to_date: str) -> dict[str, Any]:
        result = self._get(
            f"/csgo/players/{player_id}/stats",
            {"from": from_date, "to": to_date, "videogame_title": "cs-2"},
        )
        return result if isinstance(result, dict) else {}


@st.cache_data(ttl=900, show_spinner=False)
def cached_upcoming(token: str, days: int) -> list[dict[str, Any]]:
    return PandaScoreClient(token).upcoming_matches(days)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_past(token: str, days: int, max_pages: int) -> list[dict[str, Any]]:
    return PandaScoreClient(token).past_matches(days, max_pages)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cached_team(token: str, team_id: int | str) -> dict[str, Any]:
    return PandaScoreClient(token).team(team_id)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cached_player_stats(token: str, player_id: int | str, from_date: str, to_date: str) -> dict[str, Any]:
    return PandaScoreClient(token).player_stats(player_id, from_date, to_date)


# -----------------------------
# Input schemas and parsing
# -----------------------------

MAP_COLUMNS = [
    "date", "match_id", "event", "tier", "lan", "best_of", "map", "team_a", "team_b",
    "team_a_rounds", "team_b_rounds", "team_a_pick", "team_b_pick", "winner",
    "team_a_players", "team_b_players",
]
PLAYER_COLUMNS = ["as_of", "team", "player", "rating", "adr", "kast", "kd", "opening_rating", "maps", "role"]
VETO_COLUMNS = ["date", "match_id", "team", "opponent", "map", "action", "order"]


def read_csv(uploaded: Any, expected: list[str]) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame(columns=expected)
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
    frame = pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else raw)
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return frame


def clean_map_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=MAP_COLUMNS)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out = out.dropna(subset=["date", "team_a", "team_b", "map"])
    out["team_a"] = out["team_a"].map(normalise_team)
    out["team_b"] = out["team_b"].map(normalise_team)
    out["winner"] = out["winner"].map(normalise_team)
    out["map"] = out["map"].map(normalise_map)
    for col in ["lan", "best_of", "team_a_rounds", "team_b_rounds", "team_a_pick", "team_b_pick"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["tier"] = out["tier"].fillna("Unknown").astype(str)
    out["event"] = out["event"].fillna("Unknown").astype(str)
    out["team_a_win"] = (out["winner"] == out["team_a"]).astype(int)
    return out.sort_values("date").reset_index(drop=True)


def clean_player_form(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PLAYER_COLUMNS)
    out = df.copy()
    out["as_of"] = pd.to_datetime(out["as_of"], utc=True, errors="coerce")
    out["team"] = out["team"].map(normalise_team)
    for col in ["rating", "adr", "kast", "kd", "opening_rating", "maps"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["role"] = out["role"].fillna("unknown").astype(str)
    return out.dropna(subset=["as_of", "team", "player"]).sort_values("as_of")


def clean_veto_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=VETO_COLUMNS)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out["team"] = out["team"].map(normalise_team)
    out["opponent"] = out["opponent"].map(normalise_team)
    out["map"] = out["map"].map(normalise_map)
    out["action"] = out["action"].astype(str).str.lower().str.strip()
    out["order"] = pd.to_numeric(out["order"], errors="coerce").fillna(99).astype(int)
    return out.dropna(subset=["date", "team", "map"]).sort_values(["date", "match_id", "order"])


def match_to_row(match: dict[str, Any]) -> dict[str, Any] | None:
    opponents = match.get("opponents") or []
    if len(opponents) < 2:
        return None
    teams = []
    for op in opponents[:2]:
        obj = op.get("opponent") or {}
        teams.append({"id": obj.get("id"), "name": normalise_team(obj.get("name")), "image": obj.get("image_url")})
    if not teams[0]["name"] or not teams[1]["name"]:
        return None
    tournament = match.get("tournament") or {}
    league = match.get("league") or {}
    serie = match.get("serie") or {}
    best_of = safe_int(match.get("number_of_games"), 3)
    return {
        "id": match.get("id"),
        "begin_at": parse_dt(match.get("begin_at")),
        "team_a": teams[0]["name"],
        "team_b": teams[1]["name"],
        "team_a_id": teams[0]["id"],
        "team_b_id": teams[1]["id"],
        "team_a_image": teams[0]["image"],
        "team_b_image": teams[1]["image"],
        "event": tournament.get("name") or serie.get("full_name") or league.get("name") or "Unknown event",
        "tier": str(tournament.get("tier") or serie.get("tier") or "Unknown"),
        "best_of": best_of if best_of in (1, 3, 5) else 3,
        "lan": int(bool(match.get("is_lan") or tournament.get("is_lan") or match.get("location") or tournament.get("location"))),
        "status": match.get("status") or "not_started",
        "raw": match,
    }


def rows_from_matches(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [row for m in matches if (row := match_to_row(m)) is not None]
    return pd.DataFrame(rows)


# -----------------------------
# Player form aggregation
# -----------------------------

PLAYER_DEFAULTS = {
    "rating": 1.00,
    "adr": 72.0,
    "kast": 70.0,
    "kd": 1.00,
    "opening_rating": 1.00,
    "maps": 0.0,
}


def latest_team_players(player_form: pd.DataFrame, team: str, as_of: pd.Timestamp) -> pd.DataFrame:
    if player_form.empty:
        return pd.DataFrame(columns=PLAYER_COLUMNS)
    subset = player_form[(player_form["team"] == team) & (player_form["as_of"] <= as_of)].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values("as_of").groupby("player", as_index=False).tail(1)
    return subset.sort_values(["maps", "rating"], ascending=False).head(5)


def aggregate_lineup(player_form: pd.DataFrame, team: str, as_of: pd.Timestamp) -> dict[str, float]:
    players = latest_team_players(player_form, team, as_of)
    if players.empty:
        return {
            "lineup_rating": 1.0,
            "lineup_adr": 72.0,
            "lineup_kast": 70.0,
            "lineup_kd": 1.0,
            "lineup_opening": 1.0,
            "star_rating": 1.0,
            "weak_rating": 1.0,
            "lineup_maps": 0.0,
            "lineup_known": 0.0,
        }
    weights = np.sqrt(np.maximum(players["maps"].fillna(0).to_numpy(dtype=float), 1.0))
    weights /= weights.sum()
    def wavg(col: str, default: float) -> float:
        values = players[col].fillna(default).to_numpy(dtype=float)
        return float(np.sum(values * weights))
    ratings = players["rating"].fillna(1.0).to_numpy(dtype=float)
    return {
        "lineup_rating": wavg("rating", 1.0),
        "lineup_adr": wavg("adr", 72.0),
        "lineup_kast": wavg("kast", 70.0),
        "lineup_kd": wavg("kd", 1.0),
        "lineup_opening": wavg("opening_rating", 1.0),
        "star_rating": float(np.max(ratings)),
        "weak_rating": float(np.min(ratings)),
        "lineup_maps": float(players["maps"].fillna(0).sum()),
        "lineup_known": min(len(players) / 5.0, 1.0),
    }


def panda_stat_value(obj: dict[str, Any], keys: Iterable[str], default: float) -> float:
    lowered = {str(k).lower(): v for k, v in obj.items()}
    for key in keys:
        if key.lower() in lowered:
            return safe_float(lowered[key.lower()], default)
    return default


def player_rows_from_api(token: str, team_id: int | str, team_name: str) -> pd.DataFrame:
    team_obj = cached_team(token, team_id)
    players = team_obj.get("players") or []
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=90)
    rows = []
    for p in players[:7]:
        pid = p.get("id")
        if not pid:
            continue
        stats: dict[str, Any] = {}
        try:
            stats = cached_player_stats(token, pid, start.isoformat(), end.isoformat())
        except Exception:
            stats = {}
        maps = panda_stat_value(stats, ["games_count", "maps_played", "games"], 0.0)
        kills = panda_stat_value(stats, ["kills"], 0.0)
        deaths = panda_stat_value(stats, ["deaths"], 0.0)
        rounds = panda_stat_value(stats, ["rounds_played", "rounds"], 0.0)
        damage = panda_stat_value(stats, ["damage_dealt", "damage"], 0.0)
        adr = panda_stat_value(stats, ["average_damage_per_round", "adr"], damage / rounds if rounds else 72.0)
        kd = panda_stat_value(stats, ["kill_death_ratio", "kd"], kills / deaths if deaths else 1.0)
        kast = panda_stat_value(stats, ["kast", "kast_percent"], 70.0)
        rating = panda_stat_value(stats, ["rating", "performance_rating"], 1.0)
        opening = panda_stat_value(stats, ["opening_rating", "opening_kill_rating"], 1.0)
        rows.append({
            "as_of": pd.Timestamp.now(tz="UTC"),
            "team": team_name,
            "player": p.get("name") or p.get("slug") or str(pid),
            "rating": rating,
            "adr": adr,
            "kast": kast,
            "kd": kd,
            "opening_rating": opening,
            "maps": maps,
            "role": p.get("role") or "unknown",
        })
    return clean_player_form(pd.DataFrame(rows, columns=PLAYER_COLUMNS))


# -----------------------------
# Historical state and feature engineering
# -----------------------------

@dataclass
class TeamState:
    elo: float = 1500.0
    map_elo: dict[str, float] | None = None
    recent: deque | None = None
    map_recent: dict[str, deque] | None = None
    last_date: pd.Timestamp | None = None
    maps_last_7: deque | None = None

    def __post_init__(self) -> None:
        self.map_elo = self.map_elo or defaultdict(lambda: 1500.0)
        self.recent = self.recent or deque(maxlen=40)
        self.map_recent = self.map_recent or defaultdict(lambda: deque(maxlen=30))
        self.maps_last_7 = self.maps_last_7 or deque(maxlen=100)


FEATURE_COLUMNS = [
    "elo_diff", "map_elo_diff", "form90_diff", "map_form90_diff", "round_form_diff",
    "map_experience_diff", "rest_diff", "load7_diff", "h2h_diff",
    "lineup_rating_diff", "lineup_adr_diff", "lineup_kast_diff", "lineup_kd_diff",
    "lineup_opening_diff", "star_diff", "weak_diff", "lineup_maps_diff", "lineup_known_diff",
    "team_a_pick", "team_b_pick", "lan", "best_of", "tier_score", "map_name",
]


def tier_score(tier: Any) -> float:
    text = str(tier or "").strip().lower()
    if text in {"s", "s-tier", "1", "tier 1", "a"}:
        return 1.0
    if text in {"b", "2", "tier 2"}:
        return 0.6
    if text in {"c", "3", "tier 3"}:
        return 0.3
    return 0.45


def weighted_rate(records: Iterable[tuple[pd.Timestamp, float, float]], now: pd.Timestamp, days: int = 90) -> float:
    vals = []
    weights = []
    for when, result, margin in records:
        age = max((now - when).days, 0)
        if age > days:
            continue
        weight = math.exp(-age / 42.0) * (1.0 + min(abs(margin), 8.0) / 16.0)
        vals.append(result)
        weights.append(weight)
    if not weights:
        return 0.5
    prior_w = 3.0
    return (sum(v * w for v, w in zip(vals, weights)) + 0.5 * prior_w) / (sum(weights) + prior_w)


def experience(records: Iterable[tuple[pd.Timestamp, float, float]], now: pd.Timestamp, days: int = 90) -> float:
    return float(sum(1 for when, _, _ in records if max((now - when).days, 0) <= days))


def h2h_rate(h2h: dict[tuple[str, str], deque], team_a: str, team_b: str, now: pd.Timestamp) -> float:
    key = tuple(sorted((team_a, team_b)))
    records = h2h.get(key, deque())
    relevant = [(d, result if team_a == key[0] else 1 - result, margin) for d, result, margin in records]
    return weighted_rate(relevant, now, 365)


def build_feature_row(
    state: dict[str, TeamState],
    h2h: dict[tuple[str, str], deque],
    player_form: pd.DataFrame,
    when: pd.Timestamp,
    team_a: str,
    team_b: str,
    map_name: str,
    best_of: int,
    lan: int,
    tier: Any,
    team_a_pick: int = 0,
    team_b_pick: int = 0,
) -> dict[str, Any]:
    a = state.setdefault(team_a, TeamState())
    b = state.setdefault(team_b, TeamState())
    map_name = normalise_map(map_name)
    a_form = weighted_rate(a.recent, when)
    b_form = weighted_rate(b.recent, when)
    a_mform = weighted_rate(a.map_recent[map_name], when)
    b_mform = weighted_rate(b.map_recent[map_name], when)
    a_round = np.average([m for d, _, m in a.recent if (when - d).days <= 90], weights=None) if any((when - d).days <= 90 for d, _, _ in a.recent) else 0.0
    b_round = np.average([m for d, _, m in b.recent if (when - d).days <= 90], weights=None) if any((when - d).days <= 90 for d, _, _ in b.recent) else 0.0
    a_rest = min(max((when - a.last_date).days if a.last_date is not None else 7, 0), 21)
    b_rest = min(max((when - b.last_date).days if b.last_date is not None else 7, 0), 21)
    a_load = sum(1 for d in a.maps_last_7 if (when - d).days <= 7)
    b_load = sum(1 for d in b.maps_last_7 if (when - d).days <= 7)
    la = aggregate_lineup(player_form, team_a, when)
    lb = aggregate_lineup(player_form, team_b, when)
    row: dict[str, Any] = {
        "elo_diff": a.elo - b.elo,
        "map_elo_diff": a.map_elo[map_name] - b.map_elo[map_name],
        "form90_diff": a_form - b_form,
        "map_form90_diff": a_mform - b_mform,
        "round_form_diff": float(a_round - b_round),
        "map_experience_diff": math.log1p(experience(a.map_recent[map_name], when)) - math.log1p(experience(b.map_recent[map_name], when)),
        "rest_diff": float(a_rest - b_rest),
        "load7_diff": float(a_load - b_load),
        "h2h_diff": h2h_rate(h2h, team_a, team_b, when) - 0.5,
        "lineup_rating_diff": la["lineup_rating"] - lb["lineup_rating"],
        "lineup_adr_diff": la["lineup_adr"] - lb["lineup_adr"],
        "lineup_kast_diff": la["lineup_kast"] - lb["lineup_kast"],
        "lineup_kd_diff": la["lineup_kd"] - lb["lineup_kd"],
        "lineup_opening_diff": la["lineup_opening"] - lb["lineup_opening"],
        "star_diff": la["star_rating"] - lb["star_rating"],
        "weak_diff": la["weak_rating"] - lb["weak_rating"],
        "lineup_maps_diff": math.log1p(la["lineup_maps"]) - math.log1p(lb["lineup_maps"]),
        "lineup_known_diff": la["lineup_known"] - lb["lineup_known"],
        "team_a_pick": int(team_a_pick),
        "team_b_pick": int(team_b_pick),
        "lan": int(lan),
        "best_of": int(best_of),
        "tier_score": tier_score(tier),
        "map_name": map_name,
    }
    return row


def update_state(
    state: dict[str, TeamState],
    h2h: dict[tuple[str, str], deque],
    when: pd.Timestamp,
    team_a: str,
    team_b: str,
    map_name: str,
    a_win: int,
    round_margin: float,
    tier: Any,
    best_of: int,
) -> None:
    a = state.setdefault(team_a, TeamState())
    b = state.setdefault(team_b, TeamState())
    expected = logistic(a.elo - b.elo)
    outcome = float(a_win)
    margin_mult = math.log1p(abs(round_margin) + 1.0) * (2.2 / (abs(a.elo - b.elo) * 0.001 + 2.2))
    k = 22.0 * (0.85 + 0.35 * tier_score(tier)) * (1.0 + 0.08 * (best_of - 1))
    delta = k * max(margin_mult, 0.55) * (outcome - expected)
    a.elo += delta
    b.elo -= delta
    map_name = normalise_map(map_name)
    expected_map = logistic(a.map_elo[map_name] - b.map_elo[map_name])
    map_delta = 28.0 * max(margin_mult, 0.55) * (outcome - expected_map)
    a.map_elo[map_name] += map_delta
    b.map_elo[map_name] -= map_delta
    a.recent.append((when, outcome, round_margin))
    b.recent.append((when, 1.0 - outcome, -round_margin))
    a.map_recent[map_name].append((when, outcome, round_margin))
    b.map_recent[map_name].append((when, 1.0 - outcome, -round_margin))
    a.last_date = when
    b.last_date = when
    a.maps_last_7.append(when)
    b.maps_last_7.append(when)
    key = tuple(sorted((team_a, team_b)))
    first_outcome = outcome if team_a == key[0] else 1.0 - outcome
    h2h.setdefault(key, deque(maxlen=20)).append((when, first_outcome, round_margin if team_a == key[0] else -round_margin))


def mirrored(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    invert_cols = [
        "elo_diff", "map_elo_diff", "form90_diff", "map_form90_diff", "round_form_diff",
        "map_experience_diff", "rest_diff", "load7_diff", "h2h_diff",
        "lineup_rating_diff", "lineup_adr_diff", "lineup_kast_diff", "lineup_kd_diff",
        "lineup_opening_diff", "star_diff", "weak_diff", "lineup_maps_diff", "lineup_known_diff",
    ]
    for col in invert_cols:
        out[col] = -safe_float(out.get(col))
    out["team_a_pick"], out["team_b_pick"] = out.get("team_b_pick", 0), out.get("team_a_pick", 0)
    return out


@dataclass
class ModelBundle:
    model: xgb.XGBClassifier
    columns: list[str]
    state: dict[str, TeamState]
    h2h: dict[tuple[str, str], deque]
    metrics: dict[str, float]
    trained_rows: int
    latest_date: pd.Timestamp

    def matrix(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        frame = pd.get_dummies(frame, columns=["map_name"], prefix="map", dtype=float)
        for col in self.columns:
            if col not in frame.columns:
                frame[col] = 0.0
        return frame[self.columns].astype(float)

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.model.predict_proba(self.matrix(rows))[:, 1]

    def contributions(self, row: dict[str, Any]) -> pd.Series:
        matrix = self.matrix([row])
        dm = xgb.DMatrix(matrix, feature_names=self.columns)
        values = self.model.get_booster().predict(dm, pred_contribs=True)[0][:-1]
        return pd.Series(values, index=self.columns).sort_values(key=lambda s: s.abs(), ascending=False)


@st.cache_resource(show_spinner=False)
def train_model_cached(map_json: str, player_json: str) -> ModelBundle:
    history = clean_map_history(pd.read_json(io.StringIO(map_json), orient="split"))
    players = clean_player_form(pd.read_json(io.StringIO(player_json), orient="split"))
    return train_model(history, players)


def train_model(history: pd.DataFrame, player_form: pd.DataFrame) -> ModelBundle:
    if len(history) < MIN_MAP_ROWS:
        raise ValueError(f"At least {MIN_MAP_ROWS} completed map rows are required; found {len(history)}.")
    state: dict[str, TeamState] = {}
    h2h: dict[tuple[str, str], deque] = {}
    rows: list[dict[str, Any]] = []
    targets: list[int] = []
    dates: list[pd.Timestamp] = []
    for r in history.itertuples(index=False):
        feature = build_feature_row(
            state, h2h, player_form, r.date, r.team_a, r.team_b, r.map,
            safe_int(r.best_of, 3), safe_int(r.lan), r.tier,
            safe_int(r.team_a_pick), safe_int(r.team_b_pick),
        )
        rows.append(feature)
        targets.append(int(r.team_a_win))
        dates.append(r.date)
        rows.append(mirrored(feature))
        targets.append(1 - int(r.team_a_win))
        dates.append(r.date)
        update_state(
            state, h2h, r.date, r.team_a, r.team_b, r.map, int(r.team_a_win),
            safe_float(r.team_a_rounds) - safe_float(r.team_b_rounds), r.tier, safe_int(r.best_of, 3),
        )
    X = pd.DataFrame(rows)
    X = pd.get_dummies(X, columns=["map_name"], prefix="map", dtype=float).astype(float)
    y = np.asarray(targets, dtype=int)
    dt = pd.Series(dates)
    order = np.argsort(dt.to_numpy())
    X = X.iloc[order].reset_index(drop=True)
    y = y[order]
    dt = dt.iloc[order].reset_index(drop=True)
    unique_dates = np.array(sorted(pd.Series(dt).drop_duplicates().tolist()))
    cutoff_idx = min(max(int(len(unique_dates) * 0.82), 1), len(unique_dates) - 1)
    split_date = unique_dates[cutoff_idx]
    train_mask = (dt < split_date).to_numpy()
    val_mask = ~train_mask
    if train_mask.sum() < 100 or val_mask.sum() < 40:
        split = max(int(len(X) * 0.82), 100)
        split = min(split, len(X) - 40)
        train_mask = np.arange(len(X)) < split
        val_mask = ~train_mask
    latest = dt.max()
    ages = np.array([(latest - d).days for d in dt], dtype=float)
    weights = np.exp(-ages / 260.0)
    params = dict(
        n_estimators=650,
        max_depth=4,
        learning_rate=0.025,
        min_child_weight=5,
        subsample=0.88,
        colsample_bytree=0.86,
        reg_alpha=0.15,
        reg_lambda=6.0,
        gamma=0.04,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )
    validation_model = xgb.XGBClassifier(**params)
    validation_model.fit(X.loc[train_mask], y[train_mask], sample_weight=weights[train_mask])
    val_prob = validation_model.predict_proba(X.loc[val_mask])[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y[val_mask], val_pred),
        "brier": brier_score_loss(y[val_mask], val_prob),
        "logloss": log_loss(y[val_mask], np.clip(val_prob, 1e-5, 1 - 1e-5)),
        "auc": roc_auc_score(y[val_mask], val_prob) if len(np.unique(y[val_mask])) == 2 else float("nan"),
    }
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, sample_weight=weights)
    return ModelBundle(final_model, list(X.columns), state, h2h, metrics, len(X), latest)


# -----------------------------
# Map pool, veto and series model
# -----------------------------


def map_pool_from_history(history: pd.DataFrame) -> list[str]:
    recent_cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=180)
    counts = history[history["date"] >= recent_cut]["map"].value_counts()
    active = [m for m in ACTIVE_POOL_FALLBACK if counts.get(m, 0) >= 2]
    extras = [m for m in counts.index if m not in active and counts[m] >= 8]
    pool = active + extras
    return pool[:7] if len(pool) >= 7 else ACTIVE_POOL_FALLBACK


def team_map_profile(history: pd.DataFrame, veto: pd.DataFrame, team: str, pool: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
    cut = as_of - pd.Timedelta(days=90)
    rows = []
    for map_name in pool:
        maps = history[(history["date"] >= cut) & (history["map"] == map_name) & ((history["team_a"] == team) | (history["team_b"] == team))]
        wins = 0
        for r in maps.itertuples(index=False):
            wins += int(r.winner == team)
        played = len(maps)
        win_rate = (wins + 2.5) / (played + 5.0)
        picks = int(((maps["team_a"] == team) & (maps["team_a_pick"] == 1)).sum() + ((maps["team_b"] == team) & (maps["team_b_pick"] == 1)).sum())
        v = veto[(veto["date"] >= cut) & (veto["team"] == team) & (veto["map"] == map_name)] if not veto.empty else pd.DataFrame()
        bans = int((v["action"] == "ban").sum()) if not v.empty else 0
        vpicks = int((v["action"] == "pick").sum()) if not v.empty else picks
        actions = max(len(v), played, 1)
        rows.append({
            "map": map_name,
            "played": played,
            "win_rate": win_rate,
            "pick_rate": (vpicks + 0.5) / (actions + 3.5),
            "ban_rate": (bans + 0.5) / (actions + 3.5),
        })
    return pd.DataFrame(rows).set_index("map")


def map_probabilities(
    bundle: ModelBundle,
    player_form: pd.DataFrame,
    match: dict[str, Any],
    pool: list[str],
    picks: dict[str, str] | None = None,
) -> dict[str, float]:
    rows = []
    for m in pool:
        a_pick = int(bool(picks) and picks.get(m) == match["team_a"])
        b_pick = int(bool(picks) and picks.get(m) == match["team_b"])
        rows.append(build_feature_row(
            bundle.state, bundle.h2h, player_form, match["begin_at"], match["team_a"], match["team_b"], m,
            match["best_of"], match["lan"], match["tier"], a_pick, b_pick,
        ))
    probs = bundle.predict(rows)
    return {m: float(np.clip(p, 0.06, 0.94)) for m, p in zip(pool, probs)}


def choose_veto(
    history: pd.DataFrame,
    veto: pd.DataFrame,
    bundle: ModelBundle,
    player_form: pd.DataFrame,
    match: dict[str, Any],
    pool: list[str],
    rng: np.random.Generator,
    first_team_a: bool,
    profiles: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    base_probs: dict[str, float] | None = None,
) -> tuple[list[str], dict[str, str], list[tuple[str, str, str]]]:
    a, b = match["team_a"], match["team_b"]
    if profiles is None:
        pa = team_map_profile(history, veto, a, pool, match["begin_at"])
        pb = team_map_profile(history, veto, b, pool, match["begin_at"])
    else:
        pa, pb = profiles
    if base_probs is None:
        base_probs = map_probabilities(bundle, player_form, match, pool)
    remaining = list(pool)
    picks: dict[str, str] = {}
    actions: list[tuple[str, str, str]] = []

    def ban(team: str, opponent: str) -> None:
        if team == a:
            own, opp = pa, pb
            team_win = base_probs
        else:
            own, opp = pb, pa
            team_win = {m: 1 - base_probs[m] for m in pool}
        scores = []
        for m in remaining:
            danger = 1 - team_win[m]
            scores.append(2.4 * own.loc[m, "ban_rate"] + 1.45 * danger + 0.35 * opp.loc[m, "pick_rate"] - 0.25 * own.loc[m, "played"] / 15.0)
        chosen = softmax_choice(remaining, scores, 0.18, rng)
        remaining.remove(chosen)
        actions.append((team, "ban", chosen))

    def pick(team: str, opponent: str) -> None:
        if team == a:
            own, opp = pa, pb
            team_win = base_probs
        else:
            own, opp = pb, pa
            team_win = {m: 1 - base_probs[m] for m in pool}
        scores = []
        for m in remaining:
            scores.append(2.0 * own.loc[m, "pick_rate"] + 1.85 * team_win[m] + 0.35 * own.loc[m, "win_rate"] - 0.55 * opp.loc[m, "win_rate"] + 0.08 * math.log1p(own.loc[m, "played"]))
        chosen = softmax_choice(remaining, scores, 0.16, rng)
        remaining.remove(chosen)
        picks[chosen] = team
        actions.append((team, "pick", chosen))

    first, second = (a, b) if first_team_a else (b, a)
    bo = match["best_of"]
    if bo == 1:
        turn = first
        other = second
        while len(remaining) > 1:
            ban(turn, other)
            turn, other = other, turn
        actions.append(("decider", "play", remaining[0]))
        return [remaining[0]], picks, actions
    if bo == 3:
        ban(first, second)
        ban(second, first)
        pick(first, second)
        pick(second, first)
        ban(first, second)
        ban(second, first)
        maps = [m for t, act, m in actions if act == "pick"] + remaining[:1]
        if remaining:
            actions.append(("decider", "play", remaining[0]))
        return maps, picks, actions
    # BO5: one ban each, alternate two picks each, last map decider
    ban(first, second)
    ban(second, first)
    pick(first, second)
    pick(second, first)
    pick(first, second)
    pick(second, first)
    maps = [m for _, act, m in actions if act == "pick"] + remaining[:1]
    if remaining:
        actions.append(("decider", "play", remaining[0]))
    return maps, picks, actions


@dataclass
class PredictionResult:
    team_a_win: float
    team_b_win: float
    score_distribution: dict[str, float]
    map_frequency: pd.DataFrame
    most_likely_maps: list[str]
    most_likely_veto: list[tuple[str, str, str]]
    map_prob: dict[str, float]
    confidence: float


def simulate_series(
    history: pd.DataFrame,
    veto: pd.DataFrame,
    bundle: ModelBundle,
    player_form: pd.DataFrame,
    match: dict[str, Any],
    simulations: int,
    seed: int = 42,
) -> PredictionResult:
    pool = map_pool_from_history(history)
    rng = np.random.default_rng(seed)
    wins_a = 0
    score_counts: dict[str, int] = defaultdict(int)
    map_counts: dict[str, int] = defaultdict(int)
    map_a_wins: dict[str, int] = defaultdict(int)
    veto_counts: dict[str, int] = defaultdict(int)
    veto_lookup: dict[str, list[tuple[str, str, str]]] = {}
    required = match["best_of"] // 2 + 1
    profiles = (
        team_map_profile(history, veto, match["team_a"], pool, match["begin_at"]),
        team_map_profile(history, veto, match["team_b"], pool, match["begin_at"]),
    )
    base_probs = map_probabilities(bundle, player_form, match, pool)
    a_pick_probs = map_probabilities(bundle, player_form, match, pool, {m: match["team_a"] for m in pool})
    b_pick_probs = map_probabilities(bundle, player_form, match, pool, {m: match["team_b"] for m in pool})
    for i in range(simulations):
        maps, picks, actions = choose_veto(
            history, veto, bundle, player_form, match, pool, rng, bool(i % 2),
            profiles=profiles, base_probs=base_probs,
        )
        a_score = 0
        b_score = 0
        for m in maps:
            map_counts[m] += 1
            if picks.get(m) == match["team_a"]:
                map_p = a_pick_probs[m]
            elif picks.get(m) == match["team_b"]:
                map_p = b_pick_probs[m]
            else:
                map_p = base_probs[m]
            if rng.random() < map_p:
                a_score += 1
                map_a_wins[m] += 1
            else:
                b_score += 1
            if a_score == required or b_score == required:
                break
        wins_a += int(a_score > b_score)
        score_counts[f"{a_score}-{b_score}"] += 1
        key = " | ".join(f"{team}:{act}:{m}" for team, act, m in actions)
        veto_counts[key] += 1
        veto_lookup[key] = actions
    team_a_win = wins_a / simulations
    score_dist = {k: v / simulations for k, v in sorted(score_counts.items(), key=lambda kv: kv[1], reverse=True)}
    rows = []
    for m, n in sorted(map_counts.items(), key=lambda kv: kv[1], reverse=True):
        rows.append({
            "Map": m,
            "Played probability": n / simulations,
            f"{match['team_a']} map win": map_a_wins[m] / max(n, 1),
            f"{match['team_b']} map win": 1 - map_a_wins[m] / max(n, 1),
        })
    map_df = pd.DataFrame(rows)
    best_veto_key = max(veto_counts, key=veto_counts.get)
    likely_maps = list(map_df.head(match["best_of"])["Map"]) if not map_df.empty else []
    direct_probs = base_probs
    certainty = abs(team_a_win - 0.5) * 2
    data_factor = min(len(history) / 1200.0, 1.0)
    lineup_a = aggregate_lineup(player_form, match["team_a"], match["begin_at"])["lineup_known"]
    lineup_b = aggregate_lineup(player_form, match["team_b"], match["begin_at"])["lineup_known"]
    confidence = float(np.clip(0.35 + 0.35 * certainty + 0.18 * data_factor + 0.12 * (lineup_a + lineup_b) / 2, 0.35, 0.96))
    return PredictionResult(team_a_win, 1 - team_a_win, score_dist, map_df, likely_maps, veto_lookup[best_veto_key], direct_probs, confidence)


# -----------------------------
# Elo ranking and explanations
# -----------------------------


def elo_table(bundle: ModelBundle, history: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=120)
    active = set(history[history["date"] >= cutoff]["team_a"]) | set(history[history["date"] >= cutoff]["team_b"])
    rows = []
    for team, s in bundle.state.items():
        if team not in active:
            continue
        recent = weighted_rate(s.recent, pd.Timestamp.now(tz="UTC"), 90)
        rows.append({"Team": team, "Oracle Elo": round(s.elo), "90-day map form": recent, "Last seen": s.last_date})
    if not rows:
        return pd.DataFrame(columns=["Rank", "Team", "Oracle Elo", "90-day map form", "Last seen"])
    out = pd.DataFrame(rows).sort_values(["Oracle Elo", "90-day map form"], ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out


def human_feature(name: str) -> str:
    labels = {
        "elo_diff": "overall Elo",
        "map_elo_diff": "map-specific Elo",
        "form90_diff": "90-day form",
        "map_form90_diff": "90-day form on this map",
        "round_form_diff": "recent round margin",
        "map_experience_diff": "recent map experience",
        "rest_diff": "rest advantage",
        "load7_diff": "seven-day workload",
        "h2h_diff": "head-to-head history",
        "lineup_rating_diff": "lineup rating",
        "lineup_adr_diff": "lineup ADR",
        "lineup_kast_diff": "lineup KAST",
        "lineup_kd_diff": "lineup K/D",
        "lineup_opening_diff": "opening-duel strength",
        "star_diff": "star-player ceiling",
        "weak_diff": "weakest-link floor",
        "lineup_maps_diff": "lineup sample size",
        "lineup_known_diff": "roster data completeness",
        "team_a_pick": "team A map pick",
        "team_b_pick": "team B map pick",
        "lan": "LAN context",
        "best_of": "series format",
        "tier_score": "event tier",
    }
    if name.startswith("map_") and name not in labels:
        return f"map identity ({name.replace('map_', '')})"
    return labels.get(name, name.replace("_", " "))


def reasons_for_map(
    bundle: ModelBundle,
    player_form: pd.DataFrame,
    match: dict[str, Any],
    map_name: str,
    pick_team: str | None = None,
) -> tuple[list[str], list[str]]:
    row = build_feature_row(
        bundle.state, bundle.h2h, player_form, match["begin_at"], match["team_a"], match["team_b"], map_name,
        match["best_of"], match["lan"], match["tier"], int(pick_team == match["team_a"]), int(pick_team == match["team_b"]),
    )
    contrib = bundle.contributions(row)
    for_a = []
    for_b = []
    for feature, value in contrib.items():
        if abs(value) < 0.005:
            continue
        text = human_feature(feature)
        if value > 0 and len(for_a) < 4:
            for_a.append(text)
        elif value < 0 and len(for_b) < 4:
            for_b.append(text)
        if len(for_a) >= 4 and len(for_b) >= 4:
            break
    return for_a, for_b


def prediction_receipt(match: dict[str, Any], result: PredictionResult, bundle: ModelBundle) -> dict[str, Any]:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_version": APP_VERSION,
        "model": "single XGBoost map-win model + rules-based veto simulation",
        "match": {k: match[k] for k in ["id", "begin_at", "team_a", "team_b", "event", "best_of"]},
        "team_a_win": result.team_a_win,
        "team_b_win": result.team_b_win,
        "score_distribution": result.score_distribution,
        "most_likely_maps": result.most_likely_maps,
        "trained_rows": bundle.trained_rows,
        "validation": bundle.metrics,
    }
    serial = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
    payload["sha256"] = hashlib.sha256(serial).hexdigest()
    return payload


# -----------------------------
# Auto-build detailed map history when plan permits
# -----------------------------


def infer_winner_name(game: dict[str, Any], team_ids: dict[Any, str]) -> str:
    winner = game.get("winner")
    if isinstance(winner, dict):
        return normalise_team(winner.get("name"))
    winner_id = game.get("winner_id") or (winner if isinstance(winner, (int, str)) else None)
    return team_ids.get(winner_id, "")


def game_score(game: dict[str, Any], team_id: Any) -> int:
    for key in ["results", "scores", "team_stats"]:
        items = game.get(key)
        if isinstance(items, list):
            for item in items:
                if item.get("team_id") == team_id or (item.get("team") or {}).get("id") == team_id:
                    value = item.get("score") if item.get("score") is not None else item.get("rounds")
                    if value is None:
                        value = item.get("value")
                    return safe_int(value, 0)
    return 0


def build_map_history_from_api(token: str, days: int = 120, max_matches: int = 240, progress=None) -> pd.DataFrame:
    client = PandaScoreClient(token)
    matches = client.past_matches(days=days, max_pages=max(2, math.ceil(max_matches / 100)))[:max_matches]
    rows = []
    for idx, match in enumerate(reversed(matches)):
        if progress is not None:
            progress.progress((idx + 1) / max(len(matches), 1), text=f"Reading match {idx + 1} of {len(matches)}")
        opponents = match.get("opponents") or []
        if len(opponents) < 2:
            continue
        ta = opponents[0].get("opponent") or {}
        tb = opponents[1].get("opponent") or {}
        team_a = normalise_team(ta.get("name"))
        team_b = normalise_team(tb.get("name"))
        if not team_a or not team_b:
            continue
        ids = {ta.get("id"): team_a, tb.get("id"): team_b}
        try:
            games = client.match_games(match.get("id"))
        except APIError:
            raise
        for g in games:
            map_obj = g.get("map") or {}
            map_name = normalise_map(map_obj.get("name") or g.get("map_name"))
            if not map_name or map_name.lower() in {"tbd", "unknown"}:
                continue
            winner = infer_winner_name(g, ids)
            if winner not in {team_a, team_b}:
                continue
            rows.append({
                "date": match.get("begin_at"),
                "match_id": match.get("id"),
                "event": (match.get("tournament") or {}).get("name") or (match.get("league") or {}).get("name") or "Unknown",
                "tier": (match.get("tournament") or {}).get("tier") or "Unknown",
                "lan": int(bool(match.get("is_lan") or (match.get("tournament") or {}).get("is_lan") or match.get("location") or (match.get("tournament") or {}).get("location"))),
                "best_of": safe_int(match.get("number_of_games"), 3),
                "map": map_name,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_rounds": game_score(g, ta.get("id")),
                "team_b_rounds": game_score(g, tb.get("id")),
                "team_a_pick": int((g.get("pick") or {}).get("team_id") == ta.get("id")),
                "team_b_pick": int((g.get("pick") or {}).get("team_id") == tb.get("id")),
                "winner": winner,
                "team_a_players": "",
                "team_b_players": "",
            })
        time.sleep(0.035)
    return clean_map_history(pd.DataFrame(rows, columns=MAP_COLUMNS))


# -----------------------------
# UI utilities
# -----------------------------


def hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">CS Oracle Pro · {APP_VERSION}</div>
          <h1>Predict the veto.<br>Then predict the match.</h1>
          <p>One map-win model combines team and map Elo, 90-day form, current lineups, player rating, ADR, KAST, opening strength, workload, event context and opponent-adjusted history. The veto simulator then produces the most likely maps, exact series score and a ranked daily match list.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def match_label(row: pd.Series) -> str:
    local = row["begin_at"].tz_convert(None).strftime("%d %b %H:%M UTC")
    return f"{local} · {row['team_a']} vs {row['team_b']} · BO{row['best_of']}"


def render_rank_list(rankings: pd.DataFrame) -> None:
    if rankings.empty:
        return
    html = '<div class="card">'
    for r in rankings.itertuples(index=False):
        html += f"""
        <div class="rank-row">
          <div class="rank-n">#{int(r.Rank)}</div>
          <div class="team">{r.Team}</div>
          <div><b>{int(getattr(r, '_2', r[2]))}</b><br><span class="muted">Oracle Elo</span></div>
          <div class="hide-mobile"><b>{pct(float(getattr(r, '_3', r[3])))}</b><br><span class="muted">90-day form</span></div>
          <div class="hide-mobile"><span class="pill">active</span></div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_lineup(players: pd.DataFrame, team: str) -> None:
    st.markdown(f"#### {team} lineup — last 90 days")
    if players.empty:
        st.caption("No detailed player statistics loaded. The model will reduce confidence rather than invent values.")
        return
    display = players[["player", "rating", "adr", "kast", "kd", "opening_rating", "maps"]].copy()
    display.columns = ["Player", "Rating", "ADR", "KAST", "K/D", "Opening", "Maps"]
    st.dataframe(display, hide_index=True, use_container_width=True)


def score_prediction_text(result: PredictionResult, match: dict[str, Any]) -> tuple[str, str]:
    best_score = max(result.score_distribution, key=result.score_distribution.get)
    a_score, b_score = [int(x) for x in best_score.split("-")]
    winner = match["team_a"] if a_score > b_score else match["team_b"]
    return winner, best_score


# -----------------------------
# Application
# -----------------------------

hero()

with st.sidebar:
    st.header("Data setup")
    default_token = token_from_secrets()
    token = st.text_input("PandaScore token", value=default_token, type="password", help="Keep it private. For Streamlit Cloud, store it in app Secrets as PANDASCORE_TOKEN.")
    st.caption("Fixtures work on all PandaScore plans. Detailed map and player statistics require the relevant historical plan or your own permitted CSV data.")
    uploaded_maps = st.file_uploader("Map history CSV", type=["csv"])
    uploaded_players = st.file_uploader("Player form CSV", type=["csv"])
    uploaded_veto = st.file_uploader("Veto history CSV", type=["csv"])
    simulations = st.slider("Series simulations", 5_000, 50_000, DEFAULT_SIMULATIONS, 5_000)

if "api_map_history" not in st.session_state:
    st.session_state.api_map_history = pd.DataFrame(columns=MAP_COLUMNS)
if "api_player_form" not in st.session_state:
    st.session_state.api_player_form = pd.DataFrame(columns=PLAYER_COLUMNS)

try:
    user_maps = clean_map_history(read_csv(uploaded_maps, MAP_COLUMNS))
    user_players = clean_player_form(read_csv(uploaded_players, PLAYER_COLUMNS))
    user_veto = clean_veto_history(read_csv(uploaded_veto, VETO_COLUMNS))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

history = user_maps if not user_maps.empty else st.session_state.api_map_history.copy()
player_form = user_players if not user_players.empty else st.session_state.api_player_form.copy()
veto_history = user_veto

setup_tab, daily_tab, elo_tab, method_tab = st.tabs(["Setup & data", "Daily predictions", "Oracle Elo", "How it calculates"])

with setup_tab:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="card"><div class="eyebrow">Map history</div><div class="metric-big">{len(history):,}</div><div class="muted">completed map rows loaded</div></div>', unsafe_allow_html=True)
    with c2:
        teams_loaded = len(set(history.get("team_a", [])) | set(history.get("team_b", []))) if not history.empty else 0
        st.markdown(f'<div class="card"><div class="eyebrow">Teams</div><div class="metric-big">{teams_loaded}</div><div class="muted">represented in history</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="eyebrow">Player snapshots</div><div class="metric-big">{len(player_form):,}</div><div class="muted">3-month form records</div></div>', unsafe_allow_html=True)

    st.markdown("### Fastest correct setup")
    st.markdown(
        "Use a private PandaScore token for the daily fixture list. For genuinely detailed predictions, either use a PandaScore plan that exposes historical games and player stats, or upload permitted CSV data using the included templates. The app intentionally does not scrape HLTV: HLTV's current terms explicitly prohibit data mining and web scraping."
    )
    a, b, c = st.columns([1, 1, 1])
    with a:
        if st.button("Build map database from API", use_container_width=True, disabled=not bool(token)):
            progress = st.progress(0, text="Preparing history")
            try:
                built = build_map_history_from_api(token, days=150, max_matches=280, progress=progress)
                progress.empty()
                if len(built) < MIN_MAP_ROWS:
                    st.error(f"Only {len(built)} map rows were returned. The token may not include detailed historical game access.")
                else:
                    st.session_state.api_map_history = built
                    st.success(f"Loaded {len(built)} map rows. Rerunning now.")
                    st.rerun()
            except APIError as exc:
                progress.empty()
                st.error(str(exc))
    with b:
        if not history.empty:
            st.download_button("Download current map database", history.to_csv(index=False).encode(), "cs_oracle_map_history.csv", "text/csv", use_container_width=True)
    with c:
        st.download_button("Download data templates", data=(
            "Use the three template files included in the ZIP: map_history_template.csv, player_form_template.csv and veto_history_template.csv."
        ), file_name="read_templates.txt", use_container_width=True)

    if len(history) < MIN_MAP_ROWS:
        st.warning(f"The single XGBoost model needs at least {MIN_MAP_ROWS} completed map rows. It currently has {len(history)}. Daily fixtures can load, but real predictions remain locked until enough history is present.")
    elif history["date"].max() < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=45):
        st.warning("The newest map data is more than 45 days old. Update it before trusting predictions.")
    else:
        st.success("The map database is large and recent enough to train the prediction model.")

with daily_tab:
    if not token:
        st.info("Add your private PandaScore token in the sidebar or in Streamlit app Secrets to load the daily schedule.")
    else:
        try:
            with st.spinner("Loading today's CS2 matches…"):
                upcoming = rows_from_matches(cached_upcoming(token, 3))
        except APIError as exc:
            st.error(str(exc))
            upcoming = pd.DataFrame()
        if upcoming.empty:
            st.info("No upcoming CS2 matches were returned for the next three days.")
        elif len(history) < MIN_MAP_ROWS:
            st.warning("Matches loaded, but predictions are locked until the map history database contains enough rows.")
            st.dataframe(upcoming[["begin_at", "team_a", "team_b", "event", "best_of"]], hide_index=True, use_container_width=True)
        else:
            with st.spinner("Training the single map-win model…"):
                bundle = train_model_cached(history.to_json(orient="split", date_format="iso"), player_form.to_json(orient="split", date_format="iso"))

            st.markdown("### Daily prediction board")
            st.caption("Generate every listed match with a lighter simulation pass, then open one match below for the full deep analysis.")
            if st.button("Generate all daily predictions", use_container_width=True):
                board_rows = []
                board_progress = st.progress(0, text="Generating daily board")
                for board_i, (_, board_match_row) in enumerate(upcoming.head(24).iterrows()):
                    board_match = board_match_row.to_dict()
                    board_result = simulate_series(
                        history, veto_history, bundle, player_form, board_match,
                        simulations=min(5_000, simulations), seed=100 + board_i,
                    )
                    board_winner, board_score = score_prediction_text(board_result, board_match)
                    winner_probability = board_result.team_a_win if board_winner == board_match["team_a"] else board_result.team_b_win
                    board_rows.append({
                        "Time UTC": board_match["begin_at"].strftime("%d %b %H:%M"),
                        "Match": f"{board_match['team_a']} vs {board_match['team_b']}",
                        "Event": board_match["event"],
                        "BO": board_match["best_of"],
                        "Predicted winner": board_winner,
                        "Win probability": winner_probability,
                        "Predicted score": board_score,
                        "Likely maps": ", ".join(board_result.most_likely_maps[:board_match["best_of"]]),
                        "Confidence": board_result.confidence,
                    })
                    board_progress.progress((board_i + 1) / min(len(upcoming), 24), text=f"Match {board_i + 1} of {min(len(upcoming), 24)}")
                board_progress.empty()
                st.session_state.daily_prediction_board = pd.DataFrame(board_rows).sort_values("Win probability", ascending=False)
            if "daily_prediction_board" in st.session_state and not st.session_state.daily_prediction_board.empty:
                board_display = st.session_state.daily_prediction_board.copy()
                board_display["Win probability"] = board_display["Win probability"].map(pct)
                board_display["Confidence"] = board_display["Confidence"].map(pct)
                st.dataframe(board_display, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download daily prediction board",
                    st.session_state.daily_prediction_board.to_csv(index=False).encode(),
                    "cs_oracle_daily_predictions.csv",
                    "text/csv",
                )

            st.markdown("### Deep match analysis")
            labels = [match_label(r) for _, r in upcoming.iterrows()]
            selected_label = st.selectbox("Choose match", labels)
            selected = upcoming.iloc[labels.index(selected_label)].to_dict()

            # Try to enrich current roster/player data. Failure is safe and visible.
            api_players = []
            if selected.get("team_a_id"):
                try:
                    api_players.append(player_rows_from_api(token, selected["team_a_id"], selected["team_a"]))
                except Exception:
                    pass
            if selected.get("team_b_id"):
                try:
                    api_players.append(player_rows_from_api(token, selected["team_b_id"], selected["team_b"]))
                except Exception:
                    pass
            if api_players:
                auto_players = pd.concat(api_players, ignore_index=True)
                if not auto_players.empty and auto_players["maps"].sum() > 0:
                    player_form_live = pd.concat([player_form, auto_players], ignore_index=True)
                else:
                    player_form_live = player_form
            else:
                player_form_live = player_form

            with st.spinner("Simulating vetoes and complete series…"):
                result = simulate_series(history, veto_history, bundle, player_form_live, selected, simulations)
            winner, exact_score = score_prediction_text(result, selected)

            left, middle, right = st.columns([1.25, 1, 1])
            with left:
                st.markdown(f"""
                <div class="card">
                  <div class="eyebrow">Most likely winner</div>
                  <div class="metric-big">{winner}</div>
                  <div class="muted">Exact score: <b>{exact_score}</b> · model confidence {pct(result.confidence)}</div>
                </div>
                """, unsafe_allow_html=True)
            with middle:
                st.markdown(f"""
                <div class="card">
                  <div class="eyebrow">{selected['team_a']}</div>
                  <div class="metric-big">{pct(result.team_a_win)}</div>
                  <div class="muted">series win probability</div>
                </div>
                """, unsafe_allow_html=True)
            with right:
                st.markdown(f"""
                <div class="card">
                  <div class="eyebrow">{selected['team_b']}</div>
                  <div class="metric-big">{pct(result.team_b_win)}</div>
                  <div class="muted">series win probability</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Most likely map veto")
            veto_cols = st.columns(max(len(result.most_likely_veto), 1))
            for col, action in zip(veto_cols, result.most_likely_veto):
                team, act, map_name = action
                with col:
                    st.markdown(f'<div class="card"><div class="eyebrow">{act}</div><h3>{map_name}</h3><div class="muted">{team}</div></div>', unsafe_allow_html=True)

            st.markdown("### Map-by-map prediction")
            if not result.map_frequency.empty:
                map_display = result.map_frequency.copy()
                for col in map_display.columns[1:]:
                    map_display[col] = map_display[col].map(lambda x: pct(float(x)))
                st.dataframe(map_display, hide_index=True, use_container_width=True)

            st.markdown("### Exact score distribution")
            score_df = pd.DataFrame({"Score": list(result.score_distribution), "Probability": list(result.score_distribution.values())})
            score_df = score_df.head(6)
            fig = go.Figure(go.Bar(x=score_df["Score"], y=score_df["Probability"], text=[pct(x) for x in score_df["Probability"]], textposition="outside"))
            fig.update_layout(height=320, margin=dict(l=15, r=15, t=15, b=15), yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            likely_pick_map = result.most_likely_maps[0] if result.most_likely_maps else map_pool_from_history(history)[0]
            a_reasons, b_reasons = reasons_for_map(bundle, player_form_live, selected, likely_pick_map)
            st.markdown(f"### Why the model leans this way on {likely_pick_map}")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"**{selected['team_a']} advantages**")
                st.write(" · ".join(a_reasons) if a_reasons else "No strong positive feature contribution.")
            with rc2:
                st.markdown(f"**{selected['team_b']} advantages**")
                st.write(" · ".join(b_reasons) if b_reasons else "No strong positive feature contribution.")

            la = latest_team_players(player_form_live, selected["team_a"], selected["begin_at"])
            lb = latest_team_players(player_form_live, selected["team_b"], selected["begin_at"])
            p1, p2 = st.columns(2)
            with p1:
                render_lineup(la, selected["team_a"])
            with p2:
                render_lineup(lb, selected["team_b"])

            st.markdown("### Model health")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Time-split accuracy", pct(bundle.metrics["accuracy"]))
            h2.metric("Brier score", f"{bundle.metrics['brier']:.3f}", help="Lower is better.")
            h3.metric("Log loss", f"{bundle.metrics['logloss']:.3f}", help="Lower is better.")
            h4.metric("Training rows", f"{bundle.trained_rows:,}")
            receipt = prediction_receipt(selected, result, bundle)
            st.download_button("Download timestamped prediction receipt", json.dumps(receipt, indent=2, default=str), f"prediction_{selected['id']}.json", "application/json")

            st.caption("Probabilities are model estimates, not guarantees. The app lowers confidence when lineups, veto history or recent detailed stats are missing.")

with elo_tab:
    if len(history) < MIN_MAP_ROWS:
        st.info("Load enough map history to generate the Oracle Elo list.")
    else:
        bundle = train_model_cached(history.to_json(orient="split", date_format="iso"), player_form.to_json(orient="split", date_format="iso"))
        rankings = elo_table(bundle, history)
        st.markdown("### Current team strength ranking")
        render_rank_list(rankings.head(40))
        st.download_button("Download Elo list", rankings.to_csv(index=False).encode(), "cs_oracle_elo.csv", "text/csv")

with method_tab:
    st.markdown("## One prediction model, not a pile of conflicting models")
    st.markdown(
        "The only learned predictor is an **XGBoost map-win probability model**. Every historical map is processed chronologically, so features are calculated using only information available before that map. Each training row is mirrored to remove arbitrary Team A bias, recent maps receive more weight, and the final probability is tested on the newest unseen portion of the timeline."
    )
    st.markdown("### What enters each map prediction")
    st.markdown(
        "Overall team Elo; map-specific Elo; 90-day form; 90-day map form; opponent-adjusted head-to-head; recent round margin; map experience; rest; seven-day workload; event tier; LAN context; series format; map-pick ownership; and current five-player aggregates for rating, ADR, KAST, K/D, opening strength, star ceiling, weakest-player floor and sample size."
    )
    st.markdown("### How the maps and score are generated")
    st.markdown(
        "The model first calculates a win probability on every map in the active pool. A veto simulator then uses each team's recent pick, ban, play and win tendencies to run both possible veto orders thousands of times. Every simulated veto is played map by map until one team wins the series. The result is a winner probability, exact-score distribution and probability that each map appears."
    )
    st.markdown("### Accuracy rules built into the site")
    st.markdown(
        "No random train/test mixing, no future leakage, no invented roster statistics, no silent fallback to fake data, no claim of certainty, and every prediction can be downloaded with a timestamp and SHA-256 fingerprint. To prove world-leading accuracy, keep those receipts and evaluate at least 100 genuinely pre-match predictions."
    )
    st.markdown("### Data-source note")
    st.markdown(
        "Daily fixtures and permitted statistics use PandaScore's authenticated server-side API. The app does not automate HLTV because HLTV's terms prohibit data mining and web scraping. GRID Open Access is another suitable official-data source if you obtain access and later add an adapter."
    )

st.markdown("<div class='source-note'>CS Oracle Pro · predictions are informational estimates · never treat any model as certain</div>", unsafe_allow_html=True)
