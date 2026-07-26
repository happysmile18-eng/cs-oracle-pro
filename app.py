from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
import unicodedata
from difflib import SequenceMatcher
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

APP_VERSION = "5.0.0-market-edge"
TARGET_PRECISION = 0.70
API_ROOT = "https://api.pandascore.co"
ODDS_API_ROOT = "https://api.odds-api.io/v3"
DEFAULT_HISTORY_DAYS = 820
DEFAULT_HISTORY_PAGES = 26
MIN_MODEL_MATCHES = 160
ACTIVE_POOL = ["Ancient", "Anubis", "Cache", "Dust 2", "Inferno", "Mirage", "Nuke"]
SCORE_CLASSES = ["1-0", "0-1", "2-0", "2-1", "0-2", "1-2", "3-0", "3-1", "3-2", "0-3", "1-3", "2-3"]

st.set_page_config(page_title="CS Oracle Market Edge", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

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


def odds_token_from_secrets() -> str:
    try:
        return str(st.secrets.get("ODDS_API_KEY", "")).strip()
    except Exception:
        return ""


def canonical_team_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
    aliases = {
        "natus vincere": "navi", "team vitality": "vitality", "g2 esports": "g2",
        "faze clan": "faze", "mousesports": "mouz", "team liquid": "liquid",
        "ninjas in pyjamas": "nip", "virtus pro": "virtuspro", "virtus.pro": "virtuspro",
        "astralis talent": "astralistalent", "spirit academy": "spiritacademy",
    }
    text = aliases.get(text.strip(), text)
    text = re.sub(r"\b(team|esports|e-sports|gaming|club|clan|organization|organisation)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def team_name_similarity(left: Any, right: Any) -> float:
    a, b = canonical_team_key(left), canonical_team_key(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    containment = min(len(a), len(b)) / max(len(a), len(b)) if a in b or b in a else 0.0
    prefix = 0.92 * min(len(a), len(b)) / max(len(a), len(b)) if a[:5] == b[:5] and min(len(a), len(b)) >= 5 else 0.0
    return float(max(ratio, containment, prefix))


class OddsAPIError(RuntimeError):
    pass


@dataclass
class OddsClient:
    api_key: str
    timeout: int = 25

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            raise OddsAPIError("Odds API key missing.")
        query = dict(params or {})
        query["apiKey"] = self.api_key
        last_error = "Unknown Odds-API.io error"
        for attempt in range(4):
            try:
                response = requests.get(f"{ODDS_API_ROOT}{path}", params=query, timeout=self.timeout, headers={"Accept": "application/json"})
            except requests.RequestException as exc:
                last_error = f"Odds connection failed: {exc}"
                time.sleep(0.8 + attempt)
                continue
            if response.status_code == 429:
                last_error = "Odds-API.io rate limit reached"
                time.sleep(1.2 + 1.2 * attempt)
                continue
            if response.status_code in (401, 403):
                raise OddsAPIError(f"Odds-API.io returned {response.status_code}. Check ODDS_API_KEY and selected bookmakers.")
            if response.status_code >= 400:
                raise OddsAPIError(f"Odds-API.io returned {response.status_code}: {response.text[:200]}")
            return response.json()
        raise OddsAPIError(last_error)

    @staticmethod
    def _rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("data", "events", "results", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            if payload.get("id") is not None:
                return [payload]
        return []

    def events(self) -> list[dict[str, Any]]:
        payload = self._get("/events", {"sport": "esports", "status": "pending", "limit": 100})
        return self._rows(payload)

    def odds_for_events(self, event_ids: list[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for start in range(0, len(event_ids), 10):
            batch = event_ids[start:start + 10]
            if not batch:
                continue
            try:
                payload = self._get("/odds/multi", {"eventIds": ",".join(batch)})
                rows = self._rows(payload)
                if rows:
                    output.extend(rows)
                    continue
            except OddsAPIError:
                pass
            for event_id in batch:
                try:
                    output.extend(self._rows(self._get("/odds", {"eventId": event_id})))
                except OddsAPIError:
                    continue
        return output


@st.cache_data(ttl=420, max_entries=8, show_spinner=False)
def cached_odds_events(api_key: str) -> list[dict[str, Any]]:
    return OddsClient(api_key).events()


@st.cache_data(ttl=60, max_entries=16, show_spinner=False)
def cached_odds_payloads(api_key: str, event_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    return OddsClient(api_key).odds_for_events(list(event_ids))


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


WINNER_MODEL_CONFIGS = [
    {"name": "Conservative depth-2", "n_estimators": 440, "max_depth": 2, "learning_rate": 0.035, "min_child_weight": 9.0, "subsample": 0.92, "colsample_bytree": 0.90, "reg_alpha": 0.30, "reg_lambda": 4.8, "gamma": 0.02},
    {"name": "Balanced depth-3", "n_estimators": 520, "max_depth": 3, "learning_rate": 0.030, "min_child_weight": 8.0, "subsample": 0.90, "colsample_bytree": 0.86, "reg_alpha": 0.38, "reg_lambda": 5.2, "gamma": 0.04},
    {"name": "Highly regularised depth-4", "n_estimators": 580, "max_depth": 4, "learning_rate": 0.024, "min_child_weight": 13.0, "subsample": 0.88, "colsample_bytree": 0.82, "reg_alpha": 0.55, "reg_lambda": 6.5, "gamma": 0.07},
]

GATE_COLUMNS = [
    "model_confidence", "entropy", "elo_confidence", "model_elo_agree", "model_elo_gap",
    "symmetry_quality", "sample_quality", "tier", "abs_oracle_elo", "abs_fast_elo",
    "abs_momentum_30", "abs_form_30", "rest_abs", "best_of",
]


def new_winner_model(config: dict[str, Any], seed: int = 27) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **{key: value for key, value in config.items() if key != "name"},
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
class ModelBundle:
    winner_model: xgb.XGBClassifier | None
    score_model: xgb.XGBClassifier | None
    class_labels: list[str]
    score_temperature: float
    winner_calibrator: LogisticRegression | None
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
    final_winner: xgb.XGBClassifier | None = None
    final_score: xgb.XGBClassifier | None = None
    winner_calibrator: LogisticRegression | None = None
    gate_model: Any = None
    gate_threshold = 0.90
    gate_validation: dict[str, Any] = {}
    score_temperature = 1.0
    selected_config = WINNER_MODEL_CONFIGS[1]

    if len(history) >= MIN_MODEL_MATCHES:
        unique_dates = np.sort(history["date"].dt.tz_localize(None).unique())
        train_cut = unique_dates[max(1, int(len(unique_dates) * 0.70)) - 1]
        calib_cut = unique_dates[max(2, int(len(unique_dates) * 0.84)) - 1]
        dates = training.dates
        train_mask = dates <= train_cut
        calib_mask = (dates > train_cut) & (dates <= calib_cut) & training.original
        test_mask = (dates > calib_cut) & training.original

        if train_mask.sum() >= 300 and calib_mask.sum() >= 70 and test_mask.sum() >= 80:
            calib_x = training.x.loc[calib_mask, FEATURES]
            calib_y = winner_y_all[calib_mask]
            winner_candidates: list[tuple[float, dict[str, Any], xgb.XGBClassifier, np.ndarray, np.ndarray]] = []
            for candidate_i, config in enumerate(WINNER_MODEL_CONFIGS):
                model = new_winner_model(config, 40 + candidate_i)
                model.fit(training.x.loc[train_mask, FEATURES], winner_y_all[train_mask], sample_weight=training.weights[train_mask])
                raw_p, sym_gap = symmetric_binary_probs(model, calib_x)
                score = float(brier_score_loss(calib_y, raw_p) + 0.12 * log_loss(calib_y, raw_p, labels=[0, 1]) - 0.025 * accuracy_score(calib_y, raw_p >= 0.5))
                winner_candidates.append((score, config, model, raw_p, sym_gap))
            _, selected_config, eval_winner_model, calib_raw_p, calib_sym_gap = min(winner_candidates, key=lambda item: item[0])
            winner_calibrator = fit_platt(calib_raw_p, calib_y)
            calib_p = apply_platt(winner_calibrator, calib_raw_p)
            calib_elo_p = np.asarray([logistic_elo(value) for value in calib_x["oracle_elo_diff"]], dtype=float)
            calib_gate_x = gate_frame(calib_x, calib_p, calib_elo_p, calib_sym_gap)
            calib_pred = (calib_p >= 0.5).astype(int)
            calib_correct = (calib_pred == calib_y).astype(int)
            gate_split = max(35, int(len(calib_gate_x) * 0.62))
            gate_split = min(gate_split, len(calib_gate_x) - 20)
            if gate_split >= 30 and len(np.unique(calib_correct[:gate_split])) >= 2:
                gate_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.45, max_iter=1200))
                gate_model.fit(calib_gate_x.iloc[:gate_split], calib_correct[:gate_split])
                valid_scores = gate_model.predict_proba(calib_gate_x.iloc[gate_split:])[:, 1]
                gate_threshold, gate_validation = choose_precision_threshold(valid_scores, calib_correct[gate_split:])
                gate_model.fit(calib_gate_x, calib_correct)

            # Separate exact-score head, with winner probability later forced to the stronger binary head.
            score_train_labels = sorted(set(labels_array[train_mask]), key=lambda label: SCORE_CLASSES.index(label))
            score_encoder = {label: i for i, label in enumerate(score_train_labels)}
            score_train_mask = train_mask & np.asarray([label in score_encoder for label in labels_array])
            eval_score_model = None
            if len(score_train_labels) >= 4 and score_train_mask.sum() >= 280:
                eval_score_model = new_score_model(len(score_train_labels), 61, final=False)
                score_y_train = np.asarray([score_encoder[label] for label in labels_array[score_train_mask]], dtype=int)
                eval_score_model.fit(training.x.loc[score_train_mask, FEATURES], score_y_train, sample_weight=training.weights[score_train_mask])
                score_calib_known = calib_mask & np.asarray([label in score_encoder for label in labels_array])
                if score_calib_known.sum() >= 50:
                    score_calib_x = training.x.loc[score_calib_known, FEATURES]
                    score_calib_probs = symmetric_multiclass_probs(eval_score_model, score_calib_x, score_train_labels, 1.0)
                    score_calib_y = np.asarray([score_encoder[label] for label in labels_array[score_calib_known]], dtype=int)
                    score_temperature = best_temperature(score_calib_probs, score_calib_y)

            test_x = training.x.loc[test_mask, FEATURES]
            test_y = winner_y_all[test_mask]
            test_raw_p, test_sym_gap = symmetric_binary_probs(eval_winner_model, test_x)
            test_p = apply_platt(winner_calibrator, test_raw_p)
            test_pred = (test_p >= 0.5).astype(int)
            test_elo_p = np.asarray([logistic_elo(value) for value in test_x["oracle_elo_diff"]], dtype=float)
            test_gate_x = gate_frame(test_x, test_p, test_elo_p, test_sym_gap)
            gate_scores = gate_model.predict_proba(test_gate_x)[:, 1] if gate_model is not None else np.maximum(test_p, 1 - test_p)
            elite = gate_scores >= gate_threshold

            exact_accuracy = float("nan")
            exact_logloss = float("nan")
            if eval_score_model is not None:
                test_score_known = test_mask & np.asarray([label in score_encoder for label in labels_array])
                if test_score_known.sum() >= 50:
                    test_score_x = training.x.loc[test_score_known, FEATURES]
                    score_probs = symmetric_multiclass_probs(eval_score_model, test_score_x, score_train_labels, score_temperature)
                    binary_for_score_raw, _ = symmetric_binary_probs(eval_winner_model, test_score_x)
                    binary_for_score = apply_platt(winner_calibrator, binary_for_score_raw)
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
            metrics = {
                "winner_accuracy": float(accuracy_score(test_y, test_pred)),
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
                "winner_model_name": selected_config["name"],
            }

        final_winner = new_winner_model(selected_config, 77)
        final_winner.fit(training.x[FEATURES], winner_y_all, sample_weight=training.weights)
        if len(all_score_labels) >= 4:
            score_encoder_all = {label: i for i, label in enumerate(all_score_labels)}
            score_y_all = np.asarray([score_encoder_all[label] for label in labels_array], dtype=int)
            final_score = new_score_model(len(all_score_labels), 79, final=True)
            final_score.fit(training.x[FEATURES], score_y_all, sample_weight=training.weights)
        raw_gain = final_winner.get_booster().get_score(importance_type="gain")
        total_gain = sum(raw_gain.values()) or 1.0
        feature_gain = {feature: float(raw_gain.get(feature, 0.0) / total_gain) for feature in FEATURES}

    return ModelBundle(final_winner, final_score, all_score_labels, score_temperature, winner_calibrator,
                       gate_model, gate_threshold, gate_validation, training.states, training.h2h,
                       training.ledger, len(training.x), metrics, feature_gain,
                       history["date"].max() if not history.empty else None, selected_config["name"])


def winner_reliability(bundle: ModelBundle, features: dict[str, float]) -> dict[str, float]:
    frame = pd.DataFrame([features], columns=FEATURES)
    if bundle.winner_model is None:
        p = logistic_elo(features["oracle_elo_diff"])
        return {"p": p, "raw_p": p, "elo_p": p, "symmetry_gap": 0.0, "precision_score": max(p, 1 - p)}
    raw, gap = symmetric_binary_probs(bundle.winner_model, frame)
    calibrated = apply_platt(bundle.winner_calibrator, raw)
    elo_p = np.asarray([logistic_elo(features["oracle_elo_diff"])])
    gate_x = gate_frame(frame, calibrated, elo_p, gap)
    precision = float(bundle.gate_model.predict_proba(gate_x)[:, 1][0]) if bundle.gate_model is not None else float(max(calibrated[0], 1 - calibrated[0]))
    return {"p": float(calibrated[0]), "raw_p": float(raw[0]), "elo_p": float(elo_p[0]),
            "symmetry_gap": float(gap[0]), "precision_score": precision}


def model_distribution(bundle: ModelBundle, match: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], str, dict[str, float]]:
    features = build_features(bundle.states, bundle.h2h, match["date"], match["team_a"], match["team_b"], int(match["best_of"]), int(match["lan"]), float(match["tier"]))
    reliability = winner_reliability(bundle, features)
    p_a = reliability["p"]
    valid = valid_scores(int(match["best_of"]))
    if bundle.score_model is None or not bundle.class_labels:
        return analytic_score_distribution(p_a, int(match["best_of"])), features, "Calibrated binary winner model with analytic score fallback", reliability
    frame = pd.DataFrame([features], columns=FEATURES)
    matrix = symmetric_multiclass_probs(bundle.score_model, frame, bundle.class_labels, bundle.score_temperature)
    matrix = rescale_score_matrix(matrix, bundle.class_labels, np.asarray([p_a]))
    raw_dist = {label: float(prob) for label, prob in zip(bundle.class_labels, matrix[0])}
    combined = {score: raw_dist.get(score, 0.0) for score in valid}
    total = sum(combined.values())
    if total <= 1e-10:
        return analytic_score_distribution(p_a, int(match["best_of"])), features, "Calibrated winner model with format fallback", reliability
    return {score: value / total for score, value in combined.items()}, features, "Two-head precision engine: calibrated binary winner + conditional exact-score XGBoost", reliability


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
            "symmetry_gap": reliability.get("symmetry_gap", 0.0)}


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




def odds_event_teams(event: dict[str, Any]) -> tuple[str, str]:
    home = event.get("home") or event.get("homeTeam") or event.get("participant1") or ""
    away = event.get("away") or event.get("awayTeam") or event.get("participant2") or ""
    if isinstance(home, dict):
        home = home.get("name") or home.get("title") or ""
    if isinstance(away, dict):
        away = away.get("name") or away.get("title") or ""
    return str(home).strip(), str(away).strip()


def odds_event_date(event: dict[str, Any]) -> pd.Timestamp:
    return parse_dt(event.get("date") or event.get("startTime") or event.get("startsAt") or event.get("begin_at"))


def odds_event_is_probable_cs2(event: dict[str, Any]) -> bool:
    sport = event.get("sport") or {}
    league = event.get("league") or {}
    text = " ".join([
        str(sport.get("name") if isinstance(sport, dict) else sport),
        str(sport.get("slug") if isinstance(sport, dict) else ""),
        str(league.get("name") if isinstance(league, dict) else league),
        str(league.get("slug") if isinstance(league, dict) else ""),
        str(event.get("name") or event.get("title") or ""),
    ]).lower()
    explicit = any(key in text for key in ("counter-strike", "counter strike", "cs2", "cs:go", "csgo"))
    return explicit or "esport" in text


def pair_odds_events(upcoming: pd.DataFrame, events: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    candidates: list[tuple[float, Any, str, bool, float]] = []
    for match in upcoming.to_dict("records"):
        for event in events:
            event_id = event.get("id")
            if event_id is None or not odds_event_is_probable_cs2(event):
                continue
            home, away = odds_event_teams(event)
            if not home or not away:
                continue
            direct_a = team_name_similarity(match["team_a"], home)
            direct_b = team_name_similarity(match["team_b"], away)
            reverse_a = team_name_similarity(match["team_a"], away)
            reverse_b = team_name_similarity(match["team_b"], home)
            direct = 0.5 * (direct_a + direct_b)
            reverse = 0.5 * (reverse_a + reverse_b)
            reversed_orientation = reverse > direct
            name_score = max(direct, reverse)
            weakest_name = min((reverse_a, reverse_b) if reversed_orientation else (direct_a, direct_b))
            hours = abs((match["date"] - odds_event_date(event)).total_seconds()) / 3600.0
            if hours > 20 or name_score < 0.66 or weakest_name < 0.48:
                continue
            time_score = math.exp(-hours / 5.5)
            total = 0.88 * name_score + 0.12 * time_score
            candidates.append((total, match["match_id"], str(event_id), reversed_orientation, hours))
    candidates.sort(reverse=True, key=lambda row: row[0])
    used_matches: set[Any] = set()
    used_events: set[str] = set()
    output: dict[Any, dict[str, Any]] = {}
    for score, match_id, event_id, reversed_orientation, hours in candidates:
        if match_id in used_matches or event_id in used_events:
            continue
        used_matches.add(match_id); used_events.add(event_id)
        output[match_id] = {"event_id": event_id, "reversed": reversed_orientation, "match_confidence": score, "time_gap_hours": hours}
    return output


def iter_bookmakers(payload: dict[str, Any]) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    bookmakers = payload.get("bookmakers") or {}
    if isinstance(bookmakers, dict):
        for name, markets in bookmakers.items():
            if isinstance(markets, list):
                yield str(name), [market for market in markets if isinstance(market, dict)]
    elif isinstance(bookmakers, list):
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                continue
            name = bookmaker.get("name") or bookmaker.get("title") or bookmaker.get("key") or "Unknown"
            markets = bookmaker.get("markets") or bookmaker.get("odds") or []
            if isinstance(markets, list):
                yield str(name), [market for market in markets if isinstance(market, dict)]


def match_winner_market(markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred = []
    for market in markets:
        name = str(market.get("name") or market.get("key") or market.get("market") or "").strip().lower()
        if "map" in name or "handicap" in name or "spread" in name or "correct score" in name:
            continue
        if name in {"ml", "moneyline", "match winner", "match result", "winner", "2-way", "2 way"} or ("match" in name and "winner" in name):
            preferred.append(market)
    return preferred[0] if preferred else None


def decimal_pair_from_market(market: dict[str, Any], home_name: str, away_name: str) -> tuple[float, float] | None:
    odds_rows = market.get("odds") or market.get("prices") or []
    if isinstance(odds_rows, dict):
        odds_rows = [odds_rows]
    if isinstance(odds_rows, list) and odds_rows:
        row = odds_rows[0] if isinstance(odds_rows[0], dict) else {}
        home = safe_float(row.get("home") or row.get("1") or row.get("team1"), 0.0)
        away = safe_float(row.get("away") or row.get("2") or row.get("team2"), 0.0)
        if home > 1.0 and away > 1.0:
            return home, away
    outcomes = market.get("outcomes") or []
    if isinstance(outcomes, list):
        found: dict[str, float] = {}
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            name = outcome.get("name") or outcome.get("participant") or outcome.get("label") or ""
            price = safe_float(outcome.get("price") or outcome.get("odds") or outcome.get("decimal"), 0.0)
            if price <= 1.0:
                continue
            if team_name_similarity(name, home_name) >= 0.70:
                found["home"] = price
            elif team_name_similarity(name, away_name) >= 0.70:
                found["away"] = price
        if found.get("home", 0) > 1.0 and found.get("away", 0) > 1.0:
            return found["home"], found["away"]
    return None


def parse_market_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    home_name, away_name = odds_event_teams(payload)
    records = []
    for bookmaker, markets in iter_bookmakers(payload):
        market = match_winner_market(markets)
        if market is None:
            continue
        pair = decimal_pair_from_market(market, home_name, away_name)
        if pair is None:
            continue
        home_odds, away_odds = pair
        raw_home, raw_away = 1.0 / home_odds, 1.0 / away_odds
        margin = raw_home + raw_away
        if margin <= 0:
            continue
        records.append({
            "bookmaker": bookmaker, "home_odds": home_odds, "away_odds": away_odds,
            "fair_home": raw_home / margin, "fair_away": raw_away / margin,
            "overround": margin - 1.0,
            "updated": parse_dt(market.get("updatedAt") or market.get("updated_at") or payload.get("updatedAt") or payload.get("updated_at")),
        })
    if not records:
        return None
    frame = pd.DataFrame(records)
    best_home_i, best_away_i = frame["home_odds"].idxmax(), frame["away_odds"].idxmax()
    return {
        "event_id": str(payload.get("id")), "home": home_name, "away": away_name,
        "best_home_odds": float(frame.loc[best_home_i, "home_odds"]), "best_home_book": str(frame.loc[best_home_i, "bookmaker"]),
        "best_away_odds": float(frame.loc[best_away_i, "away_odds"]), "best_away_book": str(frame.loc[best_away_i, "bookmaker"]),
        "market_home": float(frame["fair_home"].median()), "market_away": float(frame["fair_away"].median()),
        "mean_overround": float(frame["overround"].mean()), "book_count": int(len(frame)),
        "latest_update": frame["updated"].max(), "bookmaker_rows": records,
    }


def build_match_odds(upcoming: pd.DataFrame, pairings: dict[Any, dict[str, Any]], payloads: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    parsed = {}
    for payload in payloads:
        item = parse_market_payload(payload)
        if item:
            parsed[str(item["event_id"])] = item
    output = {}
    for match_id, pairing in pairings.items():
        item = parsed.get(str(pairing["event_id"]))
        if not item:
            continue
        record = dict(item)
        if pairing["reversed"]:
            record.update({
                "best_a_odds": item["best_away_odds"], "best_a_book": item["best_away_book"],
                "best_b_odds": item["best_home_odds"], "best_b_book": item["best_home_book"],
                "market_a": item["market_away"], "market_b": item["market_home"],
            })
        else:
            record.update({
                "best_a_odds": item["best_home_odds"], "best_a_book": item["best_home_book"],
                "best_b_odds": item["best_away_odds"], "best_b_book": item["best_away_book"],
                "market_a": item["market_home"], "market_b": item["market_away"],
            })
        record.update(pairing)
        output[match_id] = record
    return output


def rescale_distribution_to_winner(distribution: dict[str, float], target_a: float) -> dict[str, float]:
    target_a = float(np.clip(target_a, 1e-5, 1 - 1e-5))
    win_scores = [score for score in distribution if score_is_a_win(score)]
    lose_scores = [score for score in distribution if not score_is_a_win(score)]
    win_total = sum(distribution[score] for score in win_scores)
    lose_total = sum(distribution[score] for score in lose_scores)
    adjusted = {}
    for score, probability in distribution.items():
        if score in win_scores:
            adjusted[score] = probability * target_a / max(win_total, 1e-12)
        else:
            adjusted[score] = probability * (1 - target_a) / max(lose_total, 1e-12)
    total = sum(adjusted.values())
    return {score: value / max(total, 1e-12) for score, value in adjusted.items()}


def market_assisted_probability(model_p: float, market_p: float, precision_score: float, book_count: int, match_confidence: float) -> tuple[float, float]:
    # The market is useful for winner accuracy, while the independent model remains the only basis for value calculations.
    model_weight = 0.34 + 0.24 * np.clip((precision_score - 0.50) / 0.35, 0.0, 1.0)
    if book_count <= 1:
        model_weight += 0.10
    model_weight += 0.10 * np.clip((0.82 - match_confidence) / 0.20, 0.0, 1.0)
    model_weight = float(np.clip(model_weight, 0.32, 0.68))
    combined = sigmoid(model_weight * logit(model_p) + (1.0 - model_weight) * logit(market_p))
    return float(np.clip(combined, 0.02, 0.98)), model_weight


def value_label(ev: float, edge: float, model_p: float, precision: float) -> str:
    if ev >= 0.12 and edge >= 0.06 and model_p >= 0.58 and precision >= 0.58:
        return "A+ value"
    if ev >= 0.075 and edge >= 0.04:
        return "A value"
    if ev >= 0.04 and edge >= 0.025:
        return "B value"
    if ev > 0:
        return "Small edge"
    return "No value"


def decorate_with_market(prediction: dict[str, Any], match: dict[str, Any], odds: dict[str, Any] | None) -> dict[str, Any]:
    output = dict(prediction)
    output.update({
        "odds_available": False, "final_team_a_p": prediction["team_a_p"], "final_winner": prediction["winner"],
        "final_winner_p": prediction["winner_p"], "final_score": prediction["score"], "market_model_weight": 1.0,
        "value_side": None, "value_ev": float("nan"), "value_edge": float("nan"), "value_odds": float("nan"),
        "value_book": None, "value_grade": "No odds", "banker_side": None, "banker_odds": float("nan"),
        "banker_book": None, "banker_probability": float("nan"), "banker_score": float("nan"),
    })
    if not odds:
        return output
    model_a = float(prediction["team_a_p"])
    market_a = float(odds["market_a"])
    final_a, model_weight = market_assisted_probability(model_a, market_a, float(prediction["precision_score"]), int(odds["book_count"]), float(odds["match_confidence"]))
    final_dist = rescale_distribution_to_winner(prediction["distribution"], final_a)
    final_a_pick = final_a >= 0.5
    final_side_scores = [score for score in final_dist if score_is_a_win(score) == final_a_pick]
    final_score = max(final_side_scores, key=lambda score: final_dist[score])
    final_winner = match["team_a"] if final_a_pick else match["team_b"]
    final_winner_p = final_a if final_a_pick else 1 - final_a

    choices = []
    for side, model_p, market_p, price, book in [
        (match["team_a"], model_a, odds["market_a"], odds["best_a_odds"], odds["best_a_book"]),
        (match["team_b"], 1 - model_a, odds["market_b"], odds["best_b_odds"], odds["best_b_book"]),
    ]:
        edge = float(model_p - market_p)
        ev = float(model_p * price - 1.0)
        choices.append({"side": side, "model_p": model_p, "market_p": market_p, "odds": price, "book": book, "edge": edge, "ev": ev})
    best_value = max(choices, key=lambda row: row["ev"])
    if final_a_pick:
        banker_side, banker_odds, banker_book = match["team_a"], odds["best_a_odds"], odds["best_a_book"]
    else:
        banker_side, banker_odds, banker_book = match["team_b"], odds["best_b_odds"], odds["best_b_book"]
    banker_quality = float(np.clip(0.68 * final_winner_p + 0.22 * prediction["precision_score"] + 0.10 * odds["match_confidence"], 0.0, 1.0))
    output.update({
        "odds_available": True, "odds": odds, "final_team_a_p": final_a, "final_winner": final_winner,
        "final_winner_p": final_winner_p, "final_score": final_score, "final_distribution": final_dist,
        "market_model_weight": model_weight, "market_a": market_a, "market_b": 1 - market_a,
        "value_side": best_value["side"], "value_model_p": best_value["model_p"], "value_market_p": best_value["market_p"],
        "value_ev": best_value["ev"], "value_edge": best_value["edge"], "value_odds": best_value["odds"],
        "value_book": best_value["book"], "value_grade": value_label(best_value["ev"], best_value["edge"], best_value["model_p"], prediction["precision_score"]),
        "banker_side": banker_side, "banker_odds": float(banker_odds), "banker_book": banker_book,
        "banker_probability": final_winner_p, "banker_score": banker_quality,
    })
    return output


def store_session_odds_snapshots(market_by_match: dict[Any, dict[str, Any]]) -> None:
    snapshots = st.session_state.setdefault("odds_snapshots", {})
    now = pd.Timestamp.now(tz="UTC")
    for match_id, odds in market_by_match.items():
        history = snapshots.setdefault(str(match_id), [])
        item = {"time": now, "a": odds["best_a_odds"], "b": odds["best_b_odds"]}
        if not history or history[-1]["a"] != item["a"] or history[-1]["b"] != item["b"]:
            history.append(item)
            del history[:-30]

# ----------------------------- App shell -----------------------------
st.markdown(f"""
<div class="hero"><div class="eyebrow">CS Oracle · Market Edge Engine {APP_VERSION}</div>
<h1>Precision, bankers and real value</h1>
<p>The independent Oracle model predicts the winner and exact score. Live no-vig bookmaker prices provide a separate market view. A market-assisted forecast targets the most likely winner, while value is calculated only from the independent model so the tool never marks its own copied market opinion as an edge.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Accuracy settings")
    history_days = st.slider("History window (days)", 420, 1100, DEFAULT_HISTORY_DAYS, 30)
    history_pages = st.slider("Maximum API pages", 12, 32, DEFAULT_HISTORY_PAGES, 1)
    st.caption("More history improves stability but takes longer on the first load. Recent matches receive much higher weight.")
    st.markdown("### Optional data depth")
    map_upload = st.file_uploader("Map history CSV", type=["csv"], help="Optional. Enables projected veto and map-specific context.")
    player_upload = st.file_uploader("3-month player form CSV", type=["csv"], help="Optional columns: team, player, rating, adr, kast, impact, maps.")
    st.caption("PandaScore supplies fixtures and series history. Odds-API.io supplies current prices. Optional map/player files improve the unavailable free-feed details.")

try:
    player_data = clean_player_data(player_upload)
except Exception as exc:
    st.sidebar.error(str(exc)); player_data = pd.DataFrame()
try:
    uploaded_maps = clean_map_data(map_upload)
except Exception as exc:
    st.sidebar.error(str(exc)); uploaded_maps = pd.DataFrame()

panda_token = token_from_secrets()
odds_token = odds_token_from_secrets()
if not panda_token:
    st.error("PandaScore token missing.")
    st.code('PANDASCORE_TOKEN = "your-private-token"\nODDS_API_KEY = "your-private-odds-key"', language="toml")
    st.stop()

try:
    with st.spinner("Loading completed CS2 matches and rebuilding the prediction engine…"):
        raw_history = cached_past(panda_token, history_days, history_pages)
        raw_upcoming = cached_upcoming(panda_token, 4)
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
with st.spinner("Training the winner model, exact-score head, precision gate and Elo history…"):
    bundle = train_bundle_cached(history_json)

market_by_match: dict[Any, dict[str, Any]] = {}
odds_status = "Not configured"
if odds_token and not upcoming.empty:
    try:
        with st.spinner("Matching fixtures to bookmakers and fetching current match-winner odds…"):
            odds_events = cached_odds_events(odds_token)
            pairings = pair_odds_events(upcoming.head(60), odds_events)
            event_ids = tuple(sorted({str(row["event_id"]) for row in pairings.values()}))
            payloads = cached_odds_payloads(odds_token, event_ids) if event_ids else []
            market_by_match = build_match_odds(upcoming, pairings, payloads)
            store_session_odds_snapshots(market_by_match)
            odds_status = f"{len(market_by_match)} matches priced"
    except OddsAPIError as exc:
        odds_status = "Feed error"
        st.warning(f"Predictions still work, but odds could not load: {exc}")
elif not odds_token:
    st.info('Odds integration is ready. Add ODDS_API_KEY to Streamlit Secrets to activate Bankers and Value Edge.')

prediction_rows: list[dict[str, Any]] = []
predictions_by_match: dict[Any, dict[str, Any]] = {}
for match in upcoming.head(60).to_dict("records"):
    independent = prediction_for_match(bundle, match, player_data)
    enriched = decorate_with_market(independent, match, market_by_match.get(match["match_id"]))
    predictions_by_match[match["match_id"]] = enriched
    prediction_rows.append({**match, **enriched})

tabs = st.tabs(["Daily board", "Top bankers", "Value edge", "Match laboratory", "Oracle Elo & hot form", "Accuracy proof", "Data & method"])
daily_tab, banker_tab, value_tab, match_tab, elo_tab, accuracy_tab, method_tab = tabs

with daily_tab:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Completed matches", f"{len(history):,}")
    m2.metric("Teams rated", f"{len(bundle.states):,}")
    m3.metric("Winner model", bundle.winner_model_name)
    elite_acc = 100 * bundle.metrics.get("precision_gate_accuracy", 0.0) if bundle.metrics else 0.0
    m4.metric("Elite proof", f"{elite_acc:.1f}%", delta=f"{bundle.metrics.get('precision_gate_matches', 0) if bundle.metrics else 0} matches")
    m5.metric("Odds feed", odds_status)
    if bundle.metrics.get("precision_gate_target_met", False):
        st.success("The learned Elite gate exceeded 70% on the newest untouched historical period. This applies only to the selected subset—not every match.")
    else:
        st.info("The engine ranks every match but refuses to claim 70% unless the untouched test proves it. Bankers are sorted by estimated chance, not presented as guarantees.")

    if not prediction_rows:
        st.info("No upcoming CS2 fixtures were returned for the next four days.")
    else:
        rows = []
        for row in prediction_rows:
            rows.append({
                "Start": row["date"], "Match": f'{row["team_a"]} vs {row["team_b"]}', "Event": row["event"], "BO": row["best_of"],
                "Independent Oracle": row["winner"], "Oracle win %": 100 * row["winner_p"],
                "Final forecast": row["final_winner"], "Final win %": 100 * row["final_winner_p"], "Exact score": row["final_score"],
                "Best odds": row["banker_odds"] if row["odds_available"] else np.nan,
                "Bookmaker": row["banker_book"] if row["odds_available"] else "—",
                "Precision": 100 * row["precision_score"], "Signal": row["confidence_label"],
            })
        board = pd.DataFrame(rows).sort_values(["Precision", "Final win %"], ascending=False)
        st.markdown("### Ranked daily prediction board")
        st.dataframe(board, hide_index=True, use_container_width=True,
                     column_config={"Start": st.column_config.DatetimeColumn(format="DD MMM HH:mm"),
                                    "Oracle win %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                    "Final win %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                    "Precision": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                    "Best odds": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("Download daily board", board.to_csv(index=False).encode(), "cs_oracle_daily_v5.csv", "text/csv", use_container_width=True)

with banker_tab:
    st.markdown("### 🏦 Top Bankers")
    st.caption("Definition: best available match-winner odds at or above the chosen minimum, ranked by the highest market-assisted probability of winning. This is a probability ranking—not certainty.")
    if not odds_token:
        st.warning("Add ODDS_API_KEY in Streamlit Secrets to activate this list.")
    else:
        c1, c2 = st.columns(2)
        min_banker_odds = c1.number_input("Minimum decimal odds", min_value=1.01, max_value=10.0, value=1.40, step=0.05)
        min_banker_probability = c2.slider("Minimum estimated win chance", 45, 80, 52, 1)
        bankers = []
        for row in prediction_rows:
            if not row["odds_available"] or row["banker_odds"] < min_banker_odds or 100 * row["banker_probability"] < min_banker_probability:
                continue
            market_side_p = row["odds"]["market_a"] if row["banker_side"] == row["team_a"] else row["odds"]["market_b"]
            bankers.append({
                "Start": row["date"], "Match": f'{row["team_a"]} vs {row["team_b"]}', "Banker": row["banker_side"],
                "Exact score": row["final_score"], "Estimated win %": 100 * row["banker_probability"],
                "No-vig market %": 100 * market_side_p, "Best odds": row["banker_odds"], "Bookmaker": row["banker_book"],
                "Banker quality": 100 * row["banker_score"], "Oracle precision": 100 * row["precision_score"],
                "Model-market agreement": "Yes" if row["final_winner"] == row["winner"] else "No",
            })
        banker_frame = pd.DataFrame(bankers)
        if banker_frame.empty:
            st.info("No priced match currently meets the banker filters.")
        else:
            banker_frame = banker_frame.sort_values(["Estimated win %", "Banker quality", "Best odds"], ascending=[False, False, False]).reset_index(drop=True)
            banker_frame.insert(0, "Rank", np.arange(1, len(banker_frame) + 1))
            top = banker_frame.iloc[0]
            st.markdown(f"""<div class="match-card"><div class="eyebrow">Highest-chance banker ≥ {min_banker_odds:.2f}</div><div class="winner">{top['Banker']} @ {top['Best odds']:.2f}</div><div><span class="pill">{top['Estimated win %']:.1f}% final chance</span><span class="pill">{top['Exact score']}</span><span class="pill">{top['Bookmaker']}</span></div></div>""", unsafe_allow_html=True)
            st.dataframe(banker_frame, hide_index=True, use_container_width=True,
                         column_config={"Start": st.column_config.DatetimeColumn(format="DD MMM HH:mm"),
                                        "Estimated win %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                        "No-vig market %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                        "Banker quality": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                        "Oracle precision": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                        "Best odds": st.column_config.NumberColumn(format="%.2f")})
            st.download_button("Download bankers", banker_frame.to_csv(index=False).encode(), "cs_oracle_bankers_v5.csv", "text/csv", use_container_width=True)

with value_tab:
    st.markdown("### 💎 Independent Value Edge Finder")
    st.caption("Value uses the independent Oracle probability versus the bookmaker's no-vig consensus. The market-assisted forecast is deliberately excluded from EV calculations to avoid circular logic.")
    if not odds_token:
        st.warning("Add ODDS_API_KEY in Streamlit Secrets to activate value analysis.")
    else:
        c1, c2, c3 = st.columns(3)
        min_ev = c1.slider("Minimum expected value", -5, 30, 2, 1) / 100.0
        min_edge = c2.slider("Minimum probability edge", -5, 20, 1, 1) / 100.0
        positive_only = c3.toggle("Only positive edges", value=True)
        values = []
        for row in prediction_rows:
            if not row["odds_available"]:
                continue
            if row["value_ev"] < min_ev or row["value_edge"] < min_edge:
                continue
            if positive_only and (row["value_ev"] <= 0 or row["value_edge"] <= 0):
                continue
            values.append({
                "Start": row["date"], "Match": f'{row["team_a"]} vs {row["team_b"]}', "Value side": row["value_side"],
                "Oracle probability": 100 * row["value_model_p"], "No-vig market": 100 * row["value_market_p"],
                "Edge": 100 * row["value_edge"], "Best odds": row["value_odds"], "Oracle fair odds": 1 / max(row["value_model_p"], 1e-9),
                "Expected value": 100 * row["value_ev"], "Bookmaker": row["value_book"], "Grade": row["value_grade"],
                "Precision": 100 * row["precision_score"], "Exact score": row["score"],
            })
        value_frame = pd.DataFrame(values)
        if value_frame.empty:
            st.info("No current match meets the chosen value filters. That is a valid result—forcing a bet is not value finding.")
        else:
            value_frame = value_frame.sort_values(["Expected value", "Edge", "Precision"], ascending=False).reset_index(drop=True)
            value_frame.insert(0, "Rank", np.arange(1, len(value_frame) + 1))
            st.dataframe(value_frame, hide_index=True, use_container_width=True,
                         column_config={"Start": st.column_config.DatetimeColumn(format="DD MMM HH:mm"),
                                        "Oracle probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                        "No-vig market": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                        "Edge": st.column_config.NumberColumn(format="%+.1f pp"),
                                        "Expected value": st.column_config.NumberColumn(format="%+.1f%%"),
                                        "Best odds": st.column_config.NumberColumn(format="%.2f"),
                                        "Oracle fair odds": st.column_config.NumberColumn(format="%.2f"),
                                        "Precision": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
            st.download_button("Download value edges", value_frame.to_csv(index=False).encode(), "cs_oracle_value_v5.csv", "text/csv", use_container_width=True)

with match_tab:
    if upcoming.empty:
        st.info("No match is available to analyse.")
    else:
        labels = [f'{row.date.strftime("%d %b %H:%M UTC")} · {row.team_a} vs {row.team_b} · BO{row.best_of}' for row in upcoming.itertuples(index=False)]
        selected_label = st.selectbox("Choose match", labels)
        selected = upcoming.iloc[labels.index(selected_label)].to_dict()
        prediction = predictions_by_match.get(selected["match_id"]) or decorate_with_market(prediction_for_match(bundle, selected, player_data), selected, market_by_match.get(selected["match_id"]))
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown(f"### {selected['team_a']} vs {selected['team_b']}")
            st.caption(f"{selected['event']} · BO{selected['best_of']} · {selected['date'].strftime('%d %b %Y %H:%M UTC')}")
            st.markdown(f"""<div class="match-card"><div class="eyebrow">Final forecast</div><div class="winner">{prediction['final_winner']} {prediction['final_score']}</div><div><span class="pill">{pct(prediction['final_winner_p'])} final chance</span><span class="pill">Independent: {prediction['winner']} {pct(prediction['winner_p'])}</span><span class="pill">{prediction['confidence_label']} · {prediction['confidence']}/100</span></div></div>""", unsafe_allow_html=True)
            if prediction["odds_available"]:
                market_a = prediction["market_a"]
                compare = pd.DataFrame({
                    "Source": ["Independent Oracle", "No-vig market", "Market-assisted final"],
                    selected["team_a"]: [100 * prediction["team_a_p"], 100 * market_a, 100 * prediction["final_team_a_p"]],
                    selected["team_b"]: [100 * (1 - prediction["team_a_p"]), 100 * (1 - market_a), 100 * (1 - prediction["final_team_a_p"])],
                })
                st.dataframe(compare, hide_index=True, use_container_width=True)
                odds = prediction["odds"]
                odds_table = pd.DataFrame([
                    {"Team": selected["team_a"], "Best odds": odds["best_a_odds"], "Bookmaker": odds["best_a_book"], "No-vig probability": 100 * odds["market_a"]},
                    {"Team": selected["team_b"], "Best odds": odds["best_b_odds"], "Bookmaker": odds["best_b_book"], "No-vig probability": 100 * odds["market_b"]},
                ])
                st.dataframe(odds_table, hide_index=True, use_container_width=True,
                             column_config={"Best odds": st.column_config.NumberColumn(format="%.2f"),
                                            "No-vig probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
                st.caption(f"Matched {odds['book_count']} bookmaker price set(s) · fixture match confidence {100 * odds['match_confidence']:.0f}% · mean overround {100 * odds['mean_overround']:.1f}%")
        with right:
            st.markdown("#### Exact-score distribution")
            score_dist = prediction.get("final_distribution", prediction["distribution"])
            score_frame = pd.DataFrame([{"Score": score, "Probability": 100 * probability} for score, probability in sorted(score_dist.items(), key=lambda item: item[1], reverse=True)])
            st.dataframe(score_frame, hide_index=True, use_container_width=True,
                         column_config={"Probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
            st.caption(prediction["overlay_note"])
            st.markdown("#### Reliability checks")
            reliability_rows = [
                {"Check": "Independent Oracle A", "Value": 100 * prediction["team_a_p"]},
                {"Check": "Raw XGBoost A", "Value": 100 * prediction["raw_model_p"]},
                {"Check": "Oracle Elo A", "Value": 100 * prediction["elo_p"]},
                {"Check": "Order-symmetry gap", "Value": 100 * prediction["symmetry_gap"]},
            ]
            if prediction["odds_available"]:
                reliability_rows += [
                    {"Check": "Market no-vig A", "Value": 100 * prediction["market_a"]},
                    {"Check": "Model weight in final", "Value": 100 * prediction["market_model_weight"]},
                ]
            st.dataframe(pd.DataFrame(reliability_rows), hide_index=True, use_container_width=True)

        st.markdown("### Elo trajectories before this match")
        chart_cols = st.columns(2)
        for col, team in zip(chart_cols, [selected["team_a"], selected["team_b"]]):
            with col:
                form = hot_form_metrics(bundle.states[team], selected["date"])
                st.markdown(f"#### {team} · {form['status']}")
                st.plotly_chart(elo_chart(bundle, team, height=270, compact=True), use_container_width=True)
                st.caption(f"30d Elo {form['momentum30']:+.0f} · Last 10 {100 * form['wr10']:.0f}% · Opponent-adjusted form {form['performance30']:+.3f}")

        st.markdown("### Why the independent model ranks it this way")
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
        veto_frame, veto_note = projected_veto(map_history, selected["team_a"], selected["team_b"], selected["date"], int(selected["best_of"]), prediction["final_team_a_p"])
        if veto_frame.empty: st.info(veto_note)
        else:
            st.dataframe(veto_frame, hide_index=True, use_container_width=True); st.caption(veto_note)

        st.markdown("### Current lineups")
        cols = st.columns(2)
        for col, team, team_id in [(cols[0], selected["team_a"], selected["team_a_id"]), (cols[1], selected["team_b"], selected["team_b_id"])]:
            with col:
                st.markdown(f"#### {team}")
                try:
                    players = team_players(cached_team(panda_token, team_id))
                    if players.empty: st.caption("Current roster was not returned by the team endpoint.")
                    else: st.dataframe(players, hide_index=True, use_container_width=True)
                except APIError: st.caption("Roster request unavailable.")

        receipt = {"generated_at": datetime.now(timezone.utc).isoformat(), "app_version": APP_VERSION, "match_id": selected["match_id"],
                   "match": f"{selected['team_a']} vs {selected['team_b']}", "independent_prediction": prediction["winner"],
                   "independent_probability": prediction["winner_p"], "final_prediction": prediction["final_winner"],
                   "final_probability": prediction["final_winner_p"], "score": prediction["final_score"],
                   "precision_score": prediction["precision_score"], "signal": prediction["confidence_label"],
                   "exact_score_distribution": prediction.get("final_distribution", prediction["distribution"]), "history_matches": len(history),
                   "odds": ({"best_a": prediction["odds"]["best_a_odds"], "best_b": prediction["odds"]["best_b_odds"],
                              "market_a": prediction["odds"]["market_a"], "books": prediction["odds"]["book_count"]} if prediction["odds_available"] else None),
                   "model": "Calibrated binary XGBoost + exact-score head + precision gate + separate no-vig market blend"}
        receipt["sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        st.download_button("Download locked prediction receipt", json.dumps(receipt, indent=2).encode(), f"prediction_{selected['match_id']}.json", "application/json", use_container_width=True)

with elo_tab:
    st.markdown("### Oracle Elo world ranking")
    st.caption("Public Elo blends 66% stable Core Elo and 34% Fast Elo. Match predictions also use LAN/online and BO-format ratings.")
    rankings = elo_table(bundle)
    render_elo_list(rankings)
    if not rankings.empty:
        st.download_button("Download full Elo table", rankings.to_csv(index=False).encode(), "cs_oracle_elo_v5.csv", "text/csv", use_container_width=True)
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
        st.markdown("### How every Elo move is calculated")
        st.markdown('<div class="formula">Expected = 1 / (1 + 10 ^ (−Elo difference / 400))<br>K = 27 × event-tier multiplier × experience multiplier × score-margin multiplier<br>Core change = K × (actual result − expected result)<br>Fast Elo uses a larger K and regresses faster toward 1500<br>Oracle match Elo = 55% Core + 25% Fast + 12% LAN/Online + 8% BO format<br>Public ranking Elo = 66% Core + 34% Fast</div>', unsafe_allow_html=True)
        st.caption("Inactive ratings regress gradually toward 1500. Hot Form combines 30/60-day Elo momentum, last-10 results, opponent-adjusted performance, opposition quality and rust.")

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
        d.metric("Calibration error", f"{100 * metrics['ece']:.1f}%")
        st.caption(f"Oldest 70% train the models, the next 14% tune calibration and the precision gate, and the newest {metrics['test_matches']:,} matches stay untouched. Winner model: {metrics['winner_model_name']}.")
        st.markdown("### 70% Precision Gate")
        gate_cols = st.columns(4)
        gate_cols[0].metric("Untouched accuracy", f"{100 * metrics['precision_gate_accuracy']:.1f}%")
        gate_cols[1].metric("Coverage", f"{100 * metrics['precision_gate_coverage']:.1f}%")
        gate_cols[2].metric("Matches", f"{metrics['precision_gate_matches']:,}")
        gate_cols[3].metric("Status", "PROVEN" if metrics["precision_gate_target_met"] else "NOT YET")
        if metrics["precision_gate_target_met"]:
            st.success("The Elite label exceeded 70% on the newest untouched period. This applies only to the gated subset, not every match.")
        else:
            st.warning("The newest untouched period did not reach 70%. The app therefore refuses to label live matches as proven 70% signals.")
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
    st.info("The market-assisted final forecast is not included in this historical proof because historical closing odds are not available in the current free dataset. Its purpose is to improve current winner ranking; value calculations remain independent.")

with method_tab:
    st.markdown("### The three probabilities are deliberately separate")
    st.markdown("""
- **Independent Oracle:** XGBoost, calibrated Elo, recent form, opponent quality, workload, format and score history. This is the probability used for value.
- **No-vig market:** bookmaker prices converted into fair probabilities after removing overround, then combined robustly across available books.
- **Final forecast:** a conservative log-odds blend of Oracle and market, with more model weight when the learned precision score is strong or market coverage is thin.
""")
    st.markdown("### Top Bankers")
    st.info("A Banker is simply the highest estimated winning chance among selections whose best available decimal odds meet the minimum—default 1.40. It is not a guarantee and it is not selected by expected value.")
    st.markdown("### Value calculation")
    st.markdown('<div class="formula">Raw implied probability = 1 / decimal odds<br>No-vig market probability = raw side probability / (raw A + raw B)<br>Oracle edge = independent Oracle probability − no-vig market probability<br>Expected value = independent Oracle probability × best odds − 1<br>Oracle fair odds = 1 / independent Oracle probability</div>', unsafe_allow_html=True)
    st.markdown("### Why an LLM is not the numerical core")
    st.info("An LLM can summarise roster news, but it cannot manufacture missing ADR, veto or map telemetry. Regularised gradient boosting remains the stronger numerical model for this structured dataset. Bookmaker consensus adds a different source of information without contaminating the independent value model.")
    st.markdown("### Honest boundaries")
    st.warning("PandaScore Free does not automatically provide complete three-month ADR, player ratings or map pick/ban history. Odds coverage depends on the two bookmakers selected on the free Odds-API.io plan. Team-name and fixture matching is scored and weak matches are rejected rather than forced.")
    st.markdown("### Streamlit Secrets")
    st.code('PANDASCORE_TOKEN = "your-private-pandascore-token"\nODDS_API_KEY = "your-private-odds-api-key"', language="toml")
    st.markdown("### Optional CSV formats")
    player_template = "team,player,rating,adr,kast,impact,maps\nTeam A,Player1,1.14,79.2,72.8,1.18,38\n"
    map_template = "date,team_a,team_b,map,winner\n2026-07-01,Team A,Team B,Mirage,Team A\n"
    x, y = st.columns(2)
    x.download_button("Download player template", player_template.encode(), "player_form_template.csv", "text/csv", use_container_width=True)
    y.download_button("Download map template", map_template.encode(), "map_history_template.csv", "text/csv", use_container_width=True)

st.caption("CS Oracle provides probabilistic estimates, not certainty. Odds change quickly. Regenerate every API key that has appeared in screenshots or chat before using the app long term.")
