"""
predict_tonight.py — Live NBA Game Predictor with Injury-Adjusted PIS
======================================================================
Fix log (v6):
  ADVANCED STATS — E_NET_RATING, E_OFF_RATING, E_DEF_RATING, E_PACE,
  TS_PCT, E_AST_RATIO, E_OREB_PCT, E_DREB_PCT, E_TM_TOV_PCT now fetched
  live from LeagueDashTeamStats at inference time, diffed (HOME - AWAY),
  and included in the feature vector. Falls back to values bundled inside
  the model pkl (adv_stats dict) if the live fetch fails.

  CONDITIONAL RESIDUAL STD — when converting predicted spread to win
  probability via norm.cdf(margin / std), we now use a spread-magnitude-
  dependent sigma (close/medium/large) stored in the model bundle rather
  than a single global std. Tighter sigma on big spreads = more decisive
  probabilities on genuine mismatches.

Fix log (v5.1):
  MARGIN FEATURES — ROLL10_POINT_DIFF_DIFF, SEASON_POINT_DIFF_DIFF, and
  BLOWOUT_RATE_DIFF computed at inference time from ROLL10_PLUS_MINUS.

Fix log (v5.0):
  MARGIN MODEL — margin_model is primary win probability source.
  Win probability = norm.cdf(predicted_margin / residual_std).

Fix log (v4.4):
  BUG 1 — Injury deduction logs returned to caller (fix out-of-order print).
  BUG 2 — SOS-weighted win percentage using opponent Elo.
  BUG 3 — Blend season PIS 40% with recent-form proxy 60%.
  BUG 4 — Elo reality check blending toward win-pct-implied ratings.

Fix log (v4.5):
  BUG 5 — Player name suffix mismatches fixed via _normalize_name().

Fix log (v6.1 inference):
  DOUBLE-NEGATIVE BUG — injury log line was printing `-{net_pis_gap:.2f}`
  which doubled the minus sign when net_pis_gap was already negative (i.e.
  the player had a negative PIS). Fixed by using abs() in the format string.

  ADV STATS KWARG PROBE — fetch_advanced_stats_live() now probes the
  LeagueDashTeamStats signature at runtime (same logic as nba.py) so the
  live fetch works across nba_api versions instead of always falling back
  to the bundle.
"""

import sys
import re
import time
import inspect
from scipy.stats import norm as scipy_norm
import joblib
import argparse
import datetime
import warnings
import json
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from nba_api.stats.endpoints import (
    ScoreboardV3,
    TeamGameLog,
    LeagueDashTeamStats,
)
from nba_api.stats.static import teams as nba_teams_static

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH      = Path("nba_model_v7.pkl")
PIS_PATH        = Path("team_pis.csv")
PLAYER_PIS_PATH = Path("player_impact_scores.csv")
ROLLING_WINDOW  = 10
CURRENT_SEASON  = "2025-26"
API_DELAY       = 0.7
API_MAX_RETRIES = 3

# BUG 3: Blend weights for season PIS vs recent-form proxy
PIS_SEASON_WEIGHT = 0.40
PIS_RECENT_WEIGHT = 0.60

# Advanced stats to fetch and diff at inference time
ADV_FEATURES = [
    "E_NET_RATING",
    "E_OFF_RATING",
    "E_DEF_RATING",
    "E_PACE",
    "TS_PCT",
    "E_AST_RATIO",
    "E_OREB_PCT",
    "E_DREB_PCT",
    "E_TM_TOV_PCT",
]

# ── Team maps ──────────────────────────────────────────────────────────────────
_ALL_TEAMS = nba_teams_static.get_teams()
ABBR_TO_ID = {t["abbreviation"]: t["id"] for t in _ALL_TEAMS}
ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in _ALL_TEAMS}
ID_TO_NAME = {t["id"]: t["full_name"] for t in _ALL_TEAMS}

ABBR_PATCH = {
    "NOH": "NOP", "BRK": "BKN", "PHO": "PHX", "CHO": "CHA", "GOS": "GSW",
}

def normalize_abbr(abbr: str) -> str:
    return ABBR_PATCH.get(abbr, abbr)

ESPN_ABBR_PATCH = {
    "WSH": "WAS", "PHO": "PHX", "NO":   "NOP", "NY":  "NYK",
    "SA":  "SAS", "GS":  "GSW", "UTAH": "UTA", "BKN": "BKN",
    "GOS": "GSW", "NOH": "NOP", "BRK":  "BKN", "CHO": "CHA",
}

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks":          "ATL",
    "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN",
    "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI",
    "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL",
    "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET",
    "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU",
    "Indiana Pacers":         "IND",
    "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}

CORE_STATS = [
    "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
    "REB", "AST", "TOV", "STL", "BLK", "PLUS_MINUS",
]

# v8: Conference/division lookup
TEAM_CONF = {
    "ATL": "E", "BOS": "E", "BKN": "E", "CHA": "E", "CHI": "E",
    "CLE": "E", "DET": "E", "IND": "E", "MIA": "E", "MIL": "E",
    "NYK": "E", "ORL": "E", "PHI": "E", "TOR": "E", "WAS": "E",
    "DAL": "W", "DEN": "W", "GSW": "W", "HOU": "W", "LAC": "W",
    "LAL": "W", "MEM": "W", "MIN": "W", "NOP": "W", "OKC": "W",
    "PHX": "W", "POR": "W", "SAC": "W", "SAS": "W", "UTA": "W",
}

TEAM_DIV = {
    "BOS": "ATL", "BKN": "ATL", "NYK": "ATL", "PHI": "ATL", "TOR": "ATL",
    "CHI": "CEN", "CLE": "CEN", "DET": "CEN", "IND": "CEN", "MIL": "CEN",
    "ATL": "SE",  "CHA": "SE",  "MIA": "SE",  "ORL": "SE",  "WAS": "SE",
    "DAL": "SW",  "HOU": "SW",  "MEM": "SW",  "NOP": "SW",  "SAS": "SW",
    "DEN": "NW",  "MIN": "NW",  "OKC": "NW",  "POR": "NW",  "UTA": "NW",
    "GSW": "PAC", "LAC": "PAC", "LAL": "PAC", "PHX": "PAC", "SAC": "PAC",
}

COMPUTABLE_FEATURES = (
    {f"ROLL10_{col}_DIFF" for col in CORE_STATS}
    | {f"{feat}_DIFF" for feat in ADV_FEATURES}
    | {
        "DAYS_REST_DIFF",
        "IS_BACK_TO_BACK_DIFF",
        "WIN_STREAK_DIFF",
        "DAYS_SINCE_HOME_DIFF",
        "ADJ_OFF_RATING_DIFF",
        "ADJ_DEF_RATING_DIFF",
        "HOME_ROLL_WIN_PCT",
        "AWAY_ROLL_WIN_PCT",
        "ROLL_WIN_PCT_DIFF",
        "SEASON_WIN_PCT_DIFF",
        "HOME_ELO",
        "AWAY_ELO",
        "ELO_DIFF",
        "PIS_DIFF",
        "TOP3_PIS_DIFF",
        "DEPTH_PIS_DIFF",
        "SOS_WIN_PCT_DIFF",
        "ROLL10_POINT_DIFF_DIFF",
        "SEASON_POINT_DIFF_DIFF",
        "BLOWOUT_RATE_DIFF",
        # v7 new features
        "ROLL5_PLUS_MINUS_DIFF",
        "ROLL20_PLUS_MINUS_DIFF",
        "GAMES_IN_LAST_7_DIFF",
        "ROAD_TRIP_LENGTH_DIFF",
        "SEASON_PROGRESS_DIFF",
        "IMPACT_MISSING_DIFF",   # computed from injury-adjusted PIS at inference
        # v8 new features
        "EWMA_PLUS_MINUS_DIFF",
        "EWMA_PTS_DIFF",
        "EWMA_FG_PCT_DIFF",
        "SAME_CONFERENCE",
        "SAME_DIVISION",
    }
)


# ── API helper ─────────────────────────────────────────────────────────────────
def api_call_with_retry(fn, *args, **kwargs):
    delay = API_DELAY
    for attempt in range(API_MAX_RETRIES):
        try:
            result = fn(*args, **kwargs)
            time.sleep(delay)
            return result
        except Exception as e:
            if attempt == API_MAX_RETRIES - 1:
                raise
            wait = delay * (2 ** attempt)
            print(f"    API call failed ({e}), retrying in {wait:.1f}s...")
            time.sleep(wait)


# ── Name normalisation (BUG 5) ─────────────────────────────────────────────────
_NAME_SUFFIX_RE = re.compile(
    r"\s+(jr\.?|sr\.?|ii+|iii+|iv|v|vi)$", re.IGNORECASE
)

def _normalize_name(name: str) -> str:
    return _NAME_SUFFIX_RE.sub("", name.strip().lower()).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 1. FETCH TONIGHT'S GAMES
# ══════════════════════════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_tonights_games(date: datetime.date) -> pd.DataFrame:
    print(f"\n[1/6] Fetching games for {date} ...")
    sb = api_call_with_retry(ScoreboardV3, game_date=date.strftime("%Y-%m-%d"))

    games_df = pd.DataFrame()
    try:
        raw       = sb.get_dict()
        game_list = raw.get("scoreboard", {}).get("games", [])
        if game_list:
            rows = []
            for g in game_list:
                rows.append({
                    "GAME_ID": g.get("gameId", ""),
                    "HOME_ID": int(g["homeTeam"]["teamId"]),
                    "AWAY_ID": int(g["awayTeam"]["teamId"]),
                })
            games_df = pd.DataFrame(rows)
    except Exception as e:
        print(f"  Note: nested parse failed ({e}), trying data frames ...")

    if games_df.empty:
        for df in sb.get_data_frames():
            if df.empty:
                continue
            gid_col  = _find_col(df, ["gameId", "GAME_ID"])
            home_col = _find_col(df, ["homeTeamId", "HOME_TEAM_ID"])
            away_col = _find_col(df, ["awayTeamId", "VISITOR_TEAM_ID", "AWAY_TEAM_ID"])
            if gid_col and home_col and away_col:
                games_df = pd.DataFrame({
                    "GAME_ID": df[gid_col],
                    "HOME_ID": df[home_col].astype(int),
                    "AWAY_ID": df[away_col].astype(int),
                })
                break

    if games_df.empty:
        print("  No games found for this date.")
        return pd.DataFrame()

    games_df["HOME_ABBR"] = games_df["HOME_ID"].map(ID_TO_ABBR)
    games_df["AWAY_ABBR"] = games_df["AWAY_ID"].map(ID_TO_ABBR)
    games_df["HOME_NAME"] = games_df["HOME_ID"].map(ID_TO_NAME)
    games_df["AWAY_NAME"] = games_df["AWAY_ID"].map(ID_TO_NAME)

    print(f"  Found {len(games_df)} games tonight:")
    for _, row in games_df.iterrows():
        print(f"    {row['AWAY_NAME']} @ {row['HOME_NAME']}")

    return games_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FETCH INJURY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def get_injury_report(date: datetime.date, debug: bool = False) -> dict:
    print("\n[2/6] Fetching injury report ...")
    injured = _scrape_espn_injuries(debug=debug)
    if injured:
        total = sum(len(v) for v in injured.values())
        print(f"  Found {total} injured/questionable players across {len(injured)} teams")
        for abbr, players in injured.items():
            if len(players) > 10:
                print(f"  ⚠ {abbr} has {len(players)} players listed — possible parse issue")
    else:
        print("  Could not fetch injury report — using unadjusted PIS.")
    return injured


def _extract_name(athlete: dict) -> str:
    name = (
        athlete.get("displayName")
        or athlete.get("fullName")
        or athlete.get("shortName")
    )
    if name:
        return name.strip()
    first = athlete.get("firstName", "")
    last  = athlete.get("lastName", "")
    combined = f"{first} {last}".strip()
    return combined if combined else ""


def _add_injury(out: dict, abbr: str, inj: dict) -> None:
    status = inj.get("status", "")
    if status not in ("Out", "Doubtful", "Questionable"):
        return
    athlete = inj.get("athlete") or inj.get("player") or {}
    name = _extract_name(athlete)
    if name and abbr:
        out.setdefault(abbr, []).append((name, status))


def _scrape_espn_injuries(debug: bool = False) -> dict:
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
        data = json.loads(raw)
    except Exception as e:
        print(f"  ESPN fetch failed: {e}")
        return {}

    if debug:
        out_path = Path("espn_injuries_debug.json")
        out_path.write_text(json.dumps(data, indent=2))
        print(f"  [DEBUG] Raw ESPN JSON written to {out_path}")

    out = {}
    injury_list = data.get("injuries", [])

    if not injury_list:
        print(f"  ⚠ ESPN 'injuries' key is empty. Top-level keys: {list(data.keys())}")
        return {}

    first = injury_list[0]

    if "displayName" in first and "injuries" in first:
        print("  ESPN shape: A (displayName + nested injuries)")
        for team_entry in injury_list:
            team_name = team_entry.get("displayName", "")
            abbr = TEAM_NAME_TO_ABBR.get(team_name)
            if not abbr:
                raw_abbr = team_entry.get("abbreviation", "")
                abbr = normalize_abbr(ESPN_ABBR_PATCH.get(raw_abbr, raw_abbr)) if raw_abbr else ""
            if not abbr:
                print(f"  ⚠ Unknown team name from ESPN: '{team_name}'")
                continue
            for inj in team_entry.get("injuries", []):
                _add_injury(out, abbr, inj)
        return out

    if "team" in first and "injuries" in first:
        print("  ESPN shape: B (nested team dict)")
        for team_entry in injury_list:
            raw_abbr = team_entry.get("team", {}).get("abbreviation", "")
            abbr = normalize_abbr(ESPN_ABBR_PATCH.get(raw_abbr, raw_abbr))
            for inj in team_entry.get("injuries", []):
                _add_injury(out, abbr, inj)
        return out

    if "athletes" in data:
        print("  ESPN shape: C (flat athletes list)")
        for athlete in data["athletes"]:
            raw_abbr = athlete.get("team", {}).get("abbreviation", "")
            abbr   = normalize_abbr(ESPN_ABBR_PATCH.get(raw_abbr, raw_abbr))
            status = athlete.get("status", "")
            name   = _extract_name(athlete)
            if status in ("Out", "Doubtful", "Questionable") and name and abbr:
                out.setdefault(abbr, []).append((name, status))
        return out

    print(f"  ⚠ No recognised ESPN shape. Keys in first item: {list(first.keys())}")
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# 3. FETCH ADVANCED STATS (live, current season)
# ══════════════════════════════════════════════════════════════════════════════

def _call_league_dash_advanced_live(season: str):
    """
    Probes the LeagueDashTeamStats signature at runtime to handle kwarg
    differences across nba_api versions. Mirrors the same logic in nba.py
    so the live fetch works regardless of installed nba_api version.
    """
    sig    = inspect.signature(LeagueDashTeamStats.__init__)
    params = set(sig.parameters.keys())

    common_kwargs = dict(season=season, league_id="00")

    if "measure_type_simple" in params:
        endpoint = LeagueDashTeamStats(
            measure_type_simple="Advanced",
            per_mode_simple="PerGame",
            **common_kwargs,
        )
    elif "measure_type_detailed_defense" in params:
        endpoint = LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced",
            per_mode_simple="PerGame",
            **common_kwargs,
        )
    elif "measure_type" in params:
        endpoint = LeagueDashTeamStats(
            measure_type="Advanced",
            per_mode_simple="PerGame",
            **common_kwargs,
        )
    else:
        # Last resort: positional
        endpoint = LeagueDashTeamStats(season, "Advanced", "PerGame", league_id="00")

    return endpoint.get_data_frames()[0]


def fetch_advanced_stats_live(bundle_fallback: dict = None) -> dict:
    """
    Fetches current-season advanced stats for all 30 teams in one API call.
    Returns dict keyed by team abbreviation: {abbr: {feat: value}}.
    Uses runtime kwarg probe to handle nba_api version differences.
    Falls back to values baked into the model bundle if the API fails.
    """
    print(f"\n[3/6] Fetching live advanced stats ({CURRENT_SEASON}) ...")
    abbr_patch = {"NOH": "NOP", "BRK": "BKN", "PHO": "PHX", "CHO": "CHA", "GOS": "GSW"}

    try:
        df = _call_league_dash_advanced_live(CURRENT_SEASON)
        time.sleep(API_DELAY)

        df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].replace(abbr_patch)

        result = {}
        for _, row in df.iterrows():
            abbr = row["TEAM_ABBREVIATION"]
            result[abbr] = {
                feat: float(row[feat])
                for feat in ADV_FEATURES
                if feat in row.index and pd.notna(row[feat])
            }

        print(f"  ✓ Advanced stats fetched for {len(result)} teams")

        # Quick sanity check on net rating
        print("  Net ratings snapshot:")
        sorted_teams = sorted(result.items(), key=lambda x: x[1].get("E_NET_RATING", 0), reverse=True)
        for abbr, stats in sorted_teams[:5]:
            print(f"    {abbr}: {stats.get('E_NET_RATING', 'N/A'):+.1f}")
        print(f"    ...")

        return result

    except Exception as e:
        print(f"  ⚠ Live fetch failed: {e}")
        if bundle_fallback:
            print(f"  Using bundled advanced stats from model ({len(bundle_fallback)} teams)")
            return bundle_fallback
        print("  Advanced stats unavailable — these features will be 0.")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# INJURY PIS ADJUSTMENT (BUG 1 + BUG 5 fix)
# ══════════════════════════════════════════════════════════════════════════════

def adjust_pis_for_injuries(
    team_pis_row: pd.Series,
    team_abbr: str,
    injured_players: dict,
    player_pis_df: pd.DataFrame,
    season: str,
) -> tuple:
    base_pis   = float(team_pis_row.get("TEAM_PIS",  0))
    base_top3  = float(team_pis_row.get("TOP3_PIS",  0))
    base_depth = float(team_pis_row.get("DEPTH_PIS", 0))

    outs = injured_players.get(team_abbr, [])
    if not outs or player_pis_df is None or player_pis_df.empty:
        return (
            {"TEAM_PIS": base_pis, "TOP3_PIS": base_top3, "DEPTH_PIS": base_depth},
            []
        )

    mask         = (player_pis_df["TEAM"] == team_abbr) & (player_pis_df["SEASON"] == season)
    team_players = player_pis_df[mask].copy()
    has_is_top3  = "IS_TOP3" in team_players.columns
    team_players["_NORM_PLAYER"] = team_players["PLAYER"].apply(_normalize_name)

    # Replacement efficiency: backups fill ~65% of missing player's value
    REPLACEMENT_EFF = 0.65

    pis_lost   = 0.0
    top3_lost  = 0.0
    depth_lost = 0.0
    log        = []

    for name, status in outs:
        exact_match = team_players[
            team_players["PLAYER"].str.lower().str.strip() == name.lower().strip()
        ]

        if not exact_match.empty:
            match     = exact_match
            fuzzy_hit = False
        else:
            norm_name = _normalize_name(name)
            match     = team_players[team_players["_NORM_PLAYER"] == norm_name]
            fuzzy_hit = not match.empty

        if not match.empty:
            player_pis  = float(match.iloc[0].get("PIS", 0))
            is_top3     = float(match.iloc[0].get("IS_TOP3", 0)) if has_is_top3 else 0.0
            is_depth    = 1.0 - is_top3

            # Net gap after replacement: star loses more than bench player
            net_pis_gap  = player_pis * (1.0 - REPLACEMENT_EFF)
            pis_lost    += net_pis_gap
            top3_lost   += is_top3  * net_pis_gap
            depth_lost  += is_depth * net_pis_gap

            tag = " [TOP3]" if is_top3 else ""

            # FIX: use abs() so negative-PIS players don't produce double-negative
            # e.g. player_pis=-0.20 → net_pis_gap=-0.07 → display as "-0.07 PIS"
            sign   = "-" if net_pis_gap >= 0 else "+"
            log.append(
                f"      ✗ {name} ({team_abbr}, {status}): {sign}{abs(net_pis_gap):.2f} PIS{tag}"
            )
            if fuzzy_hit:
                bbref_name = match.iloc[0]["PLAYER"]
                log.append(f"        ↳ suffix-strip match: '{name}' → '{bbref_name}'")

    adjusted = {
        "TEAM_PIS"  : max(0, base_pis   - pis_lost),
        "TOP3_PIS"  : max(0, base_top3  - top3_lost),
        "DEPTH_PIS" : max(0, base_depth - depth_lost),
    }
    return adjusted, log


# ── PIS recency blending (BUG 3) ──────────────────────────────────────────────
_LEAGUE_PM_MEAN = 0.0
_LEAGUE_PM_STD  = 7.5

def blend_pis_with_recent_form(season_pis: dict, rolling_stats: dict) -> dict:
    recent_pm = rolling_stats.get("ROLL10_PLUS_MINUS", None)

    if recent_pm is None or np.isnan(recent_pm):
        return season_pis

    PIS_SCALE    = 2.5
    recent_proxy = ((recent_pm - _LEAGUE_PM_MEAN) / _LEAGUE_PM_STD) * PIS_SCALE

    blended_team_pis = (
        PIS_SEASON_WEIGHT * season_pis["TEAM_PIS"]
        + PIS_RECENT_WEIGHT * recent_proxy
    )
    ratio = (blended_team_pis / season_pis["TEAM_PIS"]
             if season_pis["TEAM_PIS"] != 0 else 1.0)
    ratio = np.clip(ratio, 0.5, 2.0)

    return {
        "TEAM_PIS"  : blended_team_pis,
        "TOP3_PIS"  : season_pis["TOP3_PIS"]  * ratio,
        "DEPTH_PIS" : season_pis["DEPTH_PIS"] * ratio,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROLLING STATS
# ══════════════════════════════════════════════════════════════════════════════

_TEAM_LOG_CACHE = {}


def get_team_rolling_stats(team_id: int, n: int = ROLLING_WINDOW) -> dict:
    if team_id in _TEAM_LOG_CACHE:
        return _TEAM_LOG_CACHE[team_id]

    try:
        log = api_call_with_retry(
            TeamGameLog,
            team_id=team_id,
            season=CURRENT_SEASON,
            season_type_all_star="Regular Season",
        )
        df = log.get_data_frames()[0]

        if df.empty or len(df) < 3:
            _TEAM_LOG_CACHE[team_id] = {}
            return {}

        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df = df.sort_values("GAME_DATE", ascending=True).reset_index(drop=True)

        recent = df.tail(n)
        avgs   = {}
        for col in CORE_STATS:
            if col in recent.columns:
                avgs[f"ROLL10_{col}"] = float(recent[col].mean())

        if "WL" in recent.columns:
            wins     = (recent["WL"] == "W")
            raw_wpct = float(wins.mean())
            avgs["ROLL_WIN_PCT"] = raw_wpct

            sos_weighted_wins = []
            if "MATCHUP" in recent.columns and _GLOBAL_ELO:
                for _, row in recent.iterrows():
                    matchup  = str(row.get("MATCHUP", ""))
                    parts    = matchup.replace("vs.", "@").split("@")
                    opp_abbr = normalize_abbr(parts[1].strip()) if len(parts) == 2 else ""
                    opp_elo  = _GLOBAL_ELO.get(opp_abbr, 1500.0)
                    weight   = np.clip(1.0 + (opp_elo - 1500.0) / 1000.0, 0.5, 1.5)
                    won      = 1.0 if row.get("WL", "") == "W" else 0.0
                    sos_weighted_wins.append(won * weight)

                avgs["SOS_WIN_PCT"] = float(np.mean(sos_weighted_wins)) if sos_weighted_wins else raw_wpct
            else:
                avgs["SOS_WIN_PCT"] = raw_wpct

        avgs["GAMES_PLAYED"] = len(df)

        # v7: multi-window plus/minus
        if "PLUS_MINUS" in df.columns:
            avgs["ROLL5_PLUS_MINUS"]  = float(df.tail(5)["PLUS_MINUS"].mean())
            avgs["ROLL20_PLUS_MINUS"] = float(df.tail(20)["PLUS_MINUS"].mean())

        # v7: games in last 7 days (before today)
        if "GAME_DATE" in df.columns:
            today     = pd.Timestamp.today().normalize()
            days_ago  = (today - df["GAME_DATE"]).dt.days
            avgs["GAMES_IN_LAST_7"] = int(((days_ago > 0) & (days_ago <= 7)).sum())
        else:
            avgs["GAMES_IN_LAST_7"] = 0

        # v7: consecutive road games before tonight (road trip length)
        if "MATCHUP" in df.columns:
            trip = 0
            for matchup in reversed(df["MATCHUP"].tolist()):
                if "@" in str(matchup) and "vs." not in str(matchup):
                    trip += 1
                else:
                    break
            avgs["ROAD_TRIP_LENGTH"] = trip
        else:
            avgs["ROAD_TRIP_LENGTH"] = 0

        # v7: season progress (fraction of 82-game season completed)
        avgs["SEASON_PROGRESS"] = min(len(df) / 82.0, 1.0)

        # v8: EWMA features — recent games weighted exponentially (halflife=5)
        for stat in ["PLUS_MINUS", "PTS", "FG_PCT"]:
            if stat in df.columns and len(df) >= 5:
                avgs[f"EWMA_{stat}"] = float(
                    df[stat].ewm(halflife=5, min_periods=5).mean().iloc[-1]
                )

        _TEAM_LOG_CACHE[team_id] = avgs
        return avgs

    except Exception as e:
        print(f"    Warning: Could not fetch game log for team {team_id}: {e}")
        _TEAM_LOG_CACHE[team_id] = {}
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD FEATURE VECTOR
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(
    home_id: int,
    away_id: int,
    home_abbr: str,
    away_abbr: str,
    team_pis_df: pd.DataFrame,
    player_pis_df: pd.DataFrame,
    injured_players: dict,
    model_features: list,
    elo: dict,
    adv_stats: dict,
    show_features: bool = False,
) -> tuple:
    norm_home = home_abbr
    norm_away = away_abbr

    home_stats = get_team_rolling_stats(home_id)
    away_stats = get_team_rolling_stats(away_id)

    def get_team_pis(abbr):
        mask = (team_pis_df["TEAM"] == abbr) & (team_pis_df["SEASON"] == CURRENT_SEASON)
        rows = team_pis_df[mask]
        if rows.empty:
            rows = team_pis_df[team_pis_df["TEAM"] == abbr]
        return rows.iloc[-1] if not rows.empty else None

    home_pis_row = get_team_pis(norm_home)
    away_pis_row = get_team_pis(norm_away)

    injury_log_home = []
    injury_log_away = []

    if home_pis_row is None or away_pis_row is None:
        missing = [a for a, r in [(norm_home, home_pis_row), (norm_away, away_pis_row)] if r is None]
        print(f"    Warning: PIS data missing for {', '.join(missing)}")
        home_pis_adj = away_pis_adj = {"TEAM_PIS": 0, "TOP3_PIS": 0, "DEPTH_PIS": 0}
    else:
        home_pis_adj, injury_log_home = adjust_pis_for_injuries(
            home_pis_row, norm_home, injured_players, player_pis_df, CURRENT_SEASON
        )
        away_pis_adj, injury_log_away = adjust_pis_for_injuries(
            away_pis_row, norm_away, injured_players, player_pis_df, CURRENT_SEASON
        )
        home_pis_adj = blend_pis_with_recent_form(home_pis_adj, home_stats)
        away_pis_adj = blend_pis_with_recent_form(away_pis_adj, away_stats)

    feat = {}

    # PIS diffs
    feat["PIS_DIFF"]       = home_pis_adj["TEAM_PIS"]  - away_pis_adj["TEAM_PIS"]
    feat["TOP3_PIS_DIFF"]  = home_pis_adj["TOP3_PIS"]  - away_pis_adj["TOP3_PIS"]
    feat["DEPTH_PIS_DIFF"] = home_pis_adj["DEPTH_PIS"] - away_pis_adj["DEPTH_PIS"]

    # Injury impact diff — derived from PIS adjustments already computed above.
    # home_pis_row["TEAM_PIS"] is the pre-injury baseline; home_pis_adj["TEAM_PIS"]
    # is post-injury. The difference is the PIS lost.
    home_pis_base = float(home_pis_row.get("TEAM_PIS", 0)) if home_pis_row is not None else 0.0
    away_pis_base = float(away_pis_row.get("TEAM_PIS", 0)) if away_pis_row is not None else 0.0
    home_impact_missing = max(home_pis_base - home_pis_adj["TEAM_PIS"], 0.0)
    away_impact_missing = max(away_pis_base - away_pis_adj["TEAM_PIS"], 0.0)
    feat["IMPACT_MISSING_DIFF"] = home_impact_missing - away_impact_missing

    # Rolling stat diffs
    for col in CORE_STATS:
        h_val = home_stats.get(f"ROLL10_{col}", np.nan)
        a_val = away_stats.get(f"ROLL10_{col}", np.nan)
        feat[f"ROLL10_{col}_DIFF"] = (
            h_val - a_val if not (np.isnan(h_val) or np.isnan(a_val)) else np.nan
        )

    # v8: EWMA stat diffs
    for stat in ["PLUS_MINUS", "PTS", "FG_PCT"]:
        h_val = home_stats.get(f"EWMA_{stat}", np.nan)
        a_val = away_stats.get(f"EWMA_{stat}", np.nan)
        feat[f"EWMA_{stat}_DIFF"] = (
            h_val - a_val if not (np.isnan(h_val) or np.isnan(a_val)) else 0.0
        )

    # Scheduling — neutral defaults
    feat["DAYS_REST_DIFF"]       = 0.0
    feat["IS_BACK_TO_BACK_DIFF"] = 0.0
    feat["WIN_STREAK_DIFF"]      = 0.0
    feat["DAYS_SINCE_HOME_DIFF"] = 0.0
    feat["ADJ_OFF_RATING_DIFF"]  = np.nan
    feat["ADJ_DEF_RATING_DIFF"]  = np.nan

    # Win percentage features
    home_wpct = home_stats.get("ROLL_WIN_PCT", np.nan)
    away_wpct = away_stats.get("ROLL_WIN_PCT", np.nan)
    feat["HOME_ROLL_WIN_PCT"] = home_wpct if not np.isnan(home_wpct) else 0.5
    feat["AWAY_ROLL_WIN_PCT"] = away_wpct if not np.isnan(away_wpct) else 0.5
    feat["ROLL_WIN_PCT_DIFF"] = feat["HOME_ROLL_WIN_PCT"] - feat["AWAY_ROLL_WIN_PCT"]
    feat["SEASON_WIN_PCT_DIFF"] = 0.0

    home_sos = home_stats.get("SOS_WIN_PCT", feat["HOME_ROLL_WIN_PCT"])
    away_sos = away_stats.get("SOS_WIN_PCT", feat["AWAY_ROLL_WIN_PCT"])
    feat["SOS_WIN_PCT_DIFF"] = home_sos - away_sos

    # Margin features from rolling plus/minus
    home_pm = home_stats.get("ROLL10_PLUS_MINUS", 0.0) or 0.0
    away_pm = away_stats.get("ROLL10_PLUS_MINUS", 0.0) or 0.0
    if np.isnan(home_pm): home_pm = 0.0
    if np.isnan(away_pm): away_pm = 0.0
    pm_diff = home_pm - away_pm

    feat["ROLL10_POINT_DIFF_DIFF"] = pm_diff
    feat["SEASON_POINT_DIFF_DIFF"] = pm_diff
    feat["BLOWOUT_RATE_DIFF"]      = float(np.clip(pm_diff / 10.0, -1.0, 1.0))

    # v7: multi-window PM and fatigue diffs
    for v7_feat in ("ROLL5_PLUS_MINUS", "ROLL20_PLUS_MINUS",
                    "GAMES_IN_LAST_7", "ROAD_TRIP_LENGTH", "SEASON_PROGRESS"):
        h_val = home_stats.get(v7_feat, 0.0) or 0.0
        a_val = away_stats.get(v7_feat, 0.0) or 0.0
        feat[f"{v7_feat}_DIFF"] = h_val - a_val

    # Elo from model bundle
    feat["HOME_ELO"] = elo.get(norm_home, 1500.0)
    feat["AWAY_ELO"] = elo.get(norm_away, 1500.0)
    feat["ELO_DIFF"] = feat["HOME_ELO"] - feat["AWAY_ELO"]

    # BUG 4: Record-based Elo reality check
    home_win_pct = home_stats.get("ROLL_WIN_PCT", 0.5)
    away_win_pct = away_stats.get("ROLL_WIN_PCT", 0.5)
    home_implied = 1500.0 + (home_win_pct - 0.5) * 750.0
    away_implied = 1500.0 + (away_win_pct - 0.5) * 750.0
    gap          = abs(home_implied - away_implied)
    blend        = min(gap / 300.0, 0.6)
    feat["HOME_ELO"] = (1.0 - blend) * feat["HOME_ELO"] + blend * home_implied
    feat["AWAY_ELO"] = (1.0 - blend) * feat["AWAY_ELO"] + blend * away_implied
    feat["ELO_DIFF"] = feat["HOME_ELO"] - feat["AWAY_ELO"]

    # ── Advanced stat diffs (v6) ───────────────────────────────────────────────
    home_adv = adv_stats.get(norm_home, {})
    away_adv = adv_stats.get(norm_away, {})

    for adv_feat in ADV_FEATURES:
        diff_col = f"{adv_feat}_DIFF"
        if diff_col in model_features:
            h_val = home_adv.get(adv_feat, np.nan)
            a_val = away_adv.get(adv_feat, np.nan)
            feat[diff_col] = (
                h_val - a_val
                if (not np.isnan(h_val) and not np.isnan(a_val))
                else 0.0
            )

    # v8: Conference/division matchup context
    feat["SAME_CONFERENCE"] = int(
        TEAM_CONF.get(norm_home, "") != "" and
        TEAM_CONF.get(norm_home, "") == TEAM_CONF.get(norm_away, "")
    )
    feat["SAME_DIVISION"] = int(
        TEAM_DIV.get(norm_home, "") != "" and
        TEAM_DIV.get(norm_home, "") == TEAM_DIV.get(norm_away, "")
    )

    row     = {f: feat.get(f, np.nan) for f in model_features}
    feat_df = pd.DataFrame([row]).fillna(0)

    if show_features:
        print(f"\n  Feature vector for {away_abbr} @ {home_abbr}:")
        for k, v in row.items():
            if isinstance(v, float):
                print(f"    {k:48s} {v:+.4f}")
            else:
                print(f"    {k:48s} {v}")

    return feat_df, injury_log_home, injury_log_away


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOAD MODEL + RUN PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

_GLOBAL_ELO: dict = {}


def load_model():
    if not MODEL_PATH.exists():
        fallback = Path("nba_model_v6.pkl")
        if fallback.exists():
            print(f"  Note: {MODEL_PATH} not found, loading {fallback} (v6)")
            obj = joblib.load(fallback)
        else:
            print(f"ERROR: No model found. Run nba.py first.")
            sys.exit(1)
    else:
        obj = joblib.load(MODEL_PATH)

    bundle  = obj if isinstance(obj, dict) else {"calibrated_model": obj, "scaler": None, "feature_cols": None}
    version    = bundle.get("version", "unknown")
    has_margin = bundle.get("margin_model") is not None
    has_meta   = bundle.get("meta_model")   is not None
    has_adv    = bool(bundle.get("adv_stats"))
    print(f"  ✓ Loaded model bundle (version={version}, margin={'yes' if has_margin else 'no'}, meta={'yes' if has_meta else 'no'}, adv_stats={'yes' if has_adv else 'no'})")
    return bundle


def get_estimator(model_bundle):
    return model_bundle["calibrated_model"] if isinstance(model_bundle, dict) else model_bundle


def get_scaler(model_bundle):
    return model_bundle.get("scaler") if isinstance(model_bundle, dict) else None


def load_pis_data():
    abbr_map = {"NOH": "NOP", "BRK": "BKN", "PHO": "PHX", "CHO": "CHA", "GOS": "GSW"}
    team_pis = pd.DataFrame()
    if PIS_PATH.exists():
        team_pis = pd.read_csv(PIS_PATH)
        team_pis["TEAM"] = team_pis["TEAM"].replace(abbr_map)
    else:
        print(f"WARNING: {PIS_PATH} not found. PIS features will be zero.")
    player_pis = pd.DataFrame()
    if PLAYER_PIS_PATH.exists():
        player_pis = pd.read_csv(PLAYER_PIS_PATH)
        player_pis["TEAM"] = player_pis["TEAM"].replace(abbr_map)
    else:
        print(f"WARNING: {PLAYER_PIS_PATH} not found. Injury adjustments disabled.")
    return team_pis, player_pis


def get_model_feature_names(model_bundle) -> list:
    if isinstance(model_bundle, dict):
        cols = model_bundle.get("feature_cols")
        if cols:
            print(f"  ✓ Loaded {len(cols)} feature names from model bundle.")
            return list(cols)
    model = get_estimator(model_bundle)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    print("  Note: Could not extract feature names. Using v6 default list.")
    return [
        "PIS_DIFF", "TOP3_PIS_DIFF", "DEPTH_PIS_DIFF",
        "ROLL10_PTS_DIFF", "ROLL10_FG_PCT_DIFF", "ROLL10_FG3_PCT_DIFF",
        "ROLL10_FT_PCT_DIFF", "ROLL10_REB_DIFF", "ROLL10_AST_DIFF",
        "ROLL10_TOV_DIFF", "ROLL10_STL_DIFF", "ROLL10_BLK_DIFF",
        "ROLL10_PLUS_MINUS_DIFF",
        "DAYS_REST_DIFF", "IS_BACK_TO_BACK_DIFF", "WIN_STREAK_DIFF",
        "DAYS_SINCE_HOME_DIFF",
        "ADJ_OFF_RATING_DIFF", "ADJ_DEF_RATING_DIFF",
        "HOME_ROLL_WIN_PCT", "AWAY_ROLL_WIN_PCT", "ROLL_WIN_PCT_DIFF",
        "SEASON_WIN_PCT_DIFF", "SOS_WIN_PCT_DIFF",
        "HOME_ELO", "AWAY_ELO", "ELO_DIFF",
        "ROLL10_POINT_DIFF_DIFF", "SEASON_POINT_DIFF_DIFF", "BLOWOUT_RATE_DIFF",
        "E_NET_RATING_DIFF", "E_OFF_RATING_DIFF", "E_DEF_RATING_DIFF",
        "E_PACE_DIFF", "TS_PCT_DIFF", "E_AST_RATIO_DIFF",
        "E_OREB_PCT_DIFF", "E_DREB_PCT_DIFF", "E_TM_TOV_PCT_DIFF",
    ]


def validate_features(model_features: list) -> None:
    uncomputable = [f for f in model_features if f not in COMPUTABLE_FEATURES]
    pct = len(uncomputable) / len(model_features) if model_features else 0
    if uncomputable:
        print(f"\n  ⚠ WARNING: {len(uncomputable)} model feature(s) cannot be computed "
              f"at inference time and will be set to 0:")
        for f in uncomputable:
            print(f"      {f}")
        if pct > 0.10:
            print(f"\n  ✗ ERROR: {pct:.0%} of features are uncomputable (>{10:.0%} threshold).")
            print("    Delete the model pkl, retrain nba.py, then re-run.")
            sys.exit(1)
        print("  Consider retraining nba.py to eliminate these.\n")
    else:
        print(f"  ✓ All {len(model_features)} model features are computable at inference time.")


def _get_residual_std(model_bundle: dict, predicted_margin: float) -> float:
    """
    Returns spread-magnitude-aware residual std if conditional stds are
    stored in the bundle, otherwise falls back to the global std.
    """
    cond       = model_bundle.get("conditional_stds", {}) if isinstance(model_bundle, dict) else {}
    global_std = model_bundle.get("margin_residual_std", 12.0) if isinstance(model_bundle, dict) else 12.0

    abs_margin = abs(predicted_margin)
    if abs_margin < 5 and cond.get("close") is not None:
        return cond["close"]
    elif abs_margin < 12 and cond.get("medium") is not None:
        return cond["medium"]
    elif abs_margin >= 12 and cond.get("large") is not None:
        return cond["large"]
    return global_std


def predict_game(model_bundle, feat_df: pd.DataFrame):
    """
    Returns (home_prob, away_prob, predicted_spread).
    v7: meta-stacker blends classifier_prob + margin_prob when available.
    Fallback chain: meta → margin → classifier.
    """
    try:
        scaler = get_scaler(model_bundle)
        X = feat_df.values
        if scaler is not None:
            X = scaler.transform(X)

        margin_model = model_bundle.get("margin_model") if isinstance(model_bundle, dict) else None
        meta_model   = model_bundle.get("meta_model")   if isinstance(model_bundle, dict) else None

        predicted_margin   = None
        margin_prob        = 0.5
        classifier_prob    = 0.5

        if margin_model is not None:
            predicted_margin = float(margin_model.predict(X)[0])
            residual_std     = _get_residual_std(model_bundle, predicted_margin)
            margin_prob      = float(scipy_norm.cdf(predicted_margin / residual_std))

        try:
            estimator       = get_estimator(model_bundle)
            prob            = estimator.predict_proba(X)[0]
            classifier_prob = float(prob[1]) if len(prob) == 2 else 0.5
        except Exception:
            pass

        if meta_model is not None and predicted_margin is not None:
            meta_X    = np.array([[
                classifier_prob,
                margin_prob,
                abs(predicted_margin),
                classifier_prob - margin_prob,
            ]])
            home_prob = float(meta_model.predict_proba(meta_X)[0][1])
        elif margin_model is not None:
            home_prob = margin_prob
        else:
            home_prob = classifier_prob

        away_prob = 1.0 - home_prob
        return home_prob, away_prob, predicted_margin

    except Exception as e:
        print(f"    Prediction error: {e}")
        return 0.5, 0.5, None


def confidence_label(prob: float) -> str:
    if prob >= 0.70: return "🔥 STRONG"
    if prob >= 0.60: return "✅ LEAN"
    if prob >= 0.55: return "⚠️  SLIGHT"
    return "🪙 TOSS-UP"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _GLOBAL_ELO

    parser = argparse.ArgumentParser(description="Predict tonight's NBA games")
    parser.add_argument("--date", default=None, help="Date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--show-features", action="store_true", help="Print feature vector for each game")
    parser.add_argument("--debug-injuries", action="store_true",
                        help="Dump raw ESPN JSON to espn_injuries_debug.json")
    parser.add_argument("--skip-adv-fetch", action="store_true",
                        help="Skip live advanced stats fetch, use bundle values only")
    args = parser.parse_args()

    game_date = datetime.date.today() if not args.date else datetime.date.fromisoformat(args.date)

    print("=" * 65)
    print("  NBA GAME WINNER PREDICTOR — LIVE MODE (v7)")
    print(f"  Date: {game_date}")
    print("=" * 65)

    model_bundle          = load_model()
    team_pis, player_pis  = load_pis_data()
    features              = get_model_feature_names(model_bundle)
    elo                   = model_bundle.get("elo_ratings", {}) if isinstance(model_bundle, dict) else {}
    _GLOBAL_ELO.update(elo)
    validate_features(features)

    games = get_tonights_games(game_date)
    if games.empty:
        print("No games to predict. Exiting.")
        sys.exit(0)

    injured = get_injury_report(game_date, debug=args.debug_injuries)

    # Advanced stats: live fetch with bundle as fallback
    bundle_adv = model_bundle.get("adv_stats", {}) if isinstance(model_bundle, dict) else {}
    if args.skip_adv_fetch:
        print(f"\n[3/6] Using bundled advanced stats ({len(bundle_adv)} teams)")
        adv_stats = bundle_adv
    else:
        adv_stats = fetch_advanced_stats_live(bundle_fallback=bundle_adv)

    print(f"\n[4/6] Fetching rolling stats for {len(games) * 2} teams ...")
    print(f"[5/6] Building feature vectors ...")
    print(f"[6/6] Running predictions ...\n")

    print("=" * 65)
    print(f"  PREDICTIONS FOR {game_date}")
    print("=" * 65)

    results = []
    for _, game in games.iterrows():
        home_id   = int(game["HOME_ID"])
        away_id   = int(game["AWAY_ID"])
        home_abbr = game["HOME_ABBR"]
        away_abbr = game["AWAY_ABBR"]
        home_name = game["HOME_NAME"]
        away_name = game["AWAY_NAME"]

        result = build_feature_vector(
            home_id, away_id, home_abbr, away_abbr,
            team_pis, player_pis, injured, features, elo,
            adv_stats=adv_stats,
            show_features=args.show_features,
        )
        if result is None:
            print(f"  {away_name} @ {home_name}: SKIPPED (insufficient data)")
            continue

        feat_df, injury_log_home, injury_log_away = result

        home_prob, away_prob, predicted_spread = predict_game(model_bundle, feat_df)
        winner      = home_name if home_prob > away_prob else away_name
        winner_prob = max(home_prob, away_prob)
        conf        = confidence_label(winner_prob)

        print(f"\n  {away_name} @ {home_name}")
        print(f"  {'─' * 50}")

        if injury_log_home:
            print(f"  {home_abbr} injury deductions:")
            for line in injury_log_home:
                print(line)

        if injury_log_away:
            print(f"  {away_abbr} injury deductions:")
            for line in injury_log_away:
                print(line)

        # Show advanced stat snapshot for this matchup
        home_adv = adv_stats.get(home_abbr, {})
        away_adv = adv_stats.get(away_abbr, {})
        if home_adv and away_adv:
            h_net  = home_adv.get("E_NET_RATING", None)
            a_net  = away_adv.get("E_NET_RATING", None)
            h_pace = home_adv.get("E_PACE", None)
            a_pace = away_adv.get("E_PACE", None)
            if h_net is not None and a_net is not None:
                print(f"  Net rating:  {home_abbr} {h_net:+.1f}  |  {away_abbr} {a_net:+.1f}")
            if h_pace is not None and a_pace is not None:
                print(f"  Pace:        {home_abbr} {h_pace:.1f}   |  {away_abbr} {a_pace:.1f}")

        print(f"  Home ({home_abbr}):  {home_prob * 100:.1f}%")
        print(f"  Away ({away_abbr}):  {away_prob * 100:.1f}%")
        if predicted_spread is not None:
            spread_side = home_abbr if predicted_spread >= 0 else away_abbr
            print(f"  Predicted spread:  {spread_side} -{abs(predicted_spread):.1f}")
        print(f"  Prediction:  {winner}  ({winner_prob * 100:.1f}%)  {conf}")

        home_outs = injured.get(home_abbr, [])
        away_outs = injured.get(away_abbr, [])
        if home_outs:
            print(f"  {home_abbr} out: {', '.join(f'{n} ({s})' for n, s in home_outs[:5])}")
        if away_outs:
            print(f"  {away_abbr} out: {', '.join(f'{n} ({s})' for n, s in away_outs[:5])}")

        results.append({
            "Game"            : f"{away_abbr} @ {home_abbr}",
            "Home%"           : round(home_prob * 100, 1),
            "Away%"           : round(away_prob * 100, 1),
            "Predicted_Spread": round(predicted_spread, 1) if predicted_spread is not None else None,
            "Pick"            : winner,
            "Confidence"      : conf,
            "Home_Out"        : "; ".join(f"{n} ({s})" for n, s in home_outs[:3]),
            "Away_Out"        : "; ".join(f"{n} ({s})" for n, s in away_outs[:3]),
        })

    print("\n" + "=" * 65)
    print("  SUMMARY TABLE")
    print("=" * 65)
    summary = pd.DataFrame(results)
    if not summary.empty:
        print(summary.to_string(index=False))
        out_file = Path(f"predictions_{game_date}.csv")
        summary.to_csv(out_file, index=False)
        print(f"\n  ✓ Saved to {out_file}")

    print("\n  Confidence guide:")
    print("    🔥 STRONG  = 70%+  (model's highest edge)")
    print("    ✅ LEAN    = 60-70% (decent signal)")
    print("    ⚠️  SLIGHT  = 55-60% (lean but noisy)")
    print("    🪙 TOSS-UP = <55%   (skip or fade)")

    from export_predictions_json import export_json
    export_json(
        predictions=results,
        team_log_cache=_TEAM_LOG_CACHE,
        injured=injured,
        player_pis_df=player_pis,
        elo=elo,
    )


if __name__ == "__main__":
    main()
