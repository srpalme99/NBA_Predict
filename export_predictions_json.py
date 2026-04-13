"""
export_predictions_json.py  (v3.1 — name-only fallback for PIS lookup)
=======================================================================
Drop this file into your NBA predictor project folder.

USAGE FROM predict_tonight.py
------------------------------
At the very end of main(), replace the old two-liner with:

    from export_predictions_json import export_json
    export_json(
        predictions=results,
        team_log_cache=_TEAM_LOG_CACHE,
        injured=injured,
        player_pis_df=player_pis,
        elo=elo,
    )

STANDALONE USAGE (converts today's CSV, no rolling stats)
----------------------------------------------------------
    python export_predictions_json.py
    python export_predictions_json.py --csv predictions_2026-04-04.csv

v3.1 changes (over v3)
-----------------------
- mark_stars() now has a two-step lookup:
    1. TEAM + name match (exact, as before)
    2. Name-only fallback if step 1 finds nothing
  This fixes two common "PIS —" cases:
    a) Traded players: Luka is stored as DAL in the CSV but ESPN
       reports him under LAL — team filter blocks the match.
    b) Accent stripping: ESPN returns "Luka Doncic", CSV has
       "Luka Doncic" — handled by the accent-normalise helper.
- Terminal output now lists any players still missing PIS so you
  know exactly who is not in player_impact_scores.csv.
"""

import json
import csv
import re
import unicodedata
from pathlib import Path
from datetime import date, datetime


# ── Stat display config ────────────────────────────────────────────────────────
STAT_DISPLAY = [
    ("ROLL10_PTS",        "Points / g",    True),
    ("ROLL10_FG_PCT",     "FG %",          True),
    ("ROLL10_FG3_PCT",    "3PT %",         True),
    ("ROLL10_AST",        "Assists / g",   True),
    ("ROLL10_REB",        "Rebounds / g",  True),
    ("ROLL10_TOV",        "Turnovers / g", False),
    ("ROLL10_STL",        "Steals / g",    True),
    ("ROLL10_BLK",        "Blocks / g",    True),
    ("ROLL10_PLUS_MINUS", "Net rating",    True),
    ("ROLL_WIN_PCT",      "Win % (L10)",   True),
    ("SOS_WIN_PCT",       "SOS win %",     True),
]

TOP_N_STATS = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_accents(s: str) -> str:
    """'Luka Doncic' → 'Luka Doncic'  (removes diacritics)"""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def name_key(s: str) -> str:
    """Lowercase + strip accents + collapse whitespace for fuzzy matching."""
    return strip_accents(s).lower().strip()


def confidence_tier(pct: float) -> str:
    if pct >= 70: return "strong"
    if pct >= 60: return "lean"
    if pct >= 55: return "slight"
    return "tossup"


def fmt_stat(key: str, val) -> str:
    if val is None:
        return "—"
    v = float(val)
    if key in ("ROLL10_FG_PCT", "ROLL10_FG3_PCT", "ROLL_WIN_PCT", "SOS_WIN_PCT"):
        return f"{v:.3f}"
    if key == "ROLL10_PLUS_MINUS":
        return f"{v:+.1f}"
    return f"{v:.1f}"


def parse_injured_str(raw: str) -> list:
    """Parse 'Name (Status); Name2 (Status2)' CSV string into list of dicts."""
    if not raw or str(raw).strip() in ("", "nan"):
        return []
    players = []
    for i, chunk in enumerate(str(raw).split(";")):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^(.+?)\s*\((.+?)\)$", chunk)
        if m:
            players.append({
                "name":   m.group(1).strip(),
                "status": m.group(2).strip(),
                "star":   i == 0,
                "pis":    None,
            })
        else:
            players.append({
                "name":   chunk,
                "status": "Out",
                "star":   i == 0,
                "pis":    None,
            })
    return players


def mark_stars(abbr: str, player_list: list, pis_df) -> list:
    """
    Build the injury entry list with star and pis fields.

    Lookup strategy (two steps so traded players and accented names still match):
      1. TEAM == abbr  AND  normalised name matches   <- exact, preferred
      2. Normalised name matches across ALL teams      <- fallback for trades/accents

    Fields returned per player:
      name   : str
      status : str   ("Out" / "Doubtful" / "Questionable")
      star   : bool  — True if IS_TOP3 in their CSV row
      pis    : float | None — individual PIS shown in the dashboard
    """
    if pis_df is None or pis_df.empty:
        result = []
        for i, item in enumerate(player_list):
            name   = item[0] if isinstance(item, tuple) else item.get("name", "")
            status = item[1] if isinstance(item, tuple) else item.get("status", "Out")
            result.append({"name": name, "status": status, "star": i == 0, "pis": None})
        return result

    # Pre-compute normalised name column once per call
    pis_df = pis_df.copy()
    pis_df["_name_key"] = pis_df["PLAYER"].apply(name_key)

    has_is_top3 = "IS_TOP3" in pis_df.columns

    result = []
    for i, item in enumerate(player_list):
        if isinstance(item, tuple):
            name, status = item
        elif isinstance(item, dict):
            name   = item.get("name", "")
            status = item.get("status", "Out")
        else:
            continue

        nk = name_key(name)

        # Step 1: team + normalised name
        match = pis_df[(pis_df["TEAM"] == abbr) & (pis_df["_name_key"] == nk)]

        # Step 2: name only (handles trades + accents)
        if match.empty:
            match = pis_df[pis_df["_name_key"] == nk]

        if not match.empty:
            row = match.iloc[0]
            raw_pis = row.get("PIS", None)
            try:
                player_pis = round(float(raw_pis), 2) if raw_pis is not None else None
            except (ValueError, TypeError):
                player_pis = None
            is_top3 = bool(row["IS_TOP3"]) if has_is_top3 else (i == 0)
        else:
            # Genuinely not in the CSV (two-way / rookie / insufficient minutes)
            player_pis = None
            is_top3    = (i == 0)

        result.append({
            "name":   name,
            "status": status,
            "star":   is_top3,
            "pis":    player_pis,
        })

    return result


def build_stats(home_abbr: str, away_abbr: str,
                team_log_cache: dict, abbr_to_id: dict,
                elo: dict | None) -> list:
    """
    Pull rolling stats for both teams, pick the TOP_N_STATS with the
    biggest absolute difference (most decisive), prepend Elo.
    """
    home_id = abbr_to_id.get(home_abbr)
    away_id = abbr_to_id.get(away_abbr)
    home_s  = team_log_cache.get(home_id, {}) if home_id else {}
    away_s  = team_log_cache.get(away_id, {}) if away_id else {}

    rows = []
    for key, label, higher_better in STAT_DISPLAY:
        h = home_s.get(key)
        a = away_s.get(key)
        if h is None and a is None:
            continue
        h_f = float(h) if h is not None else None
        a_f = float(a) if a is not None else None
        diff = abs((h_f or 0) - (a_f or 0))
        rows.append({
            "name":         label,
            "home":         fmt_stat(key, h_f),
            "away":         fmt_stat(key, a_f),
            "rawHome":      round(h_f, 4) if h_f is not None else None,
            "rawAway":      round(a_f, 4) if a_f is not None else None,
            "higherBetter": higher_better,
            "_diff":        diff,
        })

    rows.sort(key=lambda r: r["_diff"], reverse=True)
    for r in rows:
        del r["_diff"]
    top = rows[:TOP_N_STATS]

    if elo:
        h_elo = elo.get(home_abbr)
        a_elo = elo.get(away_abbr)
        if h_elo and a_elo:
            top.insert(0, {
                "name":         "Elo rating",
                "home":         str(round(h_elo)),
                "away":         str(round(a_elo)),
                "rawHome":      round(h_elo, 1),
                "rawAway":      round(a_elo, 1),
                "higherBetter": True,
            })
            top = top[:TOP_N_STATS]

    return top


def build_game(row: dict,
               team_log_cache: dict | None = None,
               abbr_to_id: dict | None = None,
               injured_full: dict | None = None,
               player_pis_df=None,
               elo: dict | None = None) -> dict:
    game_str  = row.get("Game", "")
    home_pct  = float(row.get("Home%", 50))
    away_pct  = float(row.get("Away%", 50))
    pick      = row.get("Pick", "").strip()

    parts     = game_str.split("@")
    away_abbr = parts[0].strip() if len(parts) == 2 else "AWAY"
    home_abbr = parts[1].strip() if len(parts) == 2 else "HOME"

    winner_is = "home" if home_pct >= away_pct else "away"

    if injured_full is not None:
        home_out = mark_stars(home_abbr, injured_full.get(home_abbr, []), player_pis_df)
        away_out = mark_stars(away_abbr, injured_full.get(away_abbr, []), player_pis_df)
    else:
        home_out = parse_injured_str(row.get("Home_Out", ""))
        away_out = parse_injured_str(row.get("Away_Out", ""))

    stats = []
    if team_log_cache and abbr_to_id:
        stats = build_stats(home_abbr, away_abbr, team_log_cache, abbr_to_id, elo)

    return {
        "matchup":    game_str,
        "home_abbr":  home_abbr,
        "away_abbr":  away_abbr,
        "home_pct":   round(home_pct, 1),
        "away_pct":   round(away_pct, 1),
        "pick":       pick,
        "winner_is":  winner_is,
        "confidence": confidence_tier(max(home_pct, away_pct)),
        "home_out":   home_out,
        "away_out":   away_out,
        "stats":      stats,
    }


# ── Main export ───────────────────────────────────────────────────────────────

def export_json(
    predictions: list | None = None,
    csv_path: str | None = None,
    out_path: str = "predictions.json",
    team_log_cache: dict | None = None,
    injured: dict | None = None,
    player_pis_df=None,
    elo: dict | None = None,
) -> None:
    today = date.today().isoformat()

    abbr_to_id = None
    if team_log_cache:
        try:
            from nba_api.stats.static import teams as nba_teams_static
            abbr_to_id = {t["abbreviation"]: t["id"] for t in nba_teams_static.get_teams()}
        except ImportError:
            pass

    if predictions is not None:
        games = [
            build_game(p,
                       team_log_cache=team_log_cache,
                       abbr_to_id=abbr_to_id,
                       injured_full=injured,
                       player_pis_df=player_pis_df,
                       elo=elo)
            for p in predictions
        ]
    else:
        if csv_path is None:
            csv_path = f"predictions_{today}.csv"
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Could not find {csv_path}. "
                "Run predict_tonight.py first, or pass csv_path explicitly."
            )
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        games = [build_game(r) for r in rows]

    payload = {
        "date":       today,
        "generated":  datetime.now().isoformat(timespec="seconds"),
        "game_count": len(games),
        "strong":     sum(1 for g in games if g["confidence"] == "strong"),
        "games":      games,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    has_stats   = any(g.get("stats") for g in games)
    all_players = [
        p for g in games
        for side in (g.get("home_out", []), g.get("away_out", []))
        for p in side
    ]
    has_pis     = any(p.get("pis") is not None for p in all_players)
    missing_pis = [p["name"] for p in all_players if p.get("pis") iexs None]

    print(f"✓ Exported {len(games)} games → {out_path}  ({today})")
    print(f"  Rolling stats : {'included' if has_stats else 'not available — pass team_log_cache'}")
    print(f"  Player PIS    : {'included' if has_pis else 'not found — check player_impact_scores.csv'}")
    if missing_pis:
        print(f"  PIS missing   : {', '.join(missing_pis)}")
        print(f"                  (not in player_impact_scores.csv — likely rookies / two-ways)")


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to predictions CSV")
    parser.add_argument("--out", default="predictions.json", help="Output JSON path")
    args = parser.parse_args()
    export_json(csv_path=args.csv, out_path=args.out)
