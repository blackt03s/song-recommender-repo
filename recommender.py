import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

# ── Credentials ──────────────────────────────────────────────────────────────
GITHUB_TOKEN          = os.environ.get("GITHUB_TOKEN")
SPOTIFY_CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")

if not all([GITHUB_TOKEN, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET]):
    raise EnvironmentError(
        "Missing credentials. Make sure GITHUB_TOKEN, SPOTIFY_CLIENT_ID, "
        "and SPOTIFY_CLIENT_SECRET are set in your .env file."
    )

# ── GitHub AI client (Azure inference endpoint) ──────────────────────────────
ai_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

# ─────────────────────────────────────────────────────────────────────────────
# SPOTIFY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_spotify_token():
    """Fetch a Spotify client-credentials access token."""
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_spotify_track(token: str, title: str, artist: str) -> dict | None:
    """
    Search Spotify for a specific track and return enriched metadata.
    Returns None if nothing is found.
    """
    query = f"track:{title} artist:{artist}"
    resp = requests.get(
        "https://api.spotify.com/v1/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": query, "type": "track", "limit": 1},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        return None

    t = items[0]
    album = t["album"]
    # some fields are optional depending on the track, so use .get with sensible defaults
    return {
        "spotify_id":   t.get("id"),
        "title":        t.get("name"),
        "artist":       ", ".join(a.get("name", "") for a in t.get("artists", [])),
        "album":        album.get("name"),
        "art":          album.get("images", [])[0].get("url") if album.get("images") else None,
        "release_date": album.get("release_date"),
        # popularity may be missing for some reason; default to 0 so sorting/filters won't crash
        "popularity":   t.get("popularity", 0),          # 0-100
        "preview_url":  t.get("preview_url"),         # 30-sec MP3 or None
        "spotify_url":  t.get("external_urls", {}).get("spotify"),
        "type":         album.get("album_type"),      # "album" | "single" | "compilation"
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(answers: dict, exclude_ids: list[str]) -> str:
    tones = ", ".join(answers.get("tones") or ["any"])
    exclude_note = (
        f"\nDo NOT recommend songs with these Spotify IDs (already seen): {exclude_ids}"
        if exclude_ids else ""
    )
    return f"""You are a music recommendation expert.

A user has filled out a music preference survey with these answers:
- Favorite artist (for similar taste): {answers.get('artist', 'N/A')}
- Preferred genre: {answers.get('genre', 'N/A')}
- Era preference: {answers.get('era', 'N/A')}
- Mood/tone: {tones}
- Artist popularity range: {answers.get('listeners', 'N/A')}
{exclude_note}

Recommend exactly 10 songs that match this profile. Be diverse — vary artists and albums.
Respond ONLY with a valid JSON array, no extra text, no markdown. Example format:
[
  {{"title": "Song Name", "artist": "Artist Name"}},
  ...
]"""


def ask_ai_for_songs(answers: dict, exclude_ids: list[str]) -> list[dict]:
    """
    Call the GitHub-hosted GPT model and return a list of
    {{"title": ..., "artist": ...}} dicts.
    """
    prompt = _build_prompt(answers, exclude_ids)

    response = ai_client.chat.completions.create(
        model="gpt-4o",          # GitHub Models name for GPT-4o
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a music recommendation engine. "
                    "Always respond with raw JSON only — no explanation, no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,         # higher = more variety between calls
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences if the model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT  (called from app.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendations(answers: dict, exclude_ids: list[str] | None = None) -> list[dict]:
    """
    Full pipeline:
      1. Ask the AI for song suggestions (excluding already-seen IDs)
      2. Look each suggestion up on Spotify
      3. Return enriched song dicts + the new Spotify IDs (for session storage)

    Returns:
        (songs: list[dict], new_ids: list[str])
    """
    if exclude_ids is None:
        exclude_ids = []

    # 1 — Get AI suggestions
    suggestions = ask_ai_for_songs(answers, exclude_ids)

    # 2 — Enrich with Spotify data
    spotify_token = get_spotify_token()
    songs = []
    new_ids = []

    for suggestion in suggestions:
        track = search_spotify_track(
            spotify_token,
            suggestion.get("title", ""),
            suggestion.get("artist", ""),
        )
        if track and track["spotify_id"] not in exclude_ids:
            songs.append(track)
            new_ids.append(track["spotify_id"])

    return songs, new_ids