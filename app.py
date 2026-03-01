import os
from flask import Flask, render_template, request, session
from dotenv import load_dotenv

# Load .env file into environment variables before other imports
load_dotenv()

from recommender import get_recommendations

app = Flask(__name__)

# Required for Flask sessions — set a strong random secret in your .env
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")

# Max seen IDs to keep in session (prevents the AI prompt from getting huge)
MAX_SEEN = 50


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results():
    # ── Parse & normalize form answers ──────────────────────────────────────
    answers = {
        "artist":    request.form.get("artist", "").strip().title(),
        "genre":     request.form.get("genre", ""),
        "era":       request.form.get("era", ""),
        "tones":     request.form.getlist("tone"),   # list of checked moods
        "listeners": request.form.get("listeners", ""),
    }

    # ── Session memory: load seen track IDs ─────────────────────────────────
    seen_ids = session.get("seen_track_ids", [])

    # ── Get recommendations ──────────────────────────────────────────────────
    try:
        songs, new_ids = get_recommendations(answers, exclude_ids=seen_ids)
    except Exception as e:
        return render_template("results.html", songs=[], answers=answers, error=str(e))

    # ── Update session memory (trim if too long) ─────────────────────────────
    seen_ids = (seen_ids + new_ids)[-MAX_SEEN:]
    session["seen_track_ids"] = seen_ids

    return render_template("results.html", songs=songs, answers=answers)


@app.route("/reset", methods=["POST"])
def reset():
    """Clear the session so the user gets fresh recommendations."""
    session.clear()
    return ("", 204)


if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True)