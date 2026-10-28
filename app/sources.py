"""Resolve links (Spotify, YouTube, SoundCloud, ...) to a local audio file.

Spotify does not serve full audio, so for Spotify track links we:
1. read title/artist (and the 30 s preview URL) from the public embed page,
   or from the Web API if SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are set
2. search the track on YouTube via yt-dlp and download the full audio
3. fall back to the 30 s Spotify preview if YouTube fails
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

SPOTIFY_TRACK_RE = re.compile(
    r"(?:open\.spotify\.com/(?:intl-[a-z]{2}(?:-[a-z]{2})?/)?(?:embed/)?track/|spotify:track:)([A-Za-z0-9]{22})"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
MAX_DOWNLOAD_SECONDS = 15 * 60


class SourceError(Exception):
    pass


@dataclass
class TrackInfo:
    title: str | None = None
    artists: list[str] = field(default_factory=list)
    duration_ms: int | None = None
    preview_url: str | None = None
    cover_url: str | None = None

    @property
    def query(self) -> str:
        return f"{', '.join(self.artists)} - {self.title}" if self.artists else (self.title or "")


@dataclass
class ResolvedAudio:
    path: Path
    source: str  # "upload" | "youtube" | "spotify-preview" | "link"
    title: str | None = None
    detail: str | None = None
    cover_url: str | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- spotify


def spotify_track_id(url: str) -> str | None:
    m = SPOTIFY_TRACK_RE.search(url)
    return m.group(1) if m else None


def _find_entity(obj):
    """Find the dict describing the track inside the embed page JSON."""
    if isinstance(obj, dict):
        if ("name" in obj or "title" in obj) and ("artists" in obj or "audioPreview" in obj):
            return obj
        for v in obj.values():
            found = _find_entity(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_entity(v)
            if found:
                return found
    return None


def _spotify_api(track_id: str, client: httpx.Client) -> TrackInfo | None:
    cid, secret = os.getenv("SPOTIFY_CLIENT_ID"), os.getenv("SPOTIFY_CLIENT_SECRET")
    if not (cid and secret):
        return None
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    tok = client.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {auth}"},
    )
    tok.raise_for_status()
    r = client.get(
        f"https://api.spotify.com/v1/tracks/{track_id}",
        headers={"Authorization": f"Bearer {tok.json()['access_token']}"},
    )
    r.raise_for_status()
    t = r.json()
    images = t.get("album", {}).get("images") or []
    return TrackInfo(
        title=t.get("name"),
        artists=[a["name"] for a in t.get("artists", []) if a.get("name")],
        duration_ms=t.get("duration_ms"),
        preview_url=t.get("preview_url"),
        cover_url=images[0]["url"] if images else None,
    )


def _spotify_embed(track_id: str, client: httpx.Client) -> TrackInfo | None:
    r = client.get(f"https://open.spotify.com/embed/track/{track_id}")
    r.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', r.text, re.S)
    if not m:
        return None
    entity = _find_entity(json.loads(m.group(1)))
    if not entity:
        return None
    artists = [a.get("name") for a in entity.get("artists") or [] if isinstance(a, dict) and a.get("name")]
    if not artists and entity.get("subtitle"):
        artists = [s.strip() for s in str(entity["subtitle"]).split(",")]
    images = (entity.get("visualIdentity") or {}).get("image") or (entity.get("coverArt") or {}).get("sources") or []
    return TrackInfo(
        title=entity.get("name") or entity.get("title"),
        artists=artists,
        duration_ms=entity.get("duration"),
        preview_url=(entity.get("audioPreview") or {}).get("url"),
        cover_url=max(images, key=lambda i: i.get("maxWidth") or i.get("width") or 0).get("url") if images else None,
    )


def _spotify_oembed(url: str, client: httpx.Client) -> TrackInfo | None:
    r = client.get("https://open.spotify.com/oembed", params={"url": url})
    r.raise_for_status()
    d = r.json()
    return TrackInfo(title=html.unescape(d.get("title") or "") or None, cover_url=d.get("thumbnail_url"))


def spotify_track_info(url: str) -> TrackInfo:
    track_id = spotify_track_id(url)
    if not track_id:
        raise SourceError("Nur Spotify-Track-Links werden unterstützt (open.spotify.com/track/…)")
    errors = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True) as client:
        for fn, arg in ((_spotify_api, track_id), (_spotify_embed, track_id), (_spotify_oembed, url)):
            try:
                info = fn(arg, client)
            except Exception as e:  # noqa: BLE001 - try the next strategy
                errors.append(f"{fn.__name__}: {e}")
                continue
            if info and info.title:
                return info
    raise SourceError("Spotify-Track-Infos konnten nicht geladen werden. " + "; ".join(errors))


# --------------------------------------------------------------------------- downloads


def _ytdlp_download(target: str, workdir: Path, match_duration_ms: int | None = None) -> tuple[Path, dict]:
    import yt_dlp

    def match_filter(info, *, incomplete):
        dur = info.get("duration")
        if dur and dur > MAX_DOWNLOAD_SECONDS:
            return "Video ist zu lang"
        if match_duration_ms and dur and abs(dur - match_duration_ms / 1000) > 30:
            return "Länge passt nicht zum Spotify-Track"
        return None

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "dl.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "match_filter": match_filter,
        "max_downloads": 1,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target, download=True)
    except yt_dlp.utils.MaxDownloadsReached:
        info = None
    except Exception as e:  # noqa: BLE001
        raise SourceError(f"Download fehlgeschlagen: {e}") from e
    if info and info.get("entries") is not None:
        entries = [e for e in info["entries"] if e]
        info = entries[0] if entries else None
    files = [p for p in workdir.glob("dl.*") if not p.name.endswith(".part")]
    if not files:
        raise SourceError("Kein passendes Audio gefunden")
    return files[0], info or {}


def _download_url(url: str, workdir: Path) -> Path:
    path = workdir / "preview.mp3"
    with httpx.stream("GET", url, headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return path


def resolve_link(url: str, workdir: Path, prefer: str = "auto") -> ResolvedAudio:
    """prefer: 'auto' (YouTube, then preview), 'youtube' or 'preview'."""
    url = url.strip()
    if not re.match(r"^(https?://|spotify:)", url):
        raise SourceError("Bitte einen gültigen Link eingeben")

    if not spotify_track_id(url):
        if "spotify.com" in url:
            raise SourceError("Nur einzelne Spotify-Tracks werden unterstützt (keine Alben/Playlists)")
        path, info = _ytdlp_download(url, workdir)
        return ResolvedAudio(path, "link", title=info.get("title"), detail=info.get("webpage_url") or url,
                             cover_url=info.get("thumbnail"))

    track = spotify_track_info(url)
    notes: list[str] = []

    if prefer in ("auto", "youtube"):
        query = f"ytsearch5:{track.query} audio"
        try:
            path, info = _ytdlp_download(query, workdir, track.duration_ms)
            notes.append("Volle Länge über YouTube-Suche geladen – prüfe, ob das gefundene Video wirklich der Song ist.")
            return ResolvedAudio(path, "youtube", title=track.query,
                                 detail=f"{info.get('title')} ({info.get('webpage_url')})",
                                 cover_url=track.cover_url, notes=notes)
        except SourceError as e:
            if prefer == "youtube" or not track.preview_url:
                raise
            notes.append(f"YouTube nicht verfügbar ({e}); nutze Spotify-Preview.")

    if not track.preview_url:
        raise SourceError("Für diesen Track gibt es keine Spotify-Preview")
    path = _download_url(track.preview_url, workdir)
    notes.append("Nur 30-Sekunden-Preview analysiert – das Ergebnis ist weniger zuverlässig als beim ganzen Song.")
    return ResolvedAudio(path, "spotify-preview", title=track.query, detail="Spotify 30s Preview",
                         cover_url=track.cover_url, notes=notes)
