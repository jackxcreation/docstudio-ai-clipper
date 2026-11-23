from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import urllib.parse
import urllib.request
import wave
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

UA = "TermuxDocStudio/2.0 (offline-first-documentary-engine)"
STOP_WORDS = {"dark", "moody", "cinematic", "aesthetic", "dramatic", "glowing", "4k", "hd", "slow", "motion"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def ensure_starter_sfx(sfx_dir: str) -> Dict[str, str]:
    """Generates clean procedural cinematic SFX (boom, whoosh, pop) if assets/sfx is empty."""
    os.makedirs(sfx_dir, exist_ok=True)
    sr = 44100

    def save_wav(path: str, samples: np.ndarray) -> None:
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            return
        clipped = np.clip(samples, -0.95, 0.95)
        pcm = (clipped * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    # 1. Sub-bass cinematic boom (hook impact)
    t_b = np.linspace(0.0, 0.85, int(sr * 0.85), endpoint=False)
    freq_b = 105.0 * np.exp(-3.8 * t_b) + 34.0
    phase_b = 2.0 * math.pi * np.cumsum(freq_b) / sr
    env_b = np.minimum(1.0, t_b / 0.015) * np.exp(-4.2 * t_b)
    boom = 0.75 * np.sin(phase_b) * env_b
    save_wav(os.path.join(sfx_dir, "boom.wav"), boom)

    # 2. Soft air whoosh (B-roll transitions)
    rng = np.random.default_rng(42)
    t_w = np.linspace(0.0, 0.45, int(sr * 0.45), endpoint=False)
    noise = rng.normal(0.0, 1.0, len(t_w))
    # Smooth low-pass moving average for airy cinematic swoosh
    kernel = np.ones(28, dtype=np.float64) / 28.0
    soft_noise = np.convolve(noise, kernel, mode="same")
    env_w = np.sin(math.pi * (t_w / 0.45)) ** 1.8
    whoosh = 0.55 * (soft_noise / (np.max(np.abs(soft_noise)) + 1e-6)) * env_w
    save_wav(os.path.join(sfx_dir, "whoosh.wav"), whoosh)

    # 3. Crisp UI pop (stats / quotes / map pins)
    t_p = np.linspace(0.0, 0.14, int(sr * 0.14), endpoint=False)
    freq_p = 520.0 + 380.0 * np.exp(-24.0 * t_p)
    phase_p = 2.0 * math.pi * np.cumsum(freq_p) / sr
    env_p = np.minimum(1.0, t_p / 0.005) * np.exp(-26.0 * t_p)
    pop = 0.45 * np.sin(phase_p) * env_p
    save_wav(os.path.join(sfx_dir, "pop.wav"), pop)

    return {
        "boom": os.path.join(sfx_dir, "boom.wav"),
        "whoosh": os.path.join(sfx_dir, "whoosh.wav"),
        "pop": os.path.join(sfx_dir, "pop.wav"),
    }


def _slug(q: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:30]
    h = hashlib.md5(q.lower().strip().encode("utf-8")).hexdigest()[:6]
    return f"{clean or 'broll'}_{h}"


def _clean_query(q: str) -> str:
    words = [w for w in re.sub(r"[^a-z0-9\s]", " ", q.lower()).split() if len(w) >= 3 and w not in STOP_WORDS]
    return " ".join(words[:3]) or q


def _http_json(url: str, timeout: int = 18) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def _download_valid_image(url: str, dest: str, timeout: int = 25) -> bool:
    tmp = dest + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 6000:
            return False
        with open(tmp, "wb") as f:
            f.write(data)
        with Image.open(tmp) as im:
            im.verify()
        os.replace(tmp, dest)
        return True
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def _search_openverse(query: str) -> List[Tuple[str, str]]:
    q = urllib.parse.quote(_clean_query(query))
    url = f"https://api.openverse.org/v1/images/?q={q}&page_size=6&mature=false"
    data = _http_json(url)
    out: List[Tuple[str, str]] = []
    for it in data.get("results", []):
        u = it.get("url") or it.get("thumbnail")
        if not u or re.search(r"\.(svg|gif)(\?|$)", u, re.I):
            continue
        if (it.get("width") or 800) < 500 or (it.get("height") or 500) < 400:
            continue
        ext = ".png" if ".png" in u.lower() else ".jpg"
        out.append((u, ext))
    return out


def _search_wikimedia(query: str) -> List[Tuple[str, str]]:
    q = urllib.parse.quote(_clean_query(query))
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6"
        f"&gsrsearch={q}&gsrlimit=6&prop=imageinfo&iiprop=url|mime|size&iiurlwidth=1280&format=json"
    )
    data = _http_json(url)
    out: List[Tuple[str, str]] = []
    for pg in (data.get("query", {}).get("pages", {}) or {}).values():
        info = (pg.get("imageinfo") or [{}])[0]
        mime = info.get("mime", "")
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            continue
        if (info.get("width") or 0) < 500 or (info.get("height") or 0) < 400:
            continue
        dl = info.get("thumburl") or info.get("url")
        if dl:
            out.append((dl, ".png" if mime == "image/png" else ".jpg"))
    return out


def _find_local_broll(query: str, local_dir: str, used: set[str]) -> str | None:
    if not local_dir or not os.path.isdir(local_dir):
        return None
    q_toks = set(_clean_query(query).split())
    best, best_score = None, 0
    for root, _, files in os.walk(local_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            fp = os.path.join(root, fn)
            if fp in used:
                continue
            f_toks = set(re.findall(r"[a-z0-9]+", os.path.splitext(fn)[0].lower()))
            score = len(q_toks & f_toks)
            if score > best_score:
                best_score, best = score, fp
    return best


def fetch_broll_for_plan(plan: Dict[str, Any], assets_dir: str, enable_online: bool = True) -> Dict[str, Any]:
    """Resolves every B-roll query using: Local folder -> Cache -> Zero-Key Openverse/Wikimedia."""
    local_dir = os.path.join(assets_dir, "broll")
    cache_dir = os.path.join(assets_dir, "broll_cache")
    os.makedirs(cache_dir, exist_ok=True)

    tracks = plan.get("broll_tracks", [])
    if not tracks:
        return plan

    print(f"🎞️  Resolving {len(tracks)} B-roll visuals (zero API key)...", flush=True)
    used_paths: set[str] = set()

    for tr in tracks:
        q = tr.get("query", "").strip()
        if not q:
            continue

        # 1. Local custom B-roll folder match
        local_hit = _find_local_broll(q, local_dir, used_paths)
        if local_hit:
            tr["asset_path"], tr["asset_type"] = local_hit, "image"
            used_paths.add(local_hit)
            print(f"   📁 {tr['id']} [local] {os.path.basename(local_hit)}", flush=True)
            continue

        # 2. Local cache check
        sl = _slug(q)
        cached = next((os.path.join(cache_dir, sl + ext) for ext in (".jpg", ".png") if os.path.exists(os.path.join(cache_dir, sl + ext))), None)
        if cached:
            tr["asset_path"], tr["asset_type"] = cached, "image"
            print(f"   ♻️  {tr['id']} [cached] \"{q}\"", flush=True)
            continue

        if not enable_online:
            print(f"   ➖ {tr['id']} offline fallback (procedural cinema bg)", flush=True)
            continue

        # 3. Free zero-key online fetch (Openverse -> Wikimedia)
        found = False
        for src_name, searcher in (("openverse", _search_openverse), ("wikimedia", _search_wikimedia)):
            try:
                candidates = searcher(q)
            except Exception:
                candidates = []
            for dl_url, ext in candidates[:3]:
                dest = os.path.join(cache_dir, sl + ext)
                if _download_valid_image(dl_url, dest):
                    tr["asset_path"], tr["asset_type"] = dest, "image"
                    print(f"   ✅ {tr['id']} [{src_name}] \"{q}\"", flush=True)
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"   ➖ {tr['id']} no image match for \"{q}\" (cinematic bg fallback)", flush=True)

    return plan


def _parse_srt_time(ts: str) -> float:
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000.0


def parse_srt_to_words(srt_path: str, clip_start: float = 0.0, clip_end: float | None = None) -> Dict[str, Any]:
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    blocks = re.split(r"\r?\n\r?\n+", raw.strip())
    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    seen_txt = ""

    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        t_line = next((ln for ln in lines if "-->" in ln), None)
        if not t_line:
            continue
        p0, p1 = t_line.split("-->", 1)
        st, et = _parse_srt_time(p0), _parse_srt_time(p1)
        if et <= clip_start:
            continue
        if clip_end is not None and st >= clip_end:
            continue
        rel_st = max(0.0, st - clip_start)
        rel_et = (min(clip_end, et) - clip_start) if clip_end is not None else (et - clip_start)
        if rel_et - rel_st < 0.12:
            continue

        txt = " ".join(lines[lines.index(t_line) + 1:])
        txt = re.sub(r"<[^>]+>", "", txt).strip()
        if not txt or txt == seen_txt:
            continue
        seen_txt = txt

        segments.append({"start": round(rel_st, 3), "end": round(rel_et, 3), "text": txt})
        toks = txt.split()
        step = (rel_et - rel_st) / max(1, len(toks))
        for i, tok in enumerate(toks):
            words.append({"word": tok, "start": round(rel_st + i * step, 3), "end": round(rel_st + (i + 1) * step, 3)})

    dur = (clip_end - clip_start) if (clip_end is not None and clip_end > clip_start) else (segments[-1]["end"] if segments else 0.0)
    return {
        "duration": round(dur, 3),
        "word_count": len(words),
        "full_text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "words": words,
    }


def download_youtube_clip(url: str, out_mp4: str, temp_dir: str, start_sec: float | None = None, end_sec: float | None = None) -> Tuple[str, Dict[str, Any] | None]:
    """Downloads a YouTube video/clip + auto-captions using yt-dlp (zero API key)."""
    os.makedirs(temp_dir, exist_ok=True)
    print(f"⬇️ Downloading YouTube clip via yt-dlp...", flush=True)

    cmd = [
        "yt-dlp", "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format", "mp4", "--no-playlist", "-o", out_mp4,
    ]
    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        cmd.extend(["--download-sections", f"*{start_sec}-{end_sec}", "--force-keyframes-at-cuts"])
    cmd.append(url)

    p = subprocess.run(cmd)
    if p.returncode != 0 or not os.path.exists(out_mp4):
        raise RuntimeError("yt-dlp video download failed. Check URL or internet connection.")

    # Pull auto-captions (EN/HI) for zero-API-key word timing & documentary understanding
    sub_base = os.path.join(temp_dir, "yt_sub")
    for old in os.listdir(temp_dir):
        if old.startswith("yt_sub"):
            try:
                os.remove(os.path.join(temp_dir, old))
            except OSError:
                pass

    sub_cmd = [
        "yt-dlp", "--skip-download", "--no-playlist",
        "--write-subs", "--write-auto-subs",
        "--sub-langs", "en,en-orig,hi", "--convert-subs", "srt",
        "-o", sub_base, url,
    ]
    subprocess.run(sub_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    srt_file = next((os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir)) if f.startswith("yt_sub") and f.endswith(".srt")), None)
    words_data = None
    if srt_file:
        words_data = parse_srt_to_words(srt_file, clip_start=start_sec or 0.0, clip_end=end_sec)
        print(f"📝 YouTube captions extracted: {words_data['word_count']} words", flush=True)
    return out_mp4, words_data


def download_bgm_from_youtube(url: str, bgm_dir: str) -> str:
    """Downloads a copyright-free BGM track directly into ~/doc-studio/assets/bgm/."""
    os.makedirs(bgm_dir, exist_ok=True)
    out_tpl = os.path.join(bgm_dir, "%(title).40s.%(ext)s")
    cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "192K", "--no-playlist", "-o", out_tpl, url]
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError("Failed to download BGM audio via yt-dlp.")
    print(f"✅ BGM saved into {bgm_dir}", flush=True)
    return bgm_dir
