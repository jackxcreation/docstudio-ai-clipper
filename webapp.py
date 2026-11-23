#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from clip import (
    download_yt_section,
    fetch_yt_captions_only,
    load_config,
    parse_clean_srt,
    slice_words_window,
)
from engines.doc_render import DocumentaryRenderer
from engines.dubber import dub_clip_to_language
from engines.local_ai import build_local_edit_plan, extract_high_value_clips
from engines.media_fetcher import ensure_starter_sfx
from engines.vision_audio import analyze_audio_local, analyze_video_local

OUT_DIR = os.path.join(ROOT_DIR, "output")
TEMP_DIR = os.path.join(ROOT_DIR, "temp")
META_FILE = os.path.join(ROOT_DIR, "library_meta.json")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

JOB_LOCK = threading.Lock()
JOB: Dict[str, Any] = {
    "running": False,
    "cancel_requested": False,
    "progress": 0.0,
    "status": "Idle",
    "video_title": "",
    "video_thumb": "",
    "video_channel": "",
    "video_duration": 0.0,
    "total_clips": 0,
    "done_clips": 0,
    "clips": [],
    "error": None,
}


def load_library_meta() -> Dict[str, Any]:
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_clip_meta(filename: str, info: Dict[str, Any]) -> None:
    meta = load_library_meta()
    meta[filename] = info
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def fmt_time(sec: float) -> str:
    s = max(0, int(round(sec)))
    h, m, r = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{r:02d}" if h > 0 else f"{m:02d}:{r:02d}"


def recommend_clip_count(duration_sec: float) -> int:
    if duration_sec <= 150:
        return 1
    if duration_sec <= 360:
        return 2
    if duration_sec <= 720:
        return 4
    if duration_sec <= 1500:
        return 6
    if duration_sec <= 2700:
        return 9
    return 12


def fetch_video_preview(url: str) -> Dict[str, Any]:
    cmd = [
        "yt-dlp", "--no-playlist", "--skip-download",
        "--print", "%(title)s|||%(duration)s|||%(thumbnail)s|||%(uploader)s",
        url,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(p.stderr.strip()[:200] or "YouTube वीडियो जानकारी नहीं मिली")
    line = p.stdout.strip().splitlines()[-1]
    parts = line.split("|||")
    title = parts[0] if len(parts) > 0 else "YouTube Video"
    dur = float(parts[1]) if (len(parts) > 1 and parts[1].replace(".", "", 1).isdigit()) else 0.0
    thumb = parts[2] if len(parts) > 2 else ""
    uploader = parts[3] if len(parts) > 3 else "YouTube"
    rec_clips = recommend_clip_count(dur)
    return {
        "title": title,
        "duration": dur,
        "duration_str": fmt_time(dur),
        "thumbnail": thumb,
        "channel": uploader,
        "recommended_clips": rec_clips,
    }


def resolve_dimensions(aspect: str, hd: bool) -> tuple[int, int]:
    if aspect == "16:9":
        return (1920, 1080) if hd else (1280, 720)
    if aspect == "1:1":
        return (1080, 1080) if hd else (720, 720)
    return (1080, 1920) if hd else (720, 1280)


def render_advanced_short(
    clip_mp4: str,
    words_data: Dict[str, Any],
    out_mp4: str,
    opts: Dict[str, Any],
    cfg: dict,
    clip_idx: int,
) -> None:
    assets_dir = os.path.join(ROOT_DIR, "assets")
    aspect = str(opts.get("aspect", "9:16"))
    hd = bool(opts.get("hd", False))
    out_w, out_h = resolve_dimensions(aspect, hd)
    fps = 24

    audio_info = analyze_audio_local(clip_mp4)
    vision_info = analyze_video_local(clip_mp4, audio_info)
    actual_dur = float(vision_info["meta"]["duration"])
    words_data["duration"] = actual_dur

    dub_lang = str(opts.get("dub_lang", "orig"))
    dub_voice = str(opts.get("dub_voice", "male"))
    dubbed_audio_path = None
    if dub_lang and dub_lang != "orig" and words_data.get("segments"):
        dubbed_audio_path, words_data = dub_clip_to_language(
            words_data=words_data,
            clip_duration=actual_dur,
            target_lang=dub_lang,
            voice_gender=dub_voice,
            temp_dir=TEMP_DIR,
            tag=f"dub_{clip_idx}",
        )

    plan = build_local_edit_plan(words_data, vision_info, audio_info, mode="hybrid")
    plan["broll_tracks"] = []

    if not bool(opts.get("hook_on", True)):
        plan["overlays"] = [o for o in plan["overlays"] if o.get("type") != "hook_card"]
    if not bool(opts.get("overlays_on", True)):
        plan["overlays"] = [o for o in plan["overlays"] if o.get("type") == "hook_card"]
    if not bool(opts.get("zooms_on", True)):
        plan["punch_zooms"] = []

    render_words = words_data if bool(opts.get("captions_on", True)) else {**words_data, "words": []}
    color_grade = str(opts.get("color_grade", "podcast_clean"))
    bgm_vol = float(opts.get("bgm_vol", cfg.get("bgm_volume", 0.10)))
    sfx_vol = float(cfg.get("sfx_volume", 0.22)) if bool(opts.get("overlays_on", True)) else 0.0

    renderer = DocumentaryRenderer(out_w, out_h, fps=fps, color_grade=color_grade)
    renderer.render(
        output_path=out_mp4,
        plan=plan,
        words_data=render_words,
        assets_dir=assets_dir,
        audio_input_path=dubbed_audio_path,
        source_video_path=clip_mp4,
        vision_info=vision_info,
        bgm_vol=bgm_vol,
        sfx_vol=sfx_vol,
    )


def run_batch_job(url: str, opts: Dict[str, Any]) -> None:
    cfg = load_config()
    ensure_starter_sfx(os.path.join(ROOT_DIR, "assets", "sfx"))

    length_mode = str(opts.get("length_mode", "auto"))
    clip_limit = str(opts.get("clip_limit", "auto"))
    dub_lang = str(opts.get("dub_lang", "orig"))

    if length_mode == "short":
        min_d, max_d = 30.0, 60.0
    elif length_mode == "quick":
        min_d, max_d = 20.0, 38.0
    elif length_mode == "long":
        min_d, max_d = 60.0, 110.0
    else:
        min_d, max_d = 35.0, 105.0

    try:
        with JOB_LOCK:
            JOB["progress"] = 5.0
            JOB["status"] = "YouTube वीडियो जानकारी लोड हो रही है..."

        info = fetch_video_preview(url)
        with JOB_LOCK:
            JOB["video_title"] = info["title"]
            JOB["video_thumb"] = info["thumbnail"]
            JOB["video_channel"] = info["channel"]
            JOB["video_duration"] = info["duration"]
            JOB["progress"] = 12.0
            JOB["status"] = "Auto-Captions स्कैन करके कम्प्लीट टॉपिक्स खोजे जा रहे हैं..."

        srt_path, probed_dur = fetch_yt_captions_only(url, TEMP_DIR)
        total_dur = info["duration"] or probed_dur or 60.0
        all_segs = parse_clean_srt(srt_path) if srt_path else []

        if clip_limit == "auto":
            target_count = recommend_clip_count(total_dur)
        else:
            target_count = max(1, min(15, int(clip_limit)))

        targets = extract_high_value_clips(all_segs, total_dur, min_d, max_d, target_count)
        if not targets:
            raise RuntimeError("इस वीडियो से कोई क्लिप नहीं निकल सका।")

        with JOB_LOCK:
            JOB["total_clips"] = len(targets)
            JOB["progress"] = 18.0
            JOB["status"] = f"{len(targets)} हाई-वैल्यू क्लिप्स मिले! ऑटो-एडिटिंग शुरू..."

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_pct = 18.0
        per_clip_pct = 82.0 / max(1, len(targets))
        aspect_tag = str(opts.get("aspect", "9:16")).replace(":", "x")

        for idx, t in enumerate(targets, 1):
            with JOB_LOCK:
                if JOB["cancel_requested"]:
                    JOB["status"] = f"रोका गया ({idx - 1}/{len(targets)} क्लिप्स तैयार हैं)"
                    break
                JOB["status"] = f"Clip {idx}/{len(targets)} डाउनलोड हो रहा है ({fmt_time(t['start'])} - {fmt_time(t['end'])})..."
                JOB["progress"] = round(base_pct + (idx - 1) * per_clip_pct + per_clip_pct * 0.15, 1)

            clip_tmp = os.path.join(TEMP_DIR, f"web_clip_{idx}.mp4")
            download_yt_section(url, t["start"], t["end"], clip_tmp)

            dub_note = f" + {dub_lang.upper()} AI Dubbing" if dub_lang != "orig" else ""
            with JOB_LOCK:
                JOB["status"] = f"Clip {idx}/{len(targets)} एडिट हो रहा है ({opts.get('aspect', '9:16')}{dub_note})..."
                JOB["progress"] = round(base_pct + (idx - 1) * per_clip_pct + per_clip_pct * 0.45, 1)

            words_data = slice_words_window(all_segs, t["start"], t["end"]) if all_segs else {
                "duration": t["end"] - t["start"], "full_text": "", "segments": [], "words": []
            }
            out_name = f"clip_{idx:02d}_{aspect_tag}_{t['title']}_{stamp}.mp4"
            out_mp4 = os.path.join(OUT_DIR, out_name)

            render_advanced_short(clip_tmp, words_data, out_mp4, opts, cfg, idx)

            clip_card = {
                "filename": out_name,
                "url": f"/video/{out_name}",
                "title": t["hook"] or f"Viral Clip #{idx}",
                "score": t["score"],
                "range": f"{fmt_time(t['start'])} - {fmt_time(t['end'])}",
                "duration": round(t["end"] - t["start"], 1),
                "aspect": opts.get("aspect", "9:16"),
                "dub": dub_lang.upper() if dub_lang != "orig" else "ORIG",
                "source_title": info["title"],
                "created_at": datetime.now().strftime("%d %b, %H:%M"),
            }
            save_clip_meta(out_name, clip_card)

            with JOB_LOCK:
                JOB["done_clips"] = idx
                JOB["clips"].append(clip_card)
                JOB["progress"] = round(min(100.0, base_pct + idx * per_clip_pct), 1)
                JOB["status"] = f"Clip {idx}/{len(targets)} तैयार होकर Gallery में सेव हो गया!"

        with JOB_LOCK:
            if not JOB["cancel_requested"]:
                JOB["progress"] = 100.0
                JOB["status"] = f"पूरा हो गया! {JOB['done_clips']} क्लिप्स तैयार हैं।"
    except Exception as exc:
        with JOB_LOCK:
            JOB["error"] = str(exc)
            JOB["status"] = f"त्रुटि: {exc}"
    finally:
        with JOB_LOCK:
            JOB["running"] = False


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="hi">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
<title>DocStudio AI — Pro Clipper & Dubbing Studio</title>
<style>
  :root { --bg:#090b10; --card:#131722; --border:#232a3b; --green:#22c55e; --accent:#38bdf8; --text:#f8fafc; --muted:#94a3b8; }
  * { box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  body { margin:0; background:var(--bg); color:var(--text); padding-bottom:84px; }
  .wrap { max-width:540px; margin:0 auto; padding:16px 14px; }
  .top { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
  .badge { background:#14291d; color:var(--green); border:1px solid #1e462d; padding:5px 11px; border-radius:999px; font-weight:700; font-size:12.5px; }
  h1 { text-align:center; font-size:23px; margin:4px 0 4px; }
  .sub { text-align:center; color:var(--muted); font-size:13px; margin:0 0 16px; line-height:1.4; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:13px; margin-bottom:12px; }
  .sec-title { font-size:12px; font-weight:800; letter-spacing:0.6px; color:var(--accent); text-transform:uppercase; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; }
  .input-row { display:flex; gap:8px; }
  input[type=text] { flex:1; background:#0d1018; border:1px solid var(--border); color:#fff; border-radius:12px; padding:11px 12px; font-size:14px; outline:none; }
  input[type=text]:focus { border-color:var(--green); }
  .btn { background:var(--green); color:#052e16; border:none; border-radius:12px; padding:11px 15px; font-weight:800; font-size:14px; cursor:pointer; }
  .btn:disabled { opacity:0.5; cursor:not-allowed; }
  .btn-stop { background:#ef4444; color:#fff; width:100%; margin-top:10px; }
  .preview { display:none; margin-top:11px; border-radius:12px; overflow:hidden; background:#0d1018; border:1px solid var(--border); }
  .preview img { width:100%; height:185px; object-fit:cover; display:block; }
  .preview-meta { padding:9px 11px; }
  .preview-title { font-weight:700; font-size:13.5px; margin-bottom:4px; }
  .preview-sub { font-size:12px; color:var(--muted); display:flex; justify-content:space-between; }
  .rec-pill { display:inline-block; margin-top:6px; background:#14291d; color:var(--green); font-size:11.5px; font-weight:700; padding:3px 9px; border-radius:8px; }
  .ratio-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:6px; }
  .ratio-btn { background:#0d1018; border:1.5px solid var(--border); color:var(--muted); border-radius:12px; padding:9px 6px; text-align:center; cursor:pointer; font-size:12px; font-weight:700; }
  .ratio-btn.active { border-color:var(--green); background:#14291d; color:#fff; }
  .ratio-btn span { display:block; font-size:10.5px; font-weight:500; color:var(--muted); margin-top:2px; }
  .row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1b2130; gap:10px; }
  .row:last-child { border-bottom:none; }
  .row label { color:#cbd5e1; font-size:13px; }
  select { background:#0d1018; color:#fff; border:1px solid var(--border); border-radius:10px; padding:7px 10px; font-size:13px; max-width:205px; }
  .toggle-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-top:6px; }
  .tog { display:flex; justify-content:space-between; align-items:center; background:#0d1018; border:1px solid var(--border); padding:8px 10px; border-radius:11px; font-size:12.5px; color:#e2e8f0; cursor:pointer; }
  .tog input { accent-color:var(--green); width:16px; height:16px; cursor:pointer; }
  .prog-box { display:none; }
  .prog-text { text-align:center; font-size:13px; color:#cbd5e1; margin-bottom:9px; }
  .bar-bg { background:#1d2433; border-radius:999px; height:30px; overflow:hidden; }
  .bar-fill { background:linear-gradient(90deg,#16a34a,#22c55e); height:100%; width:0%; transition:width .4s ease; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12.5px; color:#052e16; min-width:44px; }
  .clip-card { background:var(--card); border:1px solid var(--border); border-radius:16px; overflow:hidden; margin-bottom:13px; }
  .clip-card video { width:100%; max-height:420px; background:#000; display:block; }
  .clip-body { padding:11px; }
  .clip-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; gap:8px; }
  .score { background:#14291d; color:var(--green); font-weight:800; font-size:11.5px; padding:4px 8px; border-radius:8px; white-space:nowrap; }
  .clip-title { font-weight:700; font-size:13.5px; line-height:1.35; }
  .clip-sub { color:var(--muted); font-size:11.5px; margin-top:4px; }
  .clip-actions { display:flex; gap:8px; margin-top:9px; }
  .clip-actions a { flex:1; text-align:center; text-decoration:none; background:#1e293b; color:#fff; padding:9px; border-radius:10px; font-size:13px; font-weight:700; }
  .nav { position:fixed; bottom:0; left:0; right:0; background:#0d1018; border-top:1px solid var(--border); display:flex; justify-content:space-around; padding:9px 0; }
  .nav button { background:none; border:none; color:var(--muted); font-weight:700; font-size:13px; cursor:pointer; padding:6px 18px; border-radius:10px; }
  .nav button.active { color:var(--green); background:#14291d; }
  .err { color:#f87171; font-size:13px; text-align:center; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div style="font-weight:800;font-size:16px;">⚡ DocStudio Pro</div>
    <div class="badge">∞ Free • AI Dub + Clip</div>
  </div>

  <div id="tabHome">
    <h1>AI Video Studio</h1>
    <p class="sub">YouTube लिंक पेस्ट करें — फॉर्मेट चुनें, 1 से 15 क्लिप्स निकालें, या किसी भी भाषा में AI वॉइसओवर डब करें।</p>

    <div class="card">
      <div class="input-row">
        <input id="ytUrl" type="text" placeholder="https://www.youtube.com/watch?v=..." />
        <button class="btn" id="btnPreview" onclick="loadPreview()">Check</button>
      </div>
      <div class="preview" id="previewBox">
        <img id="prevImg" src="" alt="Thumbnail" />
        <div class="preview-meta">
          <div class="preview-title" id="prevTitle"></div>
          <div class="preview-sub"><span id="prevChan"></span><span id="prevDur"></span></div>
          <div class="rec-pill" id="recPill" style="display:none;"></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="sec-title">1. Aspect Ratio / Format</div>
      <div class="ratio-grid">
        <div class="ratio-btn active" data-aspect="9:16" onclick="setAspect('9:16', this)">📱 9:16 Portrait<span>Shorts / Reels</span></div>
        <div class="ratio-btn" data-aspect="16:9" onclick="setAspect('16:9', this)">🖥️ 16:9 Landscape<span>YouTube Wide</span></div>
        <div class="ratio-btn" data-aspect="1:1" onclick="setAspect('1:1', this)">⬜ 1:1 Square<span>Instagram / FB</span></div>
      </div>
    </div>

    <div class="card">
      <div class="sec-title">2. Clips & Length Settings</div>
      <div class="row">
        <label>कितने क्लिप्स (1 - 15)</label>
        <select id="clipCount">
          <option value="auto" id="optAutoClips" selected>Auto (Video Length Detect)</option>
          <option value="1">1 Clip (Top Best)</option>
          <option value="2">2 Clips</option>
          <option value="3">3 Clips</option>
          <option value="4">4 Clips</option>
          <option value="5">5 Clips</option>
          <option value="6">6 Clips</option>
          <option value="7">7 Clips</option>
          <option value="8">8 Clips</option>
          <option value="9">9 Clips</option>
          <option value="10">10 Clips</option>
          <option value="11">11 Clips</option>
          <option value="12">12 Clips</option>
          <option value="13">13 Clips</option>
          <option value="14">14 Clips</option>
          <option value="15">15 Clips</option>
        </select>
      </div>
      <div class="row">
        <label>Clip Length Mode</label>
        <select id="lenMode">
          <option value="auto" selected>Topic Complete (Auto End)</option>
          <option value="short">Standard (30s - 60s)</option>
          <option value="quick">Quick Viral (20s - 38s)</option>
          <option value="long">Deep Story (60s - 110s)</option>
        </select>
      </div>
    </div>

    <div class="card">
      <div class="sec-title">3. AI Translate & Voiceover Dubbing</div>
      <div class="row">
        <label>Audio / Dub Language</label>
        <select id="dubLang" onchange="toggleVoiceRow()">
          <option value="orig" selected>Original Video Audio</option>
          <option value="hi">🇮🇳 Translate -> Hindi (हिंदी)</option>
          <option value="en">🇺🇸 Translate -> English</option>
          <option value="bn">🇮🇳 Translate -> Bengali (বাংলা)</option>
          <option value="ta">🇮🇳 Translate -> Tamil (தமிழ்)</option>
          <option value="te">🇮🇳 Translate -> Telugu (తెలుగు)</option>
          <option value="mr">🇮🇳 Translate -> Marathi (मराठी)</option>
          <option value="es">🇪🇸 Translate -> Spanish</option>
        </select>
      </div>
      <div class="row" id="voiceRow" style="display:none;">
        <label>AI Voiceover Style</label>
        <select id="dubVoice">
          <option value="male" selected>Male Neural Voice (Natural)</option>
          <option value="female">Female Neural Voice (Clear)</option>
          <option value="deep">Deep Documentary Narrator</option>
        </select>
      </div>
    </div>

    <div class="card">
      <div class="sec-title">4. Advanced Edit Controls</div>
      <div class="toggle-grid">
        <label class="tog"><span>💬 Captions</span><input type="checkbox" id="togCaptions" checked /></label>
        <label class="tog"><span>🪝 Hook Card</span><input type="checkbox" id="togHook" checked /></label>
        <label class="tog"><span>📊 3D / Stat Cards</span><input type="checkbox" id="togOverlays" checked /></label>
        <label class="tog"><span>🔍 Punch Zooms</span><input type="checkbox" id="togZooms" checked /></label>
      </div>
      <div class="row" style="margin-top:8px;">
        <label>Color Grade / Look</label>
        <select id="colorGrade">
          <option value="podcast_clean" selected>Podcast Clean</option>
          <option value="documentary">Dark Documentary</option>
          <option value="none">Original Natural</option>
        </select>
      </div>
      <div class="row">
        <label>Background Music (BGM)</label>
        <select id="bgmVol">
          <option value="0.10" selected>Low (10% Auto-Duck)</option>
          <option value="0.18">Medium (18% Cinematic)</option>
          <option value="0">Off (Mute BGM)</option>
        </select>
      </div>
      <div class="row">
        <label>Export Resolution</label>
        <select id="qualMode">
          <option value="fast" selected>720p HD (Fast on Phone)</option>
          <option value="hd">1080p Full HD (Slower)</option>
        </select>
      </div>

      <button class="btn" id="btnStart" style="width:100%;margin-top:12px;" onclick="startClipping()">🚀 Generate Clips Now</button>
      <div class="err" id="errMsg"></div>
    </div>

    <div class="card prog-box" id="progBox">
      <div class="prog-text" id="progText">प्रोसेसिंग शुरू हो रही है...</div>
      <div class="bar-bg"><div class="bar-fill" id="progBar">0%</div></div>
      <button class="btn btn-stop" id="btnStop" onclick="stopJob()">⏹ Stop After Current Clip</button>
    </div>

    <div id="liveClips"></div>
  </div>

  <div id="tabLib" style="display:none;">
    <h1>Saved Library</h1>
    <p class="sub">ये सभी क्लिप्स आपके फोन की Gallery → <b>Movies/DocStudio</b> में सेव हैं।</p>
    <div id="libList"></div>
  </div>
</div>

<div class="nav">
  <button id="navHome" class="active" onclick="switchTab('home')">🏠 Studio</button>
  <button id="navLib" onclick="switchTab('lib')">🎬 Library</button>
</div>

<script>
let pollTimer = null;
let renderedSet = new Set();
let selectedAspect = '9:16';

function setAspect(val, el) {
  selectedAspect = val;
  document.querySelectorAll('.ratio-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');
}

function toggleVoiceRow() {
  const lang = document.getElementById('dubLang').value;
  document.getElementById('voiceRow').style.display = lang === 'orig' ? 'none' : 'flex';
}

function switchTab(t) {
  document.getElementById('tabHome').style.display = t === 'home' ? 'block' : 'none';
  document.getElementById('tabLib').style.display = t === 'lib' ? 'block' : 'none';
  document.getElementById('navHome').classList.toggle('active', t === 'home');
  document.getElementById('navLib').classList.toggle('active', t === 'lib');
  if (t === 'lib') loadLibrary();
}

async function loadPreview() {
  const url = document.getElementById('ytUrl').value.trim();
  const err = document.getElementById('errMsg');
  err.textContent = '';
  if (!url) return;
  document.getElementById('btnPreview').textContent = '...';
  try {
    const r = await fetch('/api/preview', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url}) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Preview failed');
    document.getElementById('prevImg').src = d.thumbnail || '';
    document.getElementById('prevTitle').textContent = d.title;
    document.getElementById('prevChan').textContent = 'YouTube • ' + d.channel;
    document.getElementById('prevDur').textContent = d.duration_str;
    const rec = d.recommended_clips || 3;
    const pill = document.getElementById('recPill');
    pill.textContent = `✨ Video Length ${d.duration_str} → Auto Detected: ${rec} Best Clips`;
    pill.style.display = 'inline-block';
    document.getElementById('optAutoClips').textContent = `Auto (${rec} Clips Detected for ${d.duration_str})`;
    document.getElementById('previewBox').style.display = 'block';
  } catch (e) {
    err.textContent = e.message;
  } finally {
    document.getElementById('btnPreview').textContent = 'Check';
  }
}

async function startClipping() {
  const url = document.getElementById('ytUrl').value.trim();
  const err = document.getElementById('errMsg');
  err.textContent = '';
  if (!url) { err.textContent = 'कृपया YouTube लिंक पेस्ट करें।'; return; }
  renderedSet.clear();
  document.getElementById('liveClips').innerHTML = '';

  const payload = {
    url,
    aspect: selectedAspect,
    clip_limit: document.getElementById('clipCount').value,
    length_mode: document.getElementById('lenMode').value,
    dub_lang: document.getElementById('dubLang').value,
    dub_voice: document.getElementById('dubVoice').value,
    captions_on: document.getElementById('togCaptions').checked,
    hook_on: document.getElementById('togHook').checked,
    overlays_on: document.getElementById('togOverlays').checked,
    zooms_on: document.getElementById('togZooms').checked,
    color_grade: document.getElementById('colorGrade').value,
    bgm_vol: parseFloat(document.getElementById('bgmVol').value),
    hd: document.getElementById('qualMode').value === 'hd'
  };
  const r = await fetch('/api/start', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
  const d = await r.json();
  if (!r.ok) { err.textContent = d.error || 'Failed to start'; return; }
  document.getElementById('progBox').style.display = 'block';
  document.getElementById('btnStart').disabled = true;
  startPolling();
}

async function stopJob() {
  await fetch('/api/stop', { method:'POST' });
  document.getElementById('btnStop').disabled = true;
  document.getElementById('btnStop').textContent = 'Stopping...';
}

function makeClipHtml(c) {
  const badges = [c.aspect || '9:16', c.dub && c.dub !== 'ORIG' ? '🎙️ ' + c.dub : ''].filter(Boolean).join(' • ');
  return `<div class="clip-card">
    <video controls playsinline preload="metadata" src="${c.url}"></video>
    <div class="clip-body">
      <div class="clip-top">
        <div class="clip-title">${c.title}</div>
        <div class="score">🔥 ${c.score}/10</div>
      </div>
      <div class="clip-sub">⏱ ${c.range} (${c.duration}s) • ${badges} • Gallery में सेव</div>
      <div class="clip-actions">
        <a href="${c.url}" download="${c.filename}">⬇ Download MP4</a>
      </div>
    </div>
  </div>`;
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/status');
      const s = await r.json();
      if (s.video_thumb) {
        document.getElementById('prevImg').src = s.video_thumb;
        document.getElementById('prevTitle').textContent = s.video_title;
        document.getElementById('prevChan').textContent = 'YouTube • ' + s.video_channel;
        document.getElementById('previewBox').style.display = 'block';
      }
      if (s.running || s.progress > 0) {
        document.getElementById('progBox').style.display = 'block';
        document.getElementById('progText').textContent = s.status;
        document.getElementById('progBar').style.width = Math.max(5, s.progress) + '%';
        document.getElementById('progBar').textContent = s.progress.toFixed(1) + '%';
      }
      const box = document.getElementById('liveClips');
      for (const c of (s.clips || [])) {
        if (!renderedSet.has(c.filename)) {
          renderedSet.add(c.filename);
          box.insertAdjacentHTML('beforeend', makeClipHtml(c));
        }
      }
      document.getElementById('btnStart').disabled = s.running;
      document.getElementById('btnStop').style.display = s.running ? 'block' : 'none';
      if (!s.running) {
        document.getElementById('btnStop').disabled = false;
        document.getElementById('btnStop').textContent = '⏹ Stop After Current Clip';
        if (s.error) document.getElementById('errMsg').textContent = s.error;
        clearInterval(pollTimer);
      }
    } catch (_) {}
  }, 1500);
}

async function loadLibrary() {
  const r = await fetch('/api/library');
  const items = await r.json();
  const el = document.getElementById('libList');
  if (!items.length) { el.innerHTML = '<div class="card" style="text-align:center;color:#94a3b8;">अभी कोई क्लिप नहीं है।</div>'; return; }
  el.innerHTML = items.map(makeClipHtml).join('');
}

startPolling();
</script>
</body>
</html>
"""


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _send_json(self, code: int, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        ln = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(ln) if ln > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/status":
            with JOB_LOCK:
                snap = dict(JOB)
                snap["clips"] = list(JOB["clips"])
            self._send_json(200, snap)
            return

        if parsed.path == "/api/library":
            meta = load_library_meta()
            items = []
            for fn in sorted(os.listdir(OUT_DIR), key=lambda x: os.path.getmtime(os.path.join(OUT_DIR, x)), reverse=True):
                if not fn.endswith(".mp4"):
                    continue
                m = meta.get(fn, {})
                items.append({
                    "filename": fn,
                    "url": f"/video/{fn}",
                    "title": m.get("title", fn),
                    "score": m.get("score", 8.0),
                    "range": m.get("range", "Saved Clip"),
                    "duration": m.get("duration", 0),
                    "aspect": m.get("aspect", "9:16"),
                    "dub": m.get("dub", "ORIG"),
                })
            self._send_json(200, items[:40])
            return

        if parsed.path.startswith("/video/"):
            fname = os.path.basename(unquote(parsed.path[len("/video/"):]))
            fpath = os.path.join(OUT_DIR, fname)
            if not os.path.isfile(fpath):
                self.send_error(404)
                return
            self._serve_video_range(fpath)
            return

        self.send_error(404)

    def _serve_video_range(self, fpath: str) -> None:
        size = os.path.getsize(fpath)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng and rng.startswith("bytes="):
            parts = rng.replace("bytes=", "").split("-")
            if parts[0].isdigit():
                start = int(parts[0])
            if len(parts) > 1 and parts[1].isdigit():
                end = min(size - 1, int(parts[1]))
            code = 206

        length = max(0, end - start + 1)
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        with open(fpath, "rb") as f:
            f.seek(start)
            rem = length
            while rem > 0:
                chunk = f.read(min(65536, rem))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except BrokenPipeError:
                    break
                rem -= len(chunk)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/preview":
            try:
                body = self._read_json()
                info = fetch_video_preview(body.get("url", "").strip())
                self._send_json(200, info)
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
            return

        if parsed.path == "/api/start":
            body = self._read_json()
            url = str(body.get("url", "")).strip()
            if not url:
                self._send_json(400, {"error": "YouTube URL खाली है"})
                return
            with JOB_LOCK:
                if JOB["running"]:
                    self._send_json(409, {"error": "एक जॉब पहले से चल रहा है"})
                    return
                JOB.update({
                    "running": True,
                    "cancel_requested": False,
                    "progress": 2.0,
                    "status": "शुरू हो रहा है...",
                    "total_clips": 0,
                    "done_clips": 0,
                    "clips": [],
                    "error": None,
                })
            th = threading.Thread(target=run_batch_job, args=(url, body), daemon=True)
            th.start()
            self._send_json(200, {"ok": True})
            return

        if parsed.path == "/api/stop":
            with JOB_LOCK:
                JOB["cancel_requested"] = True
            self._send_json(200, {"ok": True})
            return

        self.send_error(404)


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("127.0.0.1", port), StudioHandler)
    print(f"\n🚀 DocStudio Pro Web App चालू है!", flush=True)
    print(f"👉 Chrome में खोलें: http://127.0.0.1:{port}\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")


if __name__ == "__main__":
    main()
