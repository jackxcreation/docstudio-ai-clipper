#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engines.doc_render import DocumentaryRenderer
from engines.local_ai import build_local_edit_plan, generate_local_script
from engines.media_fetcher import (
    download_bgm_from_youtube,
    download_youtube_clip,
    ensure_starter_sfx,
    fetch_broll_for_plan,
)
from engines.vision_audio import analyze_audio_local, analyze_video_local


def load_config() -> dict:
    cfg_path = os.path.join(ROOT_DIR, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:32]
    return s or "doc_video"


def main() -> int:
    ap = argparse.ArgumentParser(description="Termux Local AI Documentary + Podcast Studio (Zero API Key)")
    ap.add_argument("--topic", default=None, help="Topic to auto-generate full AI documentary with voiceover")
    ap.add_argument("--script", default=None, help="Path to a custom .txt script file for voiceover documentary")
    ap.add_argument("--url", default=None, help="YouTube video URL to smart-crop & documentary edit")
    ap.add_argument("--video", default=None, help="Local .mp4 video path to smart-crop & edit")
    ap.add_argument("--start", type=float, default=None, help="Clip start time in seconds (with --url)")
    ap.add_argument("--end", type=float, default=None, help="Clip end time in seconds (with --url)")
    ap.add_argument("--lang", choices=["en", "hi"], default="en", help="Language for script/voice (en or hi)")
    ap.add_argument("--voice", default=None, help="Voice key: en_male, en_deep, en_female, hi_male, hi_female")
    ap.add_argument("--style", choices=["auto", "documentary", "podcast", "hybrid"], default="auto")
    ap.add_argument("--horizontal", action="store_true", help="Render 1920x1080 landscape instead of 1080x1920")
    ap.add_argument("--fps", type=int, default=24, help="Output FPS (24 = cinematic & fast on phone)")
    ap.add_argument("--offline", action="store_true", help="Disable online B-roll fetching")
    ap.add_argument("--add-bgm", default=None, help="Download copyright-free BGM MP3 from YouTube URL into assets/bgm/")
    args = ap.parse_args()

    cfg = load_config()
    assets_dir = os.path.join(ROOT_DIR, "assets")
    temp_dir = os.path.join(ROOT_DIR, "temp")
    out_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Utility: add royalty-free BGM track from YouTube
    if args.add_bgm:
        download_bgm_from_youtube(args.add_bgm, os.path.join(assets_dir, "bgm"))
        return 0

    ensure_starter_sfx(os.path.join(assets_dir, "sfx"))
    out_w, out_h = (1920, 1080) if args.horizontal else (int(cfg.get("width", 1080)), int(cfg.get("height", 1920)))
    fps = int(args.fps or cfg.get("fps", 24))
    enable_online = bool(cfg.get("enable_online_broll", True)) and (not args.offline)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================================================================
    # MODE A: Topic or Custom Script -> Full AI Voiceover Documentary
    # =========================================================================
    if args.topic or args.script:
        if args.script:
            with open(args.script, "r", encoding="utf-8") as f:
                script_text = f.read().strip()
            title = os.path.splitext(os.path.basename(args.script))[0]
        else:
            s_obj = generate_local_script(args.topic, cfg, lang=args.lang)
            script_text = s_obj["script"]
            title = s_obj["title"]
            print(f"🧠 Local Script Ready ({s_obj['engine']}): {title}", flush=True)

        script_path = os.path.join(temp_dir, "script.txt")
        voice_mp3 = os.path.join(temp_dir, "voiceover.mp3")
        words_json = os.path.join(temp_dir, "words.json")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_text)

        voice_key = args.voice or ("hi_male" if args.lang == "hi" else cfg.get("voice", "en_male"))
        vo_script = os.path.join(ROOT_DIR, "engines", "voiceover.py")
        res = subprocess.run([
            sys.executable, vo_script,
            "--text-file", script_path,
            "--out-audio", voice_mp3,
            "--out-words", words_json,
            "--voice", voice_key,
            "--rate", str(cfg.get("voice_rate", "+4%")),
        ])
        if res.returncode != 0:
            raise RuntimeError("Voiceover generation failed.")

        with open(words_json, "r", encoding="utf-8") as f:
            words_data = json.load(f)

        audio_info = analyze_audio_local(voice_mp3)
        plan = build_local_edit_plan(words_data, vision_info=None, audio_info=audio_info, mode="documentary")
        plan = fetch_broll_for_plan(plan, assets_dir, enable_online=enable_online)

        plan_path = os.path.join(temp_dir, "last_edit_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)

        out_mp4 = os.path.join(out_dir, f"{slugify(title)}_{stamp}.mp4")
        renderer = DocumentaryRenderer(out_w, out_h, fps=fps, color_grade=cfg.get("color_grade", "documentary"))
        renderer.render(
            output_path=out_mp4,
            plan=plan,
            words_data=words_data,
            assets_dir=assets_dir,
            audio_input_path=voice_mp3,
            bgm_vol=float(cfg.get("bgm_volume", 0.14)),
            sfx_vol=float(cfg.get("sfx_volume", 0.30)),
        )
        print(f"\n📱 Check Phone Gallery -> Movies/DocStudio/{os.path.basename(out_mp4)}", flush=True)
        return 0

    # =========================================================================
    # MODE B: YouTube Link or Local Video -> Smart Crop + Documentary/Podcast Edit
    # =========================================================================
    if args.url or args.video:
        words_data = {"duration": 0.0, "full_text": "", "segments": [], "words": []}
        if args.url:
            raw_mp4 = os.path.join(temp_dir, "yt_input.mp4")
            if os.path.exists(raw_mp4):
                os.remove(raw_mp4)
            video_path, yt_words = download_youtube_clip(args.url, raw_mp4, temp_dir, args.start, args.end)
            if yt_words:
                words_data = yt_words
            base_name = "yt_edit"
        else:
            video_path = os.path.abspath(os.path.expanduser(args.video))
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            base_name = os.path.splitext(os.path.basename(video_path))[0]

        print("🔊 Analyzing audio voice-band & emphasis peaks...", flush=True)
        audio_info = analyze_audio_local(video_path)
        print("👁️  Analyzing scenes & subjects (Zero-OpenCV local vision)...", flush=True)
        vision_info = analyze_video_local(video_path, audio_info)
        if not words_data.get("duration"):
            words_data["duration"] = vision_info["meta"]["duration"]

        print(
            "✂️  Scenes: " + ", ".join(f"{s['scene_type']}({s['start_time']:.1f}-{s['end_time']:.1f}s)" for s in vision_info["scenes"]),
            flush=True,
        )

        plan = build_local_edit_plan(words_data, vision_info=vision_info, audio_info=audio_info, mode=args.style)
        plan = fetch_broll_for_plan(plan, assets_dir, enable_online=enable_online)

        out_mp4 = os.path.join(out_dir, f"{slugify(base_name)}_{stamp}.mp4")
        renderer = DocumentaryRenderer(out_w, out_h, fps=fps, color_grade=cfg.get("color_grade", "documentary"))
        renderer.render(
            output_path=out_mp4,
            plan=plan,
            words_data=words_data,
            assets_dir=assets_dir,
            source_video_path=video_path,
            vision_info=vision_info,
            bgm_vol=float(cfg.get("bgm_volume", 0.14)),
            sfx_vol=float(cfg.get("sfx_volume", 0.30)),
        )
        print(f"\n📱 Check Phone Gallery -> Movies/DocStudio/{os.path.basename(out_mp4)}", flush=True)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
