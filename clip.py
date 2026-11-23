#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engines.doc_render import DocumentaryRenderer
from engines.dubber import dub_clip_to_language
from engines.local_ai import build_local_edit_plan, extract_high_value_clips
from engines.media_fetcher import ensure_starter_sfx
from engines.vision_audio import analyze_audio_local, analyze_video_local

_SUP_TRANS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "ᵗ": "t", "ʰ": "h", "ˢ": "s", "ⁿ": "n", "ᵈ": "d", "ʳ": "r",
    "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
})

_ASR_FIXES = [
    (re.compile(r"\b(मेरा|अपना|अपने|बाद|में|पास|तक)\s+1200\b"), r"\1 12th"),
    (re.compile(r"\b1200\s+(में|के|की|का|तक|पास|किया)\b"), r"12th \1"),
    (re.compile(r"\b(मेरा|अपना|अपने|बाद|में|पास|तक)\s+1000\b"), r"\1 10th"),
    (re.compile(r"\b1000\s+(में|के|की|का|तक|पास|किया)\b"), r"10th \1"),
    (re.compile(r"जीप\s+रेटिंग"), "GP रेटिंग"),
]

_DANGLING = {"A", "AN", "THE", "TO", "FOR", "OF", "IN", "ON", "AT", "AND", "OR", "BUT", "WITH", "IS", "ARE", "WAS"}


def load_config() -> dict:
    cfg_p = os.path.join(ROOT_DIR, "config.json")
    if os.path.exists(cfg_p):
        with open(cfg_p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"bgm_volume": 0.10, "sfx_volume": 0.22}


def _short_upper(text: str, max_len: int = 38) -> str:
    clean = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    if len(clean) > max_len:
        clean = clean[:max_len].rsplit(" ", 1)[0] or clean[:max_len]
    words = clean.upper().split()
    while len(words) > 2 and words[-1] in _DANGLING:
        words.pop()
    return " ".join(words)


def _clean_caption_text(t: str) -> str:
    t = t.translate(_SUP_TRANS)
    out = [ch for ch in t if (32 <= ord(ch) <= 126) or (0x0900 <= ord(ch) <= 0x097F)]
    cleaned = re.sub(r"\s+", " ", "".join(out)).strip()
    for pat, repl in _ASR_FIXES:
        cleaned = pat.sub(repl, cleaned)
    return cleaned


def _parse_srt_time(ts: str) -> float:
    ts = ts.strip().split(" ")[0]
    m3 = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts)
    if m3:
        return int(m3.group(1)) * 3600 + int(m3.group(2)) * 60 + int(m3.group(3)) + int(m3.group(4)) / 1000.0
    m2 = re.match(r"(\d+):(\d+)[,.](\d+)", ts)
    if m2:
        return int(m2.group(1)) * 60 + int(m2.group(2)) + int(m2.group(3)) / 1000.0
    return 0.0


def parse_clean_srt(srt_path: str) -> List[Dict[str, Any]]:
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    blocks = re.split(r"\r?\n\r?\n+", raw.strip())
    segments: List[Dict[str, Any]] = []
    prev_words: List[str] = []

    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        t_line = next((ln for ln in lines if "-->" in ln), None)
        if not t_line:
            continue
        p0, p1 = t_line.split("-->", 1)
        st, et = _parse_srt_time(p0), _parse_srt_time(p1)
        if et - st < 0.12:
            continue

        txt = " ".join(lines[lines.index(t_line) + 1:])
        txt = re.sub(r"<[^>]+>", "", txt)
        txt = re.sub(r"\[[^\]]+\]", "", txt)
        txt = _clean_caption_text(txt)
        if not txt:
            continue

        cur_words = txt.split()
        overlap = 0
        max_k = min(len(prev_words), len(cur_words))
        for k in range(max_k, 0, -1):
            if [w.lower() for w in prev_words[-k:]] == [w.lower() for w in cur_words[:k]]:
                overlap = k
                break
        new_words = cur_words[overlap:]
        if not new_words:
            continue

        prev_words = (prev_words + new_words)[-24:]
        et_clamped = min(et, st + max(1.35, len(new_words) * 0.46))
        segments.append({
            "start": round(st, 3),
            "end": round(et_clamped, 3),
            "text": " ".join(new_words),
        })
    return segments


def slice_words_window(all_segments: List[Dict[str, Any]], clip_start: float, clip_end: float) -> Dict[str, Any]:
    segs: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    for s in all_segments:
        if s["end"] <= clip_start or s["start"] >= clip_end:
            continue
        st = max(0.0, s["start"] - clip_start)
        raw_et = min(clip_end, s["end"]) - clip_start
        et = max(st + 0.15, min(raw_et, st + max(1.2, len(s["text"].split()) * 0.42)))
        txt = s["text"].strip()
        if not txt:
            continue
        segs.append({"start": round(st, 3), "end": round(et, 3), "text": txt})
        toks = txt.split()
        step = (et - st) / max(1, len(toks))
        for i, tok in enumerate(toks):
            words.append({
                "word": tok,
                "start": round(st + i * step, 3),
                "end": round(st + (i + 1) * step, 3),
            })

    dur = max(0.5, clip_end - clip_start)
    return {
        "duration": round(dur, 3),
        "word_count": len(words),
        "full_text": " ".join(s["text"] for s in segs),
        "segments": segs,
        "words": words,
    }


def _cookie_args() -> List[str]:
    ck = os.path.join(ROOT_DIR, "cookies.txt")
    return ["--cookies", ck] if (os.path.exists(ck) and os.path.getsize(ck) > 20) else []


def fetch_yt_captions_only(url: str, temp_dir: str) -> Tuple[str | None, float]:
    for old in os.listdir(temp_dir):
        if old.startswith("clip_sub"):
            try:
                os.remove(os.path.join(temp_dir, old))
            except OSError:
                pass

    print("🔍 Scanning YouTube video & auto-captions...", flush=True)
    ck = _cookie_args()
    dur_p = subprocess.run(
        ["yt-dlp", *ck, "--get-duration", "--no-playlist", url],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    total_dur = 0.0
    raw_d = dur_p.stdout.strip().splitlines()[-1] if dur_p.stdout.strip() else ""
    if raw_d:
        parts = [float(x) for x in raw_d.split(":") if x.isdigit()]
        if len(parts) == 3:
            total_dur = parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            total_dur = parts[0] * 60 + parts[1]
        elif len(parts) == 1:
            total_dur = parts[0]

    sub_base = os.path.join(temp_dir, "clip_sub")
    subprocess.run(
        [
            "yt-dlp", *ck, "--skip-download", "--no-playlist",
            "--write-subs", "--write-auto-subs",
            "--sub-format", "vtt/srt/best",
            "--sub-langs", "hi,hi-orig,en,en-orig,en-IN,a.hi,a.en",
            "-o", sub_base, url,
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    sub_files = [
        os.path.join(temp_dir, f)
        for f in sorted(os.listdir(temp_dir))
        if f.startswith("clip_sub") and f.endswith((".srt", ".vtt"))
    ]

    def _sub_prio(fp: str) -> int:
        low = os.path.basename(fp).lower()
        if ".hi-orig." in low or ".en-orig." in low:
            return 0
        if ".hi." in low:
            return 1
        if ".en-in." in low or ".en." in low:
            return 2
        return 3

    srt = min(sub_files, key=_sub_prio) if sub_files else None
    print(f"📝 Caption file: {os.path.basename(srt) if srt else 'NOT AVAILABLE'}", flush=True)
    return srt, total_dur


def download_yt_section(url: str, start_s: float, end_s: float, out_mp4: str) -> str:
    if os.path.exists(out_mp4):
        os.remove(out_mp4)
    print(f"⬇️ Downloading clip [{start_s:.1f}s -> {end_s:.1f}s]...", flush=True)
    cmd = [
        "yt-dlp", *_cookie_args(),
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--download-sections", f"*{start_s:.2f}-{end_s:.2f}",
        "--force-keyframes-at-cuts",
        "-o", out_mp4, url,
    ]
    p = subprocess.run(cmd)
    if p.returncode != 0 or not os.path.exists(out_mp4):
        raise RuntimeError("yt-dlp clip section download failed.")
    return out_mp4


def build_viral_clip_plan(
    words_data: Dict[str, Any],
    vision_info: Dict[str, Any],
    audio_info: Dict[str, Any],
    hook_override: str | None = None,
) -> Dict[str, Any]:
    plan = build_local_edit_plan(words_data, vision_info, audio_info, mode="hybrid", hook_override=hook_override)
    plan["broll_tracks"] = []
    return plan


def render_single_short(
    clip_mp4: str,
    words_data: Dict[str, Any],
    out_mp4: str,
    width: int,
    height: int,
    fps: int,
    cfg: dict,
    hook_override: str | None = None,
    dub_lang: str = "orig",
    dub_voice: str = "male",
    clip_idx: int = 1,
) -> None:
    assets_dir = os.path.join(ROOT_DIR, "assets")
    temp_dir = os.path.join(ROOT_DIR, "temp")
    audio_info = analyze_audio_local(clip_mp4)
    vision_info = analyze_video_local(clip_mp4, audio_info)
    actual_dur = float(vision_info["meta"]["duration"])
    words_data["duration"] = actual_dur

    dub_audio = None
    if dub_lang and dub_lang != "orig" and words_data.get("segments"):
        dub_audio, words_data = dub_clip_to_language(
            words_data=words_data,
            clip_duration=actual_dur,
            target_lang=dub_lang,
            voice_gender=dub_voice,
            temp_dir=temp_dir,
            tag=f"cli_dub_{clip_idx}",
        )

    plan = build_viral_clip_plan(words_data, vision_info, audio_info, hook_override=hook_override)
    renderer = DocumentaryRenderer(width, height, fps=fps, color_grade="podcast_clean")
    renderer.render(
        output_path=out_mp4,
        plan=plan,
        words_data=words_data,
        assets_dir=assets_dir,
        audio_input_path=dub_audio,
        source_video_path=clip_mp4,
        vision_info=vision_info,
        bgm_vol=float(cfg.get("bgm_volume", 0.10)),
        sfx_vol=float(cfg.get("sfx_volume", 0.22)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="DocStudio AI Auto-Clipper CLI")
    ap.add_argument("--url", required=True, help="YouTube Video URL")
    ap.add_argument("--clips", type=int, default=3, help="Max clips (1-15)")
    ap.add_argument("--aspect", default="9:16", choices=["9:16", "16:9", "1:1"])
    ap.add_argument("--dub", default="orig", help="Target dubbing language (orig, hi, en, bn, ta, te, mr, es)")
    ap.add_argument("--voice", default="male", choices=["male", "female", "deep"])
    ap.add_argument("--hd", action="store_true", help="Export 1080p instead of 720p")
    args = ap.parse_args()

    cfg = load_config()
    temp_dir = os.path.join(ROOT_DIR, "temp")
    out_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    ensure_starter_sfx(os.path.join(ROOT_DIR, "assets", "sfx"))

    if args.aspect == "16:9":
        w, h = (1920, 1080) if args.hd else (1280, 720)
    elif args.aspect == "1:1":
        w, h = (1080, 1080) if args.hd else (720, 720)
    else:
        w, h = (1080, 1920) if args.hd else (720, 1280)

    srt_path, total_dur = fetch_yt_captions_only(args.url, temp_dir)
    all_segs = parse_clean_srt(srt_path) if srt_path else []
    targets = extract_high_value_clips(all_segs, total_dur or 60.0, 35.0, 105.0, max(1, min(15, args.clips)))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aspect_tag = args.aspect.replace(":", "x")
    for idx, t in enumerate(targets, 1):
        print(f"\n🎬 Processing Clip {idx}/{len(targets)}: {t['hook']} ({t['reason']})", flush=True)
        clip_tmp = os.path.join(temp_dir, f"cli_clip_{idx}.mp4")
        download_yt_section(args.url, t["start"], t["end"], clip_tmp)
        words_data = slice_words_window(all_segs, t["start"], t["end"]) if all_segs else {
            "duration": t["end"] - t["start"], "full_text": "", "segments": [], "words": []
        }
        out_mp4 = os.path.join(out_dir, f"clip_{idx:02d}_{aspect_tag}_{t['title']}_{stamp}.mp4")
        render_single_short(clip_tmp, words_data, out_mp4, w, h, 24, cfg, dub_lang=args.dub, dub_voice=args.voice, clip_idx=idx)


if __name__ == "__main__":
    main()
