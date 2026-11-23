from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from typing import Any, Dict, List

import edge_tts

DUB_VOICES = {
    ("hi", "male"): "hi-IN-MadhurNeural",
    ("hi", "female"): "hi-IN-SwaraNeural",
    ("hi", "deep"): "hi-IN-MadhurNeural",
    ("en", "male"): "en-US-AndrewNeural",
    ("en", "female"): "en-US-AvaNeural",
    ("en", "deep"): "en-US-ChristopherNeural",
    ("bn", "male"): "bn-IN-BashkarNeural",
    ("bn", "female"): "bn-IN-TanishaaNeural",
    ("ta", "male"): "ta-IN-ValluvarNeural",
    ("ta", "female"): "ta-IN-PallaviNeural",
    ("te", "male"): "te-IN-MohanNeural",
    ("te", "female"): "te-IN-ShrutiNeural",
    ("mr", "male"): "mr-IN-ManoharNeural",
    ("mr", "female"): "mr-IN-AarohiNeural",
    ("es", "male"): "es-ES-AlvaroNeural",
    ("es", "female"): "es-ES-ElviraNeural",
}


def translate_text_free(text: str, target_lang: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean or target_lang == "orig":
        return clean
    try:
        q = urllib.parse.quote(clean)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={urllib.parse.quote(target_lang)}&dt=t&q={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        translated = "".join(part[0] for part in (data[0] or []) if part and part[0])
        return translated.strip() or clean
    except Exception:
        return clean


def _probe_dur(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0


async def _synth_edge_tts(text: str, voice: str, out_mp3: str) -> List[Dict[str, Any]]:
    try:
        comm = edge_tts.Communicate(text, voice=voice, rate="+4%", boundary="WordBoundary")
    except TypeError:
        comm = edge_tts.Communicate(text, voice=voice, rate="+4%")

    words: List[Dict[str, Any]] = []
    with open(out_mp3, "wb") as f:
        async for chunk in comm.stream():
            ct = chunk.get("type")
            if ct == "audio":
                f.write(chunk["data"])
            elif ct == "WordBoundary":
                st = chunk["offset"] / 1e7
                dur = chunk["duration"] / 1e7
                wt = chunk["text"].strip()
                if wt:
                    words.append({"word": wt, "start": st, "end": st + max(0.06, dur)})
    return words


def dub_clip_to_language(
    words_data: Dict[str, Any],
    clip_duration: float,
    target_lang: str,
    voice_gender: str,
    temp_dir: str,
    tag: str = "dub",
) -> tuple[str | None, Dict[str, Any]]:
    segs = words_data.get("segments", [])
    if not segs:
        return None, words_data

    translated_segs: List[Dict[str, Any]] = []
    batch_size = 4
    for i in range(0, len(segs), batch_size):
        batch = segs[i:i + batch_size]
        joined = " || ".join(s["text"].strip() for s in batch)
        tr_joined = translate_text_free(joined, target_lang)
        parts = [p.strip() for p in tr_joined.split("||")]
        if len(parts) != len(batch):
            parts = [translate_text_free(s["text"], target_lang) for s in batch]
        for s, tr_txt in zip(batch, parts):
            translated_segs.append({
                "start": s["start"],
                "end": s["end"],
                "text": tr_txt or s["text"],
            })

    full_translated = " ".join(s["text"] for s in translated_segs).strip()
    if not full_translated:
        return None, words_data

    voice_name = DUB_VOICES.get((target_lang, voice_gender), DUB_VOICES.get((target_lang, "male"), "hi-IN-MadhurNeural"))
    raw_mp3 = os.path.join(temp_dir, f"{tag}_raw.mp3")
    fit_wav = os.path.join(temp_dir, f"{tag}_fit.wav")

    raw_words = asyncio.run(_synth_edge_tts(full_translated, voice_name, raw_mp3))
    tts_dur = _probe_dur(raw_mp3)
    if tts_dur <= 0.2:
        return None, words_data

    target_dur = max(1.0, float(clip_duration))
    ratio = tts_dur / target_dur
    tempo = max(0.88, min(1.32, ratio))

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", raw_mp3,
        "-af", f"atempo={tempo:.4f},apad=whole_dur={target_dur:.3f}",
        "-t", f"{target_dur:.3f}", "-ar", "44100", "-ac", "2", fit_wav,
    ]
    subprocess.run(cmd)
    if not os.path.exists(fit_wav) or os.path.getsize(fit_wav) < 1000:
        return None, words_data

    new_words: List[Dict[str, Any]] = []
    if raw_words:
        for w in raw_words:
            st = round(w["start"] / tempo, 3)
            et = round(min(target_dur, w["end"] / tempo), 3)
            if st < target_dur:
                new_words.append({"word": w["word"], "start": st, "end": max(st + 0.06, et)})
    else:
        toks = full_translated.split()
        step = target_dur / max(1, len(toks))
        for idx, tok in enumerate(toks):
            new_words.append({"word": tok, "start": round(idx * step, 3), "end": round((idx + 1) * step, 3)})

    new_segs: List[Dict[str, Any]] = []
    for s in translated_segs:
        new_segs.append({
            "start": round(min(target_dur, s["start"]), 3),
            "end": round(min(target_dur, s["end"]), 3),
            "text": s["text"],
        })

    updated = {
        "duration": round(target_dur, 3),
        "word_count": len(new_words),
        "full_text": full_translated,
        "segments": new_segs,
        "words": new_words,
    }
    return fit_wav, updated
