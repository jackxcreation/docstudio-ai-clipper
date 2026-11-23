import argparse
import asyncio
import json
import os
import subprocess
import sys
import edge_tts

VOICES = {
    "en_male": "en-US-AndrewNeural",
    "en_deep": "en-US-ChristopherNeural",
    "en_female": "en-US-AvaNeural",
    "hi_male": "hi-IN-MadhurNeural",
    "hi_female": "hi-IN-SwaraNeural",
}


def get_audio_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return round(float(p.stdout.strip()), 3)
    except Exception:
        return 0.0


async def generate_voiceover(text: str, out_mp3: str, out_json: str, voice_key: str, rate: str):
    voice = VOICES.get(voice_key, voice_key)
    os.makedirs(os.path.dirname(os.path.abspath(out_mp3)), exist_ok=True)

    print(f"🎙️  Generating neural voiceover ({voice}, rate={rate})...", flush=True)
    try:
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate, boundary="WordBoundary")
    except TypeError:
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate)

    words = []
    sentences = []
    with open(out_mp3, "wb") as f:
        async for chunk in communicate.stream():
            ctype = chunk.get("type")
            if ctype == "audio":
                f.write(chunk["data"])
            elif ctype == "WordBoundary":
                st = chunk["offset"] / 1e7
                dur = chunk["duration"] / 1e7
                wtxt = chunk["text"].strip()
                if wtxt:
                    words.append({
                        "word": wtxt,
                        "start": round(st, 3),
                        "end": round(st + max(0.06, dur), 3),
                    })
            elif ctype == "SentenceBoundary":
                st = chunk["offset"] / 1e7
                dur = chunk["duration"] / 1e7
                stxt = chunk["text"].strip()
                if stxt:
                    sentences.append({
                        "text": stxt,
                        "start": round(st, 3),
                        "end": round(st + max(0.2, dur), 3),
                    })

    duration = get_audio_duration(out_mp3)

    if not words:
        if sentences:
            for s in sentences:
                toks = s["text"].split()
                step = max(0.08, (s["end"] - s["start"]) / max(1, len(toks)))
                for i, tok in enumerate(toks):
                    words.append({
                        "word": tok,
                        "start": round(s["start"] + i * step, 3),
                        "end": round(s["start"] + (i + 1) * step, 3),
                    })
        else:
            toks = text.split()
            step = max(0.08, (duration or 10.0) / max(1, len(toks)))
            for i, tok in enumerate(toks):
                words.append({
                    "word": tok,
                    "start": round(i * step, 3),
                    "end": round((i + 1) * step, 3),
                })

    segments = []
    buf = []
    for w in words:
        buf.append(w)
        span = buf[-1]["end"] - buf[0]["start"]
        if span >= 4.2 or w["word"].endswith((".", "?", "!", "।")):
            segments.append({
                "start": buf[0]["start"],
                "end": buf[-1]["end"],
                "text": " ".join(x["word"] for x in buf),
            })
            buf = []
    if buf:
        segments.append({
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
            "text": " ".join(x["word"] for x in buf),
        })

    payload = {
        "audio_path": os.path.abspath(out_mp3),
        "voice": voice,
        "duration": duration or (words[-1]["end"] if words else 0.0),
        "word_count": len(words),
        "full_text": text.strip(),
        "segments": segments,
        "words": words,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Voiceover ready: {payload['duration']}s | {len(words)} synced words -> {out_mp3}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out-audio", required=True)
    ap.add_argument("--out-words", required=True)
    ap.add_argument("--voice", default="en_male")
    ap.add_argument("--rate", default="+4%")
    args = ap.parse_args()

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("❌ Script text is empty", file=sys.stderr)
        sys.exit(1)

    asyncio.run(generate_voiceover(text, args.out_audio, args.out_words, args.voice, args.rate))


if __name__ == "__main__":
    main()
