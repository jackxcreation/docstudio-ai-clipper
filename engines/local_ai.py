from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from typing import Any, Dict, List, Tuple

STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "what", "when", "where",
    "they", "them", "their", "there", "about", "would", "could", "should", "which",
    "into", "your", "just", "like", "know", "really", "actually", "going", "think",
    "because", "people", "something", "things", "very", "more", "most", "some", "also",
    "then", "than", "been", "were", "will", "here", "make", "made", "want", "need",
    "right", "well", "okay", "yeah", "mean", "kind", "sort", "much", "many", "even",
    "only", "over", "does", "dont", "cant", "youre", "thats", "its", "are", "was",
    "for", "not", "but", "you", "all", "can", "had", "her", "one", "our", "out",
    "aur", "hai", "hain", "mein", "kar", "kiya", "kaise", "kyun", "kya", "toh", "bhi",
    "nahi", "ek", "woh", "yeh", "hum", "tum", "aap", "par", "liye", "hota", "hoti",
    "है", "हैं", "था", "थी", "थे", "में", "के", "की", "का", "को", "से", "पर", "और", "या",
    "भी", "तो", "ही", "ने", "ये", "वो", "इस", "उस", "एक", "कि", "जो", "कर", "हो", "नहीं",
}

_DANGLING = {
    "A", "AN", "THE", "TO", "FOR", "OF", "IN", "ON", "AT", "AND", "OR", "BUT",
    "WITH", "IS", "ARE", "WAS", "WERE", "THAT", "THIS", "IT", "AS", "BY", "FROM",
}

HOOK_RE = re.compile(
    r"(?:^|\s)(?:why|how|secret|never|always|mistake|truth|stop|dangerous|crazy|problem|biggest|"
    r"hack|rule|imagine|shocking|hidden|nobody|everyone|changed|future|superintelligence|"
    r"kyun|kaise|sach|galti|sabse|khatarnak|asli|raaz|paisa|duniya|socho|"
    r"क्यों|कैसे|सच|गलती|सबसे|खतरनाक|असली|राज़|पैसा|दुनिया|सोचो|दोगुना|तेजी)(?:\s|$|[.,:?!।])|\?",
    re.IGNORECASE,
)

VALUE_INSIGHT_RE = re.compile(
    r"(?:^|\s)(?:because|therefore|principle|framework|method|system|lesson|reason|step|rule|"
    r"habit|focus|build|create|learn|understand|strategy|mistake|solution|works|means|"
    r"kyunki|isliye|tarika|niyam|seekh|samajh|system|kaam|farq|matlab|"
    r"क्योंकि|इसलिए|तरीका|नियम|सीख|समझ|सिस्टम|काम|फर्क|मतलब|रिसर्च|साइकोलॉजी|टेक्निक)(?:\s|$|[.,:?!।])",
    re.IGNORECASE,
)

PAYOFF_CLIMAX_RE = re.compile(
    r"(?:^|\s)(?:that's why|ultimately|in the end|the point is|remember|everything changes|"
    r"most important|real reason|winner|freedom|clarity|power|future|super intelligence|"
    r"isliye|ant mein|asli baat|yaad rakhna|sabse bada|zindagi|"
    r"इसलिए|अंत\s+में|असली\s+बात|याद\s+रखना|सबसे\s+बड़ा|यही\s+तरीका|यही\s+सीक्रेट|यही\s+कारण|असलियत\s+में)(?:\s|$|[.,:?!।])",
    re.IGNORECASE,
)

NEW_TOPIC_START_RE = re.compile(
    r"(?:^|\s)(?:"
    r"number\s*(?:1|2|3|4|5|6|7|8|9|10|one|two|three|four|five|six|seven|eight|nine|ten)|"
    r"(?:first|second|third|fourth|fifth|next|another|final|last)\s+(?:thing|point|step|reason|rule|mistake|secret|technique|method|habit)|"
    r"moving\s+on|now\s+let's|coming\s+to|let's\s+talk\s+about|the\s+next\s+thing|"
    r"पहला|पहली\s+बात|दूसरा|दूसरी\s+बात|तीसरा|तीसरी\s+बात|चौथा|पांचवा|"
    r"नंबर\s*(?:1|2|3|4|5|6|7|8|9|10|एक|दो|तीन|चार|पांच)|"
    r"पॉइंट\s*नंबर|स्टेप\s*नंबर|अब\s+बात\s+करते\s+हैं|अब\s+आते\s+हैं|अब\s+समझते\s+हैं|"
    r"अगला\s+पॉइंट|अगली\s+बात|इसके\s+बाद|अच्छा\s+अब|चलिए\s+अब"
    r")(?:\s|$|[.,:?!।])",
    re.IGNORECASE,
)

DANGLING_END_RE = re.compile(
    r"(?:^|\s)(?:"
    r"and|or|but|because|so|that|which|who|when|while|if|as|to|of|for|in|on|at|with|the|a|an|is|are|was|were|have|has|had|will|would|can|could|should|their|your|my|our|its|"
    r"और|या|लेकिन|मगर|पर|कि|क्योंकि|इसलिए|तो|जब|अगर|जैसे|के|की|का|में|से|को|तक|लिए|वाला|वाली|वाले|अच्छी|अच्छा|बड़ा|छोटा|बहुत|ज्यादा|कम|अपनी|अपने|अपना|उनकी|उनके|उसकी|इसके"
    r")\s*$",
    re.IGNORECASE,
)

COMPLETE_SENTENCE_END_RE = re.compile(
    r"(?:"
    r"[.?!।]+|"
    r"(?:[\u0900-\u097F]+(?:ता|ती|ते|रहा|रही|रहे|सकता|सकती|सकते|होगा|होगी|होंगे|जाएगा|जाएगी|जाएंगे|चुका|चुकी|चुके)\s+(?:है|हैं|था|थी|थे))|"
    r"(?:हैं|है|था|थी|थे|होगा|होगी|होंगे|गया|गई|गए|किया|की|लिए|लिया|दिया|मिला|मिली|मिले|चाहिए|करो|देखो|समझो|सुनो|रखो|जाओ|आओ|बोलो|बताओ|सीखो)"
    r")\s*$",
    re.IGNORECASE,
)

FILLER_FLUFF_RE = re.compile(
    r"(subscribe|like and share|sponsor|link in description|hello guys|welcome back|"
    r"check out|merch|patreon|bell icon|comment below|"
    r"सब्सक्राइब|लाइक|चैनल|बेल आइकन|कमेंट|शेयर|घंटी|वीडियो को देखने से पहले|"
    r"मोटिवेशन भी मेरे अंदर|क्योंकि भाई मैं बहुत कम वीडियो)",
    re.IGNORECASE,
)

INTRO_TEASER_RE = re.compile(
    r"(is video mein|poora dekhiyega|pura dekhiyega|kaise hain aap log|aaj baat karne wala|"
    r"baat start karne se pehle|इस वीडियो में|पूरा देखिएगा|कैसे हैं आप लोग|बात करने वाला|"
    r"बात स्टार्ट करने से पहले|in this video|watch till the end|what's up guys|before we start|"
    r"today we are going to talk|welcome to the video)",
    re.IGNORECASE,
)

STAT_RE = re.compile(
    r"(?P<val>(?:\$|₹|€|£)?\d+(?:[.,]\d+)?\s*(?:%|percent|million|billion|trillion|crore|lakh|thousand|x|times|years|minutes|mins|hours|days|km|miles|साल|सालों|गुना|महीने|घंटे|मिनट))",
    re.IGNORECASE,
)

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20[0-3]\d)\b")

COMP_RE = re.compile(
    r"\b(versus|vs\.?|compared to|instead of|rather than|two columns?|on one side|difference between)\b",
    re.IGNORECASE,
)

CARTOON_EXPLAINER_RE = re.compile(
    r"\b(imagine|brain|mind|think|overthink|habit|human|people|someone|story|example|"
    r"psychology|dopamine|anxiety|focus|creative|idea|socho|dimag|insaan|kahani)\b",
    re.IGNORECASE,
)

VISUAL_MAP: Dict[str, str] = {
    "brain": "human brain neurons",
    "anxiety": "stressed person night window",
    "stress": "stressed person desk night",
    "cortisol": "human brain science lab",
    "dopamine": "glowing neural network",
    "sleep": "sleeping dark bedroom",
    "morning": "sunrise morning window",
    "sunlight": "sunlight rays sky",
    "paper": "person writing notebook",
    "write": "hand writing notebook pen",
    "column": "notebook writing desk",
    "money": "currency finance trading",
    "business": "modern office city skyline",
    "rich": "luxury city skyline night",
    "market": "stock market chart screen",
    "war": "historical battlefield archive",
    "history": "ancient museum architecture",
    "empire": "ancient roman architecture",
    "space": "earth space galaxy",
    "earth": "planet earth space",
    "ai": "artificial intelligence server",
    "superintelligence": "futuristic data center server",
    "technology": "futuristic technology server",
    "phone": "smartphone screen night",
    "focus": "person working laptop focus",
    "habit": "clock time passing",
    "time": "vintage clock ticking",
    "health": "running fitness sunrise",
    "india": "india cityscape aerial",
    "america": "new york city skyline",
    "china": "shanghai city skyline",
    "japan": "tokyo street night",
}

GEO_COORDS: Dict[str, Tuple[float, float]] = {
    "india": (0.71, 0.48), "delhi": (0.71, 0.44), "mumbai": (0.70, 0.50),
    "usa": (0.22, 0.39), "america": (0.22, 0.39), "united states": (0.22, 0.39),
    "california": (0.16, 0.40), "new york": (0.29, 0.38), "uk": (0.48, 0.31),
    "london": (0.48, 0.32), "europe": (0.52, 0.32), "russia": (0.68, 0.24),
    "china": (0.79, 0.40), "japan": (0.87, 0.40), "tokyo": (0.88, 0.40),
    "dubai": (0.65, 0.47), "africa": (0.55, 0.56), "australia": (0.85, 0.72),
    "brazil": (0.35, 0.64), "germany": (0.53, 0.32), "france": (0.50, 0.35),
}


def _tokens(text: str) -> List[str]:
    return [
        w for w in re.findall(r"[a-zA-Z0-9\u0900-\u097F']+", text.lower())
        if len(w) >= 3 and w not in STOPWORDS and not w.isdigit()
    ]


def _short_upper(text: str, max_len: int = 38) -> str:
    clean = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    if len(clean) > max_len:
        clean = clean[:max_len].rsplit(" ", 1)[0] or clean[:max_len]
    words = clean.upper().split()
    while len(words) > 2 and words[-1] in _DANGLING:
        words.pop()
    return " ".join(words)


def _is_complete_sentence_end(text: str, next_gap: float = 0.0) -> bool:
    clean = text.strip()
    if not clean or DANGLING_END_RE.search(clean):
        return False
    if clean.endswith((".", "?", "!", "।")):
        return True
    if COMPLETE_SENTENCE_END_RE.search(clean):
        return True
    return next_gap >= 0.85


def _is_topic_boundary(cur_segments: List[Dict[str, Any]], next_seg: Dict[str, Any] | None, next_gap: float) -> bool:
    if next_seg is None:
        return True
    next_txt = next_seg["text"].strip()
    if NEW_TOPIC_START_RE.search(next_txt):
        return True
    if next_gap >= 1.25:
        return True
    recent_toks = set(_tokens(" ".join(s["text"] for s in cur_segments[-3:])))
    next_toks = set(_tokens(next_txt))
    if len(recent_toks) >= 4 and len(next_toks) >= 3:
        overlap = len(recent_toks & next_toks) / float(min(len(recent_toks), len(next_toks)))
        if overlap == 0.0 and next_gap >= 0.65:
            return True
    return False


def _try_local_llama(prompt: str, config: Dict[str, Any]) -> str | None:
    cli = os.path.expanduser(str(config.get("llama_cli_path", "")))
    model = os.path.expanduser(str(config.get("llama_model_path", "")))
    if not (cli and model and os.path.exists(cli) and os.path.exists(model)):
        return None
    try:
        p = subprocess.run(
            [cli, "-m", model, "-p", prompt, "-n", "340", "--temp", "0.4", "-ngl", "0", "--no-display-prompt"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=90,
        )
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except Exception:
        pass
    return None


def generate_local_script(topic: str, config: Dict[str, Any], lang: str = "en") -> Dict[str, str]:
    prompt = (
        f"Write a punchy 45-second documentary voiceover script about '{topic}' in "
        f"{'natural Hinglish' if lang == 'hi' else 'crisp English'}. "
        "Start with a strong hook question, include clear steps or contrast, no stage directions, plain narration only."
    )
    llm_out = _try_local_llama(prompt, config)
    if llm_out and len(llm_out.split()) >= 35:
        return {
            "title": topic.title(),
            "hook_headline": _short_upper(topic, 38),
            "script": re.sub(r"\[.*?\]", "", llm_out).strip(),
            "engine": "local_llama_cpp",
        }

    if lang == "hi":
        script = (
            f"Kya aap jaante hain {topic} ka asli sach jo zyada tar log miss kar dete hain? "
            f"Pehli nazar mein {topic} bilkul simple lagta hai, lekin iske peeche ek gehra system kaam karta hai. "
            f"Ek taraf hai purana tareeka jisme log bina soche time waste karte hain, "
            f"aur doosri taraf hai smart approach jo 10 minute mein game badal deta hai. "
            f"Asli farq sirf ek cheez se aata hai: clarity aur consistency. "
            f"Aaj se is rule ko apply karo aur khud result dekho."
        )
    else:
        script = (
            f"Why do most people completely misunderstand {topic}? "
            f"On the surface, {topic} looks simple, but beneath it lies a hidden mechanism that shapes daily results. "
            f"Compare the old reaction versus the structured method: "
            f"one creates endless noise, while the other brings instant clarity in under 10 minutes. "
            f"The real secret is focusing strictly on what you control and cutting every distraction. "
            f"Master this single shift, and everything changes."
        )
    return {
        "title": topic.title(),
        "hook_headline": _short_upper(f"THE TRUTH ABOUT {topic}", 38),
        "script": script,
        "engine": "local_template",
    }


def classify_all_scenes_deep(
    vision_info: Dict[str, Any] | None,
    segments: List[Dict[str, Any]],
    audio_info: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    scenes = (vision_info or {}).get("scenes", [])
    if not scenes:
        return []

    total_dur = max(1.0, float((vision_info or {}).get("meta", {}).get("duration") or 30.0))
    peaks = (audio_info or {}).get("emphasis_peaks", [])
    enriched: List[Dict[str, Any]] = []

    for idx, sc in enumerate(scenes):
        st, et = float(sc["start_time"]), float(sc["end_time"])
        sc_text = " ".join(
            s["text"] for s in segments if float(s["end"]) > st and float(s["start"]) < et
        ).strip()

        has_stat = bool(STAT_RE.search(sc_text) or YEAR_RE.search(sc_text))
        has_hook = bool(HOOK_RE.search(sc_text))
        has_value = bool(VALUE_INSIGHT_RE.search(sc_text))
        has_payoff = bool(PAYOFF_CLIMAX_RE.search(sc_text))
        has_peak = any(st <= pk <= et for pk in peaks)

        if idx == 0 and st < 3.0:
            role = "HOOK"
        elif has_stat:
            role = "EVIDENCE_DATA"
        elif has_payoff or (et >= total_dur * 0.82 and has_value):
            role = "CLIMAX_PAYOFF"
        elif has_value or has_peak:
            role = "VALUE_INSIGHT"
        elif has_hook:
            role = "STORY_SETUP"
        elif sc.get("scene_type") == "PODCAST":
            role = "PODCAST_DIALOGUE"
        else:
            role = "BROLL_MONTAGE" if sc.get("scene_type") == "BROLL" else "NARRATIVE"

        item = dict(sc)
        item["narrative_role"] = role
        item["has_vocal_peak"] = has_peak
        item["excerpt"] = sc_text[:110]
        enriched.append(item)

    return enriched


import json
import urllib.request

def _try_ollama_viral_clips(
    all_segments: List[Dict[str, Any]],
    total_dur: float,
    min_dur: float,
    max_dur: float,
    max_clips: int,
) -> List[Dict[str, Any]]:
    """Uses Ollama (qwen2.5:7b on GitHub or qwen2.5:1.5b on Termux) with strict JSON output + exact sentence snapping."""
    if not all_segments:
        return []
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    # Compact transcript lines with timestamps (sample if > 260 segments so prompt stays fast)
    step = max(1, len(all_segments) // 260)
    lines = [
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text'].strip()}"
        for s in all_segments[::step]
        if s.get("text", "").strip()
    ]
    transcript_txt = "\n".join(lines)[:14000]

    prompt = (
        f"You are an expert viral Shorts/Reels editor. Analyze this timestamped transcript (total duration {total_dur:.0f}s) "
        f"and pick the top {max_clips} highest-retention, high-value, motivating or viral complete topics.\n"
        f"Rules:\n"
        f"- Each clip must be between {min_dur:.0f}s and {max_dur:.0f}s long.\n"
        f"- Skip intros, sponsor reads, and subscribe requests.\n"
        f"- End each clip ONLY when the full topic/explanation finishes (never mid-sentence).\n"
        f"- Return ONLY valid JSON with key 'clips': "
        f'{{"clips": [{{"start": 12.5, "end": 68.0, "score": 9.3, "hook": "SHORT PUNCHY HOOK", "reason": "Why viral"}}]}}'
        f"\n\nTranscript:\n{transcript_txt}"
    )

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": "2m",
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw_resp = json.loads(resp.read().decode("utf-8", errors="ignore"))
        parsed = json.loads(raw_resp.get("response", "{}"))
        items = parsed.get("clips") if isinstance(parsed, dict) else []
    except Exception:
        return []

    if not items:
        return []

    selected: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        try:
            st = float(it.get("start", -1))
            et = float(it.get("end", -1))
        except (TypeError, ValueError):
            continue
        if st < 0 or et <= st:
            continue

        # Snap to exact transcript segment boundaries so sentences never cut mid-word
        s_idx = min(range(len(all_segments)), key=lambda i: abs(float(all_segments[i]["start"]) - st))
        e_idx = min(range(len(all_segments)), key=lambda i: abs(float(all_segments[i]["end"]) - et))
        if e_idx < s_idx:
            e_idx = s_idx

        snapped_st = max(0.0, float(all_segments[s_idx]["start"]) - 0.15)
        raw_et = float(all_segments[e_idx]["end"])
        next_lim = float(all_segments[e_idx + 1]["start"]) - 0.05 if (e_idx + 1 < len(all_segments)) else (total_dur or raw_et + 0.80)
        snapped_et = min(next_lim, raw_et + 0.80)
        if total_dur > 0:
            snapped_et = min(total_dur, snapped_et)

        dur = snapped_et - snapped_st
        if dur < max(18.0, min_dur * 0.65) or dur > max_dur + 25.0:
            continue

        overlaps = any(not (snapped_et + 8.0 <= p["start"] or snapped_st >= p["end"] + 8.0) for p in selected)
        if overlaps:
            continue

        hook = _short_upper(str(it.get("hook") or all_segments[s_idx]["text"]), 54)
        reason = str(it.get("reason") or "Viral Story Arc")[:44]
        score = round(min(9.9, max(6.8, float(it.get("score", 9.0)))), 2)
        title = re.sub(r"[^a-zA-Z0-9]+", "_", hook[:36]).strip("_").lower() or f"ollama_clip_{idx+1}"

        selected.append({
            "start": round(snapped_st, 2),
            "end": round(snapped_et, 2),
            "score": score,
            "hook": hook,
            "title": title,
            "reason": f"Ollama AI • {reason}",
        })
        if len(selected) >= max_clips:
            break

    selected.sort(key=lambda c: c["score"], reverse=True)
    return selected


def extract_high_value_clips(
    all_segments: List[Dict[str, Any]],
    total_dur: float,
    min_dur: float = 35.0,
    max_dur: float = 105.0,
    max_clips: int = 6,
    audio_info: Dict[str, Any] | None = None,
    vision_info: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Topic-Complete Viral Clip Extractor:
    - Never forces a cut at 51s/53s/60s while a topic or sentence is still unfinished
    - Extends smoothly until the current topic reaches its natural conclusion & payoff
    - Stops cleanly right before the next topic begins.
    """
    if not all_segments:
        clips = []
        span = max(40.0, min(60.0, max_dur))
        step = max(span + 20.0, total_dur / max(1, min(max_clips, 6)))
        t = 15.0 if total_dur > 90 else 0.0
        idx = 1
        while t + 20.0 < total_dur and len(clips) < max_clips:
            et = min(total_dur, t + span)
            clips.append({
                "start": round(t, 2),
                "end": round(et, 2),
                "score": 6.5,
                "hook": f"HIGHLIGHT PART {idx}",
                "title": f"highlight_part_{idx}",
                "reason": "visual_pacing_fallback",
            })
            t += step
            idx += 1
        return clips

    trimmed = list(all_segments)
    while len(trimmed) > 2 and FILLER_FLUFF_RE.search(trimmed[0]["text"]):
        trimmed.pop(0)
    while len(trimmed) > 2 and FILLER_FLUFF_RE.search(trimmed[-1]["text"]):
        trimmed.pop()

    full_span = trimmed[-1]["end"] - trimmed[0]["start"]
    if full_span <= max_dur and _is_complete_sentence_end(trimmed[-1]["text"], next_gap=1.0):
        txt = " ".join(s["text"] for s in trimmed)
        return [{
            "start": round(max(0.0, trimmed[0]["start"] - 0.15), 2),
            "end": round(min(total_dur or trimmed[-1]["end"] + 0.80, trimmed[-1]["end"] + 0.80), 2),
            "score": 9.0,
            "hook": _short_upper(txt, 54),
            "title": re.sub(r"[^a-zA-Z0-9]+", "_", txt[:36]).strip("_").lower() or "viral_short",
            "reason": "Complete Topic Arc",
        }]

    rms_times = (audio_info or {}).get("rms_times", [])
    rms_vals = (audio_info or {}).get("rms", [])
    scenes = (vision_info or {}).get("scenes", [])

    def _window_audio_energy(st: float, et: float) -> float:
        if not rms_times or not rms_vals:
            return 0.65
        vals = [v for t, v in zip(rms_times, rms_vals) if st <= t <= et]
        if not vals:
            return 0.55
        mean_v = sum(vals) / len(vals)
        peak_v = max(rms_vals) or 1.0
        return min(1.0, (mean_v / peak_v) * 2.2)

    def _window_scene_variety(st: float, et: float) -> float:
        if not scenes:
            return 0.6
        cnt = sum(1 for sc in scenes if float(sc["end_time"]) > st and float(sc["start_time"]) < et)
        return min(1.0, cnt / 4.0)

    ollama_picks = _try_ollama_viral_clips(all_segments, total_dur, min_dur, max_dur, max_clips)
    if ollama_picks:
        print(f"🧠 Ollama AI selected {len(ollama_picks)} high-value viral clips!", flush=True)
        return ollama_picks

    cands: List[Dict[str, Any]] = []
    n = len(all_segments)
    hard_cap = max_dur + 20.0

    for i in range(n):
        st = float(all_segments[i]["start"])
        if FILLER_FLUFF_RE.search(all_segments[i]["text"]):
            continue
        if total_dur > 90.0 and st < 55.0 and INTRO_TEASER_RE.search(all_segments[i]["text"]):
            continue

        gap_before = (st - float(all_segments[i - 1]["end"])) if i > 0 else 2.0
        is_explicit_topic_start = (i == 0) or bool(NEW_TOPIC_START_RE.search(all_segments[i]["text"])) or (gap_before >= 1.1)

        if i > 0:
            prev_txt = all_segments[i - 1]["text"].strip()
            if not is_explicit_topic_start and not _is_complete_sentence_end(prev_txt, next_gap=gap_before) and gap_before < 0.55:
                continue

        texts: List[str] = []
        used_segs: List[Dict[str, Any]] = []
        wc = 0
        best_for_start = None

        for j in range(i, n):
            seg_txt = all_segments[j]["text"].strip()
            texts.append(seg_txt)
            used_segs.append(all_segments[j])
            wc += len(seg_txt.split())
            raw_et = float(all_segments[j]["end"])
            dur = raw_et - st

            if dur > hard_cap:
                break

            next_seg = all_segments[j + 1] if (j + 1 < n) else None
            next_gap = (float(next_seg["start"]) - raw_et) if next_seg else 1.5
            sent_done = _is_complete_sentence_end(seg_txt, next_gap=next_gap)
            topic_done = sent_done and _is_topic_boundary(used_segs, next_seg, next_gap)

            if dur < min_dur or not sent_done:
                continue

            full = " ".join(texts).strip()
            if FILLER_FLUFF_RE.search(full):
                continue
            if total_dur > 90.0 and INTRO_TEASER_RE.search(full):
                continue

            opening = " ".join(texts[:3])
            closing = " ".join(texts[-2:])

            hook_hits = len(HOOK_RE.findall(opening))
            hook_sc = min(1.0, hook_hits * 0.32 + (0.28 if "?" in opening else 0.0))

            val_hits = len(VALUE_INSIGHT_RE.findall(full))
            stat_hits = len(STAT_RE.findall(full)) + len(YEAR_RE.findall(full))
            value_sc = min(1.0, val_hits * 0.18 + stat_hits * 0.25)

            payoff_hits = len(PAYOFF_CLIMAX_RE.findall(closing))
            topic_end_bonus = 0.35 if topic_done else 0.0
            topic_start_bonus = 0.18 if is_explicit_topic_start else 0.0
            climax_sc = min(1.0, 0.35 + topic_end_bonus + topic_start_bonus + payoff_hits * 0.25)

            wps = wc / max(1.0, dur)
            pace_sc = max(0.0, min(1.0, 1.0 - abs(wps - 2.65) / 2.1))
            audio_sc = _window_audio_energy(st, raw_et)
            vis_sc = _window_scene_variety(st, raw_et)

            intro_hits = len(INTRO_TEASER_RE.findall(full))
            intro_pen = min(0.65, intro_hits * 0.28)

            raw_w = (
                0.26 * hook_sc
                + 0.24 * value_sc
                + 0.32 * climax_sc
                + 0.08 * pace_sc
                + 0.05 * audio_sc
                + 0.05 * vis_sc
            ) - intro_pen
            total_score = round(max(0.5, raw_w) * 10.0, 2)

            next_limit = float(next_seg["start"]) - 0.05 if next_seg else (total_dur or raw_et + 0.80)
            snapped_end = min(next_limit, raw_et + 0.80)
            if total_dur > 0:
                snapped_end = min(total_dur, snapped_end)

            cand = {
                "start": round(max(0.0, st - 0.15), 2),
                "end": round(snapped_end, 2),
                "score": total_score,
                "hook": _short_upper(opening, 54),
                "title": re.sub(r"[^a-zA-Z0-9]+", "_", opening[:38]).strip("_").lower() or f"clip_{i}",
                "reason": "Topic Complete" if topic_done else "Sentence Complete",
            }
            if best_for_start is None or total_score > best_for_start["score"]:
                best_for_start = cand

            if topic_done:
                break
            if dur >= max_dur and sent_done:
                break

        if best_for_start and best_for_start["score"] >= 4.2:
            cands.append(best_for_start)

    cands.sort(key=lambda c: c["score"], reverse=True)
    selected: List[Dict[str, Any]] = []
    min_gap = 12.0
    for c in cands:
        overlaps = any(not (c["end"] + min_gap <= p["start"] or c["start"] >= p["end"] + min_gap) for p in selected)
        if not overlaps:
            selected.append(c)
        if len(selected) >= max_clips:
            break

    selected.sort(key=lambda c: c["score"], reverse=True)
    return selected


def _infer_broll_query(text: str, top_keywords: List[str]) -> str | None:
    low = text.lower()
    for key, query in VISUAL_MAP.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            return query
    return None


def build_local_edit_plan(
    words_data: Dict[str, Any],
    vision_info: Dict[str, Any] | None,
    audio_info: Dict[str, Any] | None,
    mode: str = "hybrid",
    hook_override: str | None = None,
) -> Dict[str, Any]:
    dur = max(1.0, float(words_data.get("duration") or (vision_info or {}).get("meta", {}).get("duration") or 30.0))
    segments = words_data.get("segments", [])
    full_text = words_data.get("full_text", "")
    enriched_scenes = classify_all_scenes_deep(vision_info, segments, audio_info)
    emph_peaks = (audio_info or {}).get("emphasis_peaks", [])

    top_kw = [w for w, _ in Counter(_tokens(full_text)).most_common(12)]
    main_topic = " ".join(top_kw[:3]).title() if top_kw else "DOCUMENTARY"

    has_podcast = any(s.get("scene_type") == "PODCAST" for s in enriched_scenes)
    if mode == "auto":
        style = "hybrid" if (has_podcast and segments) else ("podcast" if has_podcast else "documentary")
    else:
        style = mode

    overlays: List[Dict[str, Any]] = []
    broll_tracks: List[Dict[str, Any]] = []
    punch_zooms: List[Dict[str, Any]] = []

    opening_combo = " ".join(s["text"] for s in segments[:3]) if segments else main_topic
    hook_txt = hook_override or opening_combo
    if hook_txt.strip():
        overlays.append({
            "id": "OV_01",
            "type": "hook_card",
            "start": 0.0,
            "end": round(min(2.9, dur * 0.22), 3),
            "sfx": "boom",
            "data": {
                "headline": _short_upper(hook_txt, 40),
                "subtext": _short_upper(main_topic, 26),
            },
        })

    last_ov_end = overlays[-1]["end"] if overlays else 2.2
    used_types: set[str] = set()
    used_facts: set[str] = set()

    if style == "documentary" and not enriched_scenes:
        t_cur = 0.0
        idx = 0
        while t_cur < dur - 0.5:
            t_end = min(dur, t_cur + 4.5)
            seg_txt = " ".join(s["text"] for s in segments if s["end"] > t_cur and s["start"] < t_end) or full_text
            q = _infer_broll_query(seg_txt, top_kw) or (f"{top_kw[idx % len(top_kw)]} documentary" if top_kw else "cinematic abstract dark")
            broll_tracks.append({
                "id": f"BR_{len(broll_tracks) + 1:02d}",
                "start": round(t_cur, 3),
                "end": round(t_end, 3),
                "query": q,
                "transition": "crossfade",
                "sfx": "whoosh" if t_cur > 0 else "none",
                "asset_path": None,
                "asset_type": None,
            })
            t_cur = t_end
            idx += 1

    for s in segments:
        st, et, txt = float(s["start"]), float(s["end"]), str(s["text"]).strip()
        if st < last_ov_end + 2.4 or st > dur - 3.6:
            continue

        ev_end = round(min(dur - 2.4, st + min(3.5, max(2.4, et - st))), 3)
        if ev_end - st < 1.6:
            continue

        low = txt.lower()
        placed = False

        sm = STAT_RE.search(txt)
        if sm and sm.group("val").lower() not in used_facts:
            val = sm.group("val").upper()
            used_facts.add(val.lower())
            if "chart_3d" not in used_types and "%" in val:
                used_types.add("chart_3d")
                num_m = re.search(r"(\d+(?:\.\d+)?)", val)
                pct_val = min(100.0, float(num_m.group(1))) if num_m else 75.0
                overlays.append({
                    "id": f"OV_{len(overlays) + 1:02d}",
                    "type": "chart_3d",
                    "start": round(st, 3),
                    "end": ev_end,
                    "sfx": "pop",
                    "data": {
                        "title": _short_upper(txt, 30),
                        "bars": [
                            {"label": "BEFORE", "value": max(12.0, round(100.0 - pct_val, 1))},
                            {"label": "TARGET", "value": pct_val},
                        ],
                    },
                })
            else:
                overlays.append({
                    "id": f"OV_{len(overlays) + 1:02d}",
                    "type": "stat_callout",
                    "start": round(st, 3),
                    "end": ev_end,
                    "sfx": "pop",
                    "data": {"value": val, "label": _short_upper(txt, 34)},
                })
            last_ov_end = ev_end
            placed = True

        if not placed:
            for loc, (mx, my) in GEO_COORDS.items():
                if re.search(rf"\b{re.escape(loc)}\b", low) and loc not in used_facts:
                    used_facts.add(loc)
                    overlays.append({
                        "id": f"OV_{len(overlays) + 1:02d}",
                        "type": "globe_3d",
                        "start": round(st, 3),
                        "end": ev_end,
                        "sfx": "pop",
                        "data": {"location": loc.title(), "map_x": mx, "map_y": my, "label": _short_upper(txt, 30)},
                    })
                    last_ov_end = ev_end
                    placed = True
                    break

        if not placed and COMP_RE.search(txt) and "two_column" not in used_types:
            parts = re.split(r"\b(?:versus|vs\.?|compared to|instead of|rather than)\b", txt, maxsplit=1, flags=re.I)
            left_t = _short_upper(parts[0], 22) if len(parts) == 2 and parts[0].strip() else "OLD WAY"
            right_t = _short_upper(parts[1], 22) if len(parts) == 2 and parts[1].strip() else "NEW METHOD"
            used_types.add("two_column")
            overlays.append({
                "id": f"OV_{len(overlays) + 1:02d}",
                "type": "two_column",
                "start": round(st, 3),
                "end": ev_end,
                "sfx": "pop",
                "data": {"left_title": left_t, "right_title": right_t},
            })
            last_ov_end = ev_end
            placed = True

        if not placed and not enriched_scenes and CARTOON_EXPLAINER_RE.search(txt) and "cartoon_explainer" not in used_types:
            used_types.add("cartoon_explainer")
            overlays.append({
                "id": f"OV_{len(overlays) + 1:02d}",
                "type": "cartoon_explainer",
                "start": round(st, 3),
                "end": ev_end,
                "sfx": "pop",
                "data": {"bubble": _short_upper(txt, 34), "tag": "CONCEPT VISUAL"},
            })
            last_ov_end = ev_end
            placed = True

        if not placed and not enriched_scenes and VALUE_INSIGHT_RE.search(txt) and "article_cutout" not in used_types:
            used_types.add("article_cutout")
            overlays.append({
                "id": f"OV_{len(overlays) + 1:02d}",
                "type": "article_cutout",
                "start": round(st, 3),
                "end": ev_end,
                "sfx": "pop",
                "data": {
                    "source": "KEY INSIGHT",
                    "headline": _short_upper(txt, 46),
                    "highlight": " ".join(_tokens(txt)[:3]).upper(),
                },
            })
            last_ov_end = ev_end
            placed = True

        if not placed and not enriched_scenes and HOOK_RE.search(txt) and (st - last_ov_end) >= 5.5:
            overlays.append({
                "id": f"OV_{len(overlays) + 1:02d}",
                "type": "kinetic_quote",
                "start": round(st, 3),
                "end": ev_end,
                "sfx": "pop",
                "data": {"quote": " ".join(txt.split()[:8])},
            })
            last_ov_end = ev_end

    for pk in emph_peaks:
        if 2.2 <= pk <= dur - 2.2:
            punch_zooms.append({
                "start": round(max(0.0, pk - 0.15), 3),
                "end": round(min(dur, pk + 1.6), 3),
                "scale": 1.12,
            })

    if style == "documentary" and not enriched_scenes and dur >= 6.0:
        end_st = round(max(last_ov_end + 0.3, dur - 2.4), 3)
        if dur - end_st >= 1.2:
            overlays.append({
                "id": f"OV_{len(overlays) + 1:02d}",
                "type": "ending_card",
                "start": end_st,
                "end": round(dur, 3),
                "sfx": "none",
                "data": {"headline": "FOLLOW FOR MORE", "subtext": _short_upper(main_topic, 28)},
            })

    return {
        "engine": "local_advance_ai_v3_topic_complete",
        "duration": round(dur, 3),
        "style": style,
        "main_topic": main_topic,
        "emphasis_words": top_kw[:10],
        "scenes_classified": enriched_scenes,
        "broll_tracks": broll_tracks,
        "overlays": overlays,
        "punch_zooms": punch_zooms,
    }
