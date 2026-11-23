from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np


def probe_video(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()}")
    data = json.loads(p.stdout)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not v:
        raise RuntimeError("No video stream found")

    def parse_fr(fr: str | None) -> float:
        if not fr or fr == "0/0":
            return 0.0
        if "/" in fr:
            n, d = fr.split("/", 1)
            return float(n) / float(d) if float(d) else 0.0
        return float(fr)

    fps = parse_fr(v.get("avg_frame_rate")) or parse_fr(v.get("r_frame_rate")) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0
    dur = float(v.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": round(fps, 3),
        "duration": round(dur, 3),
        "has_audio": a is not None,
    }


def analyze_audio_local(media_path: str, step_sec: float = 0.25) -> Dict[str, Any]:
    sr = 16000
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", media_path,
        "-vn", "-ac", "1", "-ar", str(sr),
        "-af", "highpass=f=300,lowpass=f=3400",
        "-f", "f32le", "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or len(p.stdout) < 64:
        return {"has_audio": False, "duration": 0.0, "rms_times": [], "rms": [], "emphasis_peaks": [], "speech_ratio": 0.0}

    samples = np.nan_to_num(np.frombuffer(p.stdout, dtype=np.float32), nan=0.0)
    dur = len(samples) / float(sr)
    chunk = max(160, int(sr * step_sec))
    n_chunks = max(1, len(samples) // chunk)

    rms = np.zeros(n_chunks, dtype=np.float32)
    times = np.zeros(n_chunks, dtype=np.float64)
    for i in range(n_chunks):
        seg = samples[i * chunk:(i + 1) * chunk]
        rms[i] = float(np.sqrt(np.mean(seg * seg)))
        times[i] = (i + 0.5) * step_sec

    noise_floor = float(np.percentile(rms, 15))
    peak_95 = float(np.percentile(rms, 95)) + 1e-6
    speech_thresh = max(0.008, noise_floor * 2.2)
    is_speech = rms >= speech_thresh

    emph_thresh = noise_floor + (peak_95 - noise_floor) * 0.80
    peaks: List[float] = []
    last_pk = -10.0
    for i in range(1, n_chunks - 1):
        if rms[i] >= emph_thresh and rms[i] >= rms[i - 1] and rms[i] >= rms[i + 1]:
            t = float(times[i])
            if t >= 3.5 and (t - last_pk) >= 6.0 and t <= dur - 3.0:
                peaks.append(round(t, 2))
                last_pk = t

    return {
        "has_audio": True,
        "duration": round(dur, 3),
        "rms_times": [round(float(t), 3) for t in times],
        "rms": [round(float(v), 5) for v in rms],
        "speech_thresh": round(speech_thresh, 5),
        "speech_ratio": round(float(np.mean(is_speech)), 3),
        "emphasis_peaks": peaks[:10],
    }


def _integral_box_sum(grid: np.ndarray, bh: int, bw: int) -> np.ndarray:
    ii = np.pad(grid, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return ii[bh:, bw:] - ii[:-bh, bw:] - ii[bh:, :-bw] + ii[:-bh, :-bw]


def _detect_face_candidates(rgb: np.ndarray) -> List[Dict[str, float]]:
    """
    High-precision face detector:
    - Rejects beige/tan doors & walls (doors lack r-b warmth & dark eye/hair features)
    - Keeps solo vloggers full-screen centered even when they sit off-center.
    """
    h, w, _ = rgb.shape
    roi_h = int(h * 0.72)
    sub = rgb[:roi_h, :, :]

    r = sub[:, :, 0].astype(np.float32)
    g = sub[:, :, 1].astype(np.float32)
    b = sub[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    skin = (
        (y > 58.0) & (y < 225.0)
        & (cb >= 84.0) & (cb <= 119.0)
        & (cr >= 137.0) & (cr <= 165.0)
        & (r > 85.0) & (g > 45.0) & (b > 25.0)
        & (r > g) & (g > b)
        & ((r - g) >= 15.0) & ((r - g) <= 66.0)
        & ((r - b) >= 26.0)
    ).astype(np.float32)

    dark_feat = (y < 95.0).astype(np.float32)
    gy = np.abs(np.diff(y, axis=0, prepend=y[:1, :]))
    edge_feat = ((gy > 11.0) & (y < 175.0)).astype(np.float32)

    bw, bh = 38, 50
    if roi_h <= bh or w <= bw:
        return []

    area = float(bw * bh)
    skin_sum = _integral_box_sum(skin, bh, bw) / area
    dark_sum = _integral_box_sum(dark_feat, bh, bw) / area
    edge_sum = _integral_box_sum(edge_feat, bh, bw) / area

    valid = (skin_sum >= 0.36) & (dark_sum >= 0.08) & (edge_sum >= 0.07)
    score_map = np.where(valid, skin_sum * (1.0 + 2.2 * np.minimum(edge_sum, 0.30)), 0.0)

    candidates: List[Dict[str, float]] = []
    work = score_map.copy()
    for _ in range(3):
        idx = int(np.argmax(work))
        best_sc = float(work.ravel()[idx])
        if best_sc < 0.48:
            break
        py, px = divmod(idx, work.shape[1])
        cx = (px + bw * 0.5) / float(w)
        cy = (py + bh * 0.5) / float(h)
        candidates.append({"cx": cx, "cy": cy, "score": best_sc})
        y0, y1 = max(0, py - bh), min(work.shape[0], py + bh + 1)
        x0, x1 = max(0, px - bw), min(work.shape[1], px + bw + 1)
        work[y0:y1, x0:x1] = 0.0

    return candidates


def _classify_frame_faces(rgb: np.ndarray) -> Dict[str, Any]:
    faces = _detect_face_candidates(rgb)
    if len(faces) >= 2:
        top_sc = faces[0]["score"]
        sec_sc = faces[1]["score"]
        f1, f2 = sorted(faces[:2], key=lambda f: f["cx"])
        if (
            sec_sc >= 0.58
            and (sec_sc / max(1e-5, top_sc)) >= 0.80
            and (f1["cx"] < 0.35)
            and (f2["cx"] > 0.65)
            and abs(f1["cy"] - f2["cy"]) <= 0.14
        ):
            return {"kind": "TWO", "left": (f1["cx"], f1["cy"]), "right": (f2["cx"], f2["cy"])}
    if len(faces) >= 1:
        top = faces[0]
        return {"kind": "ONE", "center": (top["cx"], top["cy"])}
    return {"kind": "NONE"}


def _build_stabilized_track(ones: List[Dict[str, Any]], med_cx: float, med_cy: float) -> List[Tuple[float, float, float]]:
    inliers = [
        s for s in ones
        if abs(s["center"][0] - med_cx) <= 0.16 and abs(s["center"][1] - med_cy) <= 0.16
    ]
    pts = inliers if len(inliers) >= 2 else ones
    if not pts:
        return []

    smooth_x = med_cx
    smooth_y = med_cy
    track: List[Tuple[float, float, float]] = []
    for s in pts:
        tx = max(med_cx - 0.08, min(med_cx + 0.08, float(s["center"][0])))
        ty = max(med_cy - 0.06, min(med_cy + 0.06, float(s["center"][1])))
        smooth_x = smooth_x * 0.72 + tx * 0.28
        smooth_y = smooth_y * 0.72 + ty * 0.28
        track.append((round(float(s["t"]), 3), round(smooth_x, 4), round(smooth_y, 4)))
    return track


def analyze_video_local(video_path: str, audio_info: Dict[str, Any], sample_fps: float = 5.0) -> Dict[str, Any]:
    meta = probe_video(video_path)
    aw, ah = 320, 180
    frame_bytes = aw * ah * 3

    cmd = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-vf", f"fps={sample_fps},scale={aw}:{ah}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    scenes: List[Dict[str, Any]] = []
    curr: Dict[str, Any] = {"start_time": 0.0, "samples": [], "motions": []}
    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    idx = 0

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            t = idx / sample_fps
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape((ah, aw, 3))
            gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)

            q = (rgb[:, :, 0] // 64) * 16 + (rgb[:, :, 1] // 64) * 4 + (rgb[:, :, 2] // 64)
            hist = np.bincount(q.ravel(), minlength=64).astype(np.float32)
            hist /= max(1.0, float(np.sum(hist)))

            is_cut = False
            frame_mad = 0.0
            if prev_gray is not None and prev_hist is not None:
                diff = np.abs(gray - prev_gray)
                frame_mad = float(np.mean(diff))
                hist_l1 = float(np.sum(np.abs(hist - prev_hist))) * 0.5
                active = (gray > 18.0) | (prev_gray > 18.0)
                active_mad = float(np.mean(diff[active])) if np.any(active) else frame_mad

                g1, g2 = gray.ravel(), prev_gray.ravel()
                s1, s2 = float(np.std(g1)), float(np.std(g2))
                ncc = float(np.mean((g1 - np.mean(g1)) * (g2 - np.mean(g2))) / (s1 * s2)) if (s1 > 1 and s2 > 1) else 1.0

                is_cut = (hist_l1 > 0.36) or (ncc < 0.64 and active_mad > 22.0) or (active_mad > 42.0)

            if is_cut and (t - curr["start_time"]) >= 0.65:
                curr["end_time"] = round(t, 3)
                scenes.append(curr)
                curr = {"start_time": round(t, 3), "samples": [], "motions": []}

            subj = _classify_frame_faces(rgb)
            subj["t"] = round(t, 3)
            curr["samples"].append(subj)
            curr["motions"].append(frame_mad)

            prev_gray, prev_hist = gray, hist
            idx += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()

    curr["end_time"] = round(max(meta["duration"], idx / sample_fps), 3)
    scenes.append(curr)

    out_scenes = []
    for s_id, sc in enumerate(scenes):
        samps = sc["samples"]
        sc_dur = float(sc["end_time"]) - float(sc["start_time"])
        med_motion = float(np.median(sc["motions"])) if sc["motions"] else 0.0
        total = max(1, len(samps))
        twos = [s for s in samps if s["kind"] == "TWO"]
        ones = [s for s in samps if s["kind"] == "ONE"]

        # Only split on genuine static 2-person podcast shots (>= 82% TWO frames)
        if sc_dur >= 4.0 and med_motion < 6.0 and len(twos) >= max(14, int(total * 0.82)):
            lcx = float(np.median([s["left"][0] for s in twos]))
            lcy = float(np.median([s["left"][1] for s in twos]))
            rcx = float(np.median([s["right"][0] for s in twos]))
            rcy = float(np.median([s["right"][1] for s in twos]))
            out_scenes.append({
                "scene_id": s_id,
                "start_time": sc["start_time"],
                "end_time": sc["end_time"],
                "scene_type": "PODCAST",
                "mode": "SPLIT",
                "left_cx": round(lcx, 4), "left_cy": round(lcy, 4),
                "right_cx": round(rcx, 4), "right_cy": round(rcy, 4),
            })
        elif len(ones) >= max(2, int(total * 0.22)):
            cx = max(0.08, min(0.92, float(np.median([s["center"][0] for s in ones]))))
            cy = max(0.24, min(0.55, float(np.median([s["center"][1] for s in ones]))))
            track = _build_stabilized_track(ones, cx, cy)
            out_scenes.append({
                "scene_id": s_id,
                "start_time": sc["start_time"],
                "end_time": sc["end_time"],
                "scene_type": "TALKING_HEAD",
                "mode": "SINGLE",
                "cx": round(cx, 4), "cy": round(cy, 4),
                "track": track,
            })
        else:
            out_scenes.append({
                "scene_id": s_id,
                "start_time": sc["start_time"],
                "end_time": sc["end_time"],
                "scene_type": "BROLL",
                "mode": "SINGLE",
                "cx": 0.50, "cy": 0.45,
            })

    return {"meta": meta, "scenes": out_scenes}
