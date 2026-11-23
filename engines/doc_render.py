from __future__ import annotations

import math
import os
import re
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}


def _ease_out(t: float) -> float:
    x = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - x) ** 3


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


import glob
import urllib.request

_FONTS_DIR = os.path.expanduser("~/doc-studio/assets/fonts")
_DEV_FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"

_DEV_VOWELS = {"अ":"a","आ":"aa","इ":"i","ई":"ee","उ":"u","ऊ":"oo","ऋ":"ri","ए":"e","ऐ":"ai","ओ":"o","औ":"au","अं":"an","अः":"ah"}
_DEV_MATRAS = {"ा":"a","ि":"i","ी":"ee","ु":"u","ू":"oo","ॉ":"o","ृ":"ri","े":"e","ै":"ai","ो":"o","ौ":"au","ं":"n","ँ":"n","ः":"h","्":"","़":""}
_DEV_CONS = {"क":"k","ख":"kh","ग":"g","घ":"gh","ङ":"ng","च":"ch","छ":"chh","ज":"j","झ":"jh","ञ":"ny","ट":"t","ठ":"th","ड":"d","ढ":"dh","ण":"n","त":"t","थ":"th","द":"d","ध":"dh","न":"n","प":"p","फ":"ph","ब":"b","भ":"bh","म":"m","य":"y","र":"r","ल":"l","व":"v","श":"sh","ष":"sh","स":"s","ह":"h","ळ":"l","क़":"q","ख़":"kh","ग़":"gh","ज़":"z","ड़":"r","ढ़":"rh","फ़":"f"}
_HAS_DEV_FONT: bool | None = None

def _ensure_devanagari_font() -> str | None:
    os.makedirs(_FONTS_DIR, exist_ok=True)
    local_ttf = os.path.join(_FONTS_DIR, "NotoSansDevanagari.ttf")
    if os.path.exists(local_ttf) and os.path.getsize(local_ttf) > 40000:
        return local_ttf
    for pat in ("/system/fonts/*Devanagari*.ttf", "/system/fonts/*Devanagari*.otf", "/data/data/com.termux/files/usr/share/fonts/**/*Devanagari*.ttf"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    try:
        urllib.request.urlretrieve(_DEV_FONT_URL, local_ttf)
        if os.path.exists(local_ttf) and os.path.getsize(local_ttf) > 40000:
            return local_ttf
    except Exception:
        pass
    return None

_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "ᵗ": "t", "ʰ": "h", "ˢ": "s", "ⁿ": "n", "ᵈ": "d", "ʳ": "r",
    "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
})

def _strip_unrenderable(text: str) -> str:
    t = text.translate(_SUPERSCRIPT_MAP)
    cleaned = [
        ch for ch in t
        if (32 <= ord(ch) <= 126) or (0x0900 <= ord(ch) <= 0x097F) or ch in ("\n", "\t")
    ]
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()

def _safe_text(text: str) -> str:
    global _HAS_DEV_FONT
    text = _strip_unrenderable(text)
    if not _has_devanagari(text):
        return text
    if _HAS_DEV_FONT is None:
        _HAS_DEV_FONT = _ensure_devanagari_font() is not None
    if _HAS_DEV_FONT:
        return text
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _DEV_VOWELS:
            out.append(_DEV_VOWELS[ch]); i += 1
        elif ch in _DEV_CONS:
            base = _DEV_CONS[ch]
            nxt = text[i + 1] if (i + 1) < n else ""
            if nxt in _DEV_MATRAS:
                out.append(base + _DEV_MATRAS[nxt]); i += 2
            elif nxt == "्":
                out.append(base); i += 2
            else:
                end_w = (i + 1 >= n) or not ("\u0900" <= nxt <= "\u097F")
                out.append(base if end_w else base + "a"); i += 1
        elif ch in _DEV_MATRAS:
            out.append(_DEV_MATRAS[ch]); i += 1
        elif ch == "।":
            out.append("."); i += 1
        else:
            out.append(ch); i += 1
    return "".join(out).upper()

_FONT_CACHE: Dict[Tuple[int, bool, bool], Any] = {}


def _get_font(size: int, bold: bool = True, hindi: bool = False):
    key = (max(12, int(size)), bold, hindi)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    candidates = []
    if hindi:
        df = _ensure_devanagari_font()
        if df:
            candidates.append(df)
    if bold:
        candidates.extend([
            "/system/fonts/Roboto-Bold.ttf",
            "/system/fonts/NotoSans-Bold.ttf",
            "/system/fonts/DroidSans-Bold.ttf",
            "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ])
    candidates.extend([
        "/system/fonts/Roboto-Regular.ttf",
        "/system/fonts/NotoSans-Regular.ttf",
        "/system/fonts/DroidSans.ttf",
    ])

    font = None
    for fp in candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size=key[0])
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default(size=key[0])
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _fit_single_line(draw: ImageDraw.ImageDraw, text: str, base_size: int, max_w: int, bold: bool = True, hindi: bool = False):
    sz = max(14, int(base_size))
    font = _get_font(sz, bold, hindi)
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    if tw > max_w and tw > 0:
        sz = max(14, int(sz * (max_w / float(tw))))
        font = _get_font(sz, bold, hindi)
    return font


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, max_lines: int = 3) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    curr = words[0]
    for w in words[1:]:
        test = f"{curr} {w}"
        bb = draw.textbbox((0, 0), test, font=font)
        if (bb[2] - bb[0]) <= max_w:
            curr = test
        else:
            lines.append(curr)
            curr = w
            if len(lines) >= max_lines - 1:
                break
    if curr and len(lines) < max_lines:
        lines.append(curr)
    return lines[:max_lines]


class DocumentaryRenderer:
    """Fast Termux-Native Documentary + Cartoon + 3D Renderer."""

    def __init__(self, width: int, height: int, fps: int = 24, color_grade: str = "documentary"):
        self.w = int(width)
        self.h = int(height)
        self.fps = int(fps)
        self.is_vertical = self.h > self.w
        self.scale = min(self.w, self.h) / 1080.0
        self.color_grade = color_grade

        self._proc_bg = self._make_procedural_bg()
        self._broll_cache: Dict[str, Image.Image] = {}
        self._cap_cache_key: Tuple[int, int] | None = None
        self._cap_cache_strip: Image.Image | None = None
        self._ov_static_cache: Dict[str, Image.Image] = {}

    def _s(self, px: float) -> int:
        return max(1, int(round(px * self.scale)))

    def _grade_once(self, img: Image.Image) -> Image.Image:
        if self.color_grade == "none":
            return img
        arr = np.asarray(img, dtype=np.float32)
        if self.color_grade == "documentary":
            arr = (arr - 128.0) * 1.08 + 118.0
            arr[:, :, 0] *= 1.03
            arr[:, :, 2] *= 1.03
            strength = 0.30
        else:
            arr = (arr - 128.0) * 1.04 + 126.0
            strength = 0.14

        ih, iw = arr.shape[:2]
        yy, xx = np.ogrid[:ih, :iw]
        cy, cx = ih * 0.5, iw * 0.5
        d2 = ((xx - cx) / max(1.0, cx)) ** 2 + ((yy - cy) / max(1.0, cy)) ** 2
        vig = np.clip(1.0 - strength * np.maximum(0.0, d2 - 0.35), 0.50, 1.0).astype(np.float32)
        arr *= vig[:, :, None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    def _make_procedural_bg(self) -> Image.Image:
        yy = np.linspace(0.0, 1.0, self.h, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, self.w, dtype=np.float32)[None, :]
        dist = np.sqrt((xx - 0.5) ** 2 + (yy - 0.38) ** 2)
        glow = np.clip(1.0 - dist * 1.35, 0.0, 1.0)
        r = np.clip(8 + glow * 18, 0, 255).astype(np.uint8)
        g = np.clip(14 + glow * 32, 0, 255).astype(np.uint8)
        b = np.clip(26 + glow * 54, 0, 255).astype(np.uint8)
        img = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
        draw = ImageDraw.Draw(img)
        step = self._s(72)
        for gx in range(0, self.w, step):
            draw.line([(gx, 0), (gx, self.h)], fill=(22, 34, 54), width=1)
        for gy in range(0, self.h, step):
            draw.line([(0, gy), (self.w, gy)], fill=(22, 34, 54), width=1)
        return img

    def _get_broll_frame(self, path: str | None, prog: float, alt_dir: int = 1) -> Image.Image:
        if not path or not os.path.exists(path):
            return self._proc_bg.copy()

        if path not in self._broll_cache:
            try:
                im = Image.open(path).convert("RGB")
                pad_w, pad_h = int(self.w * 1.12), int(self.h * 1.12)
                sc = max(pad_w / float(im.width), pad_h / float(im.height))
                nw, nh = max(pad_w, int(round(im.width * sc))), max(pad_h, int(round(im.height * sc)))
                scaled = im.resize((nw, nh), Image.Resampling.BILINEAR)
                self._broll_cache[path] = self._grade_once(scaled)
            except Exception:
                return self._proc_bg.copy()

        src = self._broll_cache[path]
        zoom = 1.0 + 0.08 * max(0.0, min(1.0, prog))
        cw = min(src.width, max(64, int(round(self.w / zoom * 1.09))))
        ch = min(src.height, max(64, int(round(self.h / zoom * 1.09))))
        max_dx = max(0, src.width - cw)
        pan_frac = (0.35 + 0.30 * prog) if alt_dir > 0 else (0.65 - 0.30 * prog)
        x0 = int(round(max_dx * pan_frac))
        y0 = (src.height - ch) // 2
        return src.crop((x0, y0, x0 + cw, y0 + ch)).resize((self.w, self.h), Image.Resampling.BILINEAR)

    def _frame_source_video(self, src_img: Image.Image, scene: Dict[str, Any], t: float, punch_zooms: List[Dict[str, Any]]) -> Image.Image:
        sw, sh = src_img.size
        mode = scene.get("mode", "SINGLE")

        zoom = 1.0
        for pz in punch_zooms:
            if pz["start"] <= t <= pz["end"]:
                p = (t - pz["start"]) / max(0.2, pz["end"] - pz["start"])
                zoom = 1.0 + (float(pz.get("scale", 1.12)) - 1.0) * math.sin(math.pi * min(1.0, p))
                break

        if mode == "SPLIT" and self.is_vertical:
            half_h = self.h // 2
            lcx = float(scene.get("left_cx", 0.25)) * sw
            lcy = float(scene.get("left_cy", 0.38)) * sh
            rcx = float(scene.get("right_cx", 0.75)) * sw
            rcy = float(scene.get("right_cy", 0.38)) * sh
            mid_x = max(64, min(sw - 64, int(round((lcx + rcx) * 0.5))))

            top = self._crop_sub(src_img.crop((0, 0, mid_x, sh)), lcx, lcy, self.w, half_h)
            bot = self._crop_sub(src_img.crop((mid_x, 0, sw, sh)), rcx - mid_x, rcy, self.w, self.h - half_h)
            canvas = Image.new("RGB", (self.w, self.h), (12, 12, 12))
            canvas.paste(top, (0, 0))
            canvas.paste(bot, (0, half_h))
            ImageDraw.Draw(canvas).line([(0, half_h), (self.w, half_h)], fill=(14, 14, 14), width=8)
            return canvas

        cx_n = float(scene.get("cx", 0.5))
        cy_n = float(scene.get("cy", 0.38))
        track = scene.get("track") or []
        if track:
            p0, p1 = track[0], track[-1]
            for pt in track:
                if pt[0] <= t: p0 = pt
                if pt[0] >= t: p1 = pt; break
            dt = p1[0] - p0[0]
            if dt > 1e-4:
                f_ = max(0.0, min(1.0, (t - p0[0]) / dt))
                cx_n = p0[1] + (p1[1] - p0[1]) * f_
                cy_n = p0[2] + (p1[2] - p0[2]) * f_
            else:
                cx_n, cy_n = p0[1], p0[2]
        cx = cx_n * sw
        cy = cy_n * sh
        st = float(scene.get("start_time", 0.0))
        et = float(scene.get("end_time", st + 5.0))
        subtle = 1.0 + 0.05 * max(0.0, min(1.0, (t - st) / max(0.5, et - st)))
        return self._crop_sub(src_img, cx, cy, self.w, self.h, zoom=max(zoom, subtle))

    @staticmethod
    def _crop_sub(sub: Image.Image, cx: float, cy: float, out_w: int, out_h: int, zoom: float = 1.0) -> Image.Image:
        sw, sh = sub.size
        aspect = out_w / float(out_h)
        base_h = float(sh)
        base_w = base_h * aspect
        if base_w > sw:
            base_w, base_h = float(sw), float(sw) / aspect
        cw = min(sw, max(32, int(round(base_w / max(1.0, zoom)))))
        ch = min(sh, max(32, int(round(base_h / max(1.0, zoom)))))
        x0 = max(0, min(sw - cw, int(round(cx - cw * 0.5))))
        y0 = max(0, min(sh - ch, int(round(cy - ch * 0.38))))
        return sub.crop((x0, y0, x0 + cw, y0 + ch)).resize((out_w, out_h), Image.Resampling.BILINEAR)

    def _draw_globe_3d(self, d: ImageDraw.ImageDraw, data: Dict[str, Any], p: float, enter: float, am: float) -> None:
        loc = str(data.get("location", "GLOBAL")).upper()
        px = self._s(56)
        bw, bh = self.w - px * 2, self._s(340)
        y0 = int(self.h * 0.11)

        d.rounded_rectangle(
            [px, y0, px + bw, y0 + bh], radius=self._s(20),
            fill=(8, 14, 28, int(234 * am)), outline=(56, 189, 248, int(230 * am)), width=self._s(3),
        )
        f_hdr = _fit_single_line(d, f"3D GEO FOCUS: {loc}", self._s(26), bw - self._s(48), True, False)
        d.text((px + self._s(24), y0 + self._s(16)), f"3D GEO FOCUS: {loc}", font=f_hdr, fill=(245, 196, 48, int(255 * am)))

        gcx = px + bw // 2
        gcy = y0 + self._s(195)
        radius = self._s(112) * enter
        yaw = p * 1.45
        tilt = 0.32

        def proj(lat: float, lon: float) -> Tuple[float, float, float]:
            x = math.cos(lat) * math.sin(lon + yaw)
            y = math.sin(lat)
            z = math.cos(lat) * math.cos(lon + yaw)
            y2 = y * math.cos(tilt) - z * math.sin(tilt)
            z2 = y * math.sin(tilt) + z * math.cos(tilt)
            return gcx + x * radius, gcy - y2 * radius, z2

        d.ellipse([gcx - radius, gcy - radius, gcx + radius, gcy + radius],
                  fill=(14, 32, 62, int(220 * am)), outline=(96, 165, 250, int(240 * am)), width=self._s(3))

        for m_i in range(8):
            lon = (m_i / 8.0) * math.pi * 2.0
            pts = []
            for s_i in range(25):
                lat = -math.pi * 0.5 + (s_i / 24.0) * math.pi
                sx, sy, sz = proj(lat, lon)
                if sz >= -0.05:
                    pts.append((sx, sy))
                else:
                    if len(pts) >= 2:
                        d.line(pts, fill=(56, 189, 248, int(150 * am)), width=self._s(2))
                    pts = []
            if len(pts) >= 2:
                d.line(pts, fill=(56, 189, 248, int(150 * am)), width=self._s(2))

        for lat_deg in (-50, -25, 0, 25, 50):
            lat = math.radians(lat_deg)
            pts = []
            for s_i in range(33):
                lon = (s_i / 32.0) * math.pi * 2.0
                sx, sy, sz = proj(lat, lon)
                if sz >= -0.05:
                    pts.append((sx, sy))
                else:
                    if len(pts) >= 2:
                        col = (245, 196, 48, int(190 * am)) if lat_deg == 0 else (56, 189, 248, int(130 * am))
                        d.line(pts, fill=col, width=self._s(2))
                    pts = []
            if len(pts) >= 2:
                col = (245, 196, 48, int(190 * am)) if lat_deg == 0 else (56, 189, 248, int(130 * am))
                d.line(pts, fill=col, width=self._s(2))

        pin_x, pin_y, _ = proj(math.radians(20), -yaw + 0.25)
        pulse = (p * 3.0) % 1.0
        pr = self._s(10 + 28 * pulse)
        d.ellipse([pin_x - pr, pin_y - pr, pin_x + pr, pin_y + pr],
                  outline=(245, 196, 48, int(230 * (1.0 - pulse) * am)), width=self._s(3))
        r = self._s(9)
        d.ellipse([pin_x - r, pin_y - r, pin_x + r, pin_y + r],
                  fill=(244, 63, 94, int(255 * am)), outline=(255, 255, 255, int(255 * am)), width=self._s(2))

    def _draw_chart_3d(self, d: ImageDraw.ImageDraw, data: Dict[str, Any], p: float, enter: float, am: float) -> None:
        title = str(data.get("title", "DATA IMPACT")).upper()
        bars = data.get("bars", [{"label": "BEFORE", "value": 25}, {"label": "AFTER", "value": 85}])
        px = self._s(58)
        bw, bh = self.w - px * 2, self._s(310)
        y0 = int(self.h * 0.12)

        d.rounded_rectangle(
            [px, y0, px + bw, y0 + bh], radius=self._s(20),
            fill=(10, 15, 28, int(232 * am)), outline=(56, 189, 248, int(230 * am)), width=self._s(3),
        )
        f_t = _fit_single_line(d, title, self._s(26), bw - self._s(48), True, _has_devanagari(title))
        d.text((px + self._s(24), y0 + self._s(18)), title, font=f_t, fill=(245, 196, 48, int(255 * am)))

        grow = _ease_out(min(1.0, p / 0.50))
        max_val = max(float(b.get("value", 1.0)) for b in bars) or 1.0
        base_y = y0 + bh - self._s(56)
        max_bar_h = self._s(165)
        bar_w = self._s(76)
        depth = self._s(22)

        colors = [
            ((71, 85, 105), (100, 116, 139), (51, 65, 85)),
            ((245, 158, 11), (253, 224, 71), (180, 83, 9)),
        ]
        n = len(bars)
        for idx, b in enumerate(bars[:3]):
            val = float(b.get("value", 50.0))
            lbl = str(b.get("label", ""))[:12].upper()
            cur_h = max(self._s(14), int(max_bar_h * (val / max_val) * grow))
            cx = px + int(bw * (idx + 1) / float(n + 1)) - bar_w // 2
            cy = base_y - cur_h
            c_front, c_top, c_side = colors[idx % len(colors)]

            d.rectangle([cx, cy, cx + bar_w, base_y], fill=(*c_front, int(245 * am)))
            d.polygon([(cx, cy), (cx + depth, cy - depth), (cx + bar_w + depth, cy - depth), (cx + bar_w, cy)],
                      fill=(*c_top, int(250 * am)))
            d.polygon([(cx + bar_w, cy), (cx + bar_w + depth, cy - depth), (cx + bar_w + depth, base_y - depth), (cx + bar_w, base_y)],
                      fill=(*c_side, int(245 * am)))

            val_str = f"{int(round(val * grow))}%"
            f_v = _get_font(self._s(26), True, False)
            vb = d.textbbox((0, 0), val_str, font=f_v)
            d.text((cx + (bar_w - (vb[2] - vb[0])) // 2, cy - depth - self._s(34)), val_str, font=f_v, fill=(255, 255, 255, int(255 * am)))

            f_l = _get_font(self._s(22), True, False)
            lb = d.textbbox((0, 0), lbl, font=f_l)
            d.text((cx + (bar_w - (lb[2] - lb[0])) // 2, base_y + self._s(10)), lbl, font=f_l, fill=(203, 213, 225, int(245 * am)))

    def _draw_cartoon_explainer(self, d: ImageDraw.ImageDraw, data: Dict[str, Any], p: float, enter: float, am: float) -> None:
        bubble = str(data.get("bubble", "THINK ABOUT THIS")).upper()
        px = self._s(54)
        bw, bh = self.w - px * 2, self._s(250)
        y0 = int(self.h * 0.12) + int((1.0 - enter) * self._s(35))

        d.rounded_rectangle(
            [px, y0, px + bw, y0 + bh], radius=self._s(22),
            fill=(14, 20, 34, int(232 * am)), outline=(245, 196, 48, int(230 * am)), width=self._s(3),
        )

        bob = int(math.sin(p * math.pi * 6.0) * self._s(5))
        cx = px + self._s(110)
        cy = y0 + self._s(125) + bob
        head_r = self._s(48)

        d.pieslice([cx - self._s(62), cy + self._s(36), cx + self._s(62), cy + self._s(135)],
                   start=180, end=360, fill=(56, 189, 248, int(250 * am)))
        d.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r],
                  fill=(253, 224, 171, int(255 * am)), outline=(30, 41, 59, int(255 * am)), width=self._s(3))
        d.pieslice([cx - head_r, cy - head_r - self._s(6), cx + head_r, cy + self._s(12)],
                   start=185, end=355, fill=(30, 41, 59, int(255 * am)))

        blink = (int(p * 12) % 7 == 0)
        eye_y = cy - self._s(6)
        for ex in (cx - self._s(17), cx + self._s(17)):
            if blink:
                d.line([ex - self._s(7), eye_y, ex + self._s(7), eye_y], fill=(15, 23, 42, int(255 * am)), width=self._s(3))
            else:
                d.ellipse([ex - self._s(7), eye_y - self._s(8), ex + self._s(7), eye_y + self._s(8)], fill=(255, 255, 255, int(255 * am)))
                d.ellipse([ex - self._s(3), eye_y - self._s(4), ex + self._s(4), eye_y + self._s(4)], fill=(15, 23, 42, int(255 * am)))

        mouth_open = abs(math.sin(p * math.pi * 14.0))
        mh = self._s(4 + 11 * mouth_open)
        mw = self._s(14)
        d.ellipse([cx - mw, cy + self._s(14), cx + mw, cy + self._s(14) + mh], fill=(190, 24, 93, int(255 * am)))

        bx, by = cx + self._s(44), cy - self._s(66)
        br = self._s(15)
        d.ellipse([bx - br, by - br, bx + br, by + br], fill=(250, 204, 21, int(255 * am)), outline=(255, 255, 255, int(240 * am)), width=self._s(2))

        tx0 = px + self._s(195)
        tw_avail = (px + bw - self._s(24)) - tx0
        f_b = _get_font(self._s(30), True, _has_devanagari(bubble))
        lns = _wrap(d, bubble, f_b, tw_avail, 3)
        ty = y0 + (bh - len(lns) * self._s(40)) // 2
        for ln in lns:
            d.text((tx0, ty), ln, font=f_b, fill=(255, 255, 255, int(255 * am)))
            ty += self._s(40)

    def _draw_article_cutout(self, d: ImageDraw.ImageDraw, data: Dict[str, Any], p: float, enter: float, am: float) -> None:
        src = str(data.get("source", "KEY INSIGHT")).upper()
        head = str(data.get("headline", "")).upper()
        px = self._s(58)
        bw = self.w - px * 2
        y0 = int(self.h * 0.13) + int((1.0 - enter) * self._s(30))

        f_s = _get_font(self._s(22), True, False)
        f_h = _get_font(self._s(34), True, _has_devanagari(head))
        lns = _wrap(d, head, f_h, bw - self._s(52), 3)
        bh = self._s(74) + len(lns) * self._s(44)

        d.rounded_rectangle(
            [px, y0, px + bw, y0 + bh], radius=self._s(14),
            fill=(248, 250, 252, int(242 * am)), outline=(203, 213, 225, int(255 * am)), width=self._s(3),
        )
        d.text((px + self._s(24), y0 + self._s(16)), f"INSIGHT • {src}", font=f_s, fill=(100, 116, 139, int(255 * am)))

        sweep = _ease_out(min(1.0, p / 0.55))
        cy = y0 + self._s(52)
        for idx, ln in enumerate(lns):
            bb = d.textbbox((0, 0), ln, font=f_h)
            lw = bb[2] - bb[0]
            if idx == 0 and lw > 0:
                hw = int(lw * sweep)
                d.rectangle([px + self._s(22), cy + self._s(4), px + self._s(26) + hw, cy + self._s(38)],
                            fill=(253, 224, 71, int(225 * am)))
            d.text((px + self._s(24), cy), ln, font=f_h, fill=(15, 23, 42, int(255 * am)))
            cy += self._s(44)

    def _draw_overlay(self, base: Image.Image, ov: Dict[str, Any], prog: float) -> Image.Image:
        p = max(0.0, min(1.0, prog))
        enter = _ease_out(min(1.0, p / 0.18)) if p < 0.18 else 1.0
        exit_a = _ease_out((1.0 - p) / 0.15) if p > 0.85 else 1.0
        am = enter * exit_a
        if am < 0.02:
            return base

        otype = ov.get("type", "")
        ov_id = str(ov.get("id", otype))
        data = ov.get("data", {})

        is_steady = (0.18 <= p <= 0.85) and otype in ("hook_card", "two_column", "kinetic_quote")
        if otype == "ending_card" and p >= 0.18:
            is_steady = True
        if is_steady and ov_id in self._ov_static_cache:
            cached = self._ov_static_cache[ov_id]
            base.paste(cached, (0, 0), cached)
            return base

        band_h = self.h if otype == "ending_card" else int(self.h * 0.52)
        layer = Image.new("RGBA", (self.w, band_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        if otype == "hook_card":
            head = _safe_text(str(data.get("headline", "THE REAL TRUTH"))).upper()
            sub = _safe_text(str(data.get("subtext", "DOCUMENTARY"))).upper()
            hi = _has_devanagari(head + sub)
            px = self._s(58)
            inner_w = self.w - px * 2 - self._s(48)
            f_sub = _fit_single_line(d, sub, self._s(26), inner_w, True, hi)
            f_hd = _get_font(self._s(48 if self.is_vertical else 58), True, hi)
            lines = _wrap(d, head, f_hd, inner_w, 3)
            max_line_w = max((d.textbbox((0, 0), ln, font=f_hd)[2] for ln in lines), default=0)
            if max_line_w > inner_w and max_line_w > 0:
                f_hd = _get_font(max(20, int(self._s(48) * (inner_w / float(max_line_w)))), True, hi)
                lines = _wrap(d, head, f_hd, inner_w, 3)

            lh = self._s(60)
            bh = self._s(82) + len(lines) * lh
            y0 = int(self.h * 0.10) + int((1.0 - enter) * self._s(38))
            d.rounded_rectangle([px, y0, self.w - px, y0 + bh], radius=self._s(18),
                                fill=(10, 14, 24, int(222 * am)), outline=(245, 196, 48, int(235 * am)), width=self._s(3))
            bar_w = int(inner_w * enter)
            d.rectangle([px + self._s(24), y0 + self._s(18), px + self._s(24) + bar_w, y0 + self._s(24)], fill=(245, 196, 48, int(255 * am)))
            d.text((px + self._s(24), y0 + self._s(34)), sub, font=f_sub, fill=(245, 196, 48, int(245 * am)))
            cy = y0 + self._s(72)
            for ln in lines:
                d.text((px + self._s(24), cy), ln, font=f_hd, fill=(255, 255, 255, int(255 * am)))
                cy += lh

        elif otype == "stat_callout":
            raw_v = str(data.get("value", "0"))
            lbl = str(data.get("label", "KEY FIGURE")).upper()
            m = re.match(r"^([^0-9]*)(\d+(?:\.\d+)?)(.*)$", raw_v)
            if m:
                cur = float(m.group(2)) * _ease_out(min(1.0, p / 0.45))
                shown = f"{cur:.1f}" if "." in m.group(2) else f"{int(round(cur))}"
                val_txt = f"{m.group(1)}{shown}{m.group(3)}"
            else:
                val_txt = raw_v
            bw = int(self.w * (0.84 if self.is_vertical else 0.46))
            bh = self._s(235)
            x0 = (self.w - bw) // 2
            y0 = int(self.h * 0.13) + int((1.0 - enter) * self._s(32))
            d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=self._s(22),
                                fill=(10, 15, 26, int(230 * am)), outline=(56, 189, 248, int(235 * am)), width=self._s(4))
            f_v = _fit_single_line(d, val_txt, self._s(86), bw - self._s(48), True, False)
            f_l = _get_font(self._s(28), True, _has_devanagari(lbl))
            vb = d.textbbox((0, 0), val_txt, font=f_v)
            d.text((x0 + (bw - (vb[2] - vb[0])) // 2, y0 + self._s(26)), val_txt, font=f_v, fill=(245, 196, 48, int(255 * am)))
            cy = y0 + self._s(136)
            for ln in _wrap(d, lbl, f_l, bw - self._s(52), 2):
                lb = d.textbbox((0, 0), ln, font=f_l)
                d.text((x0 + (bw - (lb[2] - lb[0])) // 2, cy), ln, font=f_l, fill=(240, 245, 255, int(245 * am)))
                cy += self._s(36)

        elif otype == "two_column":
            lt = str(data.get("left_title", "OLD WAY")).upper()
            rt = str(data.get("right_title", "NEW METHOD")).upper()
            px, gap = self._s(52), self._s(22)
            cw = (self.w - px * 2 - gap) // 2
            ch = self._s(205)
            y0 = int(self.h * 0.13)
            slide = int((1.0 - enter) * self._s(70))
            lx0, rx0 = px - slide, px + cw + gap + slide
            d.rounded_rectangle([lx0, y0, lx0 + cw, y0 + ch], radius=self._s(18), fill=(15, 23, 42, int(230 * am)), outline=(56, 189, 248, int(240 * am)), width=self._s(4))
            d.rounded_rectangle([rx0, y0, rx0 + cw, y0 + ch], radius=self._s(18), fill=(32, 16, 26, int(230 * am)), outline=(244, 63, 94, int(240 * am)), width=self._s(4))
            f_c = _get_font(self._s(30), True, _has_devanagari(lt + rt))
            for bx, txt, col in ((lx0, lt, (186, 230, 253)), (rx0, rt, (254, 205, 211))):
                lns = _wrap(d, txt, f_c, cw - self._s(32), 3)
                cy = y0 + (ch - len(lns) * self._s(38)) // 2
                for ln in lns:
                    bb = d.textbbox((0, 0), ln, font=f_c)
                    d.text((bx + (cw - (bb[2] - bb[0])) // 2, cy), ln, font=f_c, fill=(*col, int(255 * am)))
                    cy += self._s(38)
            cx, cm = self.w // 2, y0 + ch // 2
            r = self._s(30)
            d.ellipse([cx - r, cm - r, cx + r, cm + r], fill=(245, 196, 48, int(255 * am)))
            f_vs = _get_font(self._s(24), True, False)
            vb = d.textbbox((0, 0), "VS", font=f_vs)
            d.text((cx - (vb[2] - vb[0]) // 2, cm - (vb[3] - vb[1]) // 2 - self._s(3)), "VS", font=f_vs, fill=(10, 14, 22, int(255 * am)))

        elif otype in ("globe_3d", "map_pin"):
            self._draw_globe_3d(d, data, p, enter, am)

        elif otype == "chart_3d":
            self._draw_chart_3d(d, data, p, enter, am)

        elif otype == "cartoon_explainer":
            self._draw_cartoon_explainer(d, data, p, enter, am)

        elif otype == "article_cutout":
            self._draw_article_cutout(d, data, p, enter, am)

        elif otype == "kinetic_quote":
            q = _safe_text(str(data.get("quote", ""))).strip()
            if q:
                px = self._s(60)
                bw = self.w - px * 2
                f_q = _get_font(self._s(38), True, _has_devanagari(q))
                lns = _wrap(d, f'"{q}"', f_q, bw - self._s(52), 3)
                bh = self._s(48) + len(lns) * self._s(48)
                y0 = int(self.h * 0.13) + int((1.0 - enter) * self._s(28))
                d.rounded_rectangle([px, y0, px + bw, y0 + bh], radius=self._s(16), fill=(12, 16, 28, int(224 * am)))
                d.rectangle([px, y0, px + self._s(10), y0 + bh], fill=(245, 196, 48, int(255 * am)))
                cy = y0 + self._s(24)
                for ln in lns:
                    d.text((px + self._s(28), cy), ln, font=f_q, fill=(255, 255, 255, int(255 * am)))
                    cy += self._s(48)

        elif otype == "ending_card":
            hd = str(data.get("headline", "FOLLOW FOR MORE")).upper()
            sub = str(data.get("subtext", "")).upper()
            d.rectangle([0, 0, self.w, self.h], fill=(6, 10, 18, int(155 * enter)))
            f_m = _fit_single_line(d, hd, self._s(58), self.w - self._s(100), True, _has_devanagari(hd))
            f_s = _fit_single_line(d, sub, self._s(28), self.w - self._s(100), False, _has_devanagari(sub))
            mb = d.textbbox((0, 0), hd, font=f_m)
            my = int(self.h * 0.44)
            d.text(((self.w - (mb[2] - mb[0])) // 2, my), hd, font=f_m, fill=(245, 196, 48, int(255 * enter)))
            if sub:
                sb = d.textbbox((0, 0), sub, font=f_s)
                d.text(((self.w - (sb[2] - sb[0])) // 2, my + self._s(82)), sub, font=f_s, fill=(226, 232, 240, int(235 * enter)))

        if is_steady:
            self._ov_static_cache[ov_id] = layer
        base.paste(layer, (0, 0), layer)
        return base

    def _draw_captions(self, base: Image.Image, chunks: List[List[Dict[str, Any]]], t: float, emphasis: set[str]) -> Image.Image:
        c_idx = -1
        w_idx = -1
        for ci, ch in enumerate(chunks):
            if ch[0]["start"] <= t <= ch[-1]["end"] + 0.08:
                c_idx = ci
                for idx, w in enumerate(ch):
                    if w["start"] <= t <= w["end"] + 0.06:
                        w_idx = idx
                        break
                if w_idx == -1:
                    w_idx = len(ch) - 1
                break
        if c_idx == -1:
            return base

        cap_y = int(self.h * (0.74 if self.is_vertical else 0.78))
        key = (c_idx, w_idx)

        if self._cap_cache_key != key or self._cap_cache_strip is None:
            active_chunk = chunks[c_idx]
            cap_h = min(self.h - cap_y, self._s(155))
            strip = Image.new("RGBA", (self.w, cap_h), (0, 0, 0, 0))
            d = ImageDraw.Draw(strip)

            full_line = " ".join(w["word"] for w in active_chunk)
            hi = _has_devanagari(full_line)
            fsize = self._s(54 if self.is_vertical else 46)
            font = _get_font(fsize, True, hi)
            sp_w = self._s(15)

            widths = []
            for w in active_chunk:
                txt = _safe_text(w["word"]) if hi else w["word"].upper()
                bb = d.textbbox((0, 0), txt, font=font)
                widths.append(bb[2] - bb[0])
            total_w = sum(widths) + sp_w * max(0, len(widths) - 1)

            safe_max_w = self.w - self._s(100)
            if total_w > safe_max_w and total_w > 0:
                scale_down = safe_max_w / float(total_w)
                font = _get_font(max(20, int(fsize * scale_down)), True, hi)
                sp_w = max(6, int(sp_w * scale_down))
                widths = [d.textbbox((0, 0), w["word"] if hi else w["word"].upper(), font=font)[2] for w in active_chunk]
                total_w = sum(widths) + sp_w * max(0, len(widths) - 1)

            x = max(self._s(50), (self.w - total_w) // 2)
            ly = self._s(26)
            pad_x, pad_y = self._s(24), self._s(14)
            d.rounded_rectangle(
                [max(self._s(24), x - pad_x), max(2, ly - pad_y), min(self.w - self._s(24), x + total_w + pad_x), min(cap_h - 4, ly + self._s(72) + pad_y)],
                radius=self._s(16), fill=(6, 10, 18, 184),
            )

            stroke = self._s(4)
            for idx, w in enumerate(active_chunk):
                txt = _safe_text(w["word"]) if hi else w["word"].upper()
                clean = re.sub(r"[^a-z0-9\u0900-\u097F]", "", w["word"].lower())
                if idx == w_idx:
                    col = (245, 196, 48, 255)
                elif clean in emphasis:
                    col = (56, 189, 248, 255)
                else:
                    col = (248, 250, 252, 255)
                d.text((x, ly), txt, font=font, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0, 240))
                x += widths[idx] + sp_w

            self._cap_cache_key = key
            self._cap_cache_strip = strip

        base.paste(self._cap_cache_strip, (0, cap_y), self._cap_cache_strip)
        return base

    def _find_bgm(self, assets_dir: str) -> str | None:
        bgm_dir = os.path.join(assets_dir, "bgm")
        if os.path.isdir(bgm_dir):
            for fn in sorted(os.listdir(bgm_dir)):
                if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                    return os.path.join(bgm_dir, fn)
        return None

    def _build_audio_cmd_parts(
        self, has_voice: bool, dur: float, plan: Dict[str, Any], assets_dir: str, bgm_vol: float, sfx_vol: float
    ) -> Tuple[List[str], str | None, str | None]:
        extra_in: List[str] = []
        filters: List[str] = []
        mix_labels: List[str] = []
        next_idx = 2

        dlg_lbl = None
        if has_voice:
            filters.append(
                "[1:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                "highpass=f=80,equalizer=f=3200:t=q:w=1.2:g=2.0,"
                "acompressor=threshold=-18dB:ratio=2.3:attack=15:release=180:makeup=1.5[dlg_raw]"
            )
            dlg_lbl = "dlg_raw"

        bgm = self._find_bgm(assets_dir)
        if bgm:
            extra_in.extend(["-stream_loop", "-1", "-i", bgm])
            f_out = max(0.0, dur - min(2.0, dur * 0.25))
            filters.append(
                f"[{next_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                f"atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,volume={bgm_vol:.3f},"
                f"afade=t=in:st=0:d=1.2,afade=t=out:st={f_out:.2f}:d=1.8[bgm_pre]"
            )
            next_idx += 1
            if dlg_lbl:
                filters.append(f"[{dlg_lbl}]asplit=2[dlg_main][dlg_sc]")
                filters.append("[bgm_pre][dlg_sc]sidechaincompress=threshold=0.03:ratio=4.5:attack=35:release=350[bgm_duck]")
                mix_labels.extend(["[dlg_main]", "[bgm_duck]"])
            else:
                mix_labels.append("[bgm_pre]")
        elif dlg_lbl:
            mix_labels.append(f"[{dlg_lbl}]")

        sfx_dir = os.path.join(assets_dir, "sfx")
        cues: List[Tuple[str, float]] = []
        for ov in plan.get("overlays", []):
            if ov.get("sfx") in ("boom", "whoosh", "pop"):
                cues.append((ov["sfx"], float(ov["start"])))
        for br in plan.get("broll_tracks", []):
            if br.get("sfx") in ("boom", "whoosh", "pop") and float(br["start"]) > 0.4:
                cues.append((br["sfx"], float(br["start"])))

        for i, (name, st) in enumerate(cues[:10]):
            fp = next((os.path.join(sfx_dir, name + e) for e in (".wav", ".mp3") if os.path.exists(os.path.join(sfx_dir, name + e))), None)
            if not fp or st < 0 or st >= dur:
                continue
            extra_in.extend(["-i", fp])
            ms = int(round(st * 1000))
            lbl = f"sfx_{i}"
            filters.append(f"[{next_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={sfx_vol:.3f},adelay={ms}|{ms}[{lbl}]")
            mix_labels.append(f"[{lbl}]")
            next_idx += 1

        if not mix_labels:
            return [], None, None

        if len(mix_labels) == 1:
            comb = mix_labels[0]
        else:
            filters.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0[amix_out]")
            comb = "[amix_out]"
        filters.append(f"{comb}alimiter=limit=0.95:attack=5:release=50[aout]")
        return extra_in, ";".join(filters), "[aout]"

    def render(
        self,
        output_path: str,
        plan: Dict[str, Any],
        words_data: Dict[str, Any],
        assets_dir: str,
        audio_input_path: str | None = None,
        source_video_path: str | None = None,
        vision_info: Dict[str, Any] | None = None,
        bgm_vol: float = 0.14,
        sfx_vol: float = 0.30,
    ) -> str:
        dur = max(1.0, float(plan.get("duration") or words_data.get("duration") or 10.0))
        total_frames = max(1, int(round(dur * self.fps)))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        self._cap_cache_key = None
        self._cap_cache_strip = None
        self._ov_static_cache.clear()

        words = words_data.get("words", [])
        chunk_sz = 3 if self.is_vertical else 5
        chunks = [words[i:i + chunk_sz] for i in range(0, len(words), chunk_sz)]
        emphasis = {re.sub(r"[^a-z0-9\u0900-\u097F]", "", w.lower()) for w in plan.get("emphasis_words", [])}

        media_audio_src = audio_input_path or source_video_path
        has_voice = bool(media_audio_src and os.path.exists(media_audio_src))
        extra_in, f_complex, a_map = self._build_audio_cmd_parts(has_voice, dur, plan, assets_dir, bgm_vol, sfx_vol)

        enc_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.w}x{self.h}", "-r", str(self.fps), "-i", "pipe:0",
        ]
        if has_voice:
            enc_cmd.extend(["-i", media_audio_src])
        else:
            enc_cmd.extend(["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"])
        enc_cmd.extend(extra_in)
        if f_complex and a_map:
            enc_cmd.extend(["-filter_complex", f_complex, "-map", "0:v:0", "-map", a_map])
        else:
            enc_cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])

        enc_cmd.extend([
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart", "-shortest", output_path,
        ])

        dec_proc = None
        src_w, src_h = 0, 0
        if source_video_path and os.path.exists(source_video_path) and vision_info:
            src_w = int(vision_info["meta"]["width"])
            src_h = int(vision_info["meta"]["height"])
            vf_dec = f"fps={self.fps}"
            if self.color_grade == "documentary":
                vf_dec += ",eq=contrast=1.06:brightness=-0.02:saturation=1.05"
            dec_cmd = [
                "ffmpeg", "-v", "error", "-i", source_video_path,
                "-vf", vf_dec, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ]
            dec_proc = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        broll_tracks = plan.get("broll_tracks", [])
        overlays = plan.get("overlays", [])
        punch_zooms = plan.get("punch_zooms", [])
        scenes = (vision_info or {}).get("scenes", [])

        prev_frame: Image.Image | None = None
        last_src_img: Image.Image | None = None
        src_frame_bytes = src_w * src_h * 3
        next_pct = 20

        print(f"⚡ Fast Rendering {self.w}x{self.h} @ {self.fps}fps ({dur:.1f}s / {total_frames} frames)...", flush=True)

        try:
            for fi in range(total_frames):
                t = fi / float(self.fps)

                if dec_proc is not None and src_frame_bytes > 0:
                    raw = dec_proc.stdout.read(src_frame_bytes)
                    if len(raw) == src_frame_bytes:
                        last_src_img = Image.frombytes("RGB", (src_w, src_h), raw)

                active_br = next((b for b in broll_tracks if b["start"] <= t <= b["end"]), None)
                if active_br and (active_br.get("asset_path") or last_src_img is None):
                    bp = (t - active_br["start"]) / max(0.2, active_br["end"] - active_br["start"])
                    alt_dir = 1 if (int(active_br["id"].split("_")[-1]) % 2 == 1) else -1
                    frame = self._get_broll_frame(active_br.get("asset_path"), bp, alt_dir=alt_dir)
                elif last_src_img is not None:
                    sc = next((s for s in scenes if s["start_time"] <= t <= s["end_time"]), scenes[-1] if scenes else {"mode": "SINGLE", "cx": 0.5, "cy": 0.4})
                    frame = self._frame_source_video(last_src_img, sc, t, punch_zooms)
                else:
                    frame = self._proc_bg.copy()

                if active_br and prev_frame is not None and (t - active_br["start"]) < 0.35:
                    a = max(0.0, min(1.0, (t - active_br["start"]) / 0.35))
                    frame = Image.blend(prev_frame, frame, a)
                prev_frame = frame.copy() if (active_br and (active_br["end"] - t) < 0.15) else None

                active_ov = next((o for o in overlays if o["start"] <= t <= o["end"]), None)
                if active_ov:
                    op = (t - active_ov["start"]) / max(0.2, active_ov["end"] - active_ov["start"])
                    frame = self._draw_overlay(frame, active_ov, op)

                if chunks and not (active_ov and active_ov.get("type") == "ending_card"):
                    frame = self._draw_captions(frame, chunks, t, emphasis)

                try:
                    enc_proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    break

                pct = int(((fi + 1) / float(total_frames)) * 100)
                if pct >= next_pct:
                    print(f"   ⏳ Progress: {min(100, pct)}% ({fi + 1}/{total_frames})", flush=True)
                    next_pct += 20
        finally:
            if dec_proc is not None:
                if dec_proc.stdout:
                    dec_proc.stdout.close()
                dec_proc.wait()
            if enc_proc.stdin:
                try:
                    enc_proc.stdin.close()
                except Exception:
                    pass
            err_txt = enc_proc.stderr.read().decode("utf-8", errors="ignore") if enc_proc.stderr else ""
            rc = enc_proc.wait()

        if rc != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"FFmpeg encode failed (code {rc}): {err_txt.strip()}")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ DONE! Saved ({size_mb:.2f} MB) -> {output_path}", flush=True)
        return output_path
