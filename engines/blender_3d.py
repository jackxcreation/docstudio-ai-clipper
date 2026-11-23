from __future__ import annotations

import os
import re
import shutil
import subprocess


class Blender3DEngine:
    """Headless Blender 3D template bridge with automatic native Python 3D fallback."""

    def __init__(self, width: int = 720, height: int = 1280, fps: int = 24):
        self.w = int(width)
        self.h = int(height)
        self.fps = int(fps)
        self.blender_exe = shutil.which("blender")

    @property
    def is_available(self) -> bool:
        return bool(self.blender_exe)

    def build_bpy_script(self, template: str, label: str, out_mp4: str, duration_sec: float) -> str:
        frames = max(12, int(round(min(5.0, max(1.0, duration_sec)) * self.fps)))
        safe_lbl = re.sub(r"[^A-Za-z0-9 :%\-]", "", label)[:28] or "DOCUMENTARY"
        out_esc = out_mp4.replace("\\", "/")
        lines = [
            "import bpy, math",
            "bpy.ops.wm.read_factory_settings(use_empty=True)",
            "sc = bpy.context.scene",
            "sc.render.engine = 'BLENDER_EEVEE'",
            f"sc.render.resolution_x = {self.w}",
            f"sc.render.resolution_y = {self.h}",
            f"sc.render.fps = {self.fps}",
            "sc.frame_start = 1",
            f"sc.frame_end = {frames}",
            "sc.render.image_settings.file_format = 'FFMPEG'",
            "sc.render.ffmpeg.format = 'MPEG4'",
            "sc.render.ffmpeg.codec = 'H264'",
            f'sc.render.filepath = "{out_esc}"',
            "bpy.ops.object.camera_add(location=(0, -6.2, 2.0), rotation=(math.radians(72), 0, 0))",
            "sc.camera = bpy.context.active_object",
            "bpy.ops.object.light_add(type='AREA', location=(3.0, -4.0, 5.0))",
            "bpy.context.active_object.data.energy = 320.0",
            f'tpl = "{template}"',
            'if tpl == "globe":',
            "    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.55, location=(0, 0, 0.3))",
            "    obj = bpy.context.active_object",
            '    obj.keyframe_insert(data_path="rotation_euler", frame=1)',
            "    obj.rotation_euler = (0, 0, math.radians(65))",
            f'    obj.keyframe_insert(data_path="rotation_euler", frame={frames})',
            'elif tpl == "chart_3d":',
            "    for idx, h in enumerate([1.2, 2.4]):",
            "        bpy.ops.mesh.primitive_cube_add(size=0.8, location=((idx - 0.5) * 1.6, 0, h * 0.4))",
            "        bpy.context.active_object.scale = (1.0, 1.0, h)",
            "else:",
            "    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.25, location=(0, 0, 0.3))",
            "    obj = bpy.context.active_object",
            '    obj.keyframe_insert(data_path="rotation_euler", frame=1)',
            "    obj.rotation_euler = (math.radians(20), 0, math.radians(90))",
            f'    obj.keyframe_insert(data_path="rotation_euler", frame={frames})',
            "bpy.ops.object.text_add(location=(-1.4, -2.0, -1.3), rotation=(math.radians(78), 0, 0))",
            "txt = bpy.context.active_object",
            f'txt.data.body = "{safe_lbl}"',
            "txt.data.size = 0.40",
            "bpy.ops.render.render(animation=True)",
        ]
        return "\n".join(lines) + "\n"

    def render_clip(self, template: str, label: str, out_mp4: str, duration_sec: float, timeout_sec: int = 60) -> str | None:
        if not self.blender_exe:
            return None
        script_py = out_mp4 + ".py"
        with open(script_py, "w", encoding="utf-8") as f:
            f.write(self.build_bpy_script(template, label, out_mp4, duration_sec))
        try:
            p = subprocess.run(
                [self.blender_exe, "-b", "--factory-startup", "-P", script_py],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_sec,
            )
            if p.returncode == 0 and os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1024:
                return out_mp4
        except Exception:
            pass
        finally:
            if os.path.exists(script_py):
                try:
                    os.remove(script_py)
                except OSError:
                    pass
        return None
