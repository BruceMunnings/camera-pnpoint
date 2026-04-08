"""
import_photo_blender51.py
========================
Blender 5.1 helper — load a reference photo as a MovieClip for camera-pnpoint
without the "context incorrect" error that bpy.ops.clip.open() raises in 5.x.

Problem
-------
camera-pnpoint requires reference photos to be loaded as MovieClips in the
Clip Editor.  In Blender 5.0+ the automatic context promotion that older
scripts relied on was removed, so calling ``bpy.ops.clip.open()`` from the
Python console or a script now raises::

    RuntimeError: Operator bpy.ops.clip.open.poll() failed, context is incorrect

This script bypasses the operator entirely by using
``bpy.data.movieclips.load()`` (context-free) and then wiring the returned
clip into the Clip Editor directly.

Quick start
-----------
1. Open Blender 5.1 and load (or create) your scene.
2. Open the *Scripting* workspace and paste this file into a new text block,
   **or** run it from the Python console::

       exec(open(r"G:\\My Drive\\Github\\camera-pnpoint\\import_photo_blender51.py").read())

3. Call ``load_photo_for_pnpoint()`` with your photo path::

       load_photo_for_pnpoint(
           photo_path  = r"C:\\Users\\You\\Pictures\\site.jpg",
           focal_mm    = 24.0,   # your lens focal length
           sensor_mm   = 36.0,   # sensor width (36 = full-frame)
       )

4. The Clip Editor (or the largest non-3D panel) will show the photo.
5. Switch to camera-pnpoint's panel, add your 2D track markers in the Clip
   Editor and the matching 3D empties in the viewport, then run **Pose Solver**.

API compatibility
-----------------
Tested against Blender 4.2 LTS and 5.1.  The ``principal_point_pixels``
attribute was added in 4.0; older builds fall back to the normalised
``principal`` attribute automatically.
"""

import bpy
import os


def load_photo_for_pnpoint(
    photo_path: str,
    focal_mm: float = 50.0,
    sensor_mm: float = 36.0,
    clip_name: str = None,
    open_clip_editor: bool = True,
) -> dict:
    """
    Load *photo_path* as a MovieClip and configure the Clip Editor ready for
    camera-pnpoint's Pose Solver / Calibrate operators.

    Parameters
    ----------
    photo_path        : Absolute path to the image (JPEG, PNG, etc.).
    focal_mm          : Lens focal length in mm.  Seeded into the clip's
                        tracking camera so camera-pnpoint starts with the
                        correct value.
    sensor_mm         : Sensor width in mm (default 36 = full-frame).
    clip_name         : Name for the MovieClip data-block.  Defaults to the
                        file basename without extension.
    open_clip_editor  : If True (default), switch the largest non-3D panel to
                        the Clip Editor so the photo is immediately visible.

    Returns
    -------
    dict with keys: ``clip``, ``area``, ``focal_mm``, ``sensor_mm``.

    Notes
    -----
    * Uses ``bpy.data.movieclips.load()`` — **no operator context required**.
    * Sets the clip's tracking camera intrinsics (focal length, sensor width,
      principal point at image centre) so camera-pnpoint needs no manual setup.
    * Safe to call multiple times; re-uses an existing clip if the path matches.
    """
    abs_path = os.path.realpath(os.path.abspath(photo_path))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Photo not found: {abs_path}")

    # --- Load (or reuse) the MovieClip data-block ---------------------------
    name = clip_name or os.path.splitext(os.path.basename(abs_path))[0]
    clip = bpy.data.movieclips.get(name)
    if clip is None or os.path.realpath(bpy.path.abspath(clip.filepath)) != abs_path:
        clip = bpy.data.movieclips.load(abs_path)
        clip.name = name

    # --- Wire into the Clip Editor ------------------------------------------
    clip_area = None
    if open_clip_editor:
        for area in bpy.context.screen.areas:
            if area.type == 'CLIP_EDITOR':
                clip_area = area
                break
        if clip_area is None:
            best = None
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    continue
                if best is None or area.width * area.height > best.width * best.height:
                    best = area
            if best is not None:
                best.type = 'CLIP_EDITOR'
                clip_area = best
        if clip_area is not None:
            clip_area.spaces.active.clip = clip
            clip_area.tag_redraw()

    # --- Seed camera intrinsics into the clip's tracking camera -------------
    tc = clip.tracking.camera          # bpy.types.MovieTrackingCamera
    tc.lens         = float(focal_mm)
    tc.sensor_width = float(sensor_mm)

    w, h = clip.size[:]
    if w > 0 and h > 0:
        if hasattr(tc, 'principal_point_pixels'):
            # Blender 4.0+: stored in pixels
            tc.principal_point_pixels = (w / 2.0, h / 2.0)
        elif hasattr(tc, 'principal'):
            # Blender 3.x: stored as 0..1 normalised
            tc.principal = (0.5, 0.5)

    print(
        f"[camera-pnpoint helper] clip='{clip.name}' {w}x{h}px  "
        f"focal={focal_mm}mm  sensor={sensor_mm}mm  "
        f"clip_editor={'opened' if clip_area else 'not found'}"
    )
    return {
        "clip":      clip,
        "area":      clip_area,
        "focal_mm":  focal_mm,
        "sensor_mm": sensor_mm,
    }


# ---------------------------------------------------------------------------
# Run immediately when executed as a Blender text-block script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Edit the path and lens values below, then click Run Script (Alt+P).
    load_photo_for_pnpoint(
        photo_path = r"C:\Users\YourName\Pictures\reference.jpg",
        focal_mm   = 24.0,
        sensor_mm  = 36.0,
    )
