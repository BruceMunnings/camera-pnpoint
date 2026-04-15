"""
pose_solver_numpy.py — Blender 5.1 Add-on
==========================================
Solves camera world position from N >= 3 3D-to-2D point correspondences using
a linear least-squares formulation, implemented with NumPy only.

WHY NOT OPENCV?
---------------
Blender ships with NumPy but not OpenCV.  Installing OpenCV into Blender's
bundled Python is possible but fragile (version mismatch, platform builds).
For the specific sub-problem addressed here — recovering only camera *position*
when the camera *rotation* is already known (e.g. set by fSpy) — the PnP
problem reduces to a straightforward 3×3 linear system that NumPy's
`linalg.lstsq` solves robustly with no iterative initialisation.

MATHEMATICAL APPROACH
---------------------
Let R_c2w be the 3×3 camera-to-world rotation (columns are camera axes
expressed in world space), taken directly from the camera object's
matrix_world.  Then R_w2c = R_c2w.T maps world vectors into camera space.

For a 3D world point P and observed pixel (u, v):

    q  = R_w2c @ P          # rotate world point into camera frame
    r  = R_w2c @ c          # unknown: rotated camera centre

The pinhole projection gives:
    (u - cx) / f = (qx - rx) / (qz - rz)
    (v - cy) / f = (qy - ry) / (qz - rz)

Rearranging into two linear equations per point (in the unknown vector r):
    [f,  0,  du] @ r  =  f·qx + du·qz      (row A)
    [0,  f, −dv] @ r  =  f·qy − dv·qz      (row B)

where  du = u − cx,  dv = v − cy.

Stack all rows → overdetermined linear system A·r = b, solved with lstsq.
Camera world position: c = R_c2w @ r.

PIXEL COORDINATE CONVENTIONS
-----------------------------
Blender's Clip Editor stores track coordinates in normalised space:
    origin at frame *centre*, y-up, range ≈ [-0.5, +0.5].

Convert to pixels with top-left origin, y-down:
    u = (co.x + 0.5) * clip_width
    v = (0.5  − co.y) * clip_height

WORKFLOW
--------
1. Open a Movie Clip in the Clip Editor and calibrate rotation with fSpy
   (or set the active camera rotation by any other means).
2. Place Empty objects (one per known point) inside a Blender collection.
   Name them so that alphabetical sort gives the correct pairing order.
3. In the Clip Editor, add track markers with Ctrl+LMB at each corresponding
   image position.  Name the tracks so alphabetical sort matches the empties.
4. In the N-panel → "Camera PnP" tab, select the collection and click
   "Solve Camera Position".
5. The camera object is moved to the solved position.  Per-point reprojection
   errors are printed to the system console.
"""

bl_info = {
    "name":        "Camera Position Solver",
    "author":      "camera-pnpoint project",
    "version":     (1, 0, 0),
    "blender":     (4, 0, 0),
    "location":    "Clip Editor > N Panel > Camera PnP",
    "description": "Solve camera position from 3D-2D correspondences (NumPy, no OpenCV)",
    "category":    "Camera",
}

import bpy
import numpy as np
from mathutils import Vector


# ---------------------------------------------------------------------------
# Scene property
# ---------------------------------------------------------------------------

def _register_props():
    bpy.types.Scene.pnp_collection = bpy.props.PointerProperty(
        type=bpy.types.Collection,
        name="3D Points Collection",
        description="Collection whose objects supply the known 3D world positions",
    )


def _unregister_props():
    del bpy.types.Scene.pnp_collection


# ---------------------------------------------------------------------------
# Helper: gather sorted 2D tracks from the active clip at a given frame
# ---------------------------------------------------------------------------

def _get_sorted_tracks(clip, frame: int):
    """Return list of (name, co_pixel) sorted by track name.

    co_pixel is (u, v) in pixels, top-left origin, y-down.
    """
    w = clip.size[0]
    h = clip.size[1]

    result = []
    for track in clip.tracking.tracks:
        marker = track.markers.find_frame(frame)
        if marker is None or marker.mute:
            continue
        co = marker.co  # normalised, centre-origin, y-up
        u = (co.x + 0.5) * w
        v = (0.5 - co.y) * h
        result.append((track.name, np.array([u, v], dtype=float)))

    result.sort(key=lambda x: x[0])
    return result


# ---------------------------------------------------------------------------
# Helper: gather sorted 3D object positions from a collection
# ---------------------------------------------------------------------------

def _get_sorted_points(collection):
    """Return list of (name, world_xyz) sorted by object name."""
    result = []
    for obj in collection.objects:
        p = obj.matrix_world.to_translation()
        result.append((obj.name, np.array([p.x, p.y, p.z], dtype=float)))
    result.sort(key=lambda x: x[0])
    return result


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def _solve_camera_position(points_3d, points_2d, R_c2w, f_px, cx, cy):
    """
    Solve for camera world position given corresponding 3D/2D point lists.

    Parameters
    ----------
    points_3d : list of np.ndarray shape (3,)  — world coordinates
    points_2d : list of np.ndarray shape (2,)  — pixel coords (u,v), y-down
    R_c2w     : np.ndarray shape (3,3)         — camera-to-world rotation
    f_px      : float                          — focal length in pixels
    cx, cy    : float                          — principal point in pixels

    Returns
    -------
    c_world : np.ndarray shape (3,)  — camera world position
    residuals : list of float        — per-point reprojection error in pixels
    """
    R_w2c = R_c2w.T  # world-to-camera rotation

    n = len(points_3d)
    A = np.zeros((2 * n, 3), dtype=float)
    b = np.zeros(2 * n,       dtype=float)

    for i, (P_world, pix) in enumerate(zip(points_3d, points_2d)):
        u, v = pix
        q = R_w2c @ P_world          # point in camera-frame coords
        qx, qy, qz = q
        du = u - cx
        dv = v - cy

        # Row A:  [f, 0,  du] @ r = f*qx + du*qz
        A[2 * i]     = [f_px, 0.0,   du]
        b[2 * i]     = f_px * qx + du * qz

        # Row B:  [0, f, -dv] @ r = f*qy - dv*qz
        A[2 * i + 1] = [0.0,  f_px, -dv]
        b[2 * i + 1] = f_px * qy - dv * qz

    # Solve overdetermined system A·r = b  →  r = R_w2c @ c
    r, _residuals_lstsq, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)

    # Recover camera world position
    c_world = R_c2w @ r

    # --- Reprojection errors ---
    errors = []
    for P_world, pix in zip(points_3d, points_2d):
        # Project P_world through the solved camera
        P_cam = R_w2c @ P_world - r         # point in camera-centred coords
        if abs(P_cam[2]) < 1e-12:
            errors.append(float("inf"))
            continue
        u_proj = cx + f_px * P_cam[0] / P_cam[2]
        v_proj = cy - f_px * P_cam[1] / P_cam[2]  # y-down in pixel space
        err = float(np.hypot(u_proj - pix[0], v_proj - pix[1]))
        errors.append(err)

    return c_world, errors


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class CLIP_OT_solve_camera_position(bpy.types.Operator):
    """Solve camera world position from 3D-2D correspondences (NumPy only)"""

    bl_idname = "clip.solve_camera_position"
    bl_label  = "Solve Camera Position"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene

        # --- Validate: collection ---
        collection = scene.pnp_collection
        if collection is None:
            self.report({"ERROR"}, "No 3D Points Collection selected.")
            return {"CANCELLED"}

        # --- Validate: active clip ---
        clip = context.edit_movieclip
        if clip is None:
            # Fallback: try scene's active clip
            clip = scene.active_clip if hasattr(scene, "active_clip") else None
        if clip is None:
            self.report({"ERROR"}, "No active Movie Clip. Open a clip in the Clip Editor.")
            return {"CANCELLED"}

        clip_width, clip_height = clip.size[0], clip.size[1]
        if clip_width == 0 or clip_height == 0:
            self.report({"ERROR"}, "Clip has zero size. Make sure the image/video is loaded.")
            return {"CANCELLED"}

        # --- Validate: active camera ---
        cam_obj = scene.camera
        if cam_obj is None or cam_obj.type != "CAMERA":
            self.report({"ERROR"}, "Scene has no active camera.")
            return {"CANCELLED"}

        cam_data = cam_obj.data
        f_px = cam_data.lens * clip_width / cam_data.sensor_width
        cx   = clip_width  / 2.0
        cy   = clip_height / 2.0

        # --- Gather points ---
        frame = context.scene.frame_current
        sorted_3d = _get_sorted_points(collection)
        sorted_2d = _get_sorted_tracks(clip, frame)

        n3 = len(sorted_3d)
        n2 = len(sorted_2d)

        if n3 != n2:
            self.report(
                {"WARNING"},
                f"Point count mismatch: {n3} 3D objects vs {n2} 2D tracks. "
                "They will be paired in sorted order up to min(n3,n2).",
            )

        n = min(n3, n2)
        if n < 3:
            self.report(
                {"ERROR"},
                f"Need at least 3 matched pairs, found {n}. Add more points.",
            )
            return {"CANCELLED"}

        points_3d = [p for _, p in sorted_3d[:n]]
        points_2d = [p for _, p in sorted_2d[:n]]

        names_3d = [name for name, _ in sorted_3d[:n]]
        names_2d = [name for name, _ in sorted_2d[:n]]

        # --- Camera rotation ---
        # matrix_world is camera-to-world; extract the 3×3 rotation part.
        mw = cam_obj.matrix_world
        R_c2w = np.array([
            [mw[0][0], mw[0][1], mw[0][2]],
            [mw[1][0], mw[1][1], mw[1][2]],
            [mw[2][0], mw[2][1], mw[2][2]],
        ], dtype=float)

        # --- Solve ---
        try:
            c_world, errors = _solve_camera_position(
                points_3d, points_2d, R_c2w, f_px, cx, cy
            )
        except np.linalg.LinAlgError as exc:
            self.report({"ERROR"}, f"Linear algebra error during solve: {exc}")
            return {"CANCELLED"}

        # --- Apply result ---
        cam_obj.location = Vector((float(c_world[0]),
                                   float(c_world[1]),
                                   float(c_world[2])))

        # --- Console report ---
        print("\n=== Camera PnP Solver ===")
        print(f"  Clip        : {clip.name}  ({clip_width}×{clip_height} px)")
        print(f"  Camera      : {cam_obj.name}")
        print(f"  Focal length: {f_px:.2f} px  ({cam_data.lens:.1f} mm)")
        print(f"  Pairs used  : {n}")
        print(f"  Solved position: ({c_world[0]:.4f}, {c_world[1]:.4f}, {c_world[2]:.4f})")
        print()
        print(f"  {'#':>3}  {'3D Object':<24}  {'2D Track':<24}  {'Reproj err (px)':>16}")
        print(f"  {'-'*3}  {'-'*24}  {'-'*24}  {'-'*16}")
        for i, (n3d, n2d, err) in enumerate(zip(names_3d, names_2d, errors)):
            print(f"  {i+1:>3}  {n3d:<24}  {n2d:<24}  {err:>16.3f}")
        mean_err = float(np.mean(errors))
        print(f"\n  Mean reprojection error: {mean_err:.3f} px")
        print("=========================\n")

        # --- Info bar message ---
        self.report({"INFO"}, f"Solved. Mean reprojection error: {mean_err:.2f} px")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class CLIP_PT_pnp_solver(bpy.types.Panel):
    """Camera PnP solver panel in the Clip Editor N-panel."""

    bl_label      = "Camera PnP"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category   = "Camera PnP"

    def draw(self, context):
        layout = self.layout
        scene  = context.scene

        # Collection picker
        layout.prop(scene, "pnp_collection", text="3D Points")

        # Summary counts
        col = layout.column(align=True)
        n3 = 0
        n2 = 0

        if scene.pnp_collection is not None:
            n3 = len(list(scene.pnp_collection.objects))

        clip = context.edit_movieclip
        if clip is not None:
            frame = scene.frame_current
            n2 = sum(
                1 for t in clip.tracking.tracks
                if t.markers.find_frame(frame) is not None
                and not t.markers.find_frame(frame).mute
            )

        col.label(text=f"{n3} 3D point(s) / {n2} 2D track(s)")

        # Solve button
        layout.separator()
        layout.operator(
            CLIP_OT_solve_camera_position.bl_idname,
            text="Solve Camera Position",
            icon="CAMERA_DATA",
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    CLIP_OT_solve_camera_position,
    CLIP_PT_pnp_solver,
)


def register():
    _register_props()
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    _unregister_props()


if __name__ == "__main__":
    register()
