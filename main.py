#!/usr/bin/env python3
"""
Livox Avia – Direct SDK Viewer
(Posture Outline + Three-Segment Spine + Option C Inclination)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Option C — Three-Segment Angle Model
  The spine is split into three landmarks:
    bot  → pelvis / lumbar base  (bottom 25% of centroid slices)
    mid  → thoracic anchor       (middle third — most stable while sitting)
    top  → cervical / head       (top 25% of centroid slices)

  Two segments are measured independently:
    lumbar_fwd_deg  → forward/backward lean of the lower back (bot→mid)
    lumbar_lat_deg  → left/right lean of the lower back       (bot→mid)
    fhp_fwd_deg     → forward head posture angle              (mid→top)
    fhp_lat_deg     → lateral head drift                      (mid→top)
    relative_fhp_deg → fhp_fwd_deg − lumbar_fwd_deg
                       Large positive = classic tech-neck on a rounded back

  Visual arrows in the 3-D view:
    Orange  bot → mid   (lumbar segment)
    Lime    mid → top   (cervical / FHP segment)
    Red sphere   = pelvis reference
    Orange sphere = thoracic anchor
    Green sphere  = head/neck reference

  NEW — Noise reduction pipeline:
    1. Spatial NMS  (nms_spine_centroids)
       Per-frame: centroid slices that jump laterally beyond 0.12 m
       from their neighbours are suppressed before angles are computed.
       Two passes so chains of bad points are cleaned up.

    2. Temporal smoothing  (TemporalSmoother)
       Cross-frame: 5-frame rolling buffer per angle metric.
       Any frame whose value lies > 1.5 σ from the buffer mean is
       excluded before averaging (z-score outlier rejection).
       Hard clip of ±60° discards physically impossible readings
       before they even enter the buffer.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['GRPC_VERBOSITY']   = 'NONE'

import collections
import socket
import struct
import threading
import time
import numpy as np
import open3d as o3d
import binascii
import tkinter as tk
import multiprocessing
import json
import cv2

# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BROADCAST_PORT  = 55000
DEVICE_CMD_PORT = 65000
HOST_DATA_PORT  = 60001
HOST_CMD_PORT   = 60000
HOST_IMU_PORT   = 60002

HEARTBEAT_INTERVAL_S = 0.1
FRAME_TIME_S         = 3
MAX_COLOR_DIST_M     = 5


CONFIG_FILE = "roi_settings.json"
BG_PCD_FILE = "background_model.pcd"

# Number of height slices for the spine centroid curve
SPINE_SLICES = 100

# ── Option C alert thresholds ──────────────────────────────────────────────────
# Forward-plane angles only.
# Lateral (left/right) angles are computed and stored for reference but are
# intentionally excluded from all alert logic — LiDAR angular resolution and
# depth-based estimation make lateral measurements too unreliable for seated
# posture assessment.  If a future sensor/mount improves lateral accuracy,
# add `abs(ll) > LUMBAR_ALERT_DEG` back into the lumbar_alert expressions.
LUMBAR_ALERT_DEG   = 15.0
# Upper-segment RELATIVE forward-head angle vs lumbar baseline
# 12° = early tech-neck; tune downward (e.g. 8°) for stricter monitoring
FHP_RELATIVE_ALERT = 12.0

# ── Spatial NMS ────────────────────────────────────────────────────────────────
# Max lateral deviation (m) a centroid slice may have from its neighbours
NMS_LAT_THRESH  = 0.12
NMS_ITERATIONS  = 2      # passes; extra passes clean up chains of bad points

# ── Temporal smoother ──────────────────────────────────────────────────────────
SMOOTH_WINDOW    = 5     # frames kept in rolling buffer
SMOOTH_Z_THRESH  = 1.5   # σ — frames beyond this are excluded from average
SMOOTH_HARD_CLIP = 60.0  # °  — readings outside ±clip never enter buffer


# ═══════════════════════════════════════════════════════════════════════════════
#  Thick-line helper — cylinder meshes, always visible on any driver
# ═══════════════════════════════════════════════════════════════════════════════

def create_line_cylinders(pts_3d, lines_idx, color, radius=0.018):
    """Render line segments as a merged TriangleMesh of cylinders."""
    combined = o3d.geometry.TriangleMesh()
    pts = np.asarray(pts_3d, dtype=float)
    for a, b in lines_idx:
        p1, p2 = pts[a], pts[b]
        seg    = p2 - p1
        length = float(np.linalg.norm(seg))
        if length < 1e-4:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=6, split=1)
        d    = seg / length
        z_ax = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z_ax, d)
        cn    = float(np.linalg.norm(cross))
        if cn > 1e-6:
            angle = float(np.arccos(np.clip(float(np.dot(z_ax, d)), -1.0, 1.0)))
            cyl.rotate(
                o3d.geometry.get_rotation_matrix_from_axis_angle((cross / cn) * angle),
                center=np.zeros(3))
        elif d[2] < 0:
            cyl.rotate(
                o3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array([np.pi, 0.0, 0.0])),
                center=np.zeros(3))
        cyl.translate((p1 + p2) * 0.5)
        combined += cyl
    if len(np.asarray(combined.vertices)) == 0:
        return None
    combined.paint_uniform_color(color)
    combined.compute_vertex_normals()
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  Spatial NMS — per-frame centroid outlier suppression
# ═══════════════════════════════════════════════════════════════════════════════

def nms_spine_centroids(spine_pts, lat_thresh=NMS_LAT_THRESH, iterations=NMS_ITERATIONS):
    """
    Spatial NMS on the spine centroid list.

    Each interior centroid is compared against its immediate neighbours.
    If the lateral (XY-plane) deviation from the linear interpolation
    between those neighbours exceeds `lat_thresh` metres, the point is
    suppressed.  Multiple passes handle chains of consecutive bad points.

    The first and last points are never removed so the lumbar/head
    anchor zones always have at least one sample.

    Parameters
    ----------
    spine_pts   : list of [X, Y, Z] ordered bottom → top
    lat_thresh  : max allowed XY deviation from neighbour midpoint (m)
    iterations  : number of suppression passes

    Returns
    -------
    Filtered list in the same [X, Y, Z] format.
    Falls back to the original list if fewer than 4 points survive.
    """
    pts = spine_pts
    for _ in range(iterations):
        if len(pts) < 3:
            break
        arr  = np.array(pts, dtype=float)
        keep = np.ones(len(arr), dtype=bool)
        for i in range(1, len(arr) - 1):
            # Expected XY = linear interpolation of immediate neighbours
            expected_xy = (arr[i - 1, :2] + arr[i + 1, :2]) * 0.5
            dev = np.linalg.norm(arr[i, :2] - expected_xy)
            if dev > lat_thresh:
                keep[i] = False
        filtered = arr[keep].tolist()
        if len(filtered) >= 4:
            pts = filtered
        # If too many points removed this pass, stop early
        else:
            break
    return pts


# ═══════════════════════════════════════════════════════════════════════════════
#  Temporal Smoother — 5-frame rolling average with z-score outlier rejection
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalSmoother:
    """
    Rolling N-frame buffer for inclination angles with z-score outlier rejection.

    Each frame's angles are added to per-metric circular buffers.  When a
    smoothed value is requested, frames whose value lies more than
    `z_thresh` standard deviations from the buffer mean are excluded
    before the average is taken.

    This removes single-frame spikes (chair creak, sensor glitch, partial
    occlusion) without adding noticeable lag for genuine posture changes.

    A `hard_clip` gate discards physically impossible readings before they
    even enter the buffer (e.g. a 90° spike from a spurious point cluster).

    Usage
    -----
        smoother = TemporalSmoother()
        smoothed_inc = smoother.update(raw_inc_dict)

    `relative_fhp_deg` is re-derived from the smoothed fhp/lumbar values
    rather than smoothed independently, so all three displayed numbers
    stay self-consistent.
    """

    ANGLE_KEYS = (
        "lumbar_fwd_deg",
        "lumbar_lat_deg",
        "fhp_fwd_deg",
        "fhp_lat_deg",
        "relative_fhp_deg",
    )

    def __init__(
        self,
        window: int   = SMOOTH_WINDOW,
        z_thresh: float = SMOOTH_Z_THRESH,
        hard_clip: float = SMOOTH_HARD_CLIP,
    ):
        """
        Parameters
        ----------
        window    : number of recent frames to keep (rolling)
        z_thresh  : frames beyond this many σ from the buffer mean are excluded
        hard_clip : readings outside ±hard_clip degrees never enter the buffer
        """
        self.window    = window
        self.z_thresh  = z_thresh
        self.hard_clip = hard_clip
        self._bufs: dict = {
            k: collections.deque(maxlen=window) for k in self.ANGLE_KEYS
        }

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, inc: dict) -> dict:
        """
        Push a new raw inclination dict and return a smoothed copy.

        Non-angle keys (bot_pt / mid_pt / top_pt) pass through unchanged
        so the 3-D landmarks always reflect the current frame position.
        """
        result = dict(inc)  # preserve landmark positions as-is

        # Smooth fwd/lat angles independently
        for k in ("lumbar_fwd_deg", "lumbar_lat_deg", "fhp_fwd_deg", "fhp_lat_deg"):
            raw = float(inc.get(k, 0.0))
            if abs(raw) <= self.hard_clip:          # hard gate
                self._bufs[k].append(raw)
            result[k] = round(self._robust_mean(self._bufs[k]), 2)

        # Re-derive relative FHP from the already-smoothed components so the
        # three displayed numbers are always self-consistent (not independently
        # smoothed versions of each other).
        result["relative_fhp_deg"] = round(
            result["fhp_fwd_deg"] - result["lumbar_fwd_deg"], 2)

        # Keep the derived buffer in sync for alert logic, but don't feed raw
        self._bufs["relative_fhp_deg"].append(result["relative_fhp_deg"])

        return result

    def reset(self):
        """Clear all buffers (call when the person leaves the frame)."""
        for buf in self._bufs.values():
            buf.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _robust_mean(self, buf: collections.deque) -> float:
        """Mean of buffer values after removing z-score outliers."""
        if len(buf) == 0:
            return 0.0
        arr  = np.array(buf, dtype=float)
        mean = float(np.mean(arr))
        if len(arr) == 1:
            return mean
        std = float(np.std(arr))
        if std < 1e-6:          # all values identical → no outlier logic needed
            return mean
        mask = np.abs(arr - mean) <= self.z_thresh * std
        kept = arr[mask]
        # Fallback: if the filter removes everything, use the unfiltered mean
        return float(np.mean(kept)) if len(kept) > 0 else mean


# ═══════════════════════════════════════════════════════════════════════════════
#  Option C — Three-Segment Inclination
# ═══════════════════════════════════════════════════════════════════════════════

def three_segment_inclination(spine_pts):
    """
    Split the spine centroid list into three anatomical zones and compute
    independent angles for each segment.

    Parameters
    ----------
    spine_pts : list of [X, Y, Z] centroids, ordered bottom → top

    Returns
    -------
    dict with keys:
        lumbar_fwd_deg   – forward/back lean, lower segment (bot→mid)
        lumbar_lat_deg   – left/right lean,   lower segment
        fhp_fwd_deg      – forward/back lean, upper segment (mid→top)
        fhp_lat_deg      – left/right lean,   upper segment
        relative_fhp_deg – fhp_fwd − lumbar_fwd  (key bad-office metric)
        bot_pt / mid_pt / top_pt  – the three landmark positions
    """
    pts   = np.array(spine_pts, dtype=float)
    n     = len(pts)

    # Landmark averaging windows
    n_end  = max(1, n // 4)        # outer 25% for bot & top
    n_mid0 = n // 2                # start at 50% of height
    n_mid1 = 3 * n // 4           # end at 75% of height

    bot = np.mean(pts[:n_end],          axis=0)   # pelvis / lumbar base
    mid = np.mean(pts[n_mid0:n_mid1],  axis=0)
    top = np.mean(pts[n - n_end:],      axis=0)   # head / cervical

    # ── Lower segment: pelvis → thoracic ──────────────────────────────────────
    dZ_lower = float(mid[2] - bot[2])
    if abs(dZ_lower) > 0.03:
        lumbar_fwd_deg = float(np.degrees(np.arctan2(mid[1] - bot[1], dZ_lower)))
        lumbar_lat_deg = float(np.degrees(np.arctan2(mid[0] - bot[0], dZ_lower)))
    else:
        lumbar_fwd_deg = lumbar_lat_deg = 0.0

    # ── Upper segment: thoracic → head ────────────────────────────────────────
    dZ_upper = float(top[2] - mid[2])
    if abs(dZ_upper) > 0.03:
        fhp_fwd_deg = float(np.degrees(np.arctan2(top[1] - mid[1], dZ_upper)))
        fhp_lat_deg = float(np.degrees(np.arctan2(top[0] - mid[0], dZ_upper)))
    else:
        fhp_fwd_deg = fhp_lat_deg = 0.0

    # ── Relative angle — the primary bad-office-syndrome indicator ─────────────
    # Positive = head leans further forward than the lumbar baseline
    relative_fhp_deg = fhp_fwd_deg - lumbar_fwd_deg

    return {
        "lumbar_fwd_deg":   round(lumbar_fwd_deg,   2),
        "lumbar_lat_deg":   round(lumbar_lat_deg,   2),
        "fhp_fwd_deg":      round(fhp_fwd_deg,      2),
        "fhp_lat_deg":      round(fhp_lat_deg,      2),
        "relative_fhp_deg": round(relative_fhp_deg, 2),
        "bot_pt":           bot.tolist(),
        "mid_pt":           mid.tolist(),
        "top_pt":           top.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Posture Overlay — Outline + Centroid Spine + Option C Inclination
# ═══════════════════════════════════════════════════════════════════════════════

def extract_spine_and_outline(points):
    """
    Projection: fixed X-axis view → project onto the YZ plane
                (Y = horizontal, Z = vertical).

    Pipeline
    --------
    1. Slice the body into SPINE_SLICES horizontal bands.
    2. Compute per-band centroid → raw spine_pts list.
    3. Spatial NMS pass (nms_spine_centroids) to remove lateral outliers.
    4. Rebuild spine lines from the cleaned list.
    5. Compute three-segment inclination on cleaned spine.

    Outputs
    -------
    spine_mesh   : cylinder segments + sphere knots tracing each horizontal
                   slice centroid.  Colour gradient cyan (bottom) → yellow (top).
                   Two directional arrows:
                     • Orange  bot → mid   (lumbar segment)
                     • Lime    mid → top   (FHP / cervical segment)
                   Three reference spheres:
                     • Red    = pelvis (bot)
                     • Orange = thoracic anchor (mid)
                     • Green  = head/neck (top)
    outline_mesh : magenta body-contour cylinder ring.
    inclination  : Option C dict from three_segment_inclination(), or None.
                   NOTE: these are raw per-frame values; caller applies
                   TemporalSmoother before storing / displaying.
    """
    if len(points) < 40:
        return None, None, None

    # ── Project onto YZ plane (look from X axis) ──────────────────────────────
    horiz = points[:, 1]   # Y → horizontal
    vert  = points[:, 2]   # Z → vertical

    H, W = 512, 512
    h_min, h_max = np.min(horiz), np.max(horiz)
    v_min, v_max = np.min(vert),  np.max(vert)
    if h_max == h_min or v_max == v_min:
        return None, None, None

    h_pad = (h_max - h_min) * 0.1;  v_pad = (v_max - v_min) * 0.1
    h_min -= h_pad;  h_max += h_pad
    v_min -= v_pad;  v_max += v_pad

    u_px = ((horiz - h_min) / (h_max - h_min) * (W - 1)).astype(int)
    v_px = ((1.0 - (vert - v_min) / (v_max - v_min)) * (H - 1)).astype(int)

    img = np.zeros((H, W), dtype=np.uint8)
    p2d = {}
    for idx in range(len(points)):
        p2d[(u_px[idx], v_px[idx])] = idx
        img[v_px[idx], u_px[idx]] = 255

    # ── Silhouette ────────────────────────────────────────────────────────────
    kd  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kc  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sil = cv2.morphologyEx(cv2.dilate(img, kd), cv2.MORPH_CLOSE, kc)

    # ── Outline ───────────────────────────────────────────────────────────────
    outline_mesh = None
    contours, _  = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    avg_x        = float(np.mean(points[:, 0]))   # fallback depth along X axis

    if contours:
        largest       = max(contours, key=cv2.contourArea)
        subsample     = max(1, len(largest) // 200)
        outline_pts   = []
        outline_lines = []

        for i, pt in enumerate(largest[::subsample]):
            cu, cv_p = int(pt[0][0]), int(pt[0][1])
            best_idx = None;  best_d = float('inf')
            for du in range(-10, 11):
                for dv in range(-10, 11):
                    key = (cu + du, cv_p + dv)
                    if key in p2d:
                        d2 = du * du + dv * dv
                        if d2 < best_d:
                            best_d = d2;  best_idx = p2d[key]
            if best_idx is not None:
                outline_pts.append(points[best_idx].tolist())
            else:
                real_y = h_min + (cu / W) * (h_max - h_min)
                real_z = v_max - (cv_p / H) * (v_max - v_min)
                outline_pts.append([avg_x, real_y, real_z])
            if i > 0:
                outline_lines.append([i - 1, i])

        if len(outline_pts) > 2:
            outline_lines.append([len(outline_pts) - 1, 0])
            pa = np.array(outline_pts, dtype=float)
            n  = len(pa)
            if n >= 11:
                smth = np.zeros_like(pa)
                hw   = 5
                for si in range(n):
                    idxs      = [(si + j - hw) % n for j in range(2 * hw + 1)]
                    smth[si]  = pa[idxs].mean(axis=0)
                outline_pts = smth.tolist()
            outline_mesh = create_line_cylinders(
                outline_pts, outline_lines, [1.0, 0.0, 1.0], radius=0.012)

    # ── Spine: centroid per height band ───────────────────────────────────────
    z_vals          = points[:, 2]
    z_min, z_max    = float(np.min(z_vals)), float(np.max(z_vals))
    z_range         = z_max - z_min

    # Trim bottom 10% (chair base / floor noise) and top 5% (hair reflections)
    z_lo    = z_min + z_range * 0.10
    z_hi    = z_max - z_range * 0.05
    body    = points[(z_vals >= z_lo) & (z_vals <= z_hi)]

    if len(body) < 20:
        return None, outline_mesh, None

    bz             = body[:, 2]
    bz_min, bz_max = float(np.min(bz)), float(np.max(bz))
    step           = (bz_max - bz_min) / SPINE_SLICES

    spine_pts   = []

    for i in range(SPINE_SLICES):
        lo, hi = bz_min + i * step, bz_min + (i + 1) * step
        mask   = (bz >= lo) & (bz < hi)
        if np.sum(mask) < 5:
            continue
        centroid = body[mask].mean(axis=0)
        spine_pts.append(centroid.tolist())

    spine_mesh  = None
    inclination = None

    if len(spine_pts) >= 4:

        # ── Spatial NMS: suppress per-slice lateral outliers ──────────────────
        spine_pts = nms_spine_centroids(
            spine_pts, lat_thresh=NMS_LAT_THRESH, iterations=NMS_ITERATIONS)

        # Rebuild line index list from the cleaned (possibly shorter) point list
        spine_lines = [[i, i + 1] for i in range(len(spine_pts) - 1)]

        if len(spine_pts) < 4:
            return None, outline_mesh, None

        # ── Build mesh ────────────────────────────────────────────────────────
        combined = o3d.geometry.TriangleMesh()

        # White cylinder bones
        cyl = create_line_cylinders(
            spine_pts, spine_lines, [1.0, 1.0, 1.0], radius=0.022)
        if cyl is not None:
            combined += cyl

        # Coloured sphere knots — cyan (bottom) → yellow (top)
        n_pts = len(spine_pts)
        for j, pt in enumerate(spine_pts):
            t     = j / max(n_pts - 1, 1)      # 0.0 at bottom, 1.0 at top
            color = [t, 1.0, 1.0 - t]          # cyan → yellow
            sph   = o3d.geometry.TriangleMesh.create_sphere(radius=0.035, resolution=8)
            sph.translate(np.array(pt, dtype=float))
            sph.paint_uniform_color(color)
            sph.compute_vertex_normals()
            combined += sph

        # ── Option C: three-segment inclination ───────────────────────────────
        inclination = three_segment_inclination(spine_pts)
        bot = np.array(inclination["bot_pt"])
        mid = np.array(inclination["mid_pt"])
        top = np.array(inclination["top_pt"])

        # Arrow 1: Orange  bot → mid  (lumbar segment)
        arrow_lumbar = create_line_cylinders(
            [bot.tolist(), mid.tolist()],
            [[0, 1]],
            color=[1.0, 0.55, 0.0],   # orange
            radius=0.030)
        if arrow_lumbar is not None:
            combined += arrow_lumbar

        # Arrow 2: Lime  mid → top  (FHP / cervical segment)
        arrow_fhp = create_line_cylinders(
            [mid.tolist(), top.tolist()],
            [[0, 1]],
            color=[0.4, 1.0, 0.0],    # lime green
            radius=0.030)
        if arrow_fhp is not None:
            combined += arrow_fhp

        # Reference spheres: red=pelvis, orange=thoracic, green=head
        for pt, col, rad in [
            (bot, [1.0, 0.2, 0.2],  0.045),   # red    — pelvis
            (mid, [1.0, 0.55, 0.0], 0.050),   # orange — thoracic anchor
            (top, [0.2, 1.0, 0.2],  0.045),   # green  — head/neck
        ]:
            mk = o3d.geometry.TriangleMesh.create_sphere(radius=rad, resolution=8)
            mk.translate(np.array(pt, dtype=float))
            mk.paint_uniform_color(col)
            mk.compute_vertex_normals()
            combined += mk

        if len(np.asarray(combined.vertices)) > 0:
            combined.compute_vertex_normals()
            spine_mesh = combined

    return spine_mesh, outline_mesh, inclination


# ═══════════════════════════════════════════════════════════════════════════════
#  Console inclination printer (Option C)
# ═══════════════════════════════════════════════════════════════════════════════

def print_inclination(inc):
    """
    Pretty-print Option C inclination dict to stdout.

    Alert logic uses forward-plane angles only.
    Lateral (left/right) values are printed for reference but are NOT
    used for any alert decision — LiDAR lateral accuracy is insufficient
    for reliable left/right posture assessment.
    """
    if inc is None:
        return

    lf = inc["lumbar_fwd_deg"]
    ll = inc["lumbar_lat_deg"]   # reference only — not used in alerts
    ff = inc["fhp_fwd_deg"]
    fl = inc["fhp_lat_deg"]      # reference only — not used in alerts
    rf = inc["relative_fhp_deg"]

    # Forward-plane only
    lumbar_alert = abs(lf) > LUMBAR_ALERT_DEG
    fhp_alert    = abs(rf) > FHP_RELATIVE_ALERT

    tag = ""
    if fhp_alert and lumbar_alert:
        tag = "  ⚠ ALERT: Lumbar + Forward-Head"
    elif fhp_alert:
        tag = "  ⚠ ALERT: Forward-Head Posture"
    elif lumbar_alert:
        tag = "  ⚠ ALERT: Lumbar Lean"

    print(
        f"  Lumbar  │ Fwd/Back: {lf:+6.1f}°  Lat(ref): {ll:+6.1f}°\n"
        f"  FHP     │ Fwd/Back: {ff:+6.1f}°  Lat(ref): {fl:+6.1f}°\n"
        f"  Relative│ Head vs Lumbar: {rf:+6.1f}°{tag}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Human Extraction Logic (Chair Filter)
# ═══════════════════════════════════════════════════════════════════════════════

def get_human_indices(points, ransac_iters=300, remove_seat=True):
    if len(points) < 50:
        return None

    max_z = np.max(points[:, 2])
    min_z = np.min(points[:, 2])

    head_zone_mask  = (points[:, 2] >= (max_z - 0.25)) & (points[:, 2] <= max_z)
    head_candidates = points[head_zone_mask]

    best_score  = -1
    best_center = None

    if len(head_candidates) > 4:
        for _ in range(ransac_iters):
            idx = np.random.choice(len(head_candidates), 4, replace=False)
            pts = head_candidates[idx]
            A   = np.zeros((4, 4))
            b   = np.zeros(4)
            for i in range(4):
                A[i] = [2*pts[i][0], 2*pts[i][1], 2*pts[i][2], 1]
                b[i] = pts[i][0]**2 + pts[i][1]**2 + pts[i][2]**2
            try:
                params = np.linalg.solve(A, b)
                center = params[:3];  d = params[3]
                val    = center[0]**2 + center[1]**2 + center[2]**2 + d
                if val < 0: continue
                radius = np.sqrt(val)
                if not (0.09 <= radius <= 0.18): continue
                if abs(center[2] - (max_z - 0.12)) > 0.15: continue
                distances = np.linalg.norm(head_candidates - center, axis=1)
                score     = len(np.where(abs(distances - radius) <= 0.05)[0])
                if score > best_score:
                    best_score  = score
                    best_center = center
            except np.linalg.LinAlgError:
                continue

    if best_center is None:
        return None

    horiz_dist   = np.linalg.norm(points[:, :2] - best_center[:2], axis=1)
    human_mask   = (horiz_dist <= 0.45) & (points[:, 2] <= max_z) & \
                   (points[:, 2] > min_z + 0.32)

    if remove_seat and np.sum(human_mask) > 30:
        seat_mask = human_mask & (points[:, 2] < (min_z + 0.65))
        if np.sum(seat_mask) > 10:
            seat_pcd = o3d.geometry.PointCloud()
            seat_pcd.points = o3d.utility.Vector3dVector(points[seat_mask])
            plane_model, inliers = seat_pcd.segment_plane(
                distance_threshold=0.025, ransac_n=3, num_iterations=150)
            if abs(plane_model[2]) > 0.85:
                human_mask[np.where(seat_mask)[0][inliers]] = False

    return human_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI Control Panel Process
# ═══════════════════════════════════════════════════════════════════════════════

def roi_control_panel(shared_bounds):
    root = tk.Tk()
    root.title("ROI & Background Panel")
    root.geometry("360x860")
    root.attributes('-topmost', True)
    root.configure(padx=15, pady=15)

    tk.Label(root, text="Adjust ROI Bounding Box",
             font=('Arial', 12, 'bold')).pack(pady=(0, 5))

    def make_slider(name, val):
        tk.Label(root, text=f"{name} Axis",
                 font=('Arial', 9, 'bold'), fg="gray").pack(anchor='w', pady=(3, 0))
        s = tk.Scale(root, from_=-20.0, to=20.0, resolution=0.1,
                     orient='horizontal', length=300)
        s.set(val);  s.pack()
        return s

    spatial_keys       = ['X1', 'X2', 'Y1', 'Y2', 'Z1', 'Z2']
    sliders            = {k: make_slider(k, shared_bounds[k]) for k in spatial_keys}
    last_saved_values  = {k: s.get() for k, s in sliders.items()}

    tk.Frame(root, height=2, bd=1, relief="sunken").pack(fill="x", pady=10)

    # ── Background calibration ────────────────────────────────────────────────
    bg_frame  = tk.LabelFrame(root, text=" 1. Background (Floor/Walls) ",
                               font=('Arial', 10, 'bold'), padx=10, pady=5)
    bg_frame.pack(fill="x", pady=5)
    bg_status = tk.Label(bg_frame, text="Status: Inactive",
                         font=('Arial', 9), fg="gray")
    bg_status.pack(pady=(0, 5))

    def run_bg_countdown(count):
        if count > 0:
            shared_bounds['bg_state'] = 'countdown'
            bg_status.config(text=f"Calibrating in {count}s...", fg="orange")
            root.after(1000, run_bg_countdown, count - 1)
        else:
            bg_status.config(text="Scanning empty room...", fg="blue")
            shared_bounds['bg_state'] = 'capture'

    tk.Button(bg_frame, text="Calibrate (10s)",
              command=lambda: run_bg_countdown(10),
              bg="#e1f5fe").pack(side="left", expand=True, fill="x", padx=(0, 5))
    tk.Button(bg_frame, text="Clear File",
              command=lambda: shared_bounds.update({'bg_state': 'clear'}),
              bg="#ffebee").pack(side="right", fill="x")

    # ── Chair / human filter ──────────────────────────────────────────────────
    chair_frame  = tk.LabelFrame(root, text=" 2. Human Extraction (Chair Filter) ",
                                  font=('Arial', 10, 'bold'), padx=10, pady=5)
    chair_frame.pack(fill="x", pady=5)
    chair_status = tk.Label(chair_frame, text="Status: Inactive",
                             font=('Arial', 9), fg="gray")
    chair_status.pack(pady=(0, 5))

    tk.Button(chair_frame, text="Enable Filter",
              command=lambda: (shared_bounds.update({'chair_state': 'active'}),
                               chair_status.config(text="Active", fg="green")),
              bg="#e8f5e9").pack(side="left", expand=True, fill="x", padx=(0, 5))
    tk.Button(chair_frame, text="Disable",
              command=lambda: (shared_bounds.update({'chair_state': 'idle'}),
                               chair_status.config(text="Inactive", fg="gray")),
              bg="#ffebee").pack(side="right", fill="x")

    # ── Option C — Three-Segment Inclination readout ──────────────────────────
    inc_frame = tk.LabelFrame(root, text=" 3. Three-Segment Inclination (Option C) ",
                               font=('Arial', 10, 'bold'), padx=10, pady=5)
    inc_frame.pack(fill="x", pady=5)

    # Header row labels
    tk.Label(inc_frame, text="Segment         Fwd/Back    Lat (ref only)",
             font=('Courier', 8), fg="gray", anchor='w').pack(fill='x')
    tk.Frame(inc_frame, height=1, bg="lightgray").pack(fill='x', pady=2)

    # Lumbar row  (bot → thoracic anchor)
    lbl_lumbar = tk.Label(inc_frame,
                          text="● Lumbar (bot→mid)    —       —",
                          font=('Courier', 10), fg="black", anchor='w')
    lbl_lumbar.pack(fill='x')

    # FHP row  (thoracic anchor → head)
    lbl_fhp    = tk.Label(inc_frame,
                          text="● FHP    (mid→top)    —       —",
                          font=('Courier', 10), fg="black", anchor='w')
    lbl_fhp.pack(fill='x')

    tk.Frame(inc_frame, height=1, bg="lightgray").pack(fill='x', pady=2)

    # Relative angle — the key bad-office metric
    lbl_rel    = tk.Label(inc_frame,
                          text="▲ Relative FHP:   —",
                          font=('Courier', 10, 'bold'), fg="black", anchor='w')
    lbl_rel.pack(fill='x')

    lbl_alert  = tk.Label(inc_frame, text="",
                           font=('Arial', 10, 'bold'), fg="red")
    lbl_alert.pack(pady=(4, 0))

    # Lateral reference note
    tk.Label(inc_frame,
             text="⚠ Lat values are reference only — not used in alerts\n"
                  "   (LiDAR lateral accuracy insufficient for L/R assessment)",
             font=('Arial', 7), fg="#b08000", anchor='w', justify='left').pack(fill='x', pady=(4, 0))

    # Smoothing info banner
    tk.Label(inc_frame,
             text=f"Smoothing: {SMOOTH_WINDOW}-frame avg  |  z>{SMOOTH_Z_THRESH}σ cut  |  clip ±{SMOOTH_HARD_CLIP:.0f}°",
             font=('Arial', 7), fg="gray", anchor='w').pack(fill='x', pady=(3, 0))

    # Legend
    tk.Label(inc_frame,
             text="● orange=thoracic anchor  ● red=pelvis  ● green=head",
             font=('Arial', 7), fg="gray", anchor='w').pack(fill='x', pady=(2, 0))

    def update_shared_dict():
        nonlocal last_saved_values
        current_vals = {}
        has_changed  = False
        for k, s in sliders.items():
            val = s.get()
            current_vals[k] = val
            shared_bounds[k] = val
            if val != last_saved_values.get(k): has_changed = True
        if has_changed:
            try:
                file_data = {}
                if os.path.exists(CONFIG_FILE):
                    with open(CONFIG_FILE, "r") as f: file_data = json.load(f)
                file_data.update(current_vals)
                with open(CONFIG_FILE, "w") as f: json.dump(file_data, f, indent=4)
                last_saved_values = current_vals
            except Exception:
                pass

        # ── Refresh background status ──────────────────────────────────────
        b_st = shared_bounds.get('bg_state', 'idle')
        if b_st == 'active':
            bg_status.config(text="Active (Room Hidden)", fg="green")
        elif b_st == 'idle':
            bg_status.config(text="Inactive", fg="gray")
        elif b_st == 'scanning':
            bg_status.config(text="Scanning empty room...", fg="blue")

        # ── Refresh Option C inclination readout ──────────────────────────
        inc = shared_bounds.get('inclination')
        if inc:
            lf = inc.get("lumbar_fwd_deg", 0.0)
            ll = inc.get("lumbar_lat_deg", 0.0)   # reference only
            ff = inc.get("fhp_fwd_deg",    0.0)
            fl = inc.get("fhp_lat_deg",    0.0)   # reference only
            rf = inc.get("relative_fhp_deg", 0.0)

            # Forward-plane only — lateral excluded from alert logic
            lumbar_bad = abs(lf) > LUMBAR_ALERT_DEG
            fhp_bad    = abs(rf) > FHP_RELATIVE_ALERT

            # Lateral shown in muted grey to reinforce "reference only" status
            lbl_lumbar.config(
                text=f"● Lumbar (bot→mid)  {lf:+6.1f}°  {ll:+6.1f}°",
                fg="red" if lumbar_bad else "#b35a00")

            lbl_fhp.config(
                text=f"● FHP    (mid→top)  {ff:+6.1f}°  {fl:+6.1f}°",
                fg="red" if fhp_bad else "dark green")

            rel_color = "red" if fhp_bad else ("dark orange" if abs(rf) > 6 else "black")
            lbl_rel.config(
                text=f"▲ Relative FHP:  {rf:+6.1f}°",
                fg=rel_color)

            if fhp_bad and lumbar_bad:
                lbl_alert.config(text="⚠ Lumbar + Forward-Head detected!", fg="red")
            elif fhp_bad:
                lbl_alert.config(text="⚠ Forward-Head Posture detected!", fg="red")
            elif lumbar_bad:
                lbl_alert.config(text="⚠ Excessive lumbar lean detected!", fg="red")
            else:
                lbl_alert.config(text="✓ Posture within range", fg="green")

        root.after(100, update_shared_dict)

    update_shared_dict()
    root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
#  Protocol & Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def livox_crc16(data: bytes) -> int:
    crc = 0x4C49
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc

def livox_crc32(data: bytes) -> int:
    return binascii.crc32(data, 0x564F580A) & 0xFFFFFFFF

_seq_num = 0
def _next_seq() -> int:
    global _seq_num
    s, _seq_num = _seq_num, (_seq_num + 1) & 0xFFFF
    return s

def build_cmd(cmd_set, cmd_id, payload=b'', cmd_type=0x00):
    data        = bytes([cmd_set, cmd_id]) + payload
    total_len   = 9 + len(data) + 4
    pre_crc     = struct.pack('<BBHBH', 0xAA, 0x01, total_len, cmd_type, _next_seq())
    header      = pre_crc + struct.pack('<H', livox_crc16(pre_crc))
    pkt         = header + data
    return pkt + struct.pack('<I', livox_crc32(pkt))

def cmd_handshake(host_ip, dp, cp, ip):
    return build_cmd(0x00, 0x01,
                     socket.inet_aton(host_ip) + struct.pack('<HHH', dp, cp, ip),
                     cmd_type=0x01)
def cmd_heartbeat():              return build_cmd(0x00, 0x03)
def cmd_start_sampling(s=True):   return build_cmd(0x00, 0x04, bytes([0x01 if s else 0x00]))
def cmd_set_cartesian():          return build_cmd(0x00, 0x05, bytes([0x00]))

_DATA_HDR = 18
def parse_data_packet(data):
    if len(data) < _DATA_HDR + 1 or data[0] == 0xAA: return []
    dtype = data[9];  offset = _DATA_HDR;  pts = []
    if dtype == 0:
        for _ in range((len(data) - _DATA_HDR) // 13):
            if offset + 13 > len(data): break
            x, y, z, _ = struct.unpack_from('<iiiB', data, offset)
            pts.append([x*1e-3, y*1e-3, z*1e-3]);  offset += 13
    elif dtype == 1:
        for _ in range((len(data) - _DATA_HDR) // 9):
            if offset + 9 > len(data): break
            depth, theta, phi, _ = struct.unpack_from('<IHHB', data, offset)
            r  = depth * 1e-3
            ze = np.radians(theta / 100.0);  az = np.radians(phi / 100.0)
            pts.append([r*np.sin(ze)*np.cos(az), r*np.sin(ze)*np.sin(az), r*np.cos(ze)])
            offset += 9
    elif dtype == 2:
        for _ in range((len(data) - _DATA_HDR) // 14):
            if offset + 14 > len(data): break
            x, y, z, _, _ = struct.unpack_from('<iiiBB', data, offset)
            pts.append([x*1e-3, y*1e-3, z*1e-3]);  offset += 14
    return pts

def send_and_ack(sock, pkt, dest, label, timeout=2.0):
    sock.sendto(pkt, dest);  sock.settimeout(timeout)
    try:
        ack, _ = sock.recvfrom(512)
        ret = ack[11] if len(ack) > 11 else 0xFF
        print(f"  {label:25s}  {'OK' if ret == 0 else f'ret={ret}'}")
        return ret
    except socket.timeout:
        print(f"  {label:25s}  no ACK");  return None

def get_box_lineset(min_b, max_b):
    pts = [[min_b[0],min_b[1],min_b[2]], [max_b[0],min_b[1],min_b[2]],
           [min_b[0],max_b[1],min_b[2]], [max_b[0],max_b[1],min_b[2]],
           [min_b[0],min_b[1],max_b[2]], [max_b[0],min_b[1],max_b[2]],
           [min_b[0],max_b[1],max_b[2]], [max_b[0],max_b[1],max_b[2]]]
    lines = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[1, 0.2, 0.2]] * 12)
    return ls


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Visualizer Loop
# ═══════════════════════════════════════════════════════════════════════════════

def live_livox_viewer(shared_bounds, initial_camera=None):
    print("  Livox Avia – Posture Viewer (Option C: Three-Segment Spine)\n")
    print(f"  Lumbar alert:       ±{LUMBAR_ALERT_DEG}°")
    print(f"  FHP relative alert: >{FHP_RELATIVE_ALERT}°")
    print(f"  Spatial NMS:        lat_thresh={NMS_LAT_THRESH} m, {NMS_ITERATIONS} passes")
    print(f"  Temporal smoother:  {SMOOTH_WINDOW}-frame window, "
          f"z>{SMOOTH_Z_THRESH}σ cut, hard clip ±{SMOOTH_HARD_CLIP}°\n")

    bcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bcast_sock.bind(("0.0.0.0", BROADCAST_PORT))
    bcast_sock.settimeout(10.0)

    device_ip = None
    try:
        while device_ip is None:
            try:
                raw, addr = bcast_sock.recvfrom(256)
            except socket.timeout:
                return
            if len(raw) >= 34 and raw[0] == 0xAA and raw[9] == 0x00 and raw[10] == 0x00:
                device_ip = addr[0]
    finally:
        bcast_sock.close()

    dest = (device_ip, DEVICE_CMD_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((device_ip, 1));  host_ip = s.getsockname()[0]

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.bind((host_ip, HOST_CMD_PORT))
    send_and_ack(cmd_sock, cmd_handshake(host_ip, HOST_DATA_PORT, HOST_CMD_PORT, HOST_IMU_PORT), dest, "Handshake")
    time.sleep(0.05)
    send_and_ack(cmd_sock, cmd_set_cartesian(), dest, "Set Cartesian")
    time.sleep(0.05)
    send_and_ack(cmd_sock, cmd_start_sampling(True), dest, "Start sampling")

    stop_event = threading.Event()
    cmd_sock.setblocking(False)

    def _heartbeat():
        while not stop_event.is_set():
            try:
                cmd_sock.sendto(cmd_heartbeat(), dest)
                for _ in range(16):
                    try: cmd_sock.recvfrom(256)
                    except (BlockingIOError, OSError): break
            except Exception: pass
            time.sleep(HEARTBEAT_INTERVAL_S)

    threading.Thread(target=_heartbeat, daemon=True).start()

    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    data_sock.bind((host_ip, HOST_DATA_PORT))
    data_sock.setblocking(False)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Livox Avia – Posture Viewer (Option C)", 1280, 720)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    init_min    = [shared_bounds['X1'], shared_bounds['Y1'], shared_bounds['Z1']]
    init_max    = [shared_bounds['X2'], shared_bounds['Y2'], shared_bounds['Z2']]
    roi_box_vis = get_box_lineset(init_min, init_max)
    vis.add_geometry(roi_box_vis)

    ro = vis.get_render_option()
    ro.point_size         = 2.0
    ro.background_color   = np.array([0.05, 0.05, 0.05])

    if initial_camera is not None:
        try:
            with open("temp_cam_load.json", "w") as f: json.dump(initial_camera, f)
            cam = o3d.io.read_pinhole_camera_parameters("temp_cam_load.json")
            vis.get_view_control().convert_from_pinhole_camera_parameters(
                cam, allow_arbitrary=True)
            os.remove("temp_cam_load.json")
        except: pass

    background_pcd = None
    if os.path.exists(BG_PCD_FILE):
        try:
            background_pcd = o3d.io.read_point_cloud(BG_PCD_FILE)
            if not background_pcd.is_empty():
                shared_bounds['bg_state'] = 'active'
                print(f"  [Init] Loaded background: {BG_PCD_FILE}")
            else:
                background_pcd = None
        except Exception: pass

    bg_accumulating    = False
    bg_accumulated_pts = []
    bg_start_time      = 0.0

    accumulated_pts  = []
    last_render_time = time.time()

    active_spine_geom   = None
    active_outline_geom = None

    # ── Temporal smoother — one instance persists across frames ───────────────
    smoother = TemporalSmoother(
        window=SMOOTH_WINDOW,
        z_thresh=SMOOTH_Z_THRESH,
        hard_clip=SMOOTH_HARD_CLIP,
    )
    # Track whether a person was visible last frame so we can reset on disappear
    person_was_visible = False

    try:
        while True:
            # ── Drain UDP buffer ───────────────────────────────────────────────
            while True:
                try:
                    pkt, _ = data_sock.recvfrom(1500)
                    accumulated_pts.extend(parse_data_packet(pkt))
                except (BlockingIOError, OSError):
                    break

            current_time = time.time()
            if current_time - last_render_time >= FRAME_TIME_S:
                min_b = [min(shared_bounds['X1'], shared_bounds['X2']),
                         min(shared_bounds['Y1'], shared_bounds['Y2']),
                         min(shared_bounds['Z1'], shared_bounds['Z2'])]
                max_b = [max(shared_bounds['X1'], shared_bounds['X2']),
                         max(shared_bounds['Y1'], shared_bounds['Y2']),
                         max(shared_bounds['Z1'], shared_bounds['Z2'])]

                roi_box_vis.points = get_box_lineset(min_b, max_b).points
                vis.update_geometry(roi_box_vis)

                # Remove previous frame overlays
                if active_spine_geom is not None:
                    vis.remove_geometry(active_spine_geom, reset_bounding_box=False)
                    active_spine_geom = None
                if active_outline_geom is not None:
                    vis.remove_geometry(active_outline_geom, reset_bounding_box=False)
                    active_outline_geom = None

                if accumulated_pts:
                    arr   = np.asarray(accumulated_pts, dtype=np.float64)
                    dists = np.linalg.norm(arr, axis=1)

                    roi_mask = (
                        (dists > 0.01) &
                        (arr[:, 0] >= min_b[0]) & (arr[:, 0] <= max_b[0]) &
                        (arr[:, 1] >= min_b[1]) & (arr[:, 1] <= max_b[1]) &
                        (arr[:, 2] >= min_b[2]) & (arr[:, 2] <= max_b[2]))
                    arr   = arr[roi_mask]
                    dists = dists[roi_mask]

                    # ── Background cancellation ────────────────────────────────
                    bg_state = shared_bounds.get('bg_state', 'idle')
                    if bg_state == 'clear':
                        background_pcd = None;  bg_accumulating = False
                        if os.path.exists(BG_PCD_FILE):
                            try: os.remove(BG_PCD_FILE)
                            except Exception: pass
                        shared_bounds['bg_state'] = 'idle'
                    elif bg_state == 'capture':
                        bg_accumulating = True;  bg_accumulated_pts = []
                        bg_start_time   = time.time()
                        shared_bounds['bg_state'] = 'scanning'

                    if bg_accumulating:
                        bg_accumulated_pts.extend(arr.tolist())
                        if time.time() - bg_start_time >= 2.0:
                            bg_accumulating = False
                            if bg_accumulated_pts:
                                background_pcd = o3d.geometry.PointCloud()
                                background_pcd.points = o3d.utility.Vector3dVector(
                                    np.array(bg_accumulated_pts))
                                background_pcd = background_pcd.voxel_down_sample(0.03)
                                try: o3d.io.write_point_cloud(BG_PCD_FILE, background_pcd)
                                except Exception: pass
                                shared_bounds['bg_state'] = 'active'
                            else:
                                shared_bounds['bg_state'] = 'idle'

                    if (shared_bounds.get('bg_state') == 'active'
                            and background_pcd is not None and len(arr) > 0):
                        tmp = o3d.geometry.PointCloud()
                        tmp.points = o3d.utility.Vector3dVector(arr)
                        d2bg  = np.asarray(tmp.compute_point_cloud_distance(background_pcd))
                        arr   = arr[d2bg > 0.06]
                        dists = dists[d2bg > 0.06]

                    # ── Human extraction + posture overlays ────────────────────
                    chair_state = shared_bounds.get('chair_state', 'idle')

                    if chair_state == 'active' and len(arr) > 20:
                        human_mask = get_human_indices(arr, ransac_iters=50)

                        if human_mask is not None:
                            arr_h   = arr[human_mask]
                            dists_h = dists[human_mask]

                            if len(arr_h) > 20:
                                _tmp = o3d.geometry.PointCloud()
                                _tmp.points = o3d.utility.Vector3dVector(arr_h)
                                _labels = np.array(_tmp.cluster_dbscan(
                                    eps=0.12, min_points=8, print_progress=False))
                                _valid = _labels >= 0
                                if _valid.any():
                                    _biggest = np.bincount(_labels[_valid]).argmax()
                                    _body    = _labels == _biggest
                                    arr   = arr_h[_body]
                                    dists = dists_h[_body]
                                else:
                                    arr, dists = arr_h, dists_h
                            else:
                                arr, dists = arr_h, dists_h

                            # ── Spine + outline (spatial NMS applied inside) ───
                            active_spine_geom, active_outline_geom, inclination = \
                                extract_spine_and_outline(arr)

                            if inclination is not None:
                                # ── Temporal smoothing (5-frame avg + outlier cut)
                                inclination = smoother.update(inclination)
                                shared_bounds['inclination'] = inclination
                                print_inclination(inclination)
                                person_was_visible = True
                            else:
                                # Person disappeared — reset buffer so stale history
                                # doesn't pollute the next sit-down session
                                if person_was_visible:
                                    smoother.reset()
                                    person_was_visible = False
                                shared_bounds['inclination'] = None

                        else:
                            # No human detected this frame
                            if person_was_visible:
                                smoother.reset()
                                person_was_visible = False
                            shared_bounds['inclination'] = None

                    if active_spine_geom is not None:
                        vis.add_geometry(active_spine_geom,   reset_bounding_box=False)
                    if active_outline_geom is not None:
                        vis.add_geometry(active_outline_geom, reset_bounding_box=False)

                    # ── Render point cloud ─────────────────────────────────────
                    if len(arr) > 0:
                        pcd.points = o3d.utility.Vector3dVector(arr)
                        nd = np.clip(dists / MAX_COLOR_DIST_M, 0.0, 1.0)
                        colors = np.zeros((len(arr), 3))
                        colors[:, 0] = np.clip(1.5 - np.abs(4.0 * nd - 3.0), 0, 1)
                        colors[:, 1] = np.clip(1.5 - np.abs(4.0 * nd - 2.0), 0, 1)
                        colors[:, 2] = np.clip(1.5 - np.abs(4.0 * nd - 1.0), 0, 1)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                    else:
                        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                        pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    vis.update_geometry(pcd)

                accumulated_pts.clear()
                last_render_time = current_time

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters("temp_cam_save.json", cam_params)
            with open("temp_cam_save.json", "r") as f: cam_data = json.load(f)
            os.remove("temp_cam_save.json")
            file_data = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f: file_data = json.load(f)
            file_data["camera_view"] = cam_data
            with open(CONFIG_FILE, "w") as f: json.dump(file_data, f, indent=4)
        except Exception: pass
        try:
            cmd_sock.setblocking(True)
            cmd_sock.sendto(cmd_start_sampling(False), dest)
        except Exception: pass
        cmd_sock.close();  data_sock.close();  vis.destroy_window()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    multiprocessing.freeze_support()
    initial_bounds = {'X1': -5.0, 'X2': 5.0, 'Y1': -5.0, 'Y2': 5.0,
                      'Z1': -2.0, 'Z2': 5.0}
    initial_camera = None

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f: file_data = json.load(f)
            for k in initial_bounds:
                if k in file_data: initial_bounds[k] = file_data[k]
            if "camera_view" in file_data: initial_camera = file_data["camera_view"]
        except Exception: pass

    initial_bounds.update({'bg_state': 'idle', 'chair_state': 'idle',
                           'inclination': None})

    with multiprocessing.Manager() as manager:
        shared_bounds = manager.dict(initial_bounds)
        gui_process   = multiprocessing.Process(target=roi_control_panel,
                                                args=(shared_bounds,))
        gui_process.start()
        try:
            live_livox_viewer(shared_bounds, initial_camera)
        finally:
            gui_process.terminate();  gui_process.join()