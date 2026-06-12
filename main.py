#!/usr/bin/env python3
"""
Livox Avia – Direct SDK Viewer
(Posture Outline + Three-Segment Spine + Option C Inclination
 + Reference Position Capture with Head-Below Tracking
 + Lateral Axis Alerts Enabled)
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
from posture_reference_capture import (
    HEAD_BELOW_THRESH_M,
    capture_reference_position,
    compute_head_delta,
    _build_section4,
    _refresh_delta_labels,
    _update_reference_ghost,
    _remove_reference_ghost,
)

BROADCAST_PORT  = 55000
DEVICE_CMD_PORT = 65000
HOST_DATA_PORT  = 60001
HOST_CMD_PORT   = 60000
HOST_IMU_PORT   = 60002

HEARTBEAT_INTERVAL_S = 0.1
FRAME_TIME_S         = 1
MAX_COLOR_DIST_M     = 5

CONFIG_FILE  = "roi_settings.json"
BG_PCD_FILE  = "background_model.pcd"
SPINE_SLICES = 25

LUMBAR_ALERT_DEG     = 15.0
FHP_RELATIVE_ALERT   = 12.0
LUMBAR_LAT_ALERT_DEG = 15.0   # Lateral lumbar threshold (now active)
FHP_LAT_ALERT_DEG    = 12.0   # Lateral FHP threshold (now active)

# Reclining-on-chair detection
# If the head is BELOW the reference height AND the upper-spine (FHP) segment
# is tilted backward past this threshold, the person is leaning on the chair
# — lumbar alert is suppressed.  If the head drops without backward tilt,
# the person is slumping forward and the alert fires normally.
RECLINE_FHP_DEG    = -5.0   # fhp_fwd_deg must be < this (backward) for recline
RECLINE_LUMBAR_DEG = -3.0   # lumbar_fwd_deg must also be < this to confirm recline

NMS_LAT_THRESH = 0.12
NMS_ITERATIONS = 2

SMOOTH_WINDOW          = 5
SMOOTH_Z_THRESH        = 1.5
SMOOTH_HARD_CLIP       = 60.0
SMOOTH_ENTER_TRACK_DEG = 3.5   # buffer avg must move THIS FAR from last output to unfreeze
SMOOTH_EXIT_TRACK_DEG  = 1.0   # buffer avg must move LESS THAN this per frame to re-freeze
SMOOTH_LARGE_DEG       = 12.0  # raw jump ≥ this → snap immediately + flush buffer

# ── Denoising & body-extraction parameters ────────────────────────────────────
VOXEL_SIZE           = 0.025   # 2.5 cm — normalises Livox non-repetitive density
DENOISE_NB_NEIGHBORS = 20      # statistical outlier removal: neighbourhood size
DENOISE_STD_RATIO    = 2.0     # 1.0–1.5 over-removes sparse Livox body pts; keep at 2.0
HUMAN_HORIZ_RADIUS   = 0.55    # horizontal radius from head centre in X-Y plane;
                                # 0.35 cuts torso when sensor is side-mounted and body
                                # depth differs from head by more than 35 cm
HUMAN_FLOOR_MARGIN   = 0.08    # strip only the bottom 8 cm of person Z range
                                # (was 0.32 — removed torso when total Z span is ~1 m)


def denoise_pointcloud(arr, voxel=True):
    """
    Two-stage cleaning:
      1. Voxel downsample — normalises the Livox non-repetitive scan density
         so statistical outlier removal works uniformly across the cloud.
      2. Statistical outlier removal — eliminates isolated flying pixels and
         scatter that inflate the silhouette outline.
    Returns a float64 numpy array (may be smaller than input).
    """
    if len(arr) < 20:
        return arr
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    if voxel:
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    if len(np.asarray(pcd.points)) < 20:
        return arr                         # fallback: return original if over-thinned
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=DENOISE_NB_NEIGHBORS, std_ratio=DENOISE_STD_RATIO)
    result = np.asarray(pcd.points, dtype=np.float64)
    return result if len(result) >= 10 else arr


def create_line_cylinders(pts_3d, lines_idx, color, radius=0.018):
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


def nms_spine_centroids(spine_pts, lat_thresh=NMS_LAT_THRESH, iterations=NMS_ITERATIONS):
    pts = spine_pts
    for _ in range(iterations):
        if len(pts) < 3:
            break
        arr  = np.array(pts, dtype=float)
        keep = np.ones(len(arr), dtype=bool)
        for i in range(1, len(arr) - 1):
            expected_xy = (arr[i - 1, :2] + arr[i + 1, :2]) * 0.5
            dev = np.linalg.norm(arr[i, :2] - expected_xy)
            if dev > lat_thresh:
                keep[i] = False
        filtered = arr[keep].tolist()
        if len(filtered) >= 4:
            pts = filtered
        else:
            break
    return pts


class TemporalSmoother:
    """
    Hysteresis (Schmitt-trigger) smoother for posture angles.

    Two separate thresholds stop the "too sensitive vs too frozen" trade-off:

      FREEZE → TRACK   buffer avg moves ≥ SMOOTH_ENTER_TRACK_DEG from last output
                       → requires a sustained, genuine shift to unfreeze.

      TRACK  → FREEZE  buffer avg moves < SMOOTH_EXIT_TRACK_DEG in one frame
                       → re-freezes once the person settles at the new position.

      SNAP             raw value jumps ≥ SMOOTH_LARGE_DEG in one frame
                       → output snaps instantly, buffer flushed.

    Example walk-through (person moves from 0° → 8°):
      frames 1-2 : avg builds toward 3.5°, still FROZEN          (noise suppressed)
      frame 3    : avg reaches 3.5° → TRACKING, output follows
      frames 4-5 : avg climbs 5°→7°→8°, still TRACKING
      frame 6    : avg stabilises at 8°, frame-delta < 1° → FREEZE at 8°
      frame 7+   : stationary noise ±2° < enter threshold → stays FROZEN   ✓
    """

    ANGLE_KEYS = (
        "lumbar_fwd_deg", "lumbar_lat_deg",
        "fhp_fwd_deg",    "fhp_lat_deg",
        "relative_fhp_deg",
    )
    _TRACK_KEYS = ("lumbar_fwd_deg", "lumbar_lat_deg", "fhp_fwd_deg", "fhp_lat_deg")

    def __init__(self, window=SMOOTH_WINDOW, z_thresh=SMOOTH_Z_THRESH,
                 hard_clip=SMOOTH_HARD_CLIP):
        self.window    = window
        self.z_thresh  = z_thresh
        self.hard_clip = hard_clip
        self._bufs        = {k: collections.deque(maxlen=window) for k in self.ANGLE_KEYS}
        self._last_output = {k: None for k in self._TRACK_KEYS}
        self._tracking    = {k: False for k in self._TRACK_KEYS}

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, inc):
        result = dict(inc)
        for k in self._TRACK_KEYS:
            raw  = float(inc.get(k, 0.0))
            last = self._last_output[k]

            # Hard-clip: ignore physically impossible spikes
            if abs(raw) > self.hard_clip:
                result[k] = round(last if last is not None else 0.0, 2)
                continue

            # ── SNAP: large raw jump → instant response ────────────────────────
            raw_delta = abs(raw - last) if last is not None else float('inf')
            if raw_delta >= SMOOTH_LARGE_DEG:
                self._bufs[k].clear()
                self._bufs[k].append(raw)
                self._tracking[k]    = False
                self._last_output[k] = raw
                result[k] = round(raw, 2)
                continue

            # Push to buffer for running average
            self._bufs[k].append(raw)
            arr = np.array(self._bufs[k], dtype=float)
            avg = float(np.mean(arr[-self.window:]))

            if last is None:
                # First frame: initialise
                self._last_output[k] = avg
                result[k] = round(avg, 2)
                continue

            avg_delta = abs(avg - last)

            if not self._tracking[k]:
                # ── FROZEN: check whether to start tracking ────────────────────
                if avg_delta >= SMOOTH_ENTER_TRACK_DEG:
                    self._tracking[k] = True   # genuine movement detected
                out = last                      # output stays locked until tracking
            else:
                # ── TRACKING: follow the average ──────────────────────────────
                eff = self._effective_window(avg_delta)
                out = float(np.mean(arr[-eff:]))
                # Re-freeze once movement per frame drops below exit threshold
                if avg_delta < SMOOTH_EXIT_TRACK_DEG:
                    self._tracking[k] = False

            result[k] = round(out, 2)
            self._last_output[k] = out

        result["relative_fhp_deg"] = round(
            result["fhp_fwd_deg"] - result["lumbar_fwd_deg"], 2)
        self._bufs["relative_fhp_deg"].append(result["relative_fhp_deg"])
        return result

    def reset(self):
        for buf in self._bufs.values():
            buf.clear()
        for k in self._TRACK_KEYS:
            self._last_output[k] = None
            self._tracking[k]    = False

    # ── private ───────────────────────────────────────────────────────────────

    def _effective_window(self, avg_delta):
        """Window shrinks from SMOOTH_WINDOW → 1 as movement grows."""
        if avg_delta <= SMOOTH_ENTER_TRACK_DEG:
            return self.window
        if avg_delta >= SMOOTH_LARGE_DEG:
            return 1
        t = (avg_delta - SMOOTH_ENTER_TRACK_DEG) / (SMOOTH_LARGE_DEG - SMOOTH_ENTER_TRACK_DEG)
        return max(1, round(self.window * (1.0 - t)))

    def _robust_mean(self, buf):
        if len(buf) == 0: return 0.0
        arr  = np.array(buf, dtype=float)
        mean = float(np.mean(arr))
        if len(arr) == 1: return mean
        std = float(np.std(arr))
        if std < 1e-6: return mean
        mask = np.abs(arr - mean) <= self.z_thresh * std
        kept = arr[mask]
        return float(np.mean(kept)) if len(kept) > 0 else mean


def three_segment_inclination(spine_pts):
    pts   = np.array(spine_pts, dtype=float)
    n     = len(pts)
    n_end = max(1, n // 4)
    bot   = np.mean(pts[:n_end],          axis=0)
    mid   = np.mean(pts[n//2:3*n//4],    axis=0)
    top   = np.mean(pts[n - n_end:],      axis=0)
    dZ_lower = float(mid[2] - bot[2])
    if abs(dZ_lower) > 0.03:
        lumbar_fwd_deg = float(np.degrees(np.arctan2(mid[1] - bot[1], dZ_lower)))
        lumbar_lat_deg = float(np.degrees(np.arctan2(mid[0] - bot[0], dZ_lower)))
    else:
        lumbar_fwd_deg = lumbar_lat_deg = 0.0
    dZ_upper = float(top[2] - mid[2])
    if abs(dZ_upper) > 0.03:
        fhp_fwd_deg = float(np.degrees(np.arctan2(top[1] - mid[1], dZ_upper)))
        fhp_lat_deg = float(np.degrees(np.arctan2(top[0] - mid[0], dZ_upper)))
    else:
        fhp_fwd_deg = fhp_lat_deg = 0.0
    return {
        "lumbar_fwd_deg":   round(lumbar_fwd_deg,               2),
        "lumbar_lat_deg":   round(lumbar_lat_deg,               2),
        "fhp_fwd_deg":      round(fhp_fwd_deg,                  2),
        "fhp_lat_deg":      round(fhp_lat_deg,                  2),
        "relative_fhp_deg": round(fhp_fwd_deg - lumbar_fwd_deg, 2),
        "bot_pt":           bot.tolist(),
        "mid_pt":           mid.tolist(),
        "top_pt":           top.tolist(),
    }


def extract_spine_and_outline(points):
    if len(points) < 40:
        return None, None, None
    horiz = points[:, 1];  vert = points[:, 2]
    H, W  = 512, 512
    h_min, h_max = np.min(horiz), np.max(horiz)
    v_min, v_max = np.min(vert),  np.max(vert)
    if h_max == h_min or v_max == v_min:
        return None, None, None
    h_pad = (h_max - h_min) * 0.1;  v_pad = (v_max - v_min) * 0.1
    h_min -= h_pad;  h_max += h_pad;  v_min -= v_pad;  v_max += v_pad
    u_px = ((horiz - h_min) / (h_max - h_min) * (W - 1)).astype(int)
    v_px = ((1.0 - (vert - v_min) / (v_max - v_min)) * (H - 1)).astype(int)
    img  = np.zeros((H, W), dtype=np.uint8)
    p2d  = {}
    for idx in range(len(points)):
        p2d[(u_px[idx], v_px[idx])] = idx
        img[v_px[idx], u_px[idx]] = 255
    kd  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kc  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sil = cv2.morphologyEx(cv2.dilate(img, kd), cv2.MORPH_CLOSE, kc)
    outline_mesh = None
    contours, _  = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    avg_x        = float(np.mean(points[:, 0]))
    if contours:
        largest   = max(contours, key=cv2.contourArea)
        subsample = max(1, len(largest) // 200)
        outline_pts, outline_lines = [], []
        for i, pt in enumerate(largest[::subsample]):
            cu, cv_p = int(pt[0][0]), int(pt[0][1])
            best_idx = None;  best_d = float('inf')
            for du in range(-10, 11):
                for dv in range(-10, 11):
                    key = (cu + du, cv_p + dv)
                    if key in p2d:
                        d2 = du*du + dv*dv
                        if d2 < best_d:
                            best_d = d2;  best_idx = p2d[key]
            if best_idx is not None:
                outline_pts.append(points[best_idx].tolist())
            else:
                outline_pts.append([avg_x,
                                    h_min + (cu / W) * (h_max - h_min),
                                    v_max - (cv_p / H) * (v_max - v_min)])
            if i > 0:
                outline_lines.append([i - 1, i])
        if len(outline_pts) > 2:
            outline_lines.append([len(outline_pts) - 1, 0])
            pa = np.array(outline_pts, dtype=float);  n = len(pa)
            if n >= 11:
                smth = np.zeros_like(pa);  hw = 5
                for si in range(n):
                    idxs     = [(si + j - hw) % n for j in range(2 * hw + 1)]
                    smth[si] = pa[idxs].mean(axis=0)
                outline_pts = smth.tolist()
            outline_mesh = create_line_cylinders(
                outline_pts, outline_lines, [1.0, 0.0, 1.0], radius=0.012)
    z_vals       = points[:, 2]
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
    z_range      = z_max - z_min
    body = points[(z_vals >= z_min + z_range * 0.10) &
                  (z_vals <= z_max - z_range * 0.05)]
    if len(body) < 20:
        return None, outline_mesh, None
    bz             = body[:, 2]
    bz_min, bz_max = float(np.min(bz)), float(np.max(bz))
    step           = (bz_max - bz_min) / SPINE_SLICES
    spine_pts = []
    for i in range(SPINE_SLICES):
        lo, hi = bz_min + i * step, bz_min + (i + 1) * step
        mask   = (bz >= lo) & (bz < hi)
        if np.sum(mask) < 5: continue
        spine_pts.append(body[mask].mean(axis=0).tolist())
    spine_mesh  = None;  inclination = None
    if len(spine_pts) >= 4:
        spine_pts   = nms_spine_centroids(spine_pts)
        spine_lines = [[i, i + 1] for i in range(len(spine_pts) - 1)]
        if len(spine_pts) < 4:
            return None, outline_mesh, None
        combined = o3d.geometry.TriangleMesh()
        cyl = create_line_cylinders(spine_pts, spine_lines, [1.0, 1.0, 1.0], radius=0.022)
        if cyl is not None: combined += cyl
        n_pts = len(spine_pts)
        for j, pt in enumerate(spine_pts):
            t   = j / max(n_pts - 1, 1)
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.035, resolution=8)
            sph.translate(np.array(pt, dtype=float))
            sph.paint_uniform_color([t, 1.0, 1.0 - t])
            sph.compute_vertex_normals()
            combined += sph
        inclination = three_segment_inclination(spine_pts)
        bot = np.array(inclination["bot_pt"])
        mid = np.array(inclination["mid_pt"])
        top = np.array(inclination["top_pt"])
        for arrow_pts, col in [([bot, mid], [1.0, 0.55, 0.0]),
                                ([mid, top], [0.4,  1.0, 0.0])]:
            arr = create_line_cylinders([p.tolist() for p in arrow_pts],
                                        [[0, 1]], col, radius=0.030)
            if arr is not None: combined += arr
        for pt, col, rad in [(bot, [1.0, 0.2, 0.2], 0.045),
                              (mid, [1.0, 0.55, 0.0], 0.050),
                              (top, [0.2, 1.0, 0.2], 0.045)]:
            mk = o3d.geometry.TriangleMesh.create_sphere(radius=rad, resolution=8)
            mk.translate(np.array(pt, dtype=float))
            mk.paint_uniform_color(col)
            mk.compute_vertex_normals()
            combined += mk
        if len(np.asarray(combined.vertices)) > 0:
            combined.compute_vertex_normals()
            spine_mesh = combined
    return spine_mesh, outline_mesh, inclination


def calibrate_inclination(inc, ref):
    """
    Returns a copy of `inc` with all angle fields expressed as delta from the
    reference capture.  3-D landmark fields (bot_pt, mid_pt, top_pt) are kept
    unchanged so head-below tracking still works correctly.

    If no reference is set, `inc` is returned unmodified.
    """
    if ref is None or inc is None:
        return inc
    # Reference may store angles at top level or nested under 'inclination'
    ref_inc = ref if 'lumbar_fwd_deg' in ref else ref.get('inclination')
    if not ref_inc:
        return inc
    cal = dict(inc)
    for key in ('lumbar_fwd_deg', 'lumbar_lat_deg', 'fhp_fwd_deg', 'fhp_lat_deg'):
        if key in cal and key in ref_inc:
            cal[key] = round(float(cal[key]) - float(ref_inc[key]), 2)
    cal['relative_fhp_deg'] = round(
        cal.get('fhp_fwd_deg', 0.0) - cal.get('lumbar_fwd_deg', 0.0), 2)
    return cal


def print_inclination(inc, calibrated=False, reclining=False):
    if inc is None: return
    lf, ll = inc["lumbar_fwd_deg"], inc["lumbar_lat_deg"]
    ff, fl = inc["fhp_fwd_deg"],    inc["fhp_lat_deg"]
    rf     = inc["relative_fhp_deg"]

    mode = "  [Δ from ref]" if calibrated else ""

    if reclining:
        # Backward lean + head below reference → sitting back on chair, not a posture fault
        print(f"  ↩ Reclining on chair — lumbar alert suppressed{mode}\n"
              f"  Lumbar  │ Fwd/Back: {lf:+6.1f}°  Lat: {ll:+6.1f}°\n"
              f"  FHP     │ Fwd/Back: {ff:+6.1f}°  Lat: {fl:+6.1f}°\n"
              f"  Relative│ Head vs Lumbar: {rf:+6.1f}°")
        return

    lumbar_bad = abs(lf) > LUMBAR_ALERT_DEG or abs(ll) > LUMBAR_LAT_ALERT_DEG
    fhp_bad    = abs(rf) > FHP_RELATIVE_ALERT or abs(fl) > FHP_LAT_ALERT_DEG

    tag = ""
    if fhp_bad and lumbar_bad:
        tag = "  ⚠ ALERT: Lumbar + Forward-Head"
    elif fhp_bad:
        tag = "  ⚠ ALERT: Forward-Head Posture"
    elif lumbar_bad:
        tag = "  ⚠ ALERT: Lumbar Lean"

    print(f"  Lumbar  │ Fwd/Back: {lf:+6.1f}°  Lat: {ll:+6.1f}°{mode}\n"
          f"  FHP     │ Fwd/Back: {ff:+6.1f}°  Lat: {fl:+6.1f}°\n"
          f"  Relative│ Head vs Lumbar: {rf:+6.1f}°{tag}")


def get_human_indices(points, ransac_iters=300, remove_seat=True):
    if len(points) < 50: return None
    max_z = np.max(points[:, 2]);  min_z = np.min(points[:, 2])
    head_zone_mask  = (points[:, 2] >= (max_z - 0.25)) & (points[:, 2] <= max_z)
    head_candidates = points[head_zone_mask]
    best_score  = -1;  best_center = None
    if len(head_candidates) > 4:
        for _ in range(ransac_iters):
            idx = np.random.choice(len(head_candidates), 4, replace=False)
            pts = head_candidates[idx]
            A = np.zeros((4, 4));  b = np.zeros(4)
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
                    best_score = score;  best_center = center
            except np.linalg.LinAlgError:
                continue
    if best_center is None: return None
    horiz_dist = np.linalg.norm(points[:, :2] - best_center[:2], axis=1)
    human_mask = ((horiz_dist <= HUMAN_HORIZ_RADIUS) & (points[:, 2] <= max_z) &
                  (points[:, 2] > min_z + HUMAN_FLOOR_MARGIN))
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
#  Head-below-reference tracking  (runs in main viewer process each frame)
# ═══════════════════════════════════════════════════════════════════════════════

_head_below_prev = False

def _check_head_below_reference(inclination, shared_bounds):
    """
    Called every frame when a person is visible.
    - Computes head_delta against the stored reference.
    - Sets shared_bounds['head_delta'] and shared_bounds['head_below_alert'].
    - Prints a console warning on the frame the head first crosses below,
      and again each frame it remains below (so the operator sees it streaming).
    """
    global _head_below_prev

    ref = shared_bounds.get('reference_position')
    if ref is None:
        shared_bounds['head_below_alert'] = False
        shared_bounds['head_delta']       = None
        _head_below_prev = False
        return

    delta = compute_head_delta(inclination, ref)
    if delta is None:
        shared_bounds['head_below_alert'] = False
        shared_bounds['head_delta']       = None
        _head_below_prev = False
        return

    shared_bounds['head_delta'] = delta
    dh    = delta['delta_height_m']
    below = dh < -HEAD_BELOW_THRESH_M
    shared_bounds['head_below_alert'] = below

    if below:
        print(
            f"  ⚠ HEAD BELOW REFERENCE │"
            f" Δh={dh:+.3f} m ({abs(dh)*100:.1f} cm dropped) │"
            f" Δfwd={delta['delta_fhp_fwd_deg']:+.1f}° │"
            f" ΔrelFHP={delta['delta_relative_fhp_deg']:+.1f}°"
        )
    elif _head_below_prev:
        print(
            f"  ✓ HEAD BACK ABOVE REFERENCE │"
            f" Δh={dh:+.3f} m │"
            f" Δfwd={delta['delta_fhp_fwd_deg']:+.1f}°"
        )

    _head_below_prev = below


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI Control Panel
# ═══════════════════════════════════════════════════════════════════════════════

def roi_control_panel(shared_bounds):
    root = tk.Tk()
    root.title("ROI & Background Panel")
    root.geometry("370x1100")
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

    spatial_keys      = ['X1', 'X2', 'Y1', 'Y2', 'Z1', 'Z2']
    sliders           = {k: make_slider(k, shared_bounds[k]) for k in spatial_keys}
    last_saved_values = {k: s.get() for k, s in sliders.items()}

    tk.Frame(root, height=2, bd=1, relief="sunken").pack(fill="x", pady=10)

    # ── Section 1 — Background ────────────────────────────────────────────────
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

    # ── Section 2 — Chair filter ──────────────────────────────────────────────
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

    # ── Section 3 — Three-Segment Inclination ─────────────────────────────────
    inc_frame = tk.LabelFrame(root, text=" 3. Three-Segment Inclination (Option C) ",
                               font=('Arial', 10, 'bold'), padx=10, pady=5)
    inc_frame.pack(fill="x", pady=5)
    lbl_col_header = tk.Label(inc_frame, text="Segment         Fwd/Back    Lateral",
             font=('Courier', 8), fg="gray", anchor='w')
    lbl_col_header.pack(fill='x')
    tk.Frame(inc_frame, height=1, bg="lightgray").pack(fill='x', pady=2)
    lbl_lumbar = tk.Label(inc_frame, text="● Lumbar (bot→mid)    —       —",
                          font=('Courier', 10), fg="black", anchor='w')
    lbl_lumbar.pack(fill='x')
    lbl_fhp = tk.Label(inc_frame, text="● FHP    (mid→top)    —       —",
                       font=('Courier', 10), fg="black", anchor='w')
    lbl_fhp.pack(fill='x')
    tk.Frame(inc_frame, height=1, bg="lightgray").pack(fill='x', pady=2)
    lbl_rel = tk.Label(inc_frame, text="▲ Relative FHP:   —",
                       font=('Courier', 10, 'bold'), fg="black", anchor='w')
    lbl_rel.pack(fill='x')
    lbl_alert = tk.Label(inc_frame, text="", font=('Arial', 10, 'bold'), fg="red")
    lbl_alert.pack(pady=(4, 0))
    lbl_cal_mode = tk.Label(inc_frame, text="",
                            font=('Arial', 8, 'italic'), fg="#1565c0", anchor='w')
    lbl_cal_mode.pack(fill='x', pady=(2, 0))
    tk.Label(inc_frame,
             text=f"Smoothing: {SMOOTH_WINDOW}-frame avg  |  z>{SMOOTH_Z_THRESH}σ cut  |  clip ±{SMOOTH_HARD_CLIP:.0f}°",
             font=('Arial', 7), fg="gray", anchor='w').pack(fill='x', pady=(3, 0))
    tk.Label(inc_frame,
             text="● orange=thoracic anchor  ● red=pelvis  ● green=head",
             font=('Arial', 7), fg="gray", anchor='w').pack(fill='x', pady=(2, 0))

    # ── Section 4 — Reference Position Capture ────────────────────────────────
    (ref_status, ref_countdown,
     lbl_dh, lbl_df, lbl_da, lbl_drel,
     lbl_head_alert) = _build_section4(root, shared_bounds)

    # ── Polling loop ──────────────────────────────────────────────────────────
    def update_shared_dict():
        nonlocal last_saved_values
        current_vals = {};  has_changed = False
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
            except Exception: pass

        b_st = shared_bounds.get('bg_state', 'idle')
        if b_st == 'active':
            bg_status.config(text="Active (Room Hidden)", fg="green")
        elif b_st == 'idle':
            bg_status.config(text="Inactive", fg="gray")
        elif b_st == 'scanning':
            bg_status.config(text="Scanning empty room...", fg="blue")

        inc = shared_bounds.get('inclination')
        if inc:
            lf, ll = inc.get("lumbar_fwd_deg", 0.0), inc.get("lumbar_lat_deg", 0.0)
            ff, fl = inc.get("fhp_fwd_deg",    0.0), inc.get("fhp_lat_deg",    0.0)
            rf     = inc.get("relative_fhp_deg", 0.0)

            ref_active   = shared_bounds.get('reference_position') is not None
            cal_active   = shared_bounds.get('angle_calibrated', False)
            head_below   = shared_bounds.get('head_below_alert', False)
            is_reclining = shared_bounds.get('is_reclining', False)

            if cal_active:
                # Reference set AND head at/above reference — calibrated mode
                lbl_col_header.config(
                    text="Segment   Fwd/Back    Lateral   [Δ ref]")
                lbl_cal_mode.config(
                    text="● Calibrated — showing deviation from reference posture",
                    fg="#1565c0")
            elif ref_active and head_below:
                # Reference set BUT head is below it — suspend calibration
                lbl_col_header.config(
                    text="Segment         Fwd/Back    Lateral")
                lbl_cal_mode.config(
                    text="⚠ Head below reference — showing absolute angles",
                    fg="orange")
            else:
                # No reference set
                lbl_col_header.config(
                    text="Segment         Fwd/Back    Lateral")
                lbl_cal_mode.config(text="")

            # Alert logic — suppressed during reclining
            lumbar_bad = (abs(lf) > LUMBAR_ALERT_DEG or abs(ll) > LUMBAR_LAT_ALERT_DEG) \
                         and not is_reclining
            fhp_bad    = abs(rf) > FHP_RELATIVE_ALERT or abs(fl) > FHP_LAT_ALERT_DEG

            lbl_lumbar.config(
                text=f"● Lumbar (bot→mid)  {lf:+6.1f}°  {ll:+6.1f}°",
                fg="red" if lumbar_bad else "#b35a00")
            lbl_fhp.config(
                text=f"● FHP    (mid→top)  {ff:+6.1f}°  {fl:+6.1f}°",
                fg="red" if fhp_bad else "dark green")
            lbl_rel.config(
                text=f"▲ Relative FHP:  {rf:+6.1f}°",
                fg="red" if fhp_bad else ("dark orange" if abs(rf) > 6 else "black"))

            if is_reclining:
                lbl_alert.config(text="↩ Reclining — lumbar alert suppressed", fg="#7b5ea7")
            elif fhp_bad and lumbar_bad:
                lbl_alert.config(text="⚠ Lumbar + Forward-Head detected!", fg="red")
            elif fhp_bad:
                lbl_alert.config(text="⚠ Forward-Head Posture detected!", fg="red")
            elif lumbar_bad:
                lbl_alert.config(text="⚠ Excessive lumbar lean detected!", fg="red")
            else:
                lbl_alert.config(text="✓ Posture within range", fg="green")

        # Refresh reference delta labels + head-below banner
        _refresh_delta_labels(shared_bounds, lbl_dh, lbl_df, lbl_da, lbl_drel,
                              lbl_head_alert)

        root.after(100, update_shared_dict)

    update_shared_dict()
    root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
#  Protocol & Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def livox_crc16(data):
    crc = 0x4C49
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc

def livox_crc32(data):
    return binascii.crc32(data, 0x564F580A) & 0xFFFFFFFF

_seq_num = 0
def _next_seq():
    global _seq_num
    s, _seq_num = _seq_num, (_seq_num + 1) & 0xFFFF;  return s

def build_cmd(cmd_set, cmd_id, payload=b'', cmd_type=0x00):
    data = bytes([cmd_set, cmd_id]) + payload
    total_len = 9 + len(data) + 4
    pre_crc   = struct.pack('<BBHBH', 0xAA, 0x01, total_len, cmd_type, _next_seq())
    header    = pre_crc + struct.pack('<H', livox_crc16(pre_crc))
    pkt       = header + data
    return pkt + struct.pack('<I', livox_crc32(pkt))

def cmd_handshake(host_ip, dp, cp, ip):
    return build_cmd(0x00, 0x01,
                     socket.inet_aton(host_ip) + struct.pack('<HHH', dp, cp, ip),
                     cmd_type=0x01)
def cmd_heartbeat():            return build_cmd(0x00, 0x03)
def cmd_start_sampling(s=True): return build_cmd(0x00, 0x04, bytes([0x01 if s else 0x00]))
def cmd_set_cartesian():        return build_cmd(0x00, 0x05, bytes([0x00]))

_DATA_HDR = 18;  _dtype_logged = False

def parse_data_packet(data):
    global _dtype_logged
    if len(data) < _DATA_HDR + 1 or data[0] == 0xAA: return []
    dtype = data[9];  offset = _DATA_HDR;  pts = []
    if not _dtype_logged:
        print(f"  [INFO] LiDAR dtype={dtype}  packet size={len(data)}")
        _dtype_logged = True
    if dtype == 0:
        for _ in range((len(data) - _DATA_HDR) // 13):
            if offset + 13 > len(data): break
            x, y, z, _ = struct.unpack_from('<iiiB', data, offset)
            pts.append([x*1e-3, y*1e-3, z*1e-3]);  offset += 13
    elif dtype == 1:
        for _ in range((len(data) - _DATA_HDR) // 9):
            if offset + 9 > len(data): break
            depth, theta, phi, _ = struct.unpack_from('<IHHB', data, offset)
            r = depth * 1e-3
            ze = np.radians(theta / 100.0);  az = np.radians(phi / 100.0)
            pts.append([r*np.sin(ze)*np.cos(az), r*np.sin(ze)*np.sin(az), r*np.cos(ze)])
            offset += 9
    elif dtype == 2:
        for _ in range((len(data) - _DATA_HDR) // 14):
            if offset + 14 > len(data): break
            x, y, z, _, _ = struct.unpack_from('<iiiBB', data, offset)
            pts.append([x*1e-3, y*1e-3, z*1e-3]);  offset += 14
    elif dtype == 4:
        for _ in range((len(data) - _DATA_HDR) // 14):
            if offset + 14 > len(data): break
            x, y, z, _, _ = struct.unpack_from('<fffBB', data, offset)
            pts.append([x, y, z]);  offset += 14
    elif dtype == 5:
        for _ in range((len(data) - _DATA_HDR) // 16):
            if offset + 16 > len(data): break
            x, y, z, _, _ = struct.unpack_from('<fffBB', data, offset)
            pts.append([x, y, z]);  offset += 16
    elif dtype == 7:
        for _ in range((len(data) - _DATA_HDR) // 15):
            if offset + 15 > len(data): break
            x, y, z, _, _, _ = struct.unpack_from('<iiiBBB', data, offset)
            pts.append([x*1e-3, y*1e-3, z*1e-3]);  offset += 15
    else:
        print(f"  [WARN] Unhandled dtype={dtype}  size={len(data)}")
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
    print("  Livox Avia – Posture Viewer (Option C + Reference Capture)\n")

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
                print("  [ERROR] No broadcast from LiDAR within 10s."); return
            if len(raw) >= 34 and raw[0] == 0xAA and raw[9] == 0x00 and raw[10] == 0x00:
                device_ip = addr[0]
    finally:
        bcast_sock.close()

    dest = (device_ip, DEVICE_CMD_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((device_ip, 1));  host_ip = s.getsockname()[0]

    print(f"  Device IP : {device_ip}")
    print(f"  Host IP   : {host_ip}")
    print(f"  Data port : {HOST_DATA_PORT}  (0.0.0.0)\n")

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
    data_sock.bind(("0.0.0.0", HOST_DATA_PORT))
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
    ro.point_size       = 2.0
    ro.background_color = np.array([0.05, 0.05, 0.05])

    if initial_camera is not None:
        try:
            with open("temp_cam_load.json", "w") as f: json.dump(initial_camera, f)
            cam = o3d.io.read_pinhole_camera_parameters("temp_cam_load.json")
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
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
    accumulated_pts    = []
    last_render_time   = time.time()

    active_spine_geom          = None
    active_outline_geom        = None
    active_ref_geom_container  = [None]

    smoother           = TemporalSmoother(SMOOTH_WINDOW, SMOOTH_Z_THRESH, SMOOTH_HARD_CLIP)
    person_was_visible = False
    _frame_count       = [0]

    try:
        while True:
            while True:
                try:
                    pkt, _ = data_sock.recvfrom(1500)
                    accumulated_pts.extend(parse_data_packet(pkt))
                except (BlockingIOError, OSError):
                    break

            current_time = time.time()
            if current_time - last_render_time >= FRAME_TIME_S:
                _frame_count[0] += 1
                print(f"  [Frame {_frame_count[0]:04d}] Raw pts: {len(accumulated_pts)}")

                min_b = [min(shared_bounds['X1'], shared_bounds['X2']),
                         min(shared_bounds['Y1'], shared_bounds['Y2']),
                         min(shared_bounds['Z1'], shared_bounds['Z2'])]
                max_b = [max(shared_bounds['X1'], shared_bounds['X2']),
                         max(shared_bounds['Y1'], shared_bounds['Y2']),
                         max(shared_bounds['Z1'], shared_bounds['Z2'])]

                roi_box_vis.points = get_box_lineset(min_b, max_b).points
                vis.update_geometry(roi_box_vis)

                if active_spine_geom is not None:
                    vis.remove_geometry(active_spine_geom,   reset_bounding_box=False)
                    active_spine_geom = None
                if active_outline_geom is not None:
                    vis.remove_geometry(active_outline_geom, reset_bounding_box=False)
                    active_outline_geom = None

                _remove_reference_ghost(vis, active_ref_geom_container)
                _update_reference_ghost(vis, shared_bounds, active_ref_geom_container)

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
                    print(f"  [Frame {_frame_count[0]:04d}] After ROI: {len(arr)}")

                    bg_state = shared_bounds.get('bg_state', 'idle')
                    if bg_state == 'clear':
                        background_pcd = None;  bg_accumulating = False
                        if os.path.exists(BG_PCD_FILE):
                            try: os.remove(BG_PCD_FILE)
                            except Exception: pass
                        shared_bounds['bg_state'] = 'idle'
                    elif bg_state == 'capture':
                        bg_accumulating    = True
                        bg_accumulated_pts = []
                        bg_start_time      = time.time()
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

                    # ── Stage 1 denoise: scene-level voxel + outlier removal ──
                    if len(arr) > 20:
                        arr_d = denoise_pointcloud(arr, voxel=True)
                        if len(arr_d) >= 20:
                            dists = np.linalg.norm(arr_d, axis=1)
                            arr   = arr_d
                        print(f"  [Frame {_frame_count[0]:04d}] After denoise: {len(arr)}")

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
                                    arr      = arr_h[_labels == _biggest]
                                    dists    = dists_h[_labels == _biggest]
                                else:
                                    arr, dists = arr_h, dists_h
                            else:
                                arr, dists = arr_h, dists_h

                            # ── Stage 2 denoise: tight body-only pass ─────────
                            arr_b = denoise_pointcloud(arr, voxel=False)
                            if len(arr_b) >= 40:
                                dists = np.linalg.norm(arr_b, axis=1)
                                arr   = arr_b
                            print(f"  [Frame {_frame_count[0]:04d}] Body pts: {len(arr)}")

                            active_spine_geom, active_outline_geom, inclination = \
                                extract_spine_and_outline(arr)

                            if inclination is not None:
                                inclination = smoother.update(inclination)

                                # ── Head-below check runs FIRST on raw angles ──
                                # so the calibration decision is current-frame accurate.
                                _check_head_below_reference(inclination, shared_bounds)

                                ref        = shared_bounds.get('reference_position')
                                head_below = shared_bounds.get('head_below_alert', False)

                                # ── Reclining detection ────────────────────────
                                # Head below reference + BOTH spine segments leaning
                                # backward → person is sitting back on the chair.
                                # Head below + NOT backward → person is slumping forward.
                                raw_fhp_fwd    = inclination.get('fhp_fwd_deg',    0.0)
                                raw_lumbar_fwd = inclination.get('lumbar_fwd_deg', 0.0)
                                is_reclining   = (
                                    head_below
                                    and raw_fhp_fwd    < RECLINE_FHP_DEG
                                    and raw_lumbar_fwd < RECLINE_LUMBAR_DEG
                                )
                                shared_bounds['is_reclining'] = is_reclining

                                # Calibration is active only when a reference is set
                                # AND the head is at or above that reference height.
                                # While the head is dropped the raw angles are shown
                                # so the alert still fires on the real absolute lean.
                                if ref is not None and not head_below:
                                    inclination_cal = calibrate_inclination(inclination, ref)
                                    calibrated      = True
                                else:
                                    inclination_cal = inclination
                                    calibrated      = False

                                shared_bounds['inclination']      = inclination_cal
                                shared_bounds['angle_calibrated'] = calibrated
                                print_inclination(inclination_cal,
                                                  calibrated=calibrated,
                                                  reclining=is_reclining)
                                person_was_visible = True
                            else:
                                if person_was_visible:
                                    smoother.reset()
                                    person_was_visible = False
                                shared_bounds['inclination']      = None
                                shared_bounds['head_below_alert'] = False
                                shared_bounds['head_delta']       = None
                        else:
                            if person_was_visible:
                                smoother.reset()
                                person_was_visible = False
                            shared_bounds['inclination']      = None
                            shared_bounds['head_below_alert'] = False
                            shared_bounds['head_delta']       = None

                    if active_spine_geom is not None:
                        vis.add_geometry(active_spine_geom,   reset_bounding_box=False)
                    if active_outline_geom is not None:
                        vis.add_geometry(active_outline_geom, reset_bounding_box=False)

                    if len(arr) > 0:
                        pcd.points = o3d.utility.Vector3dVector(arr)
                        nd     = np.clip(dists / MAX_COLOR_DIST_M, 0.0, 1.0)
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
        _remove_reference_ghost(vis, active_ref_geom_container)
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

    initial_bounds.update({
        'bg_state':           'idle',
        'chair_state':        'idle',
        'inclination':        None,
        'reference_position': None,
        'head_delta':         None,
        'head_below_alert':   False,
        'angle_calibrated':   False,
        'is_reclining':       False,
    })

    with multiprocessing.Manager() as manager:
        shared_bounds = manager.dict(initial_bounds)
        gui_process   = multiprocessing.Process(
            target=roi_control_panel, args=(shared_bounds,))
        gui_process.start()
        try:
            live_livox_viewer(shared_bounds, initial_camera)
        finally:
            gui_process.terminate();  gui_process.join()