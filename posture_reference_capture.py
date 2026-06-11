#!/usr/bin/env python3
"""
posture_reference_capture.py
Reference Position Capture — helper module for the Livox Avia Posture Viewer.
"""

import time
import datetime
import numpy as np

try:
    import open3d as o3d
    import tkinter as tk
except ImportError:
    o3d = None
    tk  = None

# ── Alert threshold — how far the head must drop to trigger the warning ────────
HEAD_BELOW_THRESH_M = 0.03   # 3 cm


# ═══════════════════════════════════════════════════════════════════════════════
#  Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def capture_reference_position(shared_bounds) -> bool:
    """
    Snapshot the current smoothed head/spine state as a fixed reference baseline.
    Returns True if a valid inclination was available to capture.
    """
    inc = shared_bounds.get('inclination')
    if inc is None:
        return False

    shared_bounds['reference_position'] = {
        'top_pt':           list(inc['top_pt']),
        'mid_pt':           list(inc['mid_pt']),
        'bot_pt':           list(inc['bot_pt']),
        'fhp_fwd_deg':      float(inc['fhp_fwd_deg']),
        'fhp_lat_deg':      float(inc['fhp_lat_deg']),
        'lumbar_fwd_deg':   float(inc['lumbar_fwd_deg']),
        'relative_fhp_deg': float(inc['relative_fhp_deg']),
        'captured_at':      time.time(),
    }
    # Reset any stale alert state from a previous reference
    shared_bounds['head_below_alert'] = False
    shared_bounds['head_delta']       = None
    return True


def compute_head_delta(inclination: dict, reference: dict) -> dict | None:
    """
    Compute how far the current head/spine has drifted from the captured reference.

    Positive delta_height_m  = head moved UP   from reference.
    Negative delta_height_m  = head dropped DOWN (slouching / forward bend).
    """
    if inclination is None or reference is None:
        return None

    top_now = np.array(inclination['top_pt'], dtype=float)
    top_ref = np.array(reference['top_pt'],   dtype=float)

    delta_h = float(top_now[2] - top_ref[2])   # Z: vertical (+up / -down)
    delta_f = float(top_now[1] - top_ref[1])   # Y: depth    (+away / -toward)
    delta_l = float(top_now[0] - top_ref[0])   # X: lateral  (+right / -left)

    d_fhp_fwd = float(inclination['fhp_fwd_deg'])      - float(reference['fhp_fwd_deg'])
    d_fhp_lat = float(inclination['fhp_lat_deg'])      - float(reference['fhp_lat_deg'])
    d_lum     = float(inclination['lumbar_fwd_deg'])   - float(reference['lumbar_fwd_deg'])
    d_rel     = float(inclination['relative_fhp_deg']) - float(reference['relative_fhp_deg'])

    return {
        'delta_height_m':         round(delta_h,   3),
        'delta_forward_m':        round(delta_f,   3),
        'delta_lateral_m':        round(delta_l,   3),
        'delta_fhp_fwd_deg':      round(d_fhp_fwd, 2),
        'delta_fhp_lat_deg':      round(d_fhp_lat, 2),
        'delta_lumbar_fwd_deg':   round(d_lum,     2),
        'delta_relative_fhp_deg': round(d_rel,     2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Section 4 GUI Panel
# ═══════════════════════════════════════════════════════════════════════════════

def _build_section4(root, shared_bounds):
    """
    Build and pack the Section 4 'Reference Position Capture' LabelFrame.

    Returns
    -------
    tuple: (ref_status, ref_countdown, lbl_dh, lbl_df, lbl_da, lbl_drel,
            lbl_head_alert)
    """
    ref_frame = tk.LabelFrame(
        root, text=" 4. Reference Position Capture ",
        font=('Arial', 10, 'bold'), padx=10, pady=6)
    ref_frame.pack(fill="x", pady=5)

    # ── Capture status & countdown ────────────────────────────────────────────
    ref_status = tk.Label(ref_frame, text="No reference captured",
                          font=('Arial', 9), fg="gray")
    ref_status.pack(pady=(0, 2))

    ref_countdown = tk.Label(ref_frame, text="",
                             font=('Arial', 16, 'bold'), fg="darkorange")
    ref_countdown.pack()

    def do_capture_countdown(count):
        if count > 0:
            ref_countdown.config(text=f"Capturing in {count}…")
            root.after(1000, do_capture_countdown, count - 1)
        else:
            ref_countdown.config(text="")
            ok = capture_reference_position(shared_bounds)
            if ok:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                ref_status.config(text=f"✓ Reference set at {ts}", fg="dark green")
            else:
                ref_status.config(
                    text="✗ No person detected — enable chair filter first",
                    fg="red")

    def do_clear():
        shared_bounds['reference_position'] = None
        shared_bounds['head_delta']         = None
        shared_bounds['head_below_alert']   = False
        ref_status.config(text="No reference captured", fg="gray")
        lbl_head_alert.config(text="", bg=_default_bg())
        for lbl, txt in zip(
            (lbl_dh, lbl_df, lbl_da, lbl_drel),
            ("Height", "Fwd/Back drift", "Head angle Δ (fwd)", "Rel FHP Δ"),
        ):
            lbl.config(text=f"  {txt}:  —", fg="gray")

    def _default_bg():
        try:    return root.cget('bg')
        except: return "SystemButtonFace"

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_row = tk.Frame(ref_frame);  btn_row.pack(fill='x', pady=(4, 6))
    tk.Button(btn_row, text="📸  Capture Reference (3 s)",
              command=lambda: do_capture_countdown(3),
              bg="#e8f5e9", font=('Arial', 10), relief='groove'
              ).pack(side='left', expand=True, fill='x', padx=(0, 4))
    tk.Button(btn_row, text="✕ Clear",
              command=do_clear,
              bg="#ffebee", font=('Arial', 10), relief='groove'
              ).pack(side='right', fill='x')

    # ── Prominent head-below-reference alert (shown / hidden dynamically) ─────
    lbl_head_alert = tk.Label(
        ref_frame, text="",
        font=('Arial', 11, 'bold'), fg="white",
        anchor='center', pady=4)
    lbl_head_alert.pack(fill='x', pady=(0, 4))

    # ── Separator + delta labels ───────────────────────────────────────────────
    tk.Frame(ref_frame, height=1, bg="lightgray").pack(fill='x', pady=(0, 3))
    tk.Label(ref_frame,
             text="Changes since reference:  (+ = raised / forward / worsened)",
             font=('Arial', 8), fg="gray", anchor='w').pack(fill='x')

    lbl_dh = tk.Label(ref_frame, text="  Height:              —",
                      font=('Courier', 10), anchor='w')
    lbl_dh.pack(fill='x')

    lbl_df = tk.Label(ref_frame, text="  Fwd/Back drift:      —",
                      font=('Courier', 10), anchor='w')
    lbl_df.pack(fill='x')

    lbl_da = tk.Label(ref_frame, text="  Head angle Δ (fwd):  —",
                      font=('Courier', 10), anchor='w')
    lbl_da.pack(fill='x')

    lbl_drel = tk.Label(ref_frame, text="  Rel FHP Δ:           —",
                        font=('Courier', 11, 'bold'), anchor='w')
    lbl_drel.pack(fill='x')

    tk.Label(ref_frame,
             text=f"Alert threshold: head drops >{HEAD_BELOW_THRESH_M*100:.0f} cm below reference",
             font=('Arial', 7), fg="#888", anchor='w').pack(fill='x', pady=(3, 0))

    return (ref_status, ref_countdown,
            lbl_dh, lbl_df, lbl_da, lbl_drel,
            lbl_head_alert)


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI delta refresh  — called every 100 ms from update_shared_dict()
# ═══════════════════════════════════════════════════════════════════════════════

def _refresh_delta_labels(shared_bounds,
                          lbl_dh, lbl_df, lbl_da, lbl_drel,
                          lbl_head_alert):
    """
    Read pre-computed head_delta and head_below_alert from shared_bounds
    (written each frame by the main viewer process) and update all GUI labels.
    """
    ref   = shared_bounds.get('reference_position')
    delta = shared_bounds.get('head_delta')
    below = shared_bounds.get('head_below_alert', False)

    # ── Head-below-reference alert banner ─────────────────────────────────────
    if ref is None:
        lbl_head_alert.config(text="", bg=_safe_bg(lbl_head_alert))
    elif below and delta is not None:
        dh = delta['delta_height_m']
        lbl_head_alert.config(
            text=f"⚠  HEAD BELOW REFERENCE  ▼ {abs(dh)*100:.1f} cm",
            fg="white", bg="red")
    elif delta is not None:
        lbl_head_alert.config(
            text="✓  Head above reference",
            fg="#155724", bg="#d4edda")
    else:
        # reference set but person not visible
        lbl_head_alert.config(
            text="— Person not detected —",
            fg="gray", bg=_safe_bg(lbl_head_alert))

    # ── Delta value labels ────────────────────────────────────────────────────
    if delta is None:
        for lbl, base in zip(
            (lbl_dh, lbl_df, lbl_da, lbl_drel),
            ("Height", "Fwd/Back drift", "Head angle Δ (fwd)", "Rel FHP Δ"),
        ):
            lbl.config(text=f"  {base}:  —", fg="gray")
        return

    dh  = delta['delta_height_m']
    df  = delta['delta_forward_m']
    daf = delta['delta_fhp_fwd_deg']
    drl = delta['delta_relative_fhp_deg']

    def _col_h(v):
        return "red" if abs(v) > 0.08 else ("dark orange" if abs(v) > 0.04 else "dark green")
    def _col_a(v):
        return "red" if abs(v) > 12.0 else ("dark orange" if abs(v) > 7.0 else "dark green")

    h_hint = "▼ dropped" if dh < -0.01 else ("▲ raised" if dh > 0.01 else "stable")
    lbl_dh.config(
        text=f"  Height:              {dh:+.3f} m  ({h_hint})",
        fg=_col_h(dh))

    f_hint = "→ fwd" if df > 0.01 else ("← back" if df < -0.01 else "stable")
    lbl_df.config(
        text=f"  Fwd/Back drift:      {df:+.3f} m  ({f_hint})",
        fg=_col_h(df))

    lbl_da.config(
        text=f"  Head angle Δ (fwd):  {daf:+.1f}°",
        fg=_col_a(daf))

    lbl_drel.config(
        text=f"  Rel FHP Δ:           {drl:+.1f}°",
        fg=_col_a(drl))


def _safe_bg(widget):
    try:    return widget.winfo_toplevel().cget('bg')
    except: return "SystemButtonFace"


# ═══════════════════════════════════════════════════════════════════════════════
#  3-D ghost sphere — reference head position marker
# ═══════════════════════════════════════════════════════════════════════════════

def _update_reference_ghost(vis, shared_bounds, active_ref_geom_container: list):
    """
    Draw a cyan ghost sphere + crosshair at the captured reference head position.
    active_ref_geom_container is a one-element list used as mutable state.
    """
    if active_ref_geom_container[0] is not None:
        for g in active_ref_geom_container[0]:
            vis.remove_geometry(g, reset_bounding_box=False)
        active_ref_geom_container[0] = None

    ref = shared_bounds.get('reference_position')
    if ref is None:
        return

    top_ref = np.array(ref['top_pt'], dtype=float)

    ghost = o3d.geometry.TriangleMesh.create_sphere(radius=0.07, resolution=10)
    ghost.translate(top_ref)
    ghost.paint_uniform_color([0.4, 1.0, 1.0])
    ghost.compute_vertex_normals()

    # Cross-hair at reference centroid
    cross_pts = [
        (top_ref + np.array([-0.12, 0, 0])).tolist(),
        (top_ref + np.array([ 0.12, 0, 0])).tolist(),
        (top_ref + np.array([0, -0.12, 0])).tolist(),
        (top_ref + np.array([0,  0.12, 0])).tolist(),
        (top_ref + np.array([0, 0, -0.12])).tolist(),
        (top_ref + np.array([0, 0,  0.12])).tolist(),
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(cross_pts)
    ls.lines  = o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5]])
    ls.colors = o3d.utility.Vector3dVector([[0.4, 1.0, 1.0]] * 3)

    vis.add_geometry(ghost, reset_bounding_box=False)
    vis.add_geometry(ls,    reset_bounding_box=False)
    active_ref_geom_container[0] = (ghost, ls)


def _remove_reference_ghost(vis, active_ref_geom_container: list):
    if active_ref_geom_container[0] is not None:
        for g in active_ref_geom_container[0]:
            vis.remove_geometry(g, reset_bounding_box=False)
        active_ref_geom_container[0] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import multiprocessing

    print("=== posture_reference_capture self-test ===\n")

    with multiprocessing.Manager() as manager:
        sb = manager.dict({
            'inclination': None,
            'head_delta': None,
            'head_below_alert': False,
            'reference_position': None,
        })

        ok = capture_reference_position(sb)
        assert not ok
        print("✓  Returns False when inclination is None")

        sb['inclination'] = {
            'top_pt': [0.02, 0.80, 1.70], 'mid_pt': [0.01, 0.75, 1.20],
            'bot_pt': [0.00, 0.70, 0.70], 'fhp_fwd_deg': 4.5,
            'fhp_lat_deg': -1.2, 'lumbar_fwd_deg': 2.1, 'relative_fhp_deg': 2.4,
        }
        ok = capture_reference_position(sb)
        assert ok
        print(f"✓  Captured: top_pt={sb['reference_position']['top_pt']}")

        # Simulate head dropping 8 cm — should trigger below-threshold
        sb['inclination'] = {
            'top_pt': [0.02, 0.85, 1.62], 'mid_pt': [0.01, 0.78, 1.18],
            'bot_pt': [0.00, 0.72, 0.68], 'fhp_fwd_deg': 14.0,
            'fhp_lat_deg': -0.8, 'lumbar_fwd_deg': 5.0, 'relative_fhp_deg': 9.0,
        }
        delta = compute_head_delta(sb['inclination'], sb['reference_position'])
        assert delta is not None
        dh = delta['delta_height_m']
        below = dh < -HEAD_BELOW_THRESH_M
        print(f"✓  delta_height_m = {dh:+.3f} m  →  below={below}  (threshold={-HEAD_BELOW_THRESH_M})")
        assert below, "Expected head-below alert to be True"
        print("\n✓  All assertions passed.")