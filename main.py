#!/usr/bin/env python3
"""
Livox Avia – Direct SDK Viewer
(With Real-time ML Tasks API Human Skeleton Estimation & White Outline Contour)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

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
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BROADCAST_PORT = 55000
DEVICE_CMD_PORT = 65000

HOST_DATA_PORT = 60001
HOST_CMD_PORT = 60000
HOST_IMU_PORT = 60002

HEARTBEAT_INTERVAL_S = 1.0

FRAME_TIME_S = 0.5
MAX_COLOR_DIST_M = 5.0

CONFIG_FILE = "roi_settings.json"
BG_PCD_FILE = "background_model.pcd"
MODEL_TASK_FILE = "pose_landmarker_full.task"

# Standard kinematic pairs for mapping modern MediaPipe landmarks to a skeleton lineset
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Shoulders and arms
    (11, 23), (12, 24), (23, 24),  # Torso / Hips
    (23, 25), (25, 27), (24, 26), (26, 28),  # Legs (Upper & Lower)
    (27, 29), (28, 30), (29, 31), (30, 32)  # Feet
]


# ═══════════════════════════════════════════════════════════════════════════════
#  ML Tasks Skeleton Estimation & Contour Outline Generation
# ═══════════════════════════════════════════════════════════════════════════════

def extract_skeleton_and_outline(points, landmarker):
    if len(points) < 40 or landmarker is None:
        return None, None

    # --- DYNAMIC AXIS SELECTION ---
    # Calculate the spread (width) of the points on X and Y axes
    x_spread = np.max(points[:, 0]) - np.min(points[:, 0])
    y_spread = np.max(points[:, 1]) - np.min(points[:, 1])

    # Project onto whichever axis the person is currently taking up the most space on
    if x_spread > y_spread:
        depths = points[:, 1]  # Y becomes depth
        horiz = points[:, 0]   # X becomes horizontal
    else:
        depths = points[:, 0]  # X becomes depth
        horiz = points[:, 1]   # Y becomes horizontal

    vert = points[:, 2]  # Z is always vertical
    # ------------------------------

    h, w = 512, 512  # Must be square
    img = np.zeros((h, w), dtype=np.uint8)

    h_min, h_max = np.min(horiz), np.max(horiz)
    v_min, v_max = np.min(vert), np.max(vert)

    if h_max == h_min or v_max == v_min:
        return None, None

    # Pad the bounds slightly
    h_pad = (h_max - h_min) * 0.1
    v_pad = (v_max - v_min) * 0.1
    h_min -= h_pad; h_max += h_pad
    v_min -= v_pad; v_max += v_pad

    # Map spatial values directly to 2D pixel index arrays
    u = ((horiz - h_min) / (h_max - h_min) * (w - 1)).astype(int)
    v = ((1.0 - (vert - v_min) / (v_max - v_min)) * (h - 1)).astype(int)

    # Dictionary map linking 2D spatial pixels back to original 3D indices
    pixel_to_3d_idx = {}
    for idx in range(len(points)):
        pixel_to_3d_idx[(u[idx], v[idx])] = idx
        img[v[idx], u[idx]] = 255

    # ──── STEP A: Outer Body Contour Tracing ────
    # Dilate points to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img_dilated = cv2.dilate(img, kernel)

    contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    skeleton_lineset = None
    outline_lineset = None
    largest_contour = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        outline_pts_3d = []
        outline_lines = []

        subsample_rate = max(1, len(largest_contour) // 120)

        for i, pt in enumerate(largest_contour[::subsample_rate]):
            c_u, c_v = pt[0][0], pt[0][1]

            best_idx = None
            min_pixel_dist = float('inf')
            for du in range(-10, 11):
                for dv in range(-10, 11):
                    nu, nv = c_u + du, c_v + dv
                    if (nu, nv) in pixel_to_3d_idx:
                        d = du * du + dv * dv
                        if d < min_pixel_dist:
                            min_pixel_dist = d
                            best_idx = pixel_to_3d_idx[(nu, nv)]

            if best_idx is not None:
                outline_pts_3d.append(points[best_idx])
            else:
                avg_depth = np.mean(depths)
                real_y = h_min + ((c_u / w) * (h_max - h_min))
                real_z = v_max - ((c_v / h) * (v_max - v_min))
                outline_pts_3d.append([avg_depth, real_y, real_z])

            if i > 0:
                outline_lines.append([i - 1, i])

        if len(outline_pts_3d) > 2:
            outline_lines.append([len(outline_pts_3d) - 1, 0])
            outline_lineset = o3d.geometry.LineSet()
            outline_lineset.points = o3d.utility.Vector3dVector(np.array(outline_pts_3d))
            outline_lineset.lines = o3d.utility.Vector2iVector(np.array(outline_lines))
            # HIGHLIGHT: Bright Magenta for visibility
            outline_lineset.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 1.0] for _ in range(len(outline_lines))])

    # ──── STEP B: Tasks API Human Pose Estimation (Skeleton) ────

    # Create a blank RGB image for MediaPipe
    mp_input_img = np.zeros((h, w, 3), dtype=np.uint8)

    if largest_contour is not None:
        # Crucial Fix: Fill the contour to create a solid "human silhouette"
        cv2.drawContours(mp_input_img, [largest_contour], -1, (180, 180, 180), thickness=cv2.FILLED)
    else:
        mp_input_img = cv2.cvtColor(img_dilated, cv2.COLOR_GRAY2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_input_img)
    detection_result = landmarker.detect(mp_image)

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        skel_pts_3d = []
        lm_to_3d_index_map = {}

        for i, lm in enumerate(landmarks):
            # Ignore face landmarks to prevent jitter, focus on torso/limbs
            if i < 11:
                continue

                # Lowered visibility threshold since silhouettes lack texture detail
            if lm.visibility < 0.2:
                continue

            lm_u = int(lm.x * w)
            lm_v = int(lm.y * h)

            best_idx = None
            min_pixel_dist = float('inf')
            # Widen the search radius to snap to the point cloud
            for du in range(-30, 31):
                for dv in range(-30, 31):
                    nu, nv = lm_u + du, lm_v + dv
                    if (nu, nv) in pixel_to_3d_idx:
                        d = du * du + dv * dv
                        if d < min_pixel_dist:
                            min_pixel_dist = d
                            best_idx = pixel_to_3d_idx[(nu, nv)]

            if best_idx is not None:
                skel_pts_3d.append(points[best_idx])
                lm_to_3d_index_map[i] = len(skel_pts_3d) - 1
            else:
                avg_depth = np.mean(depths)
                real_y = h_min + (lm.x * (h_max - h_min))
                real_z = v_max - (lm.y * (v_max - v_min))
                skel_pts_3d.append([avg_depth, real_y, real_z])
                lm_to_3d_index_map[i] = len(skel_pts_3d) - 1

        lines = []
        for connection in POSE_CONNECTIONS:
            start, end = connection[0], connection[1]
            if start in lm_to_3d_index_map and end in lm_to_3d_index_map:
                lines.append([lm_to_3d_index_map[start], lm_to_3d_index_map[end]])

        if lines:
            skeleton_lineset = o3d.geometry.LineSet()
            skeleton_lineset.points = o3d.utility.Vector3dVector(np.array(skel_pts_3d))
            skeleton_lineset.lines = o3d.utility.Vector2iVector(np.array(lines))
            # HIGHLIGHT: Bright Red for skeleton visibility
            skeleton_lineset.colors = o3d.utility.Vector3dVector([[1.0, 0.2, 0.0] for _ in range(len(lines))])

    return skeleton_lineset, outline_lineset

# ═══════════════════════════════════════════════════════════════════════════════
#  Human Extraction Logic (Chair Filter)
# ═══════════════════════════════════════════════════════════════════════════════

def get_human_indices(points, eps=0.10, min_points=10, ransac_iters=300, remove_seat=True):
    if len(points) < 50:
        return None

    max_z = np.max(points[:, 2])
    min_z = np.min(points[:, 2])

    head_zone_mask = (points[:, 2] >= (max_z - 0.25)) & (points[:, 2] <= max_z)
    head_candidates = points[head_zone_mask]

    best_score = -1
    best_center = None

    if len(head_candidates) > 4:
        for _ in range(ransac_iters):
            idx = np.random.choice(len(head_candidates), 4, replace=False)
            pts = head_candidates[idx]

            A = np.zeros((4, 4))
            b = np.zeros(4)
            for i in range(4):
                A[i] = [2 * pts[i][0], 2 * pts[i][1], 2 * pts[i][2], 1]
                b[i] = pts[i][0] ** 2 + pts[i][1] ** 2 + pts[i][2] ** 2

            try:
                params = np.linalg.solve(A, b)
                center = params[:3]
                d = params[3]

                val = center[0] ** 2 + center[1] ** 2 + center[2] ** 2 + d
                if val < 0: continue

                radius = np.sqrt(val)
                if not (0.09 <= radius <= 0.18): continue
                if abs(center[2] - (max_z - 0.12)) > 0.15: continue

                distances = np.linalg.norm(head_candidates - center, axis=1)
                sphere_pts_idx = np.where(abs(distances - radius) <= 0.05)[0]

                score = len(sphere_pts_idx)
                if score > best_score:
                    best_score = score
                    best_center = center
            except np.linalg.LinAlgError:
                continue

    if best_center is None:
        return None

    horizontal_dist = np.linalg.norm(points[:, :2] - best_center[:2], axis=1)
    z_floor_cutoff = min_z + 0.32

    human_mask = (horizontal_dist <= 0.45) & (points[:, 2] <= max_z) & (points[:, 2] > z_floor_cutoff)

    if remove_seat and np.sum(human_mask) > 30:
        seat_zone_mask = human_mask & (points[:, 2] < (min_z + 0.65))
        if np.sum(seat_zone_mask) > 10:
            seat_pcd = o3d.geometry.PointCloud()
            seat_pcd.points = o3d.utility.Vector3dVector(points[seat_zone_mask])
            plane_model, inliers = seat_pcd.segment_plane(distance_threshold=0.025, ransac_n=3, num_iterations=150)

            if abs(plane_model[2]) > 0.85:
                global_seat_indices = np.where(seat_zone_mask)[0][inliers]
                human_mask[global_seat_indices] = False

    return human_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI Control Panel Process
# ═══════════════════════════════════════════════════════════════════════════════

def roi_control_panel(shared_bounds):
    root = tk.Tk()
    root.title("ROI & Background Panel")
    root.geometry("360x680")
    root.attributes('-topmost', True)
    root.configure(padx=15, pady=15)

    tk.Label(root, text="Adjust ROI Bounding Box", font=('Arial', 12, 'bold')).pack(pady=(0, 5))

    def make_slider(name, val):
        tk.Label(root, text=f"{name} Axis", font=('Arial', 9, 'bold'), fg="gray").pack(anchor='w', pady=(3, 0))
        s = tk.Scale(root, from_=-20.0, to=20.0, resolution=0.1, orient='horizontal', length=300)
        s.set(val)
        s.pack()
        return s

    spatial_keys = ['X1', 'X2', 'Y1', 'Y2', 'Z1', 'Z2']
    sliders = {k: make_slider(k, shared_bounds[k]) for k in spatial_keys}
    last_saved_values = {k: s.get() for k, s in sliders.items()}

    tk.Frame(root, height=2, bd=1, relief="sunken").pack(fill="x", pady=10)

    bg_frame = tk.LabelFrame(root, text=" 1. Background (Floor/Walls) ", font=('Arial', 10, 'bold'), padx=10, pady=5)
    bg_frame.pack(fill="x", pady=5)
    bg_status = tk.Label(bg_frame, text="Status: Inactive", font=('Arial', 9), fg="gray")
    bg_status.pack(pady=(0, 5))

    def run_bg_countdown(count):
        if count > 0:
            shared_bounds['bg_state'] = 'countdown'
            bg_status.config(text=f"Calibrating in {count}s...", fg="orange")
            root.after(1000, run_bg_countdown, count - 1)
        else:
            bg_status.config(text="Scanning empty room...", fg="blue")
            shared_bounds['bg_state'] = 'capture'

    def trigger_bg_calib():
        run_bg_countdown(10)

    def trigger_bg_clear():
        shared_bounds['bg_state'] = 'clear'

    tk.Button(bg_frame, text="Calibrate (10s)", command=trigger_bg_calib, bg="#e1f5fe").pack(side="left", expand=True,
                                                                                             fill="x", padx=(0, 5))
    tk.Button(bg_frame, text="Clear File", command=trigger_bg_clear, bg="#ffebee").pack(side="right", fill="x")

    chair_frame = tk.LabelFrame(root, text=" 2. Human Extraction (Chair Filter) ", font=('Arial', 10, 'bold'), padx=10,
                                pady=5)
    chair_frame.pack(fill="x", pady=5)
    chair_status = tk.Label(chair_frame, text="Status: Inactive", font=('Arial', 9), fg="gray")
    chair_status.pack(pady=(0, 5))

    def trigger_chair_enable():
        shared_bounds['chair_state'] = 'active'
        chair_status.config(text="Active (Extracting Human)", fg="green")

    def trigger_chair_disable():
        shared_bounds['chair_state'] = 'idle'
        chair_status.config(text="Inactive", fg="gray")

    tk.Button(chair_frame, text="Enable Filter", command=trigger_chair_enable, bg="#e8f5e9").pack(side="left",
                                                                                                  expand=True, fill="x",
                                                                                                  padx=(0, 5))
    tk.Button(chair_frame, text="Disable", command=trigger_chair_disable, bg="#ffebee").pack(side="right", fill="x")

    def update_shared_dict():
        nonlocal last_saved_values
        current_vals = {}
        has_changed = False

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
                with open(CONFIG_FILE, "w") as f:
                    json.dump(file_data, f, indent=4)
                last_saved_values = current_vals
            except Exception:
                pass

        b_st = shared_bounds.get('bg_state', 'idle')
        if b_st == 'active':
            bg_status.config(text="Active (Room Hidden)", fg="green")
        elif b_st == 'idle':
            bg_status.config(text="Inactive", fg="gray")
        elif b_st == 'scanning':
            bg_status.config(text="Scanning empty room...", fg="blue")

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
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc


def livox_crc32(data: bytes) -> int:
    return binascii.crc32(data, 0x564F580A) & 0xFFFFFFFF


_seq_num = 0


def _next_seq() -> int:
    global _seq_num
    s, _seq_num = _seq_num, (_seq_num + 1) & 0xFFFF
    return s


def build_cmd(cmd_set: int, cmd_id: int, payload: bytes = b'', cmd_type: int = 0x00) -> bytes:
    data = bytes([cmd_set, cmd_id]) + payload
    total_len = 9 + len(data) + 4
    pre_crc = struct.pack('<BBHBH', 0xAA, 0x01, total_len, cmd_type, _next_seq())
    header = pre_crc + struct.pack('<H', livox_crc16(pre_crc))
    packet_without_crc32 = header + data
    return packet_without_crc32 + struct.pack('<I', livox_crc32(packet_without_crc32))


def cmd_handshake(host_ip: str, data_port: int, cmd_port: int, imu_port: int) -> bytes:
    payload = socket.inet_aton(host_ip) + struct.pack('<HHH', data_port, cmd_port, imu_port)
    return build_cmd(0x00, 0x01, payload, cmd_type=0x01)


def cmd_heartbeat() -> bytes: return build_cmd(0x00, 0x03)


def cmd_start_sampling(start: bool = True) -> bytes: return build_cmd(0x00, 0x04, bytes([0x01 if start else 0x00]))


def cmd_set_cartesian() -> bytes: return build_cmd(0x00, 0x05, bytes([0x00]))


_DATA_HDR = 18


def parse_data_packet(data: bytes) -> list:
    if len(data) < _DATA_HDR + 1 or data[0] == 0xAA: return []
    dtype = data[9]
    offset = _DATA_HDR
    pts = []
    if dtype == 0:
        for _ in range((len(data) - _DATA_HDR) // 13):
            if offset + 13 > len(data): break
            x, y, z, _ = struct.unpack_from('<iiiB', data, offset)
            pts.append([x * 1e-3, y * 1e-3, z * 1e-3])
            offset += 13
    elif dtype == 1:
        for _ in range((len(data) - _DATA_HDR) // 9):
            if offset + 9 > len(data): break
            depth, theta, phi, _ = struct.unpack_from('<IHHB', data, offset)
            r = depth * 1e-3
            ze = np.radians(theta / 100.0)
            az = np.radians(phi / 100.0)
            pts.append([r * np.sin(ze) * np.cos(az), r * np.sin(ze) * np.sin(az), r * np.cos(ze)])
            offset += 9
    elif dtype == 2:
        for _ in range((len(data) - _DATA_HDR) // 14):
            if offset + 14 > len(data): break
            x, y, z, _, _ = struct.unpack_from('<iiiBB', data, offset)
            pts.append([x * 1e-3, y * 1e-3, z * 1e-3])
            offset += 14
    return pts


def send_and_ack(sock, pkt: bytes, dest: tuple, label: str, timeout: float = 2.0):
    sock.sendto(pkt, dest)
    sock.settimeout(timeout)
    try:
        ack, _ = sock.recvfrom(512)
        ret = ack[11] if len(ack) > 11 else 0xFF
        print(f"  {label:25s}  {'OK' if ret == 0 else f'ret_code={ret}'}")
        return ret
    except socket.timeout:
        print(f"  {label:25s}  no ACK (timeout)")
        return None


def get_box_lineset(min_b, max_b):
    points = [
        [min_b[0], min_b[1], min_b[2]], [max_b[0], min_b[1], min_b[2]],
        [min_b[0], max_b[1], min_b[2]], [max_b[0], max_b[1], min_b[2]],
        [min_b[0], min_b[1], max_b[2]], [max_b[0], min_b[1], max_b[2]],
        [min_b[0], max_b[1], max_b[2]], [max_b[0], max_b[1], max_b[2]],
    ]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[1, 0.2, 0.2] for _ in range(12)])
    return ls


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Visualizer Loop
# ═══════════════════════════════════════════════════════════════════════════════

def live_livox_viewer(shared_bounds, initial_camera=None):
    print("  Livox Avia – SDK Viewer (ML Tasks API Active)\n")

    # Initialize Modern Tasks API Object Detector safely inside the loop thread context
    landmarker = None
    if os.path.exists(MODEL_TASK_FILE):
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_TASK_FILE)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
            print("  [ML Init] MediaPipe Tasks PoseLandmarker loaded successfully.")
        except Exception as e:
            print(f"  [ML Init] Failed to initialize Tasks Landmarker: {e}")
    else:
        print(f"  [ML Warning] Target model asset '{MODEL_TASK_FILE}' not found.")
        print("  Skeletal lines will not render until the asset file is positioned correctly.")

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
            if (len(raw) >= 34 and raw[0] == 0xAA and raw[9] == 0x00 and raw[10] == 0x00):
                device_ip = addr[0]
    finally:
        bcast_sock.close()

    dest = (device_ip, DEVICE_CMD_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((device_ip, 1))
        host_ip = s.getsockname()[0]

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.bind((host_ip, HOST_CMD_PORT))
    send_and_ack(cmd_sock, cmd_handshake(host_ip, HOST_DATA_PORT, HOST_CMD_PORT, HOST_IMU_PORT), dest, "Handshake")
    time.sleep(0.05)
    send_and_ack(cmd_sock, cmd_set_cartesian(), dest, "Set Cartesian coords")
    time.sleep(0.05)
    send_and_ack(cmd_sock, cmd_start_sampling(True), dest, "Start sampling")

    stop_event = threading.Event()
    cmd_sock.setblocking(False)

    def _heartbeat():
        while not stop_event.is_set():
            try:
                cmd_sock.sendto(cmd_heartbeat(), dest)
                for _ in range(16):
                    try:
                        cmd_sock.recvfrom(256)
                    except (BlockingIOError, OSError):
                        break
            except Exception:
                pass
            time.sleep(HEARTBEAT_INTERVAL_S)

    threading.Thread(target=_heartbeat, daemon=True).start()

    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    data_sock.bind((host_ip, HOST_DATA_PORT))
    data_sock.setblocking(False)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Livox Avia – Direct", 1280, 720)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    init_min = [shared_bounds['X1'], shared_bounds['Y1'], shared_bounds['Z1']]
    init_max = [shared_bounds['X2'], shared_bounds['Y2'], shared_bounds['Z2']]
    roi_box_vis = get_box_lineset(init_min, init_max)
    vis.add_geometry(roi_box_vis)

    ro = vis.get_render_option()
    ro.point_size = 2.0
    ro.background_color = np.array([0.05, 0.05, 0.05])
    ro.line_width = 25.0  # <--- ADD THIS LINE TO INCREASE THICKNESS

    if initial_camera is not None:
        try:
            with open("temp_cam_load.json", "w") as f:
                json.dump(initial_camera, f)
            cam = o3d.io.read_pinhole_camera_parameters("temp_cam_load.json")
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
            os.remove("temp_cam_load.json")
        except:
            pass

    background_pcd = None
    if os.path.exists(BG_PCD_FILE):
        try:
            background_pcd = o3d.io.read_point_cloud(BG_PCD_FILE)
            if not background_pcd.is_empty():
                shared_bounds['bg_state'] = 'active'
                print(f"  [Init] Loaded existing background file: {BG_PCD_FILE}")
            else:
                background_pcd = None
        except Exception:
            background_pcd = None

    bg_accumulating = False
    bg_accumulated_pts = []
    bg_start_time = 0.0

    accumulated_pts = []
    last_render_time = time.time()

    # Track layered layout wireframes frame-by-frame
    active_skeleton_geom = None
    active_outline_geom = None

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
                min_b = [min(shared_bounds['X1'], shared_bounds['X2']), min(shared_bounds['Y1'], shared_bounds['Y2']),
                         min(shared_bounds['Z1'], shared_bounds['Z2'])]
                max_b = [max(shared_bounds['X1'], shared_bounds['X2']), max(shared_bounds['Y1'], shared_bounds['Y2']),
                         max(shared_bounds['Z1'], shared_bounds['Z2'])]

                roi_box_vis.points = get_box_lineset(min_b, max_b).points
                vis.update_geometry(roi_box_vis)

                # Clear old frame overlay geometries before computing updates
                if active_skeleton_geom is not None:
                    vis.remove_geometry(active_skeleton_geom, reset_bounding_box=False)
                    active_skeleton_geom = None
                if active_outline_geom is not None:
                    vis.remove_geometry(active_outline_geom, reset_bounding_box=False)
                    active_outline_geom = None

                if accumulated_pts:
                    arr = np.asarray(accumulated_pts, dtype=np.float64)
                    dists = np.linalg.norm(arr, axis=1)

                    roi_mask = (
                            (dists > 0.01) &
                            (arr[:, 0] >= min_b[0]) & (arr[:, 0] <= max_b[0]) &
                            (arr[:, 1] >= min_b[1]) & (arr[:, 1] <= max_b[1]) &
                            (arr[:, 2] >= min_b[2]) & (arr[:, 2] <= max_b[2])
                    )
                    arr = arr[roi_mask]
                    dists = dists[roi_mask]

                    # ── BACKGROUND CANCELLATION ──
                    bg_state = shared_bounds.get('bg_state', 'idle')
                    if bg_state == 'clear':
                        background_pcd = None
                        bg_accumulating = False
                        if os.path.exists(BG_PCD_FILE):
                            try:
                                os.remove(BG_PCD_FILE)
                            except Exception:
                                pass
                        shared_bounds['bg_state'] = 'idle'
                    elif bg_state == 'capture':
                        bg_accumulating = True
                        bg_accumulated_pts = []
                        bg_start_time = time.time()
                        shared_bounds['bg_state'] = 'scanning'

                    if bg_accumulating:
                        bg_accumulated_pts.extend(arr.tolist())
                        if time.time() - bg_start_time >= 2.0:
                            bg_accumulating = False
                            if len(bg_accumulated_pts) > 0:
                                background_pcd = o3d.geometry.PointCloud()
                                background_pcd.points = o3d.utility.Vector3dVector(np.array(bg_accumulated_pts))
                                background_pcd = background_pcd.voxel_down_sample(voxel_size=0.03)
                                try:
                                    o3d.io.write_point_cloud(BG_PCD_FILE, background_pcd)
                                except Exception:
                                    pass
                                shared_bounds['bg_state'] = 'active'
                            else:
                                shared_bounds['bg_state'] = 'idle'

                    if (shared_bounds.get(
                            'bg_state') == 'active' or bg_state == 'active') and background_pcd is not None and len(
                            arr) > 0:
                        tmp_cloud = o3d.geometry.PointCloud()
                        tmp_cloud.points = o3d.utility.Vector3dVector(arr)
                        dists_to_bg = np.asarray(tmp_cloud.compute_point_cloud_distance(background_pcd))
                        arr = arr[dists_to_bg > 0.06]
                        dists = dists[dists_to_bg > 0.06]

                    # ── LIVE HUMAN EXTRACTION & ML OVERLAYS ──
                    chair_state = shared_bounds.get('chair_state', 'idle')

                    if chair_state == 'active' and len(arr) > 20:
                        human_mask = get_human_indices(arr, eps=0.15, min_points=10, ransac_iters=50)
                        if human_mask is not None:
                            arr = arr[human_mask]
                            dists = dists[human_mask]

                            # Call refactored Tasks API visual tracker
                            active_skeleton_geom, active_outline_geom = extract_skeleton_and_outline(arr, landmarker)

                    # Append updated tracking overlays back into Open3D renderer
                    if active_skeleton_geom is not None:
                        vis.add_geometry(active_skeleton_geom, reset_bounding_box=False)
                    if active_outline_geom is not None:
                        vis.add_geometry(active_outline_geom, reset_bounding_box=False)

                    # Render Point Cloud frame
                    if len(arr) > 0:
                        pcd.points = o3d.utility.Vector3dVector(arr)
                        norm_dists = np.clip(dists / MAX_COLOR_DIST_M, 0.0, 1.0)
                        colors = np.zeros((len(arr), 3))
                        colors[:, 0] = np.clip(1.5 - np.abs(4.0 * norm_dists - 3.0), 0, 1)
                        colors[:, 1] = np.clip(1.5 - np.abs(4.0 * norm_dists - 2.0), 0, 1)
                        colors[:, 2] = np.clip(1.5 - np.abs(4.0 * norm_dists - 1.0), 0, 1)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                        vis.update_geometry(pcd)
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
            with open("temp_cam_save.json", "r") as f:
                cam_data = json.load(f)
            os.remove("temp_cam_save.json")
            file_data = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f: file_data = json.load(f)
            file_data["camera_view"] = cam_data
            with open(CONFIG_FILE, "w") as f:
                json.dump(file_data, f, indent=4)
        except Exception:
            pass
        try:
            cmd_sock.setblocking(True)
            cmd_sock.sendto(cmd_start_sampling(False), dest)
        except Exception:
            pass
        cmd_sock.close()
        data_sock.close()
        vis.destroy_window()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    initial_bounds = {'X1': -5.0, 'X2': 5.0, 'Y1': -5.0, 'Y2': 5.0, 'Z1': -2.0, 'Z2': 5.0}
    initial_camera = None

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                file_data = json.load(f)
            for k in initial_bounds.keys():
                if k in file_data: initial_bounds[k] = file_data[k]
            if "camera_view" in file_data: initial_camera = file_data["camera_view"]
        except Exception:
            pass

    initial_bounds['bg_state'] = 'idle'
    initial_bounds['chair_state'] = 'idle'

    with multiprocessing.Manager() as manager:
        shared_bounds = manager.dict(initial_bounds)
        gui_process = multiprocessing.Process(target=roi_control_panel, args=(shared_bounds,))
        gui_process.start()
        try:
            live_livox_viewer(shared_bounds, initial_camera)
        finally:
            gui_process.terminate()
            gui_process.join()