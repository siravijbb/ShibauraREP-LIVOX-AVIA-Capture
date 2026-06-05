#!/usr/bin/env python3
"""
Livox Avia – Direct SDK Viewer (With Live ROI & Background Cancellation)
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


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI Control Panel Process
# ═══════════════════════════════════════════════════════════════════════════════

def roi_control_panel(shared_bounds):
    """Runs a Tkinter window in a separate process to avoid blocking Open3D."""
    root = tk.Tk()
    root.title("ROI & Background Panel")
    root.geometry("340x560")
    root.attributes('-topmost', True)
    root.configure(padx=15, pady=15)

    tk.Label(root, text="Adjust ROI Bounding Box", font=('Arial', 12, 'bold')).pack(pady=(0, 5))

    def make_slider(name, val):
        tk.Label(root, text=f"{name} Axis", font=('Arial', 9, 'bold'), fg="gray").pack(anchor='w', pady=(3, 0))
        s = tk.Scale(root, from_=-20.0, to=20.0, resolution=0.1, orient='horizontal', length=300)
        s.set(val)
        s.pack()
        return s

    # Ensure we only generate sliders for spatial keys
    spatial_keys = ['X1', 'X2', 'Y1', 'Y2', 'Z1', 'Z2']
    sliders = {k: make_slider(k, shared_bounds[k]) for k in spatial_keys}
    last_saved_values = {k: s.get() for k, s in sliders.items()}

    # ── Background Subtraction UI Section ──
    tk.Frame(root, height=2, bd=1, relief="sunken").pack(fill="x", pady=15)

    bg_frame = tk.LabelFrame(root, text=" Background Cancellation ", font=('Arial', 10, 'bold'), padx=10, pady=10)
    bg_frame.pack(fill="x")

    # FIXED: Changed 'medium' to 'normal'
    status_label = tk.Label(bg_frame, text="Status: Inactive", font=('Arial', 10, 'normal'), fg="gray")
    status_label.pack(pady=(0, 8))

    def run_countdown(count):
        if count > 0:
            shared_bounds['bg_state'] = 'countdown'
            status_label.config(text=f"Calibrating in {count}s...", fg="orange")
            root.after(1000, run_countdown, count - 1)
        else:
            status_label.config(text="Scanning environment (Keep clear)...", fg="blue")
            shared_bounds['bg_state'] = 'capture'
            monitor_capture()

    def monitor_capture():
        state = shared_bounds['bg_state']
        if state == 'active':
            status_label.config(text="Status: Subtraction Active", fg="green")
            btn_calibrate.config(state='normal')
            btn_clear.config(state='normal')
        elif state == 'idle':
            status_label.config(text="Status: Inactive", fg="gray")
            btn_calibrate.config(state='normal')
        else:
            root.after(200, monitor_capture)

    def trigger_calibration():
        btn_calibrate.config(state='disabled')
        btn_clear.config(state='disabled')
        run_countdown(10)

    def trigger_clear():
        shared_bounds['bg_state'] = 'clear'
        status_label.config(text="Status: Inactive", fg="gray")
        btn_clear.config(state='disabled')

    btn_calibrate = tk.Button(bg_frame, text="Calibrate Background (10s)", command=trigger_calibration, bg="#e1f5fe")
    btn_calibrate.pack(side="left", expand=True, fill="x", padx=(0, 5))

    btn_clear = tk.Button(bg_frame, text="Clear", command=trigger_clear, state='disabled', bg="#ffebee")
    btn_clear.pack(side="right", expand=True, fill="x", padx=(5, 0))

    def update_shared_dict():
        nonlocal last_saved_values
        current_vals = {}
        has_changed = False

        for k, s in sliders.items():
            val = s.get()
            current_vals[k] = val
            shared_bounds[k] = val
            if val != last_saved_values.get(k):
                has_changed = True

        if has_changed:
            try:
                file_data = {}
                if os.path.exists(CONFIG_FILE):
                    with open(CONFIG_FILE, "r") as f:
                        file_data = json.load(f)

                file_data.update(current_vals)
                with open(CONFIG_FILE, "w") as f:
                    json.dump(file_data, f, indent=4)
                last_saved_values = current_vals
            except Exception as e:
                print(f"Error auto-saving settings: {e}")

        # Sync visual status if changed externally by Open3D backend
        current_state = shared_bounds['bg_state']
        if current_state == 'active' and "Active" not in status_label.cget("text"):
            status_label.config(text="Status: Subtraction Active", fg="green")
            btn_clear.config(state='normal')

        root.after(50, update_shared_dict)

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
    crc32_val = livox_crc32(packet_without_crc32)
    return packet_without_crc32 + struct.pack('<I', crc32_val)


def cmd_handshake(host_ip: str, data_port: int, cmd_port: int, imu_port: int) -> bytes:
    payload = socket.inet_aton(host_ip) + struct.pack('<HHH', data_port, cmd_port, imu_port)
    return build_cmd(0x00, 0x01, payload, cmd_type=0x01)


def cmd_heartbeat() -> bytes:
    return build_cmd(0x00, 0x03)


def cmd_start_sampling(start: bool = True) -> bytes:
    return build_cmd(0x00, 0x04, bytes([0x01 if start else 0x00]))


def cmd_set_cartesian() -> bytes:
    return build_cmd(0x00, 0x05, bytes([0x00]))


_DATA_HDR = 18


def parse_data_packet(data: bytes) -> list:
    if len(data) < _DATA_HDR + 1 or data[0] == 0xAA:
        return []
    dtype = data[9]
    offset = _DATA_HDR
    pts = []
    if dtype == 0:
        n = (len(data) - _DATA_HDR) // 13
        for _ in range(n):
            if offset + 13 > len(data): break
            x, y, z, _ = struct.unpack_from('<iiiB', data, offset)
            pts.append([x * 1e-3, y * 1e-3, z * 1e-3])
            offset += 13
    elif dtype == 1:
        n = (len(data) - _DATA_HDR) // 9
        for _ in range(n):
            if offset + 9 > len(data): break
            depth, theta, phi, _ = struct.unpack_from('<IHHB', data, offset)
            r = depth * 1e-3
            ze = np.radians(theta / 100.0)
            az = np.radians(phi / 100.0)
            pts.append([r * np.sin(ze) * np.cos(az), r * np.sin(ze) * np.sin(az), r * np.cos(ze)])
            offset += 9
    elif dtype == 2:
        n = (len(data) - _DATA_HDR) // 14
        for _ in range(n):
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
        status = "OK" if ret == 0 else f"ret_code={ret}"
        print(f"  {label:25s}  {status}")
        return ret
    except socket.timeout:
        print(f"  {label:25s}  no ACK (timeout) – verify connection")
        return None


def get_box_lineset(min_b, max_b):
    points = [
        [min_b[0], min_b[1], min_b[2]], [max_b[0], min_b[1], min_b[2]],
        [min_b[0], max_b[1], min_b[2]], [max_b[0], max_b[1], min_b[2]],
        [min_b[0], min_b[1], max_b[2]], [max_b[0], min_b[1], max_b[2]],
        [min_b[0], max_b[1], max_b[2]], [max_b[0], max_b[1], max_b[2]],
    ]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0.2, 0.2] for _ in range(12)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

def live_livox_viewer(shared_bounds, initial_camera=None):
    sep = "─" * 62
    print(sep)
    print("  Livox Avia – Direct SDK Viewer (Live ROI & Background Sync)")
    print(sep + "\n")

    print(f"[1/4] Waiting for device broadcast on port {BROADCAST_PORT}...")
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
                print("  TIMEOUT – no broadcast received. Check cables.")
                return
            if (len(raw) >= 34 and raw[0] == 0xAA and raw[9] == 0x00 and raw[10] == 0x00):
                device_ip = addr[0]
                print(f"  Device found   : {device_ip}")
    finally:
        bcast_sock.close()

    dest = (device_ip, DEVICE_CMD_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((device_ip, 1))
        host_ip = s.getsockname()[0]

    print(f"\n[2/4] Handshake  (this host = {host_ip})")
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.bind((host_ip, HOST_CMD_PORT))

    hs_pkt = cmd_handshake(host_ip, HOST_DATA_PORT, HOST_CMD_PORT, HOST_IMU_PORT)
    send_and_ack(cmd_sock, hs_pkt, dest, "Handshake")
    time.sleep(0.05)
    send_and_ack(cmd_sock, cmd_set_cartesian(), dest, "Set Cartesian coords")
    time.sleep(0.05)

    print(f"\n[3/4] Starting data stream...")
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

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    data_sock.bind((host_ip, HOST_DATA_PORT))
    data_sock.setblocking(False)

    print(f"\n[4/4] Rendering Live Stream...")

    # ── Open3D window ──
    vis = o3d.visualization.Visualizer()
    vis.create_window("Livox Avia – Direct", 1280, 720)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    init_min = [min(shared_bounds['X1'], shared_bounds['X2']), min(shared_bounds['Y1'], shared_bounds['Y2']),
                min(shared_bounds['Z1'], shared_bounds['Z2'])]
    init_max = [max(shared_bounds['X1'], shared_bounds['X2']), max(shared_bounds['Y1'], shared_bounds['Y2']),
                max(shared_bounds['Z1'], shared_bounds['Z2'])]
    roi_box_vis = get_box_lineset(init_min, init_max)
    vis.add_geometry(roi_box_vis)

    ro = vis.get_render_option()
    ro.point_size = 1.5
    ro.background_color = np.array([0.05, 0.05, 0.05])

    if initial_camera is not None:
        try:
            with open("temp_cam_load.json", "w") as f:
                json.dump(initial_camera, f)
            cam = o3d.io.read_pinhole_camera_parameters("temp_cam_load.json")
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
            os.remove("temp_cam_load.json")
        except Exception as e:
            print(f"  -> Could not load camera view: {e}")

    # ── Background Cancellation State variables ──
    background_pcd = None
    bg_accumulating = False
    bg_accumulated_pts = []
    bg_start_time = 0.0
    SUBTRACTION_RADIUS_M = 0.06  # 6 centimeters distance tolerance filter

    accumulated_pts = []
    last_render_time = time.time()

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

                new_box = get_box_lineset(min_b, max_b)
                roi_box_vis.points = new_box.points
                vis.update_geometry(roi_box_vis)

                if accumulated_pts:
                    arr = np.asarray(accumulated_pts, dtype=np.float64)
                    dists = np.linalg.norm(arr, axis=1)

                    # Apply Spatial ROI Bounds
                    roi_mask = (
                            (dists > 0.01) &
                            (arr[:, 0] >= min_b[0]) & (arr[:, 0] <= max_b[0]) &
                            (arr[:, 1] >= min_b[1]) & (arr[:, 1] <= max_b[1]) &
                            (arr[:, 2] >= min_b[2]) & (arr[:, 2] <= max_b[2])
                    )
                    arr = arr[roi_mask]
                    dists = dists[roi_mask]

                    # ── Handle Background Subtraction Logic ──
                    bg_state = shared_bounds.get('bg_state', 'idle')

                    if bg_state == 'clear':
                        background_pcd = None
                        bg_accumulating = False
                        shared_bounds['bg_state'] = 'idle'
                        print("-> Background filter cleared.")

                    elif bg_state == 'capture':
                        # Setup collection sweep window
                        bg_accumulating = True
                        bg_accumulated_pts = []
                        bg_start_time = time.time()
                        shared_bounds['bg_state'] = 'scanning'
                        print("-> Beginning 2.5s environment collection...")

                    if bg_accumulating:
                        if len(arr) > 0:
                            bg_accumulated_pts.extend(arr.tolist())
                        # Collect fields for 2.5s to dense map the non-repetitive flower patterns
                        if time.time() - bg_start_time >= 2.5:
                            bg_accumulating = False
                            if len(bg_accumulated_pts) > 0:
                                background_pcd = o3d.geometry.PointCloud()
                                background_pcd.points = o3d.utility.Vector3dVector(np.array(bg_accumulated_pts))
                                # Downsample slightly to unify the lookup map
                                background_pcd = background_pcd.voxel_down_sample(voxel_size=0.03)
                                shared_bounds['bg_state'] = 'active'
                                print(f"-> Calibration successful! Point map count: {len(background_pcd.points)}")
                            else:
                                shared_bounds['bg_state'] = 'idle'
                                print("-> Calibration failed (No visual data detected inside ROI).")

                    # Apply background suppression matrix if operational
                    if bg_state == 'active' and background_pcd is not None and len(arr) > 0:
                        tmp_cloud = o3d.geometry.PointCloud()
                        tmp_cloud.points = o3d.utility.Vector3dVector(arr)

                        # High performance distance compute using native C++ map backend
                        points_to_bg_distances = np.asarray(tmp_cloud.compute_point_cloud_distance(background_pcd))

                        # Only keep points further than the tolerance radius
                        subtraction_mask = points_to_bg_distances > SUBTRACTION_RADIUS_M
                        arr = arr[subtraction_mask]
                        dists = dists[subtraction_mask]

                    # Render Final output frame
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
        print("\nStopping...")
    finally:
        stop_event.set()

        # Save camera orientation states
        try:
            cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters("temp_cam_save.json", cam_params)
            with open("temp_cam_save.json", "r") as f:
                cam_data = json.load(f)
            os.remove("temp_cam_save.json")

            file_data = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f:
                    file_data = json.load(f)

            file_data["camera_view"] = cam_data
            with open(CONFIG_FILE, "w") as f:
                json.dump(file_data, f, indent=4)
            print("-> Camera viewpoint saved successfully.")
        except Exception as e:
            print(f"-> Notice: Could not save camera parameters ({e})")

        try:
            cmd_sock.setblocking(True)
            cmd_sock.sendto(cmd_start_sampling(False), dest)
        except Exception:
            pass
        cmd_sock.close()
        data_sock.close()
        vis.destroy_window()


# ═══════════════════════════════════════════════════════════════════════════════
#  Multiprocessing Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    multiprocessing.freeze_support()

    initial_bounds = {'X1': -5.0, 'X2': 5.0, 'Y1': -5.0, 'Y2': 5.0, 'Z1': -2.0, 'Z2': 5.0}
    initial_camera = None

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                file_data = json.load(f)

            for k in initial_bounds.keys():
                if k in file_data:
                    initial_bounds[k] = file_data[k]

            if "camera_view" in file_data:
                initial_camera = file_data["camera_view"]
            print(f"-> Loaded existing config from {CONFIG_FILE}")
        except Exception:
            print("-> Found corrupted config file. Loading defaults.")

    # Append processing variables for background state sync
    initial_bounds['bg_state'] = 'idle'

    with multiprocessing.Manager() as manager:
        shared_bounds = manager.dict(initial_bounds)

        gui_process = multiprocessing.Process(target=roi_control_panel, args=(shared_bounds,))
        gui_process.start()

        try:
            live_livox_viewer(shared_bounds, initial_camera)
        finally:
            gui_process.terminate()
            gui_process.join()