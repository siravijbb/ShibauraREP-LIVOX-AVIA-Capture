#!/usr/bin/env python3
"""
Livox Avia – Direct SDK Viewer (With Persistent Live ROI & Camera Controls)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
#  GUI Control Panel Process (With Auto-Save)
# ═══════════════════════════════════════════════════════════════════════════════

def roi_control_panel(shared_bounds):
    """Runs a Tkinter window in a separate process to avoid blocking Open3D."""
    root = tk.Tk()
    root.title("ROI Control Panel")
    root.geometry("320x450")
    root.attributes('-topmost', True)
    root.configure(padx=15, pady=15)

    tk.Label(root, text="Adjust ROI Bounding Box", font=('Arial', 12, 'bold')).pack(pady=(0, 10))

    def make_slider(name, val):
        tk.Label(root, text=f"{name} Axis", font=('Arial', 9, 'bold'), fg="gray").pack(anchor='w', pady=(5, 0))
        s = tk.Scale(root, from_=-20.0, to=20.0, resolution=0.1, orient='horizontal', length=280)
        s.set(val)
        s.pack()
        return s

    sliders = {k: make_slider(k, v) for k, v in shared_bounds.items()}
    last_saved_values = {k: s.get() for k, s in sliders.items()}

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
                # Load existing config to PRESERVE camera data
                file_data = {}
                if os.path.exists(CONFIG_FILE):
                    with open(CONFIG_FILE, "r") as f:
                        file_data = json.load(f)

                # Update only the bounding box values
                file_data.update(current_vals)

                with open(CONFIG_FILE, "w") as f:
                    json.dump(file_data, f, indent=4)
                last_saved_values = current_vals
            except Exception as e:
                print(f"Error auto-saving settings: {e}")

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
    print("  Livox Avia – Direct SDK Viewer (Live ROI & Camera Sync)")
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
    print("      Use the popup panel to adjust the ROI bounding box.")
    print("      Camera & ROI settings auto-save to 1 file on exit.")
    print("      Press Ctrl+C in the terminal to stop.\n")

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

    # APPLY SAVED CAMERA VIEW
    if initial_camera is not None:
        try:
            with open("temp_cam_load.json", "w") as f:
                json.dump(initial_camera, f)
            cam = o3d.io.read_pinhole_camera_parameters("temp_cam_load.json")
            vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
            os.remove("temp_cam_load.json")
        except Exception as e:
            print(f"  -> Could not load camera view: {e}")

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

                    roi_mask = (
                            (dists > 0.01) &
                            (arr[:, 0] >= min_b[0]) & (arr[:, 0] <= max_b[0]) &
                            (arr[:, 1] >= min_b[1]) & (arr[:, 1] <= max_b[1]) &
                            (arr[:, 2] >= min_b[2]) & (arr[:, 2] <= max_b[2])
                    )

                    arr = arr[roi_mask]
                    dists = dists[roi_mask]

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

        # ── MERGE CAMERA PARAMETERS ON EXIT ──
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
            print("-> Final camera viewpoint saved successfully.")
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

            # Load bounds if they exist
            for k in initial_bounds.keys():
                if k in file_data:
                    initial_bounds[k] = file_data[k]

            # Load camera if it exists
            if "camera_view" in file_data:
                initial_camera = file_data["camera_view"]

            print(f"-> Loaded existing config (ROI + Camera) from {CONFIG_FILE}")
        except Exception:
            print("-> Found corrupted config file. Loading defaults.")

    with multiprocessing.Manager() as manager:
        shared_bounds = manager.dict(initial_bounds)

        gui_process = multiprocessing.Process(target=roi_control_panel, args=(shared_bounds,))
        gui_process.start()

        try:
            live_livox_viewer(shared_bounds, initial_camera)
        finally:
            gui_process.terminate()
            gui_process.join()