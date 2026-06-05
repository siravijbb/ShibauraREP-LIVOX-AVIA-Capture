#!/usr/bin/env python3
"""
Livox Avia – Direct SDK Viewer (The True Protocol Fix)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import socket
import struct
import threading
import time
import numpy as np
import open3d as o3d
import binascii

# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BROADCAST_PORT = 55000
DEVICE_CMD_PORT = 65000

HOST_DATA_PORT = 60001
HOST_CMD_PORT = 60000
HOST_IMU_PORT = 60002

HEARTBEAT_INTERVAL_S = 1.0


# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
#  The Correct LSB-First CRCs
# ───────────────────────────────────────────────────────────────────────────────

def livox_crc16(data: bytes) -> int:
    """Standard LSB-first CRC16 with poly 0x8408 (reversed 0x1021) and Init 0x4C49"""
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
    """Standard LSB-first CRC32 using binascii, with Livox's custom Init 0x564F580A"""
    return binascii.crc32(data, 0x564F580A) & 0xFFFFFFFF


# ───────────────────────────────────────────────────────────────────────────────
#  Command packet builder
# ───────────────────────────────────────────────────────────────────────────────

_seq_num = 0


def _next_seq() -> int:
    global _seq_num
    s, _seq_num = _seq_num, (_seq_num + 1) & 0xFFFF
    return s


def build_cmd(cmd_set: int, cmd_id: int, payload: bytes = b'', cmd_type: int = 0x00) -> bytes:
    """Build a Livox SDK command packet."""
    data = bytes([cmd_set, cmd_id]) + payload
    total_len = 9 + len(data) + 4

    # Header: SOF, Version, Length, CmdType, Seq
    pre_crc = struct.pack('<BBHBH', 0xAA, 0x01, total_len, cmd_type, _next_seq())

    # Header CRC16
    header = pre_crc + struct.pack('<H', livox_crc16(pre_crc))

    # Payload
    packet_without_crc32 = header + data

    # Packet CRC32
    crc32_val = livox_crc32(packet_without_crc32)

    return packet_without_crc32 + struct.pack('<I', crc32_val)


# ───────────────────────────────────────────────────────────────────────────────
#  Standard commands
# ───────────────────────────────────────────────────────────────────────────────

def cmd_handshake(host_ip: str, data_port: int, cmd_port: int, imu_port: int) -> bytes:
    """
    CRITICAL FIX: Handshake MUST be cmd_type=0x01 (ACK) as it responds to Broadcast.
    """
    payload = socket.inet_aton(host_ip) + struct.pack('<HHH', data_port, cmd_port, imu_port)
    return build_cmd(0x00, 0x01, payload, cmd_type=0x01)


def cmd_heartbeat() -> bytes:
    """CMD_SET=0x00, CMD_ID=0x03 – keep-alive (Defaults to Request 0x00)."""
    return build_cmd(0x00, 0x03)


def cmd_start_sampling(start: bool = True) -> bytes:
    """CMD_SET=0x00, CMD_ID=0x04 – start/stop streaming (Defaults to Request 0x00)."""
    return build_cmd(0x00, 0x04, bytes([0x01 if start else 0x00]))


def cmd_set_cartesian() -> bytes:
    """CMD_SET=0x00, CMD_ID=0x05 – request Cartesian output (Defaults to Request 0x00)."""
    return build_cmd(0x00, 0x05, bytes([0x00]))


# ───────────────────────────────────────────────────────────────────────────────
#  Point cloud parser
# ───────────────────────────────────────────────────────────────────────────────

_DATA_HDR = 18


def parse_data_packet(data: bytes) -> list:
    if len(data) < _DATA_HDR + 1:
        return []
    if data[0] == 0xAA:
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
            pts.append([r * np.sin(ze) * np.cos(az),
                        r * np.sin(ze) * np.sin(az),
                        r * np.cos(ze)])
            offset += 9
    elif dtype == 2:
        n = (len(data) - _DATA_HDR) // 14
        for _ in range(n):
            if offset + 14 > len(data): break
            x, y, z, _, _ = struct.unpack_from('<iiiBB', data, offset)
            pts.append([x * 1e-3, y * 1e-3, z * 1e-3])
            offset += 14
    return pts


# ───────────────────────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────────────

def live_livox_viewer():
    sep = "─" * 62
    print(sep)
    print("  Livox Avia – Direct SDK Viewer (The True Protocol Fix)")
    print(sep)
    print("  ⚠  Make sure Livox Viewer is fully closed.")
    print(sep + "\n")

    # ── Step 1: Capture device broadcast ─────────────────────────────────────
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
                bcast_code = raw[11:27].decode('ascii', errors='replace').rstrip('\x00')
                device_ip = addr[0]
                print(f"  Device found   : {device_ip}")
                print(f"  Broadcast code : {bcast_code}")
    finally:
        bcast_sock.close()

    # ── Step 2: Handshake ─────────────────────────────────────────────────────
    dest = (device_ip, DEVICE_CMD_PORT)

    # Dynamically find the correct network adapter IP for the LiDAR
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((device_ip, 1))
        host_ip = s.getsockname()[0]

    print(f"\n[2/4] Handshake  (this host = {host_ip})")

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        cmd_sock.bind((host_ip, HOST_CMD_PORT))
    except OSError as e:
        print(f"  ERROR: cannot bind cmd socket to {host_ip}:{HOST_CMD_PORT}: {e}")
        return

    hs_pkt = cmd_handshake(host_ip, HOST_DATA_PORT, HOST_CMD_PORT, HOST_IMU_PORT)
    send_and_ack(cmd_sock, hs_pkt, dest, "Handshake")
    time.sleep(0.05)
    send_and_ack(cmd_sock, cmd_set_cartesian(), dest, "Set Cartesian coords")
    time.sleep(0.05)

    # ── Step 3: Start sampling ────────────────────────────────────────────────
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

    # ── Step 4: Data socket ───────────────────────────────────────────────────
    data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        data_sock.bind((host_ip, HOST_DATA_PORT))
    except OSError as e:
        print(f"\n  ERROR: cannot bind data socket to {host_ip}:{HOST_DATA_PORT}: {e}")
        stop_event.set()
        cmd_sock.close()
        return
    data_sock.setblocking(False)

    print(f"\n[4/4] Listening for point cloud on port {HOST_DATA_PORT}...")
    print("      Press Ctrl+C to stop.\n")

    # ── Open3D window ──────────────────────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window("Livox Avia – Direct", 1280, 720)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    ro = vis.get_render_option()
    ro.point_size = 1.5
    ro.background_color = np.array([0.05, 0.05, 0.05])

    vis.poll_events()
    vis.update_renderer()

    # ── Render loop ────────────────────────────────────────────────────────────
    total_pkts = 0
    try:
        while True:
            pts = []
            for _ in range(200):
                try:
                    pkt, _ = data_sock.recvfrom(1500)
                except (BlockingIOError, OSError):
                    break

                if total_pkts == 0:
                    print(f"  First DATA packet: len={len(pkt)} "
                          f"version=0x{pkt[0]:02X} data_type=0x{pkt[9]:02X}")
                total_pkts += 1
                if total_pkts % 2000 == 0:
                    print(f"  {total_pkts} data packets received...")

                pts.extend(parse_data_packet(pkt))

            if pts:
                arr = np.asarray(pts, dtype=np.float64)
                arr = arr[np.linalg.norm(arr, axis=1) > 0.01]
                if len(arr):
                    pcd.points = o3d.utility.Vector3dVector(arr)
                    pcd.paint_uniform_color([0.1, 0.9, 0.2])
                    vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        try:
            cmd_sock.setblocking(True)
            cmd_sock.sendto(cmd_start_sampling(False), dest)
        except Exception:
            pass
        time.sleep(0.1)
        cmd_sock.close()
        data_sock.close()
        vis.destroy_window()
        print("Done.")


if __name__ == "__main__":
    live_livox_viewer()