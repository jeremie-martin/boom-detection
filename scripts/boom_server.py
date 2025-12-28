#!/usr/bin/env python3
"""
Boom detection server - persistent Python process for low-latency inference.

Listens on a Unix socket and accepts prediction requests. The model is loaded
once at startup, so subsequent predictions avoid Python/model startup time.

Protocol:
    Request: 4-byte header length (uint32 LE) + JSON header + optional binary data
    Response: 4-byte length (uint32 LE) + JSON response

Request types:
    1. File path mode:
       {"type": "path", "path": "/path/to/simulation_data.bin"}

    2. Binary data mode (for in-memory simulations):
       {"type": "binary", "frames": 1000, "pendulums": 2000}
       followed by: frames * pendulums * 8 * sizeof(float32) bytes
       Data layout: [frame0_pend0_8vals, frame0_pend1_8vals, ..., frame1_pend0_8vals, ...]

    3. Shutdown:
       {"type": "shutdown"}

Response:
    {"status": "ok", "accepted": true, "boom_frame": 123, ...}
    or {"status": "error", "message": "..."}

Usage:
    # Start server (default socket: /tmp/boom_server.sock)
    uv run python scripts/boom_server.py models/boom_v1

    # Start with custom socket path
    uv run python scripts/boom_server.py models/boom_v1 --socket /tmp/my_boom.sock

    # Test with netcat (path mode)
    echo -n '{"type":"path","path":"data/simulations/run_xxx/simulation_data.bin"}' | \\
        python -c "import sys,struct; d=sys.stdin.buffer.read(); sys.stdout.buffer.write(struct.pack('<I',len(d))+d)" | \\
        nc -U /tmp/boom_server.sock | python -c "import sys,struct,json; sys.stdin.buffer.read(4); print(json.loads(sys.stdin.buffer.read()))"

C++ client example: see scripts/boom_client_example.cpp
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import sys
from pathlib import Path

import numpy as np


def recv_exactly(conn: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


def send_response(conn: socket.socket, response: dict) -> None:
    """Send JSON response with length prefix."""
    data = json.dumps(response).encode('utf-8')
    conn.sendall(struct.pack('<I', len(data)) + data)


def handle_client(conn: socket.socket, pipeline) -> bool:
    """
    Handle a single client request.

    Returns:
        True to continue serving, False to shutdown
    """
    try:
        # Read header length (4 bytes, uint32 little-endian)
        header_len_data = recv_exactly(conn, 4)
        header_len = struct.unpack('<I', header_len_data)[0]

        # Read JSON header
        header_data = recv_exactly(conn, header_len)
        header = json.loads(header_data.decode('utf-8'))

        request_type = header.get('type', 'path')

        if request_type == 'shutdown':
            send_response(conn, {'status': 'ok', 'message': 'shutting down'})
            return False

        elif request_type == 'path':
            # File path mode
            sim_path = header['path']
            result = pipeline.predict_file(sim_path)

        elif request_type == 'binary':
            # Binary data mode
            frames = header['frames']
            pendulums = header['pendulums']
            values_per_pendulum = header.get('values', 8)

            # Read binary data
            data_size = frames * pendulums * values_per_pendulum * 4  # float32
            data_bytes = recv_exactly(conn, data_size)

            # Reshape to (frames, pendulums, 8)
            data = np.frombuffer(data_bytes, dtype=np.float32)
            data = data.reshape(frames, pendulums, values_per_pendulum)

            # Run prediction
            result = pipeline.predict_simulation(data)

        else:
            send_response(conn, {'status': 'error', 'message': f'unknown type: {request_type}'})
            return True

        # Send successful response (convert numpy types to Python types)
        response = {
            'status': 'ok',
            'accepted': bool(result.accepted),
            'boom_frame': int(result.boom_frame) if result.boom_frame is not None else None,
            'cnn_pred': int(result.cnn_pred),
            'hgb_pred': int(result.hgb_pred),
            'disagreement': int(result.disagreement),
            'predicted_quality': round(float(result.predicted_quality), 4),
            'accept_score': round(float(result.accept_score), 4),
        }
        send_response(conn, response)
        return True

    except ConnectionError:
        return True
    except Exception as e:
        try:
            send_response(conn, {'status': 'error', 'message': str(e)})
        except:
            pass
        return True


def run_server(model_path: Path, socket_path: Path) -> None:
    """Run the boom detection server."""
    from boom_detection.deploy_pipeline import BoomDetectionPipeline

    # Load model
    print(f"Loading model from {model_path}...")
    pipeline = BoomDetectionPipeline.from_pretrained(model_path)
    print(f"  n_features: {pipeline.n_features}")
    print(f"  max_pendulums: {pipeline.feature_config.max_pendulums}")
    print()

    # Remove existing socket file
    if socket_path.exists():
        socket_path.unlink()

    # Create Unix socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)

    # Handle SIGINT/SIGTERM gracefully
    running = True
    def handle_signal(signum, frame):
        nonlocal running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Server listening on {socket_path}")
    print("Ready for predictions. Press Ctrl+C to stop.\n")

    # Set socket timeout so we can check running flag
    server.settimeout(1.0)

    request_count = 0
    while running:
        try:
            conn, _ = server.accept()
            request_count += 1

            try:
                should_continue = handle_client(conn, pipeline)
                if not should_continue:
                    running = False
            finally:
                conn.close()

        except socket.timeout:
            continue
        except Exception as e:
            if running:
                print(f"Error: {e}")

    print(f"\nShutdown complete. Handled {request_count} requests.")
    server.close()
    socket_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description='Boom detection server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('model_path', type=Path, help='Path to saved model directory')
    parser.add_argument('--socket', '-s', type=Path, default=Path('/tmp/boom_server.sock'),
                       help='Unix socket path (default: /tmp/boom_server.sock)')

    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)

    run_server(args.model_path, args.socket)


if __name__ == '__main__':
    main()
