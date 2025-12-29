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
import signal
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# =============================================================================
# Logging
# =============================================================================

class Logger:
    """Simple logger with timestamps and colors."""

    COLORS = {
        'reset': '\033[0m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'red': '\033[31m',
        'cyan': '\033[36m',
        'dim': '\033[2m',
    }

    def __init__(self, use_color: bool = True):
        self.use_color = use_color
        self.request_count = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.error_count = 0
        self.total_time = 0.0

    def _color(self, name: str) -> str:
        return self.COLORS[name] if self.use_color else ''

    def _timestamp(self) -> str:
        return datetime.now().strftime('%H:%M:%S')

    def info(self, msg: str) -> None:
        print(f"{self._color('dim')}[{self._timestamp()}]{self._color('reset')} {msg}")

    def success(self, msg: str) -> None:
        print(f"{self._color('dim')}[{self._timestamp()}]{self._color('reset')} {self._color('green')}✓{self._color('reset')} {msg}")

    def warning(self, msg: str) -> None:
        print(f"{self._color('dim')}[{self._timestamp()}]{self._color('reset')} {self._color('yellow')}⚠{self._color('reset')} {msg}")

    def error(self, msg: str) -> None:
        print(f"{self._color('dim')}[{self._timestamp()}]{self._color('reset')} {self._color('red')}✗{self._color('reset')} {msg}")

    def request(self, req_type: str, details: str) -> None:
        self.request_count += 1
        print(f"{self._color('dim')}[{self._timestamp()}]{self._color('reset')} "
              f"{self._color('cyan')}←{self._color('reset')} "
              f"#{self.request_count} {req_type}: {details}")

    def response(self, accepted: bool, boom_frame: int | None, elapsed_ms: float, details: str) -> None:
        self.total_time += elapsed_ms / 1000.0

        if accepted:
            self.accepted_count += 1
            status = f"{self._color('green')}ACCEPTED{self._color('reset')} boom@{boom_frame}"
        else:
            self.rejected_count += 1
            status = f"{self._color('yellow')}REJECTED{self._color('reset')}"

        print(f"{self._color('dim')}[{self._timestamp()}]{self._color('reset')} "
              f"{self._color('cyan')}→{self._color('reset')} "
              f"{status} ({elapsed_ms:.0f}ms) {details}")

    def response_error(self, error_msg: str, elapsed_ms: float) -> None:
        self.error_count += 1
        self.total_time += elapsed_ms / 1000.0
        print(f"{self._color('dim')}[{self._timestamp()}]{self._color('reset')} "
              f"{self._color('red')}→ ERROR{self._color('reset')} ({elapsed_ms:.0f}ms) {error_msg}")

    def stats(self) -> str:
        total = self.accepted_count + self.rejected_count + self.error_count
        if total == 0:
            return "No requests processed"

        accept_rate = self.accepted_count / total * 100 if total > 0 else 0
        avg_time = self.total_time / total * 1000 if total > 0 else 0

        return (f"Processed {total} requests: "
                f"{self.accepted_count} accepted ({accept_rate:.0f}%), "
                f"{self.rejected_count} rejected, "
                f"{self.error_count} errors. "
                f"Avg time: {avg_time:.0f}ms")


log = Logger()


# =============================================================================
# Protocol helpers
# =============================================================================

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


# =============================================================================
# Request handler
# =============================================================================

def handle_client(conn: socket.socket, pipeline) -> bool:
    """
    Handle a single client request.

    Returns:
        True to continue serving, False to shutdown
    """
    start_time = time.perf_counter()

    try:
        # Read header length (4 bytes, uint32 little-endian)
        header_len_data = recv_exactly(conn, 4)
        header_len = struct.unpack('<I', header_len_data)[0]

        # Read JSON header
        header_data = recv_exactly(conn, header_len)
        header = json.loads(header_data.decode('utf-8'))

        request_type = header.get('type', 'path')

        if request_type == 'shutdown':
            log.info("Shutdown requested")
            send_response(conn, {'status': 'ok', 'message': 'shutting down'})
            return False

        elif request_type == 'path':
            # File path mode
            sim_path = header['path']
            # Shorten path for display
            display_path = Path(sim_path).name if '/' in sim_path else sim_path
            log.request('path', display_path)

            result = pipeline.predict_file(sim_path)

        elif request_type == 'binary':
            # Binary data mode
            frames = header['frames']
            pendulums = header['pendulums']
            values_per_pendulum = header.get('values', 8)

            data_size = frames * pendulums * values_per_pendulum * 4  # float32
            data_mb = data_size / (1024 * 1024)
            log.request('binary', f"{frames} frames × {pendulums} pendulums ({data_mb:.1f} MB)")

            # Read binary data
            data_bytes = recv_exactly(conn, data_size)

            # Reshape to (frames, pendulums, 8)
            data = np.frombuffer(data_bytes, dtype=np.float32)
            data = data.reshape(frames, pendulums, values_per_pendulum)

            # Run prediction
            result = pipeline.predict_simulation(data)

        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            log.response_error(f"Unknown request type: {request_type}", elapsed_ms)
            send_response(conn, {'status': 'error', 'message': f'unknown type: {request_type}'})
            return True

        # Calculate elapsed time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Build response - simplified API: client mainly needs accepted + boom_frame
        response = {
            'status': 'ok',
            'accepted': bool(result.accepted),
            'boom_frame': int(result.boom_frame) if result.boom_frame is not None else None,
            'accept_score': round(float(result.accept_score), 4),
            'predicted_quality': round(float(result.predicted_quality), 4),
            'disagreement': round(float(result.disagreement), 4),
            'model_predictions': {k: int(v) for k, v in result.model_predictions.items()},
        }

        # Log response
        details = f"quality={result.predicted_quality:.2f} disagree={result.disagreement:.1f} score={result.accept_score:.2f}"
        log.response(result.accepted, response['boom_frame'], elapsed_ms, details)

        send_response(conn, response)
        return True

    except ConnectionError:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        log.warning(f"Client disconnected ({elapsed_ms:.0f}ms)")
        return True
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        log.response_error(str(e), elapsed_ms)
        try:
            send_response(conn, {'status': 'error', 'message': str(e)})
        except:
            pass
        return True


# =============================================================================
# Server
# =============================================================================

def run_server(model_path: Path, socket_path: Path) -> None:
    """Run the boom detection server."""
    from boom_detection.deploy_pipeline import BoomDetectionPipeline

    # Load model
    log.info(f"Loading model from {model_path}...")
    load_start = time.perf_counter()
    pipeline = BoomDetectionPipeline.from_pretrained(model_path)
    load_time = time.perf_counter() - load_start

    log.success(f"Model loaded in {load_time:.1f}s")
    log.info(f"  Features: {pipeline.n_features}")
    log.info(f"  Max pendulums: {pipeline.feature_config.max_pendulums or 'unlimited'}")
    log.info(f"  Acceptance: {pipeline._acceptance_formula} with params {pipeline._acceptance_params}")
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
        print()
        log.info("Received shutdown signal...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log.success(f"Server listening on {socket_path}")
    log.info("Waiting for connections... (Ctrl+C to stop)")
    print()

    # Set socket timeout so we can check running flag
    server.settimeout(1.0)

    while running:
        try:
            conn, _ = server.accept()

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
                log.error(f"Server error: {e}")

    print()
    log.info(log.stats())
    log.success("Server stopped")
    server.close()
    if socket_path.exists():
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
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')

    args = parser.parse_args()

    if args.no_color:
        log.use_color = False

    if not args.model_path.exists():
        log.error(f"Model not found at {args.model_path}")
        sys.exit(1)

    run_server(args.model_path, args.socket)


if __name__ == '__main__':
    main()
