"""Run TabPFN on a remote GPU and fly the fly from a laptop: a tiny decision server + its client.

On the GPU machine (listens on localhost only):
    MUJOCO_GL=egl .venv-fly/bin/python collect/remote_brain.py serve --context-steps 16384 --port 7777
On the laptop, tunnel the port and point the live viewer at it:
    ssh -N -L 7777:localhost:7777 <gpu-host> &
    python collect/view_flight.py --controller remote --path saccade_then_smooth

Protocol, per message: a 4-byte little-endian length, then that many bytes of float32. The client
sends the 104 flattened sensor values and gets the 12 actions back; an empty message resets the
controller's per-episode state. One client at a time.
"""

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from closed_loop import OBS_KEYS  # noqa: E402


def _send(sock, arr):
    data = np.asarray(arr, np.float32).tobytes()
    sock.sendall(struct.pack("<I", len(data)) + data)


def _recv(sock):
    head = _recv_exact(sock, 4)
    n = struct.unpack("<I", head)[0]
    return np.frombuffer(_recv_exact(sock, n), np.float32) if n else None


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return buf


class RemoteController:
    """Client side: behaves like TabPFNController (call with the observation dict, reset per episode)."""

    def __init__(self, address="localhost:7777"):
        host, port = address.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def reset(self):
        self.sock.sendall(struct.pack("<I", 0))
        _recv(self.sock)

    def __call__(self, observation):
        _send(self.sock, np.concatenate([np.ravel(observation[k]) for k in OBS_KEYS]))
        return _recv(self.sock).astype(np.float64)


def serve(args):
    from closed_loop import TabPFNController
    from flybody.fly_envs import flight_imitation  # only for the action bounds
    from preview_flight import find
    data = Path(__file__).resolve().parents[1] / "data"
    spec = flight_imitation(str(find(data, "flight-dataset_*.hdf5")), str(find(data, "wing_pattern_fmech.npy")),
                            randomize_start_step=False).action_spec()
    ctrl = TabPFNController(args.context_steps, "cuda", bounds=(spec.minimum, spec.maximum))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"serving TabPFN ({args.context_steps} context steps) on localhost:{args.port}", flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        n, t0 = 0, time.time()
        print("client connected", flush=True)
        try:
            while True:
                o = _recv(conn)
                if o is None:
                    ctrl.reset()
                    _send(conn, [])
                    continue
                _send(conn, ctrl.act_flat(o))
                n += 1
                if n % 500 == 0:
                    print(f"  {n} decisions, {(time.time() - t0) / n * 1000:.1f} ms each incl. network", flush=True)
        except ConnectionError:
            print(f"client left after {n} decisions", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["serve"])
    parser.add_argument("--context-steps", type=int, default=16384)
    parser.add_argument("--port", type=int, default=7777)
    serve(parser.parse_args())
