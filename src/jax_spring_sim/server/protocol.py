"""Binary wire protocol for streaming trajectory frames to the browser.

Control/metadata messages are JSON (sent as WebSocket *text* frames); the hot
path — per-step particle positions — is packed binary (sent as *binary*
frames). This mirrors the convention used across the sibling projects: JSON for
control, little-endian Float32 payloads for anything streamed every frame
(JSON is ~10-20x slower for that).

Per-frame binary layout (all little-endian)::

    [ header, 16 bytes ]
        uint32  magic        0x4A53494D  ("JSIM")
        uint32  frame_index
        uint32  n_particles
        uint32  flags        bit0 = has per-particle value channel
    [ positions : n_particles * 3 * float32 ]   x, y, z (y up)
    [ value     : n_particles * float32 ]        optional scalar per particle
"""

from __future__ import annotations

import struct

import numpy as np

MAGIC = 0x4A53494D  # "JSIM"
HEADER = struct.Struct("<IIII")
FLAG_HAS_VALUE = 1


def pack_frame(
    frame_index: int,
    positions: np.ndarray,
    value: np.ndarray | None = None,
) -> bytes:
    """Pack one frame.

    Args:
        frame_index: Sequential index of this frame.
        positions: ``(N, 3)`` float array (y-up world coordinates).
        value: Optional ``(N,)`` scalar per particle (e.g. speed) for colouring.

    Returns:
        The encoded frame as ``bytes``.
    """
    n = positions.shape[0]
    flags = FLAG_HAS_VALUE if value is not None else 0
    out = HEADER.pack(MAGIC, frame_index, n, flags)
    out += np.ascontiguousarray(positions, dtype="<f4").tobytes()
    if value is not None:
        out += np.ascontiguousarray(value, dtype="<f4").tobytes()
    return out


def unpack_frame(buffer: bytes) -> tuple[int, np.ndarray, np.ndarray | None]:
    """Inverse of :func:`pack_frame` (used by tests and Python clients)."""
    magic, frame_index, n, flags = HEADER.unpack_from(buffer, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic:#x}")
    offset = HEADER.size
    pos = np.frombuffer(buffer, dtype="<f4", count=n * 3, offset=offset).reshape(n, 3)
    offset += n * 3 * 4
    value = None
    if flags & FLAG_HAS_VALUE:
        value = np.frombuffer(buffer, dtype="<f4", count=n, offset=offset)
    return frame_index, pos, value
