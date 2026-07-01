"""Backend protocol + WebSocket streaming."""

from __future__ import annotations

import numpy as np
import pytest

from jax_spring_sim.server.protocol import pack_frame, unpack_frame

fastapi = pytest.importorskip("fastapi")


def test_protocol_roundtrip() -> None:
    pos = np.random.default_rng(0).normal(size=(7, 3)).astype(np.float32)
    value = np.linspace(0, 1, 7).astype(np.float32)
    idx, pos2, value2 = unpack_frame(pack_frame(3, pos, value))
    assert idx == 3
    assert np.allclose(pos, pos2)
    assert np.allclose(value, value2)


def test_protocol_roundtrip_no_value() -> None:
    pos = np.zeros((4, 3), dtype=np.float32)
    idx, pos2, value2 = unpack_frame(pack_frame(0, pos, None))
    assert idx == 0
    assert value2 is None
    assert pos2.shape == (4, 3)


def test_ws_streams_chain_scene() -> None:
    from fastapi.testclient import TestClient

    from jax_spring_sim.server import create_app

    client = TestClient(create_app())
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {"scene": "chain", "params": {"n": 18, "frames": 8, "steps": 160}, "fps": 1000}
        )
        status = ws.receive_json()
        assert status["type"] == "status"

        meta = ws.receive_json()
        assert meta["type"] == "meta"
        assert meta["n"] == 18
        assert len(meta["edges"]) == 17
        assert meta["frames"] >= 1

        idx, pos, value = unpack_frame(ws.receive_bytes())
        assert idx == 0
        assert pos.shape == (18, 3)
        assert value is not None and value.shape == (18,)

        for _ in range(meta["frames"] - 1):
            unpack_frame(ws.receive_bytes())
        done = ws.receive_json()
        assert done["type"] == "done"
