"""FastAPI app streaming JAX-computed trajectories over a WebSocket.

Protocol (see :mod:`.protocol`):

* Client -> server (JSON text): a *configure* message
  ``{"scene": "chain", "params": {...}, "fps": 60}``.
* Server -> client: a ``status`` then a ``meta`` (JSON text), a sequence of
  binary frames, then ``{"type": "done"}``. Sending a new configure message at
  any time cancels the current stream and starts the new scene.
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .protocol import pack_frame
from .scenes import SCENE_BUILDERS, build_scene


def create_app() -> FastAPI:
    app = FastAPI(title="jax-spring-sim viewer", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "scenes": list(SCENE_BUILDERS)}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        stream: asyncio.Task | None = None
        try:
            while True:
                msg = await socket.receive_json()
                if stream and not stream.done():
                    stream.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stream
                stream = asyncio.create_task(_run_scene(socket, msg))
        except WebSocketDisconnect:
            pass
        finally:
            if stream and not stream.done():
                stream.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await stream

    return app


async def _run_scene(socket: WebSocket, msg: dict) -> None:
    scene_name = msg.get("scene", "chain")
    params = msg.get("params", {})
    fps = float(msg.get("fps", settings.default_fps))
    period = 1.0 / max(1.0, fps)

    await socket.send_json({"type": "status", "state": "computing", "scene": scene_name})
    try:
        # JAX compilation + rollout is blocking; keep the event loop responsive.
        scene = await asyncio.to_thread(build_scene, scene_name, params)
    except Exception as exc:  # noqa: BLE001 — report any build failure to the client
        await socket.send_json({"type": "error", "message": str(exc)})
        return

    await socket.send_json(scene.as_meta())
    for i in range(scene.positions.shape[0]):
        await socket.send_bytes(pack_frame(i, scene.positions[i], scene.value[i]))
        await asyncio.sleep(period)
    await socket.send_json({"type": "done", "frames": int(scene.positions.shape[0])})


app = create_app()
