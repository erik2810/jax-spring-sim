"""Pre-compute scenes to static binary bundles for the offline (Pages) build.

Each scene is serialised as::

    [ uint32 magic 0x4A535342 ("JSSB") ]
    [ uint32 metaLen ]
    [ metaLen bytes : UTF-8 JSON metadata ]
    [ concatenated frames : each is one server-protocol frame (pack_frame) ]

The frontend fetches these instead of opening a WebSocket, so the viewer runs
as a fully static site with no backend.

Usage::

    uv run python scripts/bake_scenes.py [output_dir]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from jax_spring_sim.server.protocol import pack_frame
from jax_spring_sim.server.scenes import build_scene

BUNDLE_MAGIC = 0x4A535342  # "JSSB"

# Scenes baked for the static demo, with their (default) parameters.
BAKED_SCENES: dict[str, dict] = {
    "chain": {},
    "cloth": {},
    "inverse": {},
}


def bake(name: str, params: dict) -> bytes:
    scene = build_scene(name, params)
    meta = json.dumps(scene.as_meta(), separators=(",", ":")).encode("utf-8")
    out = bytearray(struct.pack("<II", BUNDLE_MAGIC, len(meta)))
    out += meta
    for i in range(scene.positions.shape[0]):
        out += pack_frame(i, scene.positions[i], scene.value[i])
    return bytes(out)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("frontend/public/scenes")
    out_dir.mkdir(parents=True, exist_ok=True)

    baked = []
    for name, params in BAKED_SCENES.items():
        data = bake(name, params)
        (out_dir / f"{name}.bin").write_bytes(data)
        baked.append(name)
        print(f"baked {name:8s} {len(data) / 1024:7.0f} KB -> {out_dir / f'{name}.bin'}")

    (out_dir / "index.json").write_text(json.dumps({"scenes": baked}))


if __name__ == "__main__":
    main()
