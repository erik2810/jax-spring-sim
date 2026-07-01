// Parse a baked scene bundle (see scripts/bake_scenes.py):
//   [u32 magic "JSSB"][u32 metaLen][metaLen JSON][concatenated protocol frames]
import { parseFrame } from './protocol.js';

const BUNDLE_MAGIC = 0x4a535342; // "JSSB"

export function parseSceneBundle(buffer) {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== BUNDLE_MAGIC) {
    throw new Error('not a scene bundle');
  }
  const metaLen = view.getUint32(4, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, metaLen)));

  const n = meta.n;
  const frameBytes = 16 + n * 3 * 4 + n * 4; // header + positions + value
  let offset = 8 + metaLen;

  const frames = [];
  for (let f = 0; f < meta.frames; f++) {
    // slice() yields a 0-offset ArrayBuffer, so the typed-array views inside
    // parseFrame are always aligned regardless of metaLen.
    const parsed = parseFrame(buffer.slice(offset, offset + frameBytes));
    frames.push({ positions: parsed.positions, value: parsed.value });
    offset += frameBytes;
  }
  return { meta, frames };
}
