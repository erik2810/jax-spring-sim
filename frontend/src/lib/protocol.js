// Mirror of the server-side binary protocol (see server/protocol.py).
// Per-frame layout, little-endian:
//   [header 16B] magic(u32) frameIndex(u32) n(u32) flags(u32)
//   [positions n*3 f32]  [value n f32 if flags&1]

export const MAGIC = 0x4a53494d; // "JSIM"
const FLAG_HAS_VALUE = 1;

export function parseFrame(buffer) {
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) {
    console.error('bad frame magic', magic.toString(16));
    return null;
  }
  const frameIndex = view.getUint32(4, true);
  const n = view.getUint32(8, true);
  const flags = view.getUint32(12, true);

  let offset = 16;
  const positions = new Float32Array(buffer, offset, n * 3);
  offset += n * 3 * 4;

  let value = null;
  if (flags & FLAG_HAS_VALUE) {
    value = new Float32Array(buffer, offset, n);
  }
  return { frameIndex, n, positions, value };
}
