import { useCallback, useEffect, useRef, useState } from 'react';
import { parseFrame } from './protocol.js';
import { parseSceneBundle } from './staticFormat.js';

// Single source of scene frames with two backends:
//   - live   : WebSocket to the FastAPI server (parameters recompute on the fly)
//   - static : fetch pre-baked .bin bundles (no backend; GitHub Pages build)
// Both expose the same interface so App is agnostic. Binary frames are kept in a
// ref (read by the render loop); only coarse status/meta/progress are React state.
export function useSceneData({ staticMode, wsUrl, baseUrl }) {
  const wsRef = useRef(null);
  const framesRef = useRef([]);
  const lastProgressRef = useRef(0);

  const [status, setStatus] = useState(staticMode ? 'idle' : 'connecting');
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);
  const [buffered, setBuffered] = useState(0);

  // Live mode: open the WebSocket and stream frames in.
  useEffect(() => {
    if (staticMode) return;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => setStatus('idle');
    ws.onclose = () => setStatus('disconnected');
    ws.onerror = () => setStatus('error');
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'status') {
          setStatus('computing');
        } else if (msg.type === 'meta') {
          framesRef.current = [];
          setBuffered(0);
          setMeta(msg);
          setStatus('streaming');
        } else if (msg.type === 'done') {
          setBuffered(framesRef.current.length);
          setStatus('done');
        } else if (msg.type === 'error') {
          setError(msg.message);
          setStatus('error');
        }
        return;
      }
      const frame = parseFrame(ev.data);
      if (!frame) return;
      framesRef.current.push({ positions: frame.positions, value: frame.value });
      const now = performance.now();
      if (now - lastProgressRef.current > 80) {
        lastProgressRef.current = now;
        setBuffered(framesRef.current.length);
      }
    };

    return () => ws.close();
  }, [staticMode, wsUrl]);

  const loadStatic = useCallback(
    async (scene) => {
      setStatus('computing');
      framesRef.current = [];
      setBuffered(0);
      setMeta(null);
      setError(null);
      try {
        const res = await fetch(`${baseUrl}scenes/${scene}.bin`);
        if (!res.ok) throw new Error(`could not load "${scene}" (${res.status})`);
        const { meta: m, frames } = parseSceneBundle(await res.arrayBuffer());
        framesRef.current = frames;
        setMeta(m);
        setBuffered(frames.length);
        setStatus('done');
      } catch (e) {
        setError(e.message);
        setStatus('error');
      }
    },
    [baseUrl],
  );

  const configure = useCallback(
    (scene, params, fps = 120) => {
      if (staticMode) {
        loadStatic(scene);
        return;
      }
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      framesRef.current = [];
      setBuffered(0);
      setMeta(null);
      setError(null);
      setStatus('computing');
      ws.send(JSON.stringify({ scene, params, fps }));
    },
    [staticMode, loadStatic],
  );

  return { status, meta, error, buffered, framesRef, configure };
}
