import { useCallback, useEffect, useRef, useState } from 'react';
import { parseFrame } from './protocol.js';

// Manages the WebSocket to the FastAPI backend. Binary frames are accumulated
// into a ref (read directly by the render loop, no React churn); only coarse
// status/meta/progress live in React state.
export function useSimSocket(url) {
  const wsRef = useRef(null);
  const framesRef = useRef([]);
  const lastProgressRef = useRef(0);

  const [status, setStatus] = useState('connecting');
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);
  const [buffered, setBuffered] = useState(0);

  useEffect(() => {
    const ws = new WebSocket(url);
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
      // Throttle progress state updates (~12/s) to avoid re-rendering per frame.
      const now = performance.now();
      if (now - lastProgressRef.current > 80) {
        lastProgressRef.current = now;
        setBuffered(framesRef.current.length);
      }
    };

    return () => ws.close();
  }, [url]);

  const configure = useCallback((scene, params, fps = 120) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    framesRef.current = [];
    setBuffered(0);
    setMeta(null);
    setError(null);
    setStatus('computing');
    ws.send(JSON.stringify({ scene, params, fps }));
  }, []);

  return { status, meta, error, buffered, framesRef, configure };
}
