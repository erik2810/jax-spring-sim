import { useCallback, useEffect, useRef, useState } from 'react';
import { SpringScene } from './SpringScene.js';

// Bridges React state to the imperative three.js scene. The render loop lives in
// SpringScene and pulls the current frame through a stable provider that reads
// refs only — so playback never re-creates the scene or the loop.
export default function SpringViewer({ framesRef, meta, playbackRef, onProgress, onBackend }) {
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const clockRef = useRef({ last: 0, acc: 0, index: 0 });
  const onProgressRef = useRef(onProgress);
  const [ready, setReady] = useState(false);

  onProgressRef.current = onProgress;

  const provideFrame = useCallback(() => {
    const sc = sceneRef.current;
    const frames = framesRef.current;
    if (!sc || !sc.meta || frames.length === 0) return null;

    const pb = playbackRef.current;
    const clk = clockRef.current;
    const now = performance.now();
    let dt = now - (clk.last || now);
    clk.last = now;
    if (dt > 200) dt = 0; // skip the gap after a hidden tab

    const total = sc.meta.frames;
    const available = frames.length;

    if (pb.scrub != null) {
      clk.index = Math.min(pb.scrub, available - 1);
    } else if (pb.playing) {
      clk.acc += (dt / 1000) * 60 * pb.speed;
      while (clk.acc >= 1) {
        clk.index += 1;
        clk.acc -= 1;
      }
      const streamed = pb.done || available >= total;
      clk.index = streamed
        ? pb.loop
          ? clk.index % available
          : Math.min(clk.index, available - 1)
        : Math.min(clk.index, available - 1);
    }

    const i = Math.max(0, Math.min(clk.index, available - 1));
    onProgressRef.current?.(i);
    return frames[i];
  }, [framesRef, playbackRef]);

  // Create + initialise the scene once.
  useEffect(() => {
    const canvas = canvasRef.current;
    const sc = new SpringScene();
    sceneRef.current = sc;
    let disposed = false;

    sc.init(canvas)
      .then(() => {
        if (disposed) {
          sc.dispose();
          return;
        }
        sc.setFrameProvider(provideFrame);
        onBackend?.(sc.backendName);
        setReady(true);
      })
      .catch((err) => {
        console.error('renderer init failed', err);
        onBackend?.('unavailable');
      });

    const ro = new ResizeObserver(() => {
      const r = canvas.getBoundingClientRect();
      sc.resize(r.width, r.height);
    });
    ro.observe(canvas);

    return () => {
      disposed = true;
      ro.disconnect();
      sc.dispose();
      sceneRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Rebuild topology whenever a new scene's metadata arrives.
  useEffect(() => {
    const sc = sceneRef.current;
    if (ready && sc && meta) {
      sc.setTopology(meta);
      clockRef.current = { last: performance.now(), acc: 0, index: 0 };
    }
  }, [ready, meta]);

  return (
    <div className="relative h-full w-full">
      <canvas ref={canvasRef} className="block h-full w-full" />
    </div>
  );
}
