import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BlockMath } from 'react-katex';
import SpringViewer from './viewer/SpringViewer.jsx';
import { useSimSocket } from './lib/useSimSocket.js';
import { viridisGradient } from './lib/colormap.js';
import { SCENES, defaultParams } from './components/scenes.js';
import { LoopIcon, PauseIcon, PlayIcon, RestartIcon } from './components/Icons.jsx';

const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

const STATUS = {
  connecting: ['#eab308', 'connecting to backend…'],
  idle: ['#22c55e', 'connected'],
  computing: ['#38bdf8', 'JAX compiling + simulating…'],
  streaming: ['#38bdf8', 'streaming frames'],
  done: ['#22c55e', 'ready'],
  error: ['#ef4444', 'error'],
  disconnected: ['#ef4444', 'backend offline — run the server'],
};

export default function App() {
  const { status, meta, error, buffered, framesRef, configure } = useSimSocket(WS_URL);

  const [scene, setScene] = useState('chain');
  const [params, setParams] = useState(() => defaultParams('chain'));
  const [playing, setPlaying] = useState(true);
  const [loop, setLoop] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [frame, setFrame] = useState(0);
  const [backend, setBackend] = useState('…');

  const playbackRef = useRef({ playing: true, loop: true, speed: 1, scrub: null, seek: null });
  const ranRef = useRef(false);
  const progressRef = useRef(0);

  // Keep the imperative playback ref in sync with React state.
  useEffect(() => {
    const pb = playbackRef.current;
    pb.playing = playing;
    pb.loop = loop;
    pb.speed = speed;
  }, [playing, loop, speed]);

  const run = useCallback(
    (nextScene, nextParams) => {
      playbackRef.current.scrub = null;
      playbackRef.current.seek = 0;
      setPlaying(true);
      setFrame(0);
      configure(nextScene, nextParams);
    },
    [configure],
  );

  // Auto-run the default scene once the socket is connected.
  useEffect(() => {
    if (!ranRef.current && (status === 'idle' || status === 'done')) {
      ranRef.current = true;
      run(scene, params);
    }
  }, [status, scene, params, run]);

  const onSelectScene = (next) => {
    const p = defaultParams(next);
    setScene(next);
    setParams(p);
    run(next, p);
  };

  const onParam = (key, value) => setParams((p) => ({ ...p, [key]: value }));

  const onProgress = useCallback((i) => {
    progressRef.current = i;
    setFrame(i); // throttled upstream by rAF cadence; cheap enough here
  }, []);

  const onScrub = (value) => {
    setPlaying(false);
    playbackRef.current.scrub = value;
    setFrame(value);
  };
  const resumeFrom = () => {
    playbackRef.current.scrub = null;
    setPlaying(true);
  };
  const restart = () => {
    playbackRef.current.scrub = null;
    playbackRef.current.seek = 0;
    setPlaying(true);
  };

  const totalFrames = meta?.frames ?? 0;
  const [statusColor, statusText] = STATUS[status] ?? STATUS.connecting;
  const sceneDef = SCENES[scene];

  const valueRange = meta?.valueRange;
  const rangeLabel = useMemo(() => {
    if (!valueRange) return null;
    return valueRange.map((v) => v.toFixed(2));
  }, [valueRange]);

  return (
    <div className="flex h-full flex-col bg-[#0a0e14] text-[#e6edf3]">
      <header className="flex items-center justify-between border-b border-white/5 px-5 py-3">
        <div>
          <h1 className="text-sm font-semibold tracking-tight">
            jax-spring-sim <span className="text-white/40">· differentiable physics viewer</span>
          </h1>
          <p className="mt-0.5 text-xs text-white/40">
            JAX computes the trajectory · streamed over WebSocket · rendered with three.js (WebGPU/TSL)
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <span className="rounded-full border border-white/10 px-2 py-0.5 text-white/60">
            renderer: <span className="text-[#5b9cff]">{backend}</span>
          </span>
          <span className="flex items-center gap-1.5 text-white/60">
            <span className="h-2 w-2 rounded-full" style={{ background: statusColor }} />
            {statusText}
          </span>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <main className="relative min-w-0 flex-1">
          <SpringViewer
            framesRef={framesRef}
            meta={meta}
            playbackRef={playbackRef}
            onProgress={onProgress}
            onBackend={setBackend}
          />
          {error && (
            <div className="absolute left-1/2 top-4 -translate-x-1/2 rounded-md border border-red-500/40 bg-red-950/70 px-3 py-1.5 text-xs text-red-200">
              {error}
            </div>
          )}
          {status === 'disconnected' && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="max-w-sm rounded-lg border border-white/10 bg-black/50 p-5 text-center text-sm text-white/70">
                <p className="mb-2 font-medium text-white/90">Backend offline</p>
                <p className="text-xs leading-relaxed text-white/50">
                  Start it with{' '}
                  <code className="rounded bg-white/10 px-1 py-0.5 text-[#9ecbff]">
                    uv run python -m jax_spring_sim.server
                  </code>{' '}
                  then reload.
                </p>
              </div>
            </div>
          )}
        </main>

        <aside className="flex w-80 shrink-0 flex-col gap-5 overflow-y-auto border-l border-white/5 p-4 text-sm">
          {/* Scene selector */}
          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/40">
              Scene
            </h2>
            <div className="grid grid-cols-3 gap-1.5">
              {Object.entries(SCENES).map(([key, def]) => (
                <button
                  key={key}
                  onClick={() => onSelectScene(key)}
                  className={`rounded-md px-2 py-1.5 text-xs transition ${
                    scene === key
                      ? 'bg-[#5b9cff] text-black'
                      : 'bg-white/5 text-white/70 hover:bg-white/10'
                  }`}
                >
                  {def.label}
                </button>
              ))}
            </div>
            <p className="mt-2 text-xs leading-relaxed text-white/45">{sceneDef.blurb}</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {sceneDef.transforms.map((t) => (
                <span
                  key={t}
                  className="rounded bg-[#5b9cff]/10 px-1.5 py-0.5 font-mono text-[10px] text-[#9ecbff]"
                >
                  {t}
                </span>
              ))}
            </div>
          </section>

          {/* Parameters */}
          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/40">
              Parameters
            </h2>
            <div className="flex flex-col gap-3">
              {sceneDef.params.map((p) => (
                <label key={p.key} className="block">
                  <div className="mb-1 flex justify-between text-xs text-white/60">
                    <span>{p.label}</span>
                    <span className="font-mono text-white/80">{params[p.key]}</span>
                  </div>
                  <input
                    type="range"
                    className="w-full"
                    min={p.min}
                    max={p.max}
                    step={p.step}
                    value={params[p.key]}
                    onChange={(e) => onParam(p.key, Number(e.target.value))}
                  />
                </label>
              ))}
            </div>
            <button
              onClick={() => run(scene, params)}
              disabled={status === 'computing'}
              className="mt-3 w-full rounded-md bg-[#5b9cff] px-3 py-2 text-xs font-medium text-black transition hover:bg-[#7fb0ff] disabled:opacity-40"
            >
              {status === 'computing' ? 'computing…' : 'Run simulation'}
            </button>
          </section>

          {/* Playback */}
          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/40">
              Playback
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => (playing ? setPlaying(false) : resumeFrom())}
                className="flex h-8 w-8 items-center justify-center rounded-md bg-white/10 hover:bg-white/20"
                title={playing ? 'pause' : 'play'}
              >
                {playing ? <PauseIcon /> : <PlayIcon />}
              </button>
              <button
                onClick={restart}
                className="flex h-8 w-8 items-center justify-center rounded-md bg-white/10 hover:bg-white/20"
                title="restart"
              >
                <RestartIcon />
              </button>
              <button
                onClick={() => setLoop((l) => !l)}
                className={`flex h-8 w-8 items-center justify-center rounded-md ${
                  loop ? 'bg-[#5b9cff] text-black' : 'bg-white/10 hover:bg-white/20'
                }`}
                title="loop"
              >
                <LoopIcon />
              </button>
              <div className="ml-auto font-mono text-xs text-white/50">
                {frame + 1}/{totalFrames || '—'}
              </div>
            </div>

            <input
              type="range"
              className="mt-3 w-full"
              min={0}
              max={Math.max(0, totalFrames - 1)}
              step={1}
              value={Math.min(frame, Math.max(0, totalFrames - 1))}
              onChange={(e) => onScrub(Number(e.target.value))}
              onMouseUp={resumeFrom}
              onTouchEnd={resumeFrom}
            />

            <label className="mt-3 block">
              <div className="mb-1 flex justify-between text-xs text-white/60">
                <span>speed</span>
                <span className="font-mono text-white/80">{speed.toFixed(2)}×</span>
              </div>
              <input
                type="range"
                className="w-full"
                min={0.1}
                max={3}
                step={0.1}
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
              />
            </label>

            {status === 'streaming' && totalFrames > 0 && (
              <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full bg-[#5b9cff] transition-[width]"
                  style={{ width: `${Math.round((buffered / totalFrames) * 100)}%` }}
                />
              </div>
            )}
          </section>

          {/* Colour legend */}
          {meta && (
            <section>
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/40">
                {meta.valueLabel}
              </h2>
              <div className="h-2.5 w-full rounded-full" style={{ background: viridisGradient() }} />
              {rangeLabel && (
                <div className="mt-1 flex justify-between font-mono text-[10px] text-white/40">
                  <span>{rangeLabel[0]}</span>
                  <span>{rangeLabel[1]}</span>
                </div>
              )}
              <div className="mt-2 flex items-center gap-1.5 text-[10px] text-white/40">
                <span className="inline-block h-2 w-2 rounded-full bg-[#ffae3b]" /> pinned particle
                {meta.target && (
                  <>
                    <span className="ml-2 inline-block h-2 w-3 bg-[#37d39a]" /> target shape
                  </>
                )}
              </div>
            </section>
          )}

          {/* Physics */}
          <section className="mt-auto">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/40">
              The physics
            </h2>
            <div className="rounded-md bg-white/[0.03] p-2 text-[11px] text-white/70">
              <BlockMath math="U(\mathbf{x}) = \tfrac12\!\sum_{(i,j)} k_{ij}\big(\lVert \mathbf{x}_i-\mathbf{x}_j\rVert - L_{ij}\big)^2 - \sum_i m_i\,\mathbf{g}\cdot\mathbf{x}_i" />
              <BlockMath math="\mathbf{F} = -\nabla_{\mathbf{x}} U \quad (\texttt{jax.grad})" />
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
