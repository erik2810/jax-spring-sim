// Compact viridis colormap (8 stops, linearly interpolated). Returns [r,g,b]
// in 0..1. Used for the per-particle value channel (speed / distance-to-target).

const VIRIDIS = [
  [0.267, 0.005, 0.329],
  [0.283, 0.141, 0.458],
  [0.254, 0.265, 0.53],
  [0.207, 0.372, 0.553],
  [0.164, 0.471, 0.558],
  [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518],
  [0.267, 0.749, 0.441],
  [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.15],
  [0.993, 0.906, 0.144],
];

export function viridis(t) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[Math.min(VIRIDIS.length - 1, i + 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// CSS gradient string for the legend bar.
export function viridisGradient() {
  const stops = VIRIDIS.map((c, i) => {
    const pct = Math.round((i / (VIRIDIS.length - 1)) * 100);
    return `rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${pct}%`;
  });
  return `linear-gradient(to right, ${stops.join(', ')})`;
}
