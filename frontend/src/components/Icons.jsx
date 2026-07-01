// Minimal inline SVG icons (no icon-library dependency).
const base = { width: 16, height: 16, viewBox: '0 0 24 24', fill: 'currentColor' };

export const PlayIcon = () => (
  <svg {...base}>
    <path d="M8 5v14l11-7z" />
  </svg>
);

export const PauseIcon = () => (
  <svg {...base}>
    <path d="M6 5h4v14H6zM14 5h4v14h-4z" />
  </svg>
);

export const RestartIcon = () => (
  <svg {...base} fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 12a9 9 0 1 0 3-6.7" strokeLinecap="round" />
    <path d="M3 4v5h5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export const LoopIcon = () => (
  <svg {...base} fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M17 2l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M3 11V9a4 4 0 0 1 4-4h14" strokeLinecap="round" />
    <path d="M7 22l-4-4 4-4" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M21 13v2a4 4 0 0 1-4 4H3" strokeLinecap="round" />
  </svg>
);
