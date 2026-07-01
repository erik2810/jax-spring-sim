// UI metadata for each scene. `params` keys match the backend builder kwargs
// (see server/scenes.py); `def` values seed the controls.
export const SCENES = {
  chain: {
    label: 'Catenary',
    blurb: 'A cable pinned at both ends settling under gravity. The whole rollout is a jit-compiled lax.scan.',
    transforms: ['jax.jit', 'lax.scan', 'jax.grad (forces)'],
    params: [
      { key: 'n', label: 'particles', min: 6, max: 100, step: 1, def: 48 },
      { key: 'stiffness', label: 'stiffness k', min: 50, max: 1200, step: 10, def: 400 },
    ],
  },
  cloth: {
    label: 'Cloth drape',
    blurb: 'A pinned cloth grid draping in 3D under gravity — the same energy + integrator, now on a 2D mesh.',
    transforms: ['jax.jit', 'lax.scan', 'jax.grad (forces)'],
    params: [
      { key: 'rows', label: 'rows', min: 4, max: 28, step: 1, def: 16 },
      { key: 'cols', label: 'cols', min: 4, max: 28, step: 1, def: 16 },
      { key: 'stiffness', label: 'stiffness k', min: 50, max: 800, step: 10, def: 220 },
    ],
  },
  inverse: {
    label: 'Inverse design',
    blurb: 'System identification: value_and_grad recovers hidden rest lengths. Watch Adam sculpt the chain onto the target (green).',
    transforms: ['jax.value_and_grad', 'jax.vmap', 'Adam through the rollout'],
    params: [
      { key: 'n', label: 'particles', min: 6, max: 60, step: 1, def: 28 },
      { key: 'opt_steps', label: 'Adam steps', min: 60, max: 400, step: 10, def: 220 },
    ],
  },
};

export function defaultParams(scene) {
  return Object.fromEntries(SCENES[scene].params.map((p) => [p.key, p.def]));
}
