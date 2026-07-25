"""Export a spring network to VTK files (VTU/PVD) for ParaView and PyVista.

The interactive viewer speaks the package's own binary trajectory format; the
scientific-visualisation world speaks VTK. This module writes the network as a
VTK unstructured grid: particles as points, springs as ``VTK_LINE`` cells, and
per-node fields (speed, pin mask) as point data. A rollout becomes a ``.pvd``
collection, one ``.vtu`` per saved frame with a time stamp, which ParaView plays
as an animation.

The writer targets the VTK XML format directly (ASCII arrays, an
``UnstructuredGrid`` piece, a ``Collection`` index for the time series) with no
dependency on the VTK stack. The test suite round-trips every file through
``meshio`` as an independent reader, so the output is checked against a
spec-conform implementation. Positions in 2D are padded to 3D, since VTK points
are always three-dimensional.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from .system import SpringSystem, State

_VTK_LINE = 3  # VTK cell type id for a two-point line element


def _points3(pos: jax.Array | np.ndarray) -> np.ndarray:
    """Node positions as ``(N, 3)`` float, zero-padding a 2D network into a plane."""
    arr = np.asarray(pos, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"positions must be (N, D), got {arr.shape}")
    n, d = arr.shape
    if d == 3:
        return arr
    if d == 2:
        return np.concatenate([arr, np.zeros((n, 1))], axis=1)
    raise ValueError(f"positions must have D in (2, 3), got D={d}")


def _ascii(a: np.ndarray) -> str:
    return " ".join(f"{v:.9g}" for v in np.asarray(a, dtype=np.float64).ravel())


def _ascii_int(a: np.ndarray) -> str:
    return " ".join(str(int(v)) for v in np.asarray(a).ravel())


def write_vtu(
    path: str | Path,
    points: jax.Array | np.ndarray,
    edges: jax.Array | np.ndarray,
    point_data: dict[str, jax.Array | np.ndarray] | None = None,
) -> Path:
    """Write one snapshot as a VTK unstructured grid (``.vtu``).

    Args:
        path: Output file; parent directories are created.
        points: Particle positions, shape ``(N, D)`` with ``D`` in ``(2, 3)``.
        edges: Spring index pairs, shape ``(E, 2)``.
        point_data: Optional per-node fields, each ``(N,)`` scalar or ``(N, 3)``.

    Returns:
        The written path.
    """
    pts = _points3(points)
    con = np.asarray(edges)
    n, e = pts.shape[0], con.shape[0]

    arrays: list[str] = []
    for name, field in (point_data or {}).items():
        arr = np.asarray(field, dtype=np.float64)
        comps = 1 if arr.ndim == 1 else arr.shape[1]
        if arr.shape[0] != n:
            raise ValueError(f"point_data '{name}' has {arr.shape[0]} entries, expected {n}")
        arrays.append(
            f'<DataArray type="Float64" Name="{name}" '
            f'NumberOfComponents="{comps}" format="ascii">{_ascii(arr)}</DataArray>'
        )
    point_data_block = f"<PointData>{''.join(arrays)}</PointData>" if arrays else ""

    offsets = np.arange(1, e + 1) * 2
    types = np.full(e, _VTK_LINE)
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        "<UnstructuredGrid>\n"
        f'<Piece NumberOfPoints="{n}" NumberOfCells="{e}">\n'
        "<Points>"
        f'<DataArray type="Float64" NumberOfComponents="3" format="ascii">{_ascii(pts)}</DataArray>'
        "</Points>\n"
        "<Cells>"
        f'<DataArray type="Int64" Name="connectivity" format="ascii">{_ascii_int(con)}</DataArray>'
        f'<DataArray type="Int64" Name="offsets" format="ascii">{_ascii_int(offsets)}</DataArray>'
        f'<DataArray type="UInt8" Name="types" format="ascii">{_ascii_int(types)}</DataArray>'
        "</Cells>\n"
        f"{point_data_block}\n"
        "</Piece>\n"
        "</UnstructuredGrid>\n"
        "</VTKFile>\n"
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    return out


def _node_fields(
    pos: np.ndarray, vel: np.ndarray | None, system: SpringSystem
) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {"pinned": np.asarray(system.fixed, dtype=np.float64)}
    if vel is not None:
        v = np.asarray(vel, dtype=np.float64)
        fields["velocity"] = _points3(v)
        fields["speed"] = np.linalg.norm(v, axis=1)
    return fields


def export_state(path: str | Path, state: State, system: SpringSystem) -> Path:
    """Write a single :class:`State` as a ``.vtu`` with the pin mask and velocity."""
    pos = np.asarray(state.pos)
    return write_vtu(path, pos, system.edges, _node_fields(pos, np.asarray(state.vel), system))


def export_trajectory(
    out_dir: str | Path, trajectory: State, system: SpringSystem, *, name: str = "trajectory"
) -> Path:
    """Write a rollout trajectory as a ``.pvd`` time series of ``.vtu`` frames.

    Args:
        out_dir: Target directory (created if needed).
        trajectory: A :class:`State` with a leading time axis, i.e. ``pos`` /
            ``vel`` of shape ``(F, N, D)`` (as returned by
            :func:`~jax_spring_sim.dynamics.simulate`).
        system: Provides the spring edges and the pin mask.
        name: Base name; frames become ``<name>_0000.vtu`` and the index
            ``<name>.pvd``.

    Returns:
        The path of the ``.pvd`` collection file. Open it in ParaView and press
        play, or load it with ``pyvista.read``.
    """
    pos = np.asarray(trajectory.pos)
    vel = np.asarray(trajectory.vel)
    if pos.ndim != 3:
        raise ValueError(f"trajectory.pos must be (F, N, D), got {pos.shape}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for f in range(pos.shape[0]):
        fname = f"{name}_{f:04d}.vtu"
        write_vtu(out / fname, pos[f], system.edges, _node_fields(pos[f], vel[f], system))
        entries.append(f'<DataSet timestep="{f}" group="" part="0" file="{fname}"/>')

    pvd = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "<Collection>\n" + "\n".join(entries) + "\n</Collection>\n</VTKFile>\n"
    )
    pvd_path = out / f"{name}.pvd"
    pvd_path.write_text(pvd)
    return pvd_path
