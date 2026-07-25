"""The VTU/PVD writer, validated against an independent reader.

The writer is hand-rolled against the VTK XML spec, so the load-bearing test is
the round trip: every file must parse in ``meshio`` with identical points, spring
connectivity, and point data. 2D networks must pad into a plane, and a rollout
must produce an ordered ``.pvd`` time series.
"""

from __future__ import annotations

import numpy as np
import pytest

from jax_spring_sim import export_state, export_trajectory, make_chain, make_cloth, simulate
from jax_spring_sim.vtk_export import write_vtu

# meshio is the independent reader; skip the whole module if it is not installed.
meshio = pytest.importorskip("meshio")


def test_vtu_round_trips_through_meshio(tmp_path) -> None:
    state, system = make_cloth(3, 4)
    path = export_state(tmp_path / "snapshot.vtu", state, system)
    mesh = meshio.read(path)

    assert np.allclose(mesh.points, np.asarray(state.pos))
    (cells,) = mesh.cells
    assert cells.type == "line"
    assert np.array_equal(cells.data, np.asarray(system.edges))
    # meshio reads a 1-component DataArray back with a trailing axis; ravel to compare.
    assert np.allclose(mesh.point_data["pinned"].ravel(), np.asarray(system.fixed))
    assert "speed" in mesh.point_data and "velocity" in mesh.point_data


def test_two_d_positions_are_padded_to_a_plane(tmp_path) -> None:
    # make_chain defaults to 2D; VTK points are 3D, so z must be zero-filled.
    state, system = make_chain(5)
    assert state.pos.shape[1] == 2
    mesh = meshio.read(export_state(tmp_path / "chain.vtu", state, system))
    assert mesh.points.shape[1] == 3
    assert np.allclose(mesh.points[:, 2], 0.0)


def test_trajectory_time_series(tmp_path) -> None:
    state, system = make_cloth(4, 5)
    _, traj = simulate(state, system, dt=1e-3, n_steps=60, save_every=10)  # 6 frames
    pvd = export_trajectory(tmp_path, traj, system)

    assert pvd.name == "trajectory.pvd"
    text = pvd.read_text()
    frames = sorted(tmp_path.glob("trajectory_*.vtu"))
    assert len(frames) == 6
    for i, frame in enumerate(frames):
        assert frame.name in text
        assert f'timestep="{i}"' in text

    # The cloth falls under gravity: a later frame moves relative to the first.
    first = meshio.read(frames[0]).points
    last = meshio.read(frames[-1]).points
    assert float(np.abs(last - first).max()) > 1e-4
    # Speed field is consistent with the stored velocity magnitude.
    m = meshio.read(frames[-1])
    speed = m.point_data["speed"].ravel()
    assert np.allclose(speed, np.linalg.norm(m.point_data["velocity"], axis=1))


def test_writer_validates_shapes(tmp_path) -> None:
    state, system = make_cloth(3, 3)
    with pytest.raises(ValueError, match="D in"):
        write_vtu(tmp_path / "bad.vtu", np.zeros((4, 4)), system.edges)
    with pytest.raises(ValueError, match="expected"):
        write_vtu(tmp_path / "bad.vtu", np.asarray(state.pos), system.edges, {"f": np.zeros(3)})
