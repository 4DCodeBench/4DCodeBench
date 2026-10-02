"""MAC velocity advection and the benchmark's common liquid seeding rule."""

import sys
import numpy as np
from config.world import TRACER_POINTS

SUBSTEPS = 4        # Heun substeps per frame
SEED = 20260821     # seed of the tracer generator
BISECTIONS = 6      # bisection steps of the seeding-spacing search


def log(*parts):
    """Print one progress line."""

    print("[liquid]", *parts)
    sys.stdout.flush()


class MacSampler:
    """Trilinear sampling of Mantaflow's staggered (MAC) velocity grid.

    Component c lives on the c-faces, offset by half a voxel along the other two
    axes. Positions are clamped to the grid, so a point outside the domain takes
    the boundary velocity.
    """

    def __init__(self, low, h, dim):
        self.low = np.asarray(low, np.float64)
        self.h = float(h)
        self.dim = np.asarray(dim, np.int64)

    def _blend(self, grid, x, component):
        i0 = np.clip(np.floor(x).astype(np.int64), 0, self.dim - 2)
        w = np.clip(x - i0, 0.0, 1.0)
        total = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    weight = (w[:, 0] if dx else 1 - w[:, 0]) \
                        * (w[:, 1] if dy else 1 - w[:, 1]) \
                        * (w[:, 2] if dz else 1 - w[:, 2])
                    corner = grid[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
                    total = total + weight * (corner if component is None else corner[:, component])
        return total

    def velocity(self, grid, points):
        q = (points - self.low) / self.h
        out = np.zeros_like(points)
        for c in range(3):
            x = q - 0.5
            x[:, c] += 0.5
            out[:, c] = self._blend(grid, x, c)
        return out



def rk2(sampler, grid, points, dt, substeps, nxt=None):
    """Heun integration, with linear temporal interpolation when the next field is known."""

    step = dt / substeps
    current = points.copy()

    def velocity(points, fraction):
        value = sampler.velocity(grid, points)
        return value if nxt is None else (1 - fraction) * value + fraction * sampler.velocity(nxt, points)

    for k in range(substeps):
        v1 = velocity(current, k / substeps)
        v2 = velocity(current + step * v1, (k + 1) / substeps)
        current = current + 0.5 * step * (v1 + v2)
    return current


def _coarse(cells, spacing):
    """The seeding cell of edge `spacing`, in grid units, that each grid cell centre lies in."""

    return np.floor((cells + 0.5) / spacing).astype(np.int64)


def _dilate(grid):
    """The grid with every true cell spread to its 26 neighbours."""

    out = grid
    for axis in range(3):
        out = out | np.roll(out, 1, axis) | np.roll(out, -1, axis)
    return out


def seed(rng, sampler, phi, alive, spacing):
    """One new tracer in every seeding cell that holds liquid but has no tracer near it.

    Seeding cells are cubes of `spacing` grid cells. A seeding cell is liquid when
    some grid cell centre inside it has `phi < 0`, and covered when a living
    tracer lies in it or in one of its 26 neighbours; the neighbourhood keeps
    tracers drifting between adjacent cells from triggering reseeding. Each
    uncovered liquid cell gets one tracer, at a uniformly drawn liquid grid cell
    inside it, jittered within that grid cell. Returns `(n, 3)` positions.
    """

    wet = np.argwhere(phi < 0.0)
    if not len(wet):
        return np.empty((0, 3), np.float64)
    shape = tuple(int(v) for v in _coarse(sampler.dim - 1, spacing) + 1)
    liquid = np.ravel_multi_index(tuple(_coarse(wet, spacing).T), shape)
    if len(alive):
        held = np.clip(np.floor((alive - sampler.low) / (sampler.h * spacing)), 0,
                       np.asarray(shape) - 1).astype(np.int64)
        covered = np.zeros(shape, bool)
        covered[tuple(held.T)] = True
        empty = ~_dilate(covered).ravel()[liquid]
        wet, liquid = wet[empty], liquid[empty]
        if not len(wet):
            return np.empty((0, 3), np.float64)
    order = rng.permutation(len(wet))                     # one liquid grid cell per seeding cell
    _, first = np.unique(liquid[order], return_index=True)
    chosen = wet[order[first]]
    return sampler.low + (chosen + rng.random((len(chosen), 3))) * sampler.h


def advect_tracers(fields, sampler, dt, spacing, seed_value):
    """Carry tracers through the take at one seeding spacing.

    Returns `(pos, births)`: `pos` is `(F, N, 3)` in grid units, NaN before birth
    and after leaving the domain, or None once more than `TRACER_POINTS` exist;
    `births` counts the tracers seeded per frame.
    """

    rng = np.random.default_rng(seed_value)
    frames = len(fields.frames)
    alive = np.empty((0, 3), np.float64)
    columns = []                                          # per frame: every column born so far
    births = []
    intervals = np.broadcast_to(dt, (frames - 1,))
    for slot in range(frames):
        if slot:
            live = np.isfinite(alive).all(axis=1)
            points = rk2(sampler, fields.velocity(slot - 1), alive[live], intervals[slot - 1],
                         SUBSTEPS, fields.velocity(slot))
            outside = ((points < sampler.low) | (points >= sampler.low + sampler.h * sampler.dim)).any(axis=1)
            points[outside] = np.nan
            alive[live] = points
        born = seed(rng, sampler, fields.phi(slot), alive[np.isfinite(alive).all(axis=1)], spacing)
        alive = np.concatenate([alive, born])
        births.append(len(born))
        # The column count only grows, so an over-budget trial stops here.
        if len(alive) > TRACER_POINTS:
            return None, np.asarray(births)
        columns.append(alive.copy())
    total = len(alive)
    pos = np.full((frames, total, 3), np.nan, np.float64)
    for slot, held in enumerate(columns):
        pos[slot, :len(held)] = held
    return pos, np.asarray(births)


def sample_tracers(fields, sampler, dt):
    """Tracers at the finest seeding spacing that stays within `TRACER_POINTS` columns.

    Uses the fixed `SEED`. Returns `(F, N, 3)` in grid units; raises `ValueError`
    when no grid cell centre is ever inside the liquid.
    """

    def run(spacing):
        pos, births = advect_tracers(fields, sampler, dt, spacing, SEED)
        count = f">{TRACER_POINTS}" if pos is None else str(pos.shape[1])
        log(f"seeding spacing {spacing:.3f} cells -> {count} tracers")
        return pos, births

    # Start from the spacing that tiles the fullest frame's liquid in TRACER_POINTS / 2
    # cells, bracket the budget by factors of 1.5, then bisect to the finest fit.
    fullest = max(int(np.count_nonzero(fields.phi(slot) < 0.0)) for slot in range(len(fields.frames)))
    spacing = max(1.0, (2.0 * fullest / TRACER_POINTS) ** (1 / 3))
    pos, births = run(spacing)
    if pos is None:
        low, high = spacing, spacing
        while pos is None:
            low, high = high, high * 1.5
            pos, births = run(high)
    else:
        low, high = spacing, spacing
        while low > 1.0:
            candidate = run(max(1.0, low / 1.5))
            if candidate[0] is None:
                low = max(1.0, low / 1.5)
                break
            low = high = max(1.0, low / 1.5)
            pos, births = candidate
    for _ in range(BISECTIONS if high > low else 0):
        middle = 0.5 * (low + high)
        candidate = run(middle)
        if candidate[0] is not None:
            high, (pos, births) = middle, candidate
        else:
            low = middle
    if not pos.shape[1]:
        raise ValueError("no cell centre is ever inside the liquid")
    log("tracers: %d columns, born per frame min %d / median %d / max %d"
        % (pos.shape[1], births.min(), int(np.median(births)), births.max()))
    return pos
