from __future__ import annotations

import importlib
import math
from typing import Iterable, Sequence

R = 8.314


def _as_2d_list(values: Sequence[Sequence[float]]) -> list[list[float]]:
    out = [list(map(float, row)) for row in values]
    if not out or not out[0]:
        raise ValueError("S must be a non-empty 2D matrix")
    width = len(out[0])
    if any(len(row) != width for row in out):
        raise ValueError("S rows must all have the same length")
    return out


def _as_1d_float(values: Iterable[float], name: str) -> list[float]:
    out = [float(v) for v in values]
    if not out:
        raise ValueError(f"{name} must be non-empty")
    if any(not math.isfinite(v) for v in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_1d_int(values: Iterable[int], name: str) -> list[int]:
    out = [int(v) for v in values]
    if not out:
        raise ValueError(f"{name} must be non-empty")
    return out


def _validate_inputs(
    S: Sequence[Sequence[float]],
    phases: Sequence[int],
    elements: Sequence[float],
    GibbsEnergies: Sequence[float],
    T: float,
    Pa: float,
) -> tuple[list[list[float]], list[int], list[float], list[float], float, float]:
    matrix = _as_2d_list(S)
    ph = _as_1d_int(phases, "phases")
    el = _as_1d_float(elements, "elements")
    gibbs = _as_1d_float(GibbsEnergies, "GibbsEnergies")

    rows = len(matrix)
    cols = len(matrix[0])

    if len(ph) != cols:
        raise ValueError("len(phases) must match number of columns in S")
    if len(gibbs) != cols:
        raise ValueError("len(GibbsEnergies) must match number of columns in S")
    if len(el) != rows:
        raise ValueError("len(elements) must match number of rows in S")

    if any(v < 0.0 for v in el):
        raise ValueError("elements must be non-negative")
    if not math.isfinite(T):
        raise ValueError("T must be finite")
    if not math.isfinite(Pa):
        raise ValueError("Pa must be finite")
    if T <= 0.0:
        raise ValueError("T must be > 0")
    if Pa <= 0.0:
        raise ValueError("Pa must be > 0")

    return matrix, ph, el, gibbs, float(T), float(Pa)


def calculate_equilibrium(
    S: Sequence[Sequence[float]],
    phases: Sequence[int],
    elements: Sequence[float],
    GibbsEnergies: Sequence[float],
    T: float,
    Pa: float = 1.0,
) -> list[float]:
    matrix, ph, el, gibbs, temp, pressure = _validate_inputs(
        S, phases, elements, GibbsEnergies, T, Pa
    )
    native = importlib.import_module("equilibrium_cpp._native")
    return list(native.calculate_equilibrium(matrix, ph, el, gibbs, temp, pressure))