# equilibrium-solver

C++ (pybind11) chemical equilibrium solver with a Python API.

The project computes equilibrium composition by minimizing a Gibbs-energy-based
objective under linear element-balance constraints.

## Features

- C++17 core solver for performance.
- Python bindings via pybind11.
- High-level Python wrappers in `equilibrium_solver/api.py`.
- Input validation for matrix/vector dimensions and thermodynamic parameters.

## Repository Layout

- `src/solver_core.cpp`: core numerical method implementation.
- `src/python_bindings.cpp`: pybind11 bridge exposing `_native.calculate_equilibrium`.
- `equilibrium_solver/api.py`: public Python functions and validation helpers.
- `include/equilibrium_solver.hpp`: C++ public data structures and interfaces.

## Requirements

- Python 3.10+
- CMake 3.18+
- A C++17 compiler:
	- Linux: GCC/Clang
	- Windows: MSVC (Build Tools or Visual Studio)
- Build dependencies are declared in `pyproject.toml`:
	- `scikit-build-core>=0.9.7`
	- `pybind11>=2.13.0`

## Installation

### 1. Create and activate a virtual environment (recommended)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install package from source

From the repository root:

```bash
pip install equilibrium-solver
```

This step builds the C++ extension module and installs it as `equilibrium_cpp._native`.

## Quick Start (Python)

```python
from equilibrium_solver import calculate_equilibrium

S = [
		[2.0, 0.0, 1.0],  # element 1 stoichiometry by species
		[0.0, 2.0, 1.0],  # element 2 stoichiometry by species
]

phases = [0, 0, 1]                 # phase id per species
elements = [1.0, 1.0]              # total element amounts
gibbs_energies = [-10000.0, -9500.0, -12000.0]  # J/mol-like values
T = 1200.0
Pa = 1.0

x_eq = calculate_equilibrium(S, phases, elements, gibbs_energies, T, Pa)
print(x_eq)
```

Public API:

- `calculate_equilibrium(...)`: equilibrium calculation function.

## Input Contract

For `calculate_equilibrium(S, phases, elements, GibbsEnergies, T, Pa=1.0)`:

- `S`: non-empty 2D matrix with shape `(n_elements, n_species)`.
- `phases`: length `n_species`, integer phase id for each species.
- `elements`: length `n_elements`, non-negative and finite.
- `GibbsEnergies`: length `n_species`, finite.
- `T > 0`, `Pa > 0`.

## Output

`calculate_equilibrium(...)` returns a list/array of equilibrium molar amounts of components
in the order corresponding to the column order in the stoichiometric matrix `S`.
The returned values are non-negative and satisfy the element-balance constraints.

The solver raises `ValueError`/`RuntimeError` (Python side) or
`std::invalid_argument`/`std::runtime_error` (C++ side) when inputs are invalid
or no feasible solution is found.

## Numerical Method (Implemented in `src/solver_core.cpp`)

This solver is a custom constrained optimization routine with simplex-like basis
updates and phase-coupled logarithmic terms.

### Problem Form

The method targets minimization of an objective of the form:

- linear Gibbs contribution per species;
- entropy-like phase mixing term `x_i * log(x_i / sum_phase)` for multi-species phases;
- linear element-balance constraints `A * x = b`.

Slack variables are introduced internally to enforce feasibility during iterations.

### Core Ideas

1. Species are reordered so each phase forms a contiguous block.
2. A basis is initialized from slack variables.
3. Reduced costs are computed for non-basis columns.
4. Candidate moves are tested in two directions (increase/decrease variable).
5. Step sizes are bounded by:
	 - positivity of variables;
	 - phase-coupled stability limits.
6. A safeguarded line search (biased large-step then bisection) chooses the step.
7. Basis replacement (`replace_basis`) performs a pivot in decomposition space.
8. If no entering column is suitable, fallback phase-specific recovery moves are applied.
9. At convergence, slack variables must be near zero; then post-processing restores
	 non-basis values in multi-species phases.

### Numerical Stabilization

- Values are clamped to `>= 1e-60` before log operations.
- Many pivot/projection decisions use tolerances around `1e-5` and `1e-12`.
- Element totals are normalized by `max(elements)` before solving and rescaled back.

### Pressure Handling

Dimensionless transformed Gibbs energies are assembled as:

- `-G / (R * T)` at `Pa == 1.0`;
- additional `-log(Pa)` correction is applied for selected phase entries in the
	current model branch.

## Build/Packaging Notes

- The extension is created by CMake via `pybind11_add_module(_native ...)`.
- Install target places `_native` into the `equilibrium_cpp` package.
- `scikit-build-core` drives CMake build from `pip install`.

## Troubleshooting

- Runtime error `Solution not found`:
	- check physical consistency of `S`, `elements`, phase assignments, and Gibbs data;
	- test with a simpler, known-feasible system first.

