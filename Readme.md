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

## Documentation

Detailed usage instructions, input/output contract, and validation cases are provided in:

- [examples/guide_and_validation.md](https://github.com/chemo-tool/equilibrium-solver/blob/main/examples/guide_and_validation.md)

This guide includes:

- full API description for `calculate_equilibrium(...)`;
- input preparation workflow and interpretation of outputs;
- validation examples (ammonia synthesis, calcium carbonate decomposition, water-gas shift, steam reforming).

## API Summary

Public API:

- `calculate_equilibrium(S, phases, elements, GibbsEnergies, T, Pa=1.0)`

Input expectations (brief):

- `S`: non-empty matrix `(n_elements, n_species)`;
- `phases`: integer phase id per species, length `n_species`;
- `elements`: total elemental amounts, length `n_elements`;
- `GibbsEnergies`: standard Gibbs energies, length `n_species`;
- `T > 0`, `Pa > 0`.

Returns:

- equilibrium species amounts in the same species order as columns in `S`.

Errors:

- raises `ValueError`/`RuntimeError` (Python) or
  `std::invalid_argument`/`std::runtime_error` (C++) for invalid input or infeasible solves.

## Numerical Method (Brief)

The C++ core (`src/solver_core.cpp`) solves a constrained Gibbs-energy minimization problem
under element-balance equations with internal safeguards for numerical stability.
For detailed usage and validation-oriented interpretation, see `examples/guide_and_validation.md`.

## Build/Packaging Notes

- The extension is created by CMake via `pybind11_add_module(_native ...)`.
- Install target places `_native` into the `equilibrium_cpp` package.
- `scikit-build-core` drives CMake build from `pip install`.

## Troubleshooting

- Runtime error `Solution not found`:
	- check physical consistency of `S`, `elements`, phase assignments, and Gibbs data;
	- test with a simpler, known-feasible system first.

