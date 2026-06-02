#include "equilibrium_solver.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
  m.doc() = "C++ equilibrium solver bindings";

  m.def(
      "calculate_equilibrium",
      [](const std::vector<std::vector<double>>& s,
         const std::vector<int>& phases,
         const std::vector<double>& elements,
         const std::vector<double>& gibbs_energies,
         double temperature,
         double pressure) {
        if (s.empty() || s[0].empty()) {
          throw std::invalid_argument("S must be non-empty");
        }

        const int rows = static_cast<int>(s.size());
        const int cols = static_cast<int>(s[0].size());
        for (const auto& row : s) {
          if (static_cast<int>(row.size()) != cols) {
            throw std::invalid_argument("S rows must all have same size");
          }
        }

        equilibrium::EquilibriumInput input;
        input.rows = rows;
        input.cols = cols;
        input.phases = phases;
        input.elements = elements;
        input.gibbs_energies = gibbs_energies;
        input.temperature = temperature;
        input.pressure = pressure;

        input.s_matrix.reserve(static_cast<size_t>(rows * cols));
        for (const auto& row : s) {
          input.s_matrix.insert(input.s_matrix.end(), row.begin(), row.end());
        }

        return equilibrium::calculate_equilibrium(input);
      },
      py::arg("S"),
      py::arg("phases"),
      py::arg("elements"),
      py::arg("gibbs_energies"),
      py::arg("temperature"),
      py::arg("pressure") = 1.0);
}
