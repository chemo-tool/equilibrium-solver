#pragma once

#include <vector>

namespace equilibrium {

struct EquilibriumInput {
  std::vector<double> s_matrix;
  int rows = 0;
  int cols = 0;
  std::vector<int> phases;
  std::vector<double> elements;
  std::vector<double> gibbs_energies;
  double temperature = 298.15;
  double pressure = 1.0;
};

struct EquilibriumOutput {
  std::vector<double> composition;
  bool converged = false;
  int iteration_count = 0;
  int error_code = 0;
};

class IEquilibriumSolver {
public:
  virtual ~IEquilibriumSolver() = default;
  virtual EquilibriumOutput solve(const EquilibriumInput& input) = 0;
};

class SolverCore final : public IEquilibriumSolver {
public:
  EquilibriumOutput solve(const EquilibriumInput& input) override;
};

std::vector<double> calculate_equilibrium(const EquilibriumInput& input);

}  // namespace equilibrium
