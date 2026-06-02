#include "equilibrium_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace equilibrium {
namespace {

constexpr double kMatrixCoeffEps = 1e-5;
constexpr double kAmountFloor = 1e-60;
constexpr double kRelativeMoveEps = 1e-12;
constexpr double kNearZeroCleanupEps = 1e-14;
constexpr double kPhaseSumActiveEps = 1e-10;
constexpr double kSpeciesActiveEps = 1e-8;
constexpr double kSlackPenalty = 1e200;
constexpr double kInitialMinimalBasisChange = 100.0;
constexpr int kMinimalBasisHalvingCount = 46;
constexpr double kLineSearchBiasNumerator = 99999.0;
constexpr double kLineSearchBiasDenominator = 100000.0;
constexpr int kMaxIterationCount = 3000;

// Holds the result of the entering-column selection step. A column == -1
// indicates that no improving candidate was found in the current iteration.
struct PivotCandidate {
  int    column          = -1;
  bool   forward         = false;
  double step_lo         = 0.0;
  double step_hi         = 0.0;
  int    basis_replace_idx = -1;
};

// Internal simplex-like solver tailored for Gibbs-energy-based equilibrium.
// The implementation keeps variables grouped by phase and applies additional
// logarithmic terms for multi-species phases.
class Solver {
public:
  Solver(const std::vector<std::vector<double>>& a,
           const std::vector<double>& b,
           const std::vector<double>& c,
           const std::vector<int>& f,
           const std::vector<bool>& is_invariant_phase) {
    load_data(a, b, c, f, is_invariant_phase);
  }

  void load_data(const std::vector<std::vector<double>>& a,
                 const std::vector<double>& b,
                 const std::vector<double>& c,
                 const std::vector<int>& f,
                 const std::vector<bool>& is_invariant_phase) {
    // n_ - number of composition variables, m_ - number of element-balance constraints.
    n_ = static_cast<int>(c.size());
    m_ = static_cast<int>(b.size());

    a_.assign(m_, std::vector<double>(n_, 0.0));
    for (int i = 0; i < m_; ++i) {
      for (int j = 0; j < n_; ++j) {
        a_[i][j] = a[i][j];
      }
    }

    b_.assign(b.begin(), b.end());
    c_.assign(c.begin(), c.end());

    // Expand invariant phases into single-species pseudo-phases. This keeps
    // the downstream indexing uniform: every phase block has explicit length.
    std::vector<int> f_expanded;
    for (int i = 0; i < static_cast<int>(f.size()); ++i) {
      if (is_invariant_phase[i]) {
        for (int k = 0; k < f[i]; ++k) {
          f_expanded.push_back(1);
        }
      } else {
        f_expanded.push_back(f[i]);
      }
    }

    f_ = std::move(f_expanded);
    nf_ = static_cast<int>(f.size());
    init();
  }

  bool solve() {
    initialize_basis();

    std::vector<double> coeff1(nm_, 0.0);
    n_iter_ = 0;
    std::vector<bool> flag_array(nm_, false);
    std::vector<double> non_basis_phase_sum(nf_ + m_, 0.0);
    double minimal_change_basis = kInitialMinimalBasisChange;
    int count_minimal_change_basis = kMinimalBasisHalvingCount;

    // Main optimization loop. Each iteration either performs a pivot-like move
    // or a fallback phase-specific correction when no valid entering column exists.
    while (n_iter_ < max_iteration_count_) {
      normalize_x(result_, result_sum_);
      compute_non_basis_phase_sum(non_basis_phase_sum);
      calc_c(result_, coeff1);

      PivotCandidate pivot = select_entering_column(
          coeff1, non_basis_phase_sum, flag_array, minimal_change_basis,
          count_minimal_change_basis);

      if (pivot.column == -1) {
        // No entering column found: run phase-specific recovery moves to escape
        // near-degenerate regions and satisfy phase coupling constraints.
        int recovery_replace_idx = -1;
        const bool still_running = apply_recovery_moves(
            coeff1, flag_array, minimal_change_basis, count_minimal_change_basis,
            recovery_replace_idx);
        if (!still_running) break;
      } else {
        // Candidate accepted: perform safeguarded line search and update basis.
        ++n_iter_;
        apply_accepted_move(pivot, flag_array, count_minimal_change_basis);
      }
    }

    if (!check_feasibility()) return false;

    normalize_x(result_, result_sum_);
    calc_c(result_, coeff1);
    post_process_result(coeff1);
    trim_result();
    compute_objective();
    return true;
  }

  const std::vector<double>& result() const {
    return result_;
  }

  int iteration_count() const {
    return n_iter_;
  }

private:
  // ---------------------------------------------------------------------------
  // Group A — Initialization
  // ---------------------------------------------------------------------------

  // Builds the initial slack-variable basis and sets up the decomposition
  // matrix, result vector, and phase-sum vector.
  void initialize_basis() {
    is_basis_column_.assign(nm_, 0);
    index_basis_column_.assign(m_ + 1, 0);
    changes_decomposition_.assign(nm_, std::vector<double>(m_, 0.0));
    result_.assign(nm_, 0.0);
    result_sum_.assign(nf_ + m_, 0.0);

    for (int i = 0; i < n_; ++i) {
      for (int j = 0; j < m_; ++j) {
        changes_decomposition_[i][j] = -a_[j][i];
      }
      is_basis_column_[i] = -1;
    }

    for (int i = 0; i < m_; ++i) {
      is_basis_column_[n_ + i] = i;
      index_basis_column_[i] = n_ + i;
      changes_decomposition_[n_ + i][i] = -1.0;

      if (b_[i] >= 0.0) {
        result_[n_ + i] = b_[i];
      } else {
        for (int j = 0; j < nm_; ++j) {
          if (j != n_ + i) {
            changes_decomposition_[j][i] = -changes_decomposition_[j][i];
          }
        }
        result_[n_ + i] = -b_[i];
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Group B — Per-iteration setup
  // ---------------------------------------------------------------------------

  // Fills out[phase] with the sum of result_[i] for all non-basis variables i
  // belonging to that phase. Used in phase-coupled step bound calculations.
  void compute_non_basis_phase_sum(std::vector<double>& out) const {
    for (int i = 0; i < nf_ + m_; ++i) {
      out[i] = 0.0;
    }
    for (int i = 0; i < nm_; ++i) {
      if (is_basis_column_[i] == -1) {
        out[index_phase_[i]] += result_[i];
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Group C — Column selection
  // ---------------------------------------------------------------------------

  // Returns the reduced cost of column col given current gradient coefficients.
  double compute_reduced_cost(int col, const std::vector<double>& coeff1) const {
    double rc = coeff1[col];
    for (int j = 0; j < m_; ++j) {
      if (std::abs(changes_decomposition_[col][j]) > kMatrixCoeffEps) {
        rc += coeff1[index_basis_column_[j]] * changes_decomposition_[col][j];
      }
    }
    return rc;
  }

  // Computes the phase-coupled upper bound on the step size for column col
  // given the aggregated phase direction and the current minimal_change_basis.
  double compute_phase_limited_step(int col,
                                    const std::vector<double>& phase_dir,
                                    double minimal_change_basis) const {
    double limit = std::numeric_limits<double>::max();
    for (int j = 0; j < m_; ++j) {
      const int basis_var   = index_basis_column_[j];
      const int basis_phase = index_phase_[basis_var];

      if (std::abs(changes_decomposition_[col][j]) > kMatrixCoeffEps) {
        const double v = std::abs(minimal_change_basis * result_[basis_var] /
                                  changes_decomposition_[col][j]);
        if (limit > v) limit = v;
      }

      double den = phase_dir[basis_phase] * result_[basis_var];
      if (std::abs(changes_decomposition_[col][j]) > kMatrixCoeffEps) {
        den -= changes_decomposition_[col][j] * result_sum_[basis_phase];
      }
      if (den != 0.0) {
        const double v = std::abs(minimal_change_basis * result_[basis_var] *
                                  result_sum_[basis_phase] / den);
        if (limit > v) limit = v;
      }
    }
    return limit;
  }

  // Evaluates a forward step (increase x_col) for column col.
  // Updates best_abs and best if the step predicts objective improvement.
  // Returns true if best was updated.
  bool evaluate_forward_step(int col,
                             double reduced_cost,
                             double phase_limited_step,
                             const std::vector<double>& non_basis_phase_sum,
                             int count_minimal_change_basis,
                             double& best_abs,
                             PivotCandidate& best,
                             std::vector<bool>& flag_array) {
    int pivot_idx = -1;
    double forward_bound = std::numeric_limits<double>::max();
    for (int j = 0; j < m_; ++j) {
      if (changes_decomposition_[col][j] < -kMatrixCoeffEps) {
        const double v = -result_[index_basis_column_[j]] / changes_decomposition_[col][j];
        if (forward_bound > v) {
          forward_bound = v;
          pivot_idx = j;
        }
      }
    }

    const double trial_step = std::min(phase_limited_step, forward_bound);
    std::vector<double> x_local(m_ + 1, 0.0);
    std::vector<double> phase_sum_local(nf_ + m_, 0.0);
    std::vector<double> coeff_local(m_ + 1, 0.0);
    index_basis_column_[m_] = col;

    x_local[m_] = result_[col] + trial_step;
    for (int j = 0; j < m_; ++j) {
      double vv = result_[index_basis_column_[j]];
      if (std::abs(changes_decomposition_[col][j]) > kMatrixCoeffEps) {
        vv += trial_step * changes_decomposition_[col][j];
      }
      x_local[j] = std::max(kAmountFloor, vv);
    }

    for (int j = 0; j < nf_ + m_; ++j) {
      phase_sum_local[j] += non_basis_phase_sum[j];
    }
    for (int j = 0; j < m_; ++j) {
      phase_sum_local[index_phase_[index_basis_column_[j]]] += x_local[j];
    }
    phase_sum_local[index_phase_[col]] += trial_step;

    for (int j = 0; j <= m_; ++j) {
      const int v_idx = index_basis_column_[j];
      const int p_idx = index_phase_[v_idx];
      coeff_local[j] = cc_[v_idx];
      if (p_idx < nf_ && f_[p_idx] > 1) {
        coeff_local[j] += std::log(x_local[j] / phase_sum_local[p_idx]);
      }
    }

    double projected = coeff_local[m_];
    for (int j = 0; j < m_; ++j) {
      if (std::abs(changes_decomposition_[col][j]) > kMatrixCoeffEps) {
        projected += coeff_local[j] * changes_decomposition_[col][j];
      }
    }

    // Keep candidate only if directional derivative predicts improvement.
    if (projected < 0.0) {
      best_abs = -reduced_cost;
      best = {col, true, trial_step, forward_bound, pivot_idx};
      return true;
    }
    if (count_minimal_change_basis > 0) {
      flag_array[col] = true;
    }
    return false;
  }

  // Evaluates a backward step (decrease x_col) for column col.
  // Updates best_abs and best if the step predicts objective improvement.
  // Returns true if best was updated.
  bool evaluate_backward_step(int col,
                              double reduced_cost,
                              double phase_limited_step,
                              int count_minimal_change_basis,
                              double& best_abs,
                              PivotCandidate& best,
                              std::vector<bool>& flag_array) {
    int pivot_idx = -1;
    double backward_bound = result_[col];
    for (int j = 0; j < m_; ++j) {
      if (changes_decomposition_[col][j] > kMatrixCoeffEps) {
        const double v = result_[index_basis_column_[j]] / changes_decomposition_[col][j];
        if (backward_bound > v) {
          backward_bound = v;
          pivot_idx = j;
        }
      }
    }

    const double trial_step = std::min(phase_limited_step, backward_bound);
    std::vector<double> x_try(nm_, 0.0);
    std::vector<double> x_sum_try(nf_ + m_, 0.0);
    std::vector<double> coeff_local(m_ + 1, 0.0);
    index_basis_column_[m_] = col;

    for (int j = 0; j < nm_; ++j) {
      x_try[j] = result_[j];
    }
    x_try[col] -= trial_step;
    for (int j = 0; j < m_; ++j) {
      if (std::abs(changes_decomposition_[col][j]) > kMatrixCoeffEps) {
        x_try[index_basis_column_[j]] -= trial_step * changes_decomposition_[col][j];
      }
    }
    normalize_x(x_try, x_sum_try);

    for (int j = 0; j <= m_; ++j) {
      const int v_idx = index_basis_column_[j];
      const int p_idx = index_phase_[v_idx];
      coeff_local[j] = cc_[v_idx];
      if (p_idx < nf_ && f_[p_idx] > 1) {
        coeff_local[j] += std::log(x_try[v_idx] / x_sum_try[p_idx]);
      }
    }

    double projected = coeff_local[m_];
    for (int j = 0; j < m_; ++j) {
      if (std::abs(changes_decomposition_[col][j]) > kMatrixCoeffEps) {
        projected += coeff_local[j] * changes_decomposition_[col][j];
      }
    }

    // Sign flips compared to forward step due to opposite move direction.
    if (projected > 0.0) {
      best_abs = reduced_cost;
      best = {col, false, trial_step, backward_bound, pivot_idx};
      return true;
    }
    if (count_minimal_change_basis > 0) {
      flag_array[col] = true;
    }
    return false;
  }

  // Scans all non-basis columns and returns the strongest reduced-cost
  // PivotCandidate that passes feasibility and directional-derivative checks.
  // Returns a candidate with column == -1 when no improving direction exists.
  PivotCandidate select_entering_column(const std::vector<double>& coeff1,
                                        const std::vector<double>& non_basis_phase_sum,
                                        std::vector<bool>& flag_array,
                                        double minimal_change_basis,
                                        int count_minimal_change_basis) {
    PivotCandidate best;
    double best_abs = 0.0;

    for (int col = 0; col < nm_; ++col) {
      if (is_basis_column_[col] >= 0 || flag_array[col]) continue;

      const double reduced_cost = compute_reduced_cost(col, coeff1);
      if (reduced_cost == 0.0) continue;

      // reduced_cost < 0: objective decreases by increasing x_col.
      if (reduced_cost < 0.0) {
        if (best_abs >= -reduced_cost) continue;

        // Degenerate floor-bound check: accept immediately without step eval.
        bool degenerate = false;
        for (int j = 0; j < m_; ++j) {
          if (changes_decomposition_[col][j] < -kMatrixCoeffEps &&
              result_[index_basis_column_[j]] <= kAmountFloor) {
            best_abs = -reduced_cost;
            best = {col, true, 0.0, 0.0, j};
            degenerate = true;
            break;
          }
        }
        if (degenerate) continue;
      } else {
        // reduced_cost > 0: objective decreases by decreasing x_col.
        if (best_abs >= reduced_cost) continue;
        if (result_[col] <= kAmountFloor) continue;

        // Degenerate floor-bound check.
        bool degenerate = false;
        for (int j = 0; j < m_; ++j) {
          if (changes_decomposition_[col][j] > kMatrixCoeffEps &&
              result_[index_basis_column_[j]] <= kAmountFloor) {
            best_abs = reduced_cost;
            best = {col, false, 0.0, 0.0, j};
            degenerate = true;
            break;
          }
        }
        if (degenerate) continue;
      }

      // Build direction aggregated by phases to respect coupled phase totals.
      std::vector<double> phase_dir(nf_ + m_, 0.0);
      for (int j = 0; j < m_; ++j) {
        if (std::abs(changes_decomposition_[col][j]) > kMatrixCoeffEps) {
          phase_dir[index_phase_[index_basis_column_[j]]] += changes_decomposition_[col][j];
        }
      }
      ++phase_dir[index_phase_[col]];

      const double phase_limited_step =
          compute_phase_limited_step(col, phase_dir, minimal_change_basis);

      if (reduced_cost < 0.0 && best_abs < -reduced_cost) {
        evaluate_forward_step(col, reduced_cost, phase_limited_step, non_basis_phase_sum,
                              count_minimal_change_basis, best_abs, best, flag_array);
      } else if (reduced_cost > 0.0 && best_abs < reduced_cost) {
        evaluate_backward_step(col, reduced_cost, phase_limited_step,
                               count_minimal_change_basis, best_abs, best, flag_array);
      }
    }
    return best;
  }

  // ---------------------------------------------------------------------------
  // Group D — Recovery moves (no entering column found)
  // ---------------------------------------------------------------------------

  // Case A: a multi-species phase is entirely zero; tests if activating this
  // phase can reduce the objective. Performs a bisection line search if so.
  // Returns true if result_ was updated (basis moved).
  bool try_activate_zero_phase(int phase_offset, int length,
                               const std::vector<double>& coeff1,
                               int& basis_replace_idx) {
    double sum_beta = 0.0;
    std::vector<double> beta(length, 0.0);
    for (int i = 0; i < length; ++i) {
      const int idx = phase_offset + i;
      double red = 0.0;
      for (int j = 0; j < m_; ++j) {
        if (std::abs(changes_decomposition_[idx][j]) > kMatrixCoeffEps) {
          red += coeff1[index_basis_column_[j]] * changes_decomposition_[idx][j];
        }
      }
      beta[i] = std::exp(-red - cc_[idx]);
      sum_beta += beta[i];
      if (beta[i] > 1.0) {
        sum_beta = 0.0;
        break;
      }
    }
    if (sum_beta <= 1.0) return false;

    ++n_iter_;
    std::vector<double> direction(nm_, 0.0);
    for (int i = 0; i < length; ++i) {
      const int idx = phase_offset + i;
      for (int j = 0; j < m_; ++j) {
        if (std::abs(changes_decomposition_[idx][j]) > kMatrixCoeffEps) {
          direction[index_basis_column_[j]] += beta[i] * changes_decomposition_[idx][j];
        }
      }
      direction[idx] += beta[i];
    }

    double lo = 0.0;
    double hi = std::numeric_limits<double>::max();
    for (int j = 0; j < m_; ++j) {
      const int idx_basis = index_basis_column_[j];
      if (direction[idx_basis] < -kMatrixCoeffEps) {
        const double cand = -result_[idx_basis] / direction[idx_basis];
        if (hi > cand) {
          hi = cand;
          basis_replace_idx = j;
        }
      }
    }

    std::vector<double> x(nm_, 0.0);
    std::vector<double> coeff2(nm_, 0.0);
    std::vector<double> x_sum(nf_ + m_, 0.0);

    // One-dimensional line search by bisection on projection sign.
    while (true) {
      const double mid = (lo + hi) / 2.0;
      if (mid == lo || mid == hi) break;
      for (int i = 0; i < nm_; ++i) {
        x[i] = result_[i] + direction[i] * mid;
      }
      normalize_x(x, x_sum);
      calc_c(x, coeff2);

      double proj = 0.0;
      for (int i = 0; i < nm_; ++i) proj += coeff2[i] * direction[i];
      if (proj > 0.0) hi = mid;
      else            lo = mid;
    }

    bool moved_basis = false;
    for (int i = 0; i < nm_; ++i) {
      const double v = result_[i] + direction[i] * hi;
      if (is_basis_column_[i] >= 0) {
        moved_basis = moved_basis ||
                      std::abs((result_[i] - v) / result_[i]) > kRelativeMoveEps;
      }
      result_[i] = v;
    }

    // step_hi_outer is always 0.0 in the recovery path (no accepted move yet).
    // The guard triggers only when the bisection collapses to hi == 0.
    constexpr double step_hi_outer = 0.0;
    if (step_hi_outer == hi && basis_replace_idx >= 0) {
      int best_i = 0;
      for (int i = 1; i < length; ++i) {
        if (beta[best_i] < beta[i]) best_i = i;
      }
      replace_basis(index_basis_column_[basis_replace_idx], phase_offset + best_i);
    }

    return moved_basis;
  }

  // Case B: phase has positive total amount but no basis variable from the same
  // phase. Attempts an in-phase redistribution move.
  // Returns true if result_ was updated.
  bool try_redistribute_phase(int phase, int phase_offset, int length,
                              const std::vector<double>& coeff1) {
    // Verify that all phase variables are non-basis (precondition for Case B).
    for (int i = 0; i < length; ++i) {
      if (is_basis_column_[phase_offset + i] >= 0) return false;
    }

    double sum_beta = 0.0;
    std::vector<double> beta(length, 0.0);
    for (int i = 0; i < length; ++i) {
      const int idx = phase_offset + i;
      double red = 0.0;
      for (int j = 0; j < m_; ++j) {
        if (std::abs(changes_decomposition_[idx][j]) > kMatrixCoeffEps) {
          red += coeff1[index_basis_column_[j]] * changes_decomposition_[idx][j];
        }
      }
      beta[i] = std::exp(-red - cc_[idx]);
      sum_beta += beta[i];
      if (beta[i] > 1.0) break;
    }
    if (sum_beta >= 1.0) return false;

    ++n_iter_;
    std::vector<double> direction(nm_, 0.0);
    for (int i = 0; i < length; ++i) {
      const int idx = phase_offset + i;
      for (int j = 0; j < m_; ++j) {
        if (std::abs(changes_decomposition_[idx][j]) > kMatrixCoeffEps) {
          direction[index_basis_column_[j]] -= result_[idx] * changes_decomposition_[idx][j];
        }
      }
      direction[idx] -= result_[idx];
    }

    std::vector<double> x(nm_, 0.0);
    std::vector<double> coeff2(nm_, 0.0);
    std::vector<double> x_sum(nf_ + m_, 0.0);
    for (int i = 0; i < nm_; ++i) {
      x[i] = result_[i] + direction[i];
    }
    normalize_x(x, x_sum);
    calc_c(x, coeff2);

    double projected = 0.0;
    for (int i = 0; i < nm_; ++i) {
      if (index_phase_[i] == phase) {
        projected += coeff1[i] * direction[i];
      } else {
        projected += coeff2[i] * direction[i];
      }
    }

    if (projected < 0.0) {
      for (int i = 0; i < nm_; ++i) result_[i] += direction[i];
      return true;
    }
    return false;
  }

  // Last-ditch correction near termination: rebalances slack variables against
  // residual element-balance constraints and fixes basis sign conventions.
  void apply_last_ditch_correction() {
    for (int i = 0; i < m_; ++i) {
      if (result_[index_basis_column_[i]] < kNearZeroCleanupEps) {
        result_[index_basis_column_[i]] = 0.0;
      }
    }

    for (int i = 0; i < m_; ++i) {
      std::vector<int> idx(nm_, 0);
      for (int j = 0; j < nm_; ++j) idx[j] = j;

      for (int j = 1; j < nm_; ++j) {
        for (int k = 0; k < j; ++k) {
          if (result_[idx[k]] < result_[idx[j]]) std::swap(idx[j], idx[k]);
        }
      }

      double residual = b_[i];
      for (int j = 0; j < nm_; ++j) {
        if (idx[j] < n_) {
          residual -= a_[i][idx[j]] * result_[idx[j]];
        } else if (n_ + i == idx[j]) {
          residual -= result_[idx[j]];
        }
      }

      if (residual > 0.0) {
        result_[n_ + i] += residual;
      } else if (residual < 0.0) {
        result_[n_ + i] += residual;
        if (result_[n_ + i] < 0.0) {
          result_[n_ + i] = -result_[n_ + i];
          if (is_basis_column_[n_ + i] == -1) {
            for (int j = 0; j < m_; ++j) {
              changes_decomposition_[n_ + i][j] = -changes_decomposition_[n_ + i][j];
            }
          } else {
            const int col = is_basis_column_[n_ + i];
            for (int j = 0; j < n_ + m_; ++j) {
              if (j != n_ + i) {
                changes_decomposition_[j][col] = -changes_decomposition_[j][col];
              }
            }
          }
        }
      }
    }
  }

  // Orchestrates all recovery moves when no entering column was found:
  //   1. Case A — zero-phase activation (across all multi-species phases)
  //   2. Case B — in-phase redistribution (across all multi-species phases)
  //   3. Last-ditch correction and step-size halving
  // Returns false when count_minimal_change_basis drops below zero (caller
  // should break the main loop); true otherwise.
  bool apply_recovery_moves(const std::vector<double>& coeff1,
                            std::vector<bool>& flag_array,
                            double& minimal_change_basis,
                            int& count_minimal_change_basis,
                            int& basis_replace_idx) {
    bool changed = false;
    int phase_offset = 0;

    // Case A: try to activate a zero phase.
    for (int phase = 0; phase < nf_; ++phase) {
      const int length = f_[phase];
      if (length > 1) {
        bool all_zero = true;
        for (int i = 0; i < length; ++i) {
          if (result_[phase_offset + i] > kAmountFloor) { all_zero = false; break; }
        }
        if (all_zero) {
          changed = try_activate_zero_phase(phase_offset, length, coeff1,
                                            basis_replace_idx);
        }
      }
      if (!changed) phase_offset += length;
      else          break;
    }

    if (!changed) {
      phase_offset = 0;
      // Case B: try in-phase redistribution for phases with positive total but
      // no basis variable from that phase.
      for (int phase = 0; phase < nf_; ++phase) {
        const int length = f_[phase];
        if (length > 1) {
          bool has_positive = false;
          for (int i = 0; i < length; ++i) {
            if (result_[phase_offset + i] > kAmountFloor) { has_positive = true; break; }
          }
          if (has_positive) {
            changed = try_redistribute_phase(phase, phase_offset, length, coeff1);
          }
        }
        if (!changed) phase_offset += length;
        else          break;
      }

      if (!changed) {
        if (count_minimal_change_basis == 1) {
          apply_last_ditch_correction();
        }
        std::fill(flag_array.begin(), flag_array.end(), false);
        minimal_change_basis /= 2.0;
        --count_minimal_change_basis;
        if (count_minimal_change_basis < 0) {
          return false;
        }
      }
    }
    return true;
  }

  // ---------------------------------------------------------------------------
  // Group E — Accepted move
  // ---------------------------------------------------------------------------

  // Builds the move direction for the accepted pivot, performs a two-stage
  // biased+bisection line search, applies the step to result_, and updates
  // the basis (direct pivot or score-based replacement).
  void apply_accepted_move(const PivotCandidate& pivot,
                           std::vector<bool>& flag_array,
                           int& count_minimal_change_basis) {
    const int  new_column        = pivot.column;
    const bool forward_move      = pivot.forward;
    const int  basis_replace_idx = pivot.basis_replace_idx;
    double lo = pivot.step_lo;
    double hi = pivot.step_hi;

    std::vector<double> direction(m_ + 1, 0.0);
    if (forward_move) {
      for (int j = 0; j < m_; ++j) {
        if (std::abs(changes_decomposition_[new_column][j]) > kMatrixCoeffEps) {
          direction[j] = changes_decomposition_[new_column][j];
        }
      }
      direction[m_] = 1.0;
    } else {
      for (int j = 0; j < m_; ++j) {
        if (std::abs(changes_decomposition_[new_column][j]) > kMatrixCoeffEps) {
          direction[j] = -changes_decomposition_[new_column][j];
        }
      }
      direction[m_] = -1.0;
    }
    index_basis_column_[m_] = new_column;

    std::vector<double> x_local(m_ + 1, 0.0);
    std::vector<double> coeff_local(m_ + 1, 0.0);
    std::vector<double> non_basis_sum_local(nf_ + m_, 0.0);
    for (int i = 0; i < nm_; ++i) {
      if (is_basis_column_[i] == -1 && i != new_column) {
        non_basis_sum_local[index_phase_[i]] += result_[i];
      }
    }

    std::vector<double> phase_sum_local(nf_ + m_, 0.0);
    bool switched_to_bisect = false;

    // Two-stage line search: first biased toward larger step, then strict
    // bisection after crossing projection sign.
    while (true) {
      double step = 0.0;
      if (!switched_to_bisect) {
        step = std::min((lo + kLineSearchBiasNumerator * hi) / kLineSearchBiasDenominator, hi);
        if (step == lo || step == hi) switched_to_bisect = true;
      }
      if (switched_to_bisect) {
        step = (lo + hi) / 2.0;
        if (step == lo || step == hi) break;
      }

      for (int i = 0; i <= m_; ++i) {
        const int v_idx = index_basis_column_[i];
        x_local[i] = std::max(kAmountFloor, result_[v_idx] + direction[i] * step);
      }
      for (int i = 0; i < nf_ + m_; ++i) {
        phase_sum_local[i] = non_basis_sum_local[i];
      }
      for (int i = 0; i <= m_; ++i) {
        phase_sum_local[index_phase_[index_basis_column_[i]]] += x_local[i];
      }
      for (int i = 0; i <= m_; ++i) {
        const int v_idx = index_basis_column_[i];
        coeff_local[i] = cc_[v_idx];
        const int p_idx = index_phase_[v_idx];
        if (p_idx < nf_ && f_[p_idx] > 1) {
          coeff_local[i] += std::log(x_local[i] / phase_sum_local[p_idx]);
        }
      }

      double proj = 0.0;
      for (int i = 0; i <= m_; ++i) proj += coeff_local[i] * direction[i];

      if (proj > 0.0) { hi = step; switched_to_bisect = true; }
      else            { lo = step; }
    }

    bool moved_any   = false;
    bool moved_basis = false;
    for (int i = 0; i <= m_; ++i) {
      const int    idx = index_basis_column_[i];
      const double v   = result_[idx] + direction[i] * hi;
      moved_any = moved_any ||
                  std::abs((result_[idx] - v) / result_[idx]) > kRelativeMoveEps;
      if (i < m_) {
        moved_basis = moved_basis ||
                      std::abs((result_[idx] - v) / result_[idx]) > kRelativeMoveEps;
      }
      result_[idx] = v;
    }
    (void)moved_any;

    if (!moved_basis) {
      flag_array[new_column] = true;
    } else if (count_minimal_change_basis == 0) {
      std::fill(flag_array.begin(), flag_array.end(), false);
    }

    // If the optimal step hits the feasibility boundary, pivot basis directly.
    if (pivot.step_hi == hi) {
      if (basis_replace_idx >= 0) {
        replace_basis(index_basis_column_[basis_replace_idx], new_column);
      }
    } else {
      // Otherwise choose replacement by maximal phase-sensitivity score.
      normalize_x(result_, result_sum_);
      std::vector<double> phase_dir(nf_ + m_, 0.0);
      for (int i = 0; i < m_; ++i) {
        if (std::abs(changes_decomposition_[new_column][i]) > kMatrixCoeffEps) {
          phase_dir[index_phase_[index_basis_column_[i]]] +=
              changes_decomposition_[new_column][i];
        }
      }
      ++phase_dir[index_phase_[new_column]];

      const int    p_new       = index_phase_[new_column];
      int          replace_idx = m_;
      double       max_score   =
          std::abs(phase_dir[p_new] / result_sum_[p_new] - 1.0 / result_[new_column]);

      for (int i = 0; i < m_; ++i) {
        if (std::abs(changes_decomposition_[new_column][i]) > kMatrixCoeffEps) {
          const int    b_idx = index_basis_column_[i];
          const int    p_idx = index_phase_[b_idx];
          const double score =
              std::abs(phase_dir[p_idx] / result_sum_[p_idx] -
                       changes_decomposition_[new_column][i] / result_[b_idx]);
          if (max_score < score) { max_score = score; replace_idx = i; }
        }
      }

      if (replace_idx < m_) {
        replace_basis(index_basis_column_[replace_idx], new_column);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Group F — Post-processing
  // ---------------------------------------------------------------------------

  // Returns false if any slack variable exceeds the feasibility tolerance.
  bool check_feasibility() const {
    for (int i = 0; i < m_; ++i) {
      if (result_[n_ + i] > kMatrixCoeffEps) return false;
    }
    return true;
  }

  // Post-processes non-basis species in multi-species phases using the
  // closed-form beta relation from phase-partition optimality conditions.
  void post_process_result(const std::vector<double>& coeff1) {
    int offset = 0;
    for (int phase = 0; phase < nf_; ++phase) {
      const int length = f_[phase];
      bool all_zero = true;
      for (int i = 0; i < length; ++i) {
        if (result_[offset + i] > kAmountFloor) { all_zero = false; break; }
      }

      if (!all_zero && length > 1) {
        for (int i = 0; i < length; ++i) {
          const int idx = offset + i;
          if (is_basis_column_[idx] < 0) {
            double reduced = 0.0;
            for (int j = 0; j < m_; ++j) {
              if (std::abs(changes_decomposition_[idx][j]) > kMatrixCoeffEps) {
                reduced += coeff1[index_basis_column_[j]] * changes_decomposition_[idx][j];
              }
            }
            const double beta =
                std::min(1.0, std::max(0.0, std::exp(-reduced - cc_[idx])));
            result_[idx] =
                (result_sum_[index_phase_[idx]] - result_[idx]) * beta / (1.0 - beta);
          }
        }
      }
      offset += length;
    }
  }

  // Zeroes species below the amount floor and truncates result_ to n_ entries.
  void trim_result() {
    for (int i = 0; i < n_; ++i) {
      if (result_[i] <= kAmountFloor) result_[i] = 0.0;
    }
    const std::vector<double> result_full = result_;
    result_.assign(n_, 0.0);
    for (int i = 0; i < n_; ++i) {
      result_[i] = result_full[i];
    }
  }

  // Recomputes objective_function_ from the final result_ for diagnostics.
  void compute_objective() {
    objective_function_ = 0.0;
    int phase_offset = 0;
    for (int phase = 0; phase < nf_; ++phase) {
      const int length = f_[phase];
      double phase_sum = 0.0;
      for (int i = phase_offset; i < phase_offset + length; ++i) {
        phase_sum += result_[i];
      }
      if (phase_sum > kPhaseSumActiveEps) {
        for (int i = phase_offset; i < phase_offset + length; ++i) {
          if (result_[i] > kSpeciesActiveEps) {
            objective_function_ -=
                result_[i] * (c_[i] + std::log(result_[i] / phase_sum));
          }
        }
      }
      phase_offset += length;
    }
  }

  // ---------------------------------------------------------------------------
  // Internal helpers (unchanged)
  // ---------------------------------------------------------------------------

  void init() {
    // If target element totals are omitted, derive b from reference composition r.
    if (b_.empty()) {
      if (r_.empty()) {
        throw std::invalid_argument("Error r is null.");
      }
      b_.assign(m_, 0.0);
      for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < m_; ++j) {
          b_[j] += a_[j][i] * r_[i];
        }
      }
    }

    // index_phase_ maps each variable to either a physical phase [0..nf_-1]
    // or a dedicated slack pseudo-phase [nf_..nf_+m_-1].
    nm_ = n_ + m_;
    index_phase_.assign(nm_, 0);

    int offset = 0;
    for (int phase = 0; phase < nf_; ++phase) {
      const int cnt = f_[phase];
      for (int i = 0; i < cnt; ++i) {
        index_phase_[offset + i] = phase;
      }
      offset += cnt;
    }
    for (int i = 0; i < m_; ++i) {
      index_phase_[n_ + i] = nf_ + i;
    }

    // Base coefficients used in reduced-cost calculations.
    // Composition variables use -c; slack variables are heavily penalized.
    cc_.assign(n_ + m_, 0.0);
    for (int i = 0; i < n_; ++i) {
      cc_[i] = -c_[i];
    }
    for (int i = 0; i < m_; ++i) {
      cc_[n_ + i] = kSlackPenalty;
    }
  }

  void calc_c(const std::vector<double>& x, std::vector<double>& coeff) const {
    // Build gradient-like coefficients for current x, including
    // log(x_i / phase_sum) entropy terms for species in the same phase.
    int phase_offset = 0;
    for (int phase = 0; phase < nf_; ++phase) {
      const int count = f_[phase];
      double phase_sum = 0.0;
      for (int i = 0; i < count; ++i) {
        phase_sum += x[phase_offset + i];
      }
      for (int i = 0; i < count; ++i) {
        coeff[phase_offset + i] =
            cc_[phase_offset + i] + std::log(x[phase_offset + i] / phase_sum);
      }
      phase_offset += count;
    }

    for (int i = n_; i < nm_; ++i) {
      coeff[i] = cc_[i];
    }
  }

  void normalize_x(std::vector<double>& x, std::vector<double>& x_sum) const {
    // Clamp tiny negatives/noise and recompute phase sums used in log terms
    // and phase-coupled step bounds.
    for (int i = 0; i < nm_; ++i) {
      x[i] = std::max(x[i], kAmountFloor);
    }

    for (int i = 0; i < nf_ + m_; ++i) {
      x_sum[i] = 0.0;
    }

    for (int i = 0; i < nm_; ++i) {
      x_sum[index_phase_[i]] += x[i];
    }
  }

  void replace_basis(int old_column, int new_column) {
    // Pivot in decomposition space: normalize pivot row, eliminate pivot column
    // from all other rows, and refresh basis index mappings.
    std::vector<double>& pivot_row = changes_decomposition_[new_column];
    const int pivot_col = is_basis_column_[old_column];
    const double pivot = pivot_row[pivot_col];

    for (int i = 0; i < m_; ++i) {
      pivot_row[i] /= pivot;
    }

    for (int row = 0; row < nm_; ++row) {
        if (row != new_column &&
          std::abs(changes_decomposition_[row][pivot_col]) > kMatrixCoeffEps) {
        const double factor = changes_decomposition_[row][pivot_col];
        for (int col = 0; col < m_; ++col) {
          changes_decomposition_[row][col] -= factor * pivot_row[col];
        }
        changes_decomposition_[row][pivot_col] -= factor / pivot;
      }
    }

    for (int i = 0; i < m_; ++i) {
      pivot_row[i] = 0.0;
    }
    pivot_row[pivot_col] = -1.0;

    is_basis_column_[index_basis_column_[pivot_col]] = -1;
    is_basis_column_[new_column] = pivot_col;
    index_basis_column_[pivot_col] = new_column;
  }

private:
  int n_ = 0;
  int m_ = 0;
  int nf_ = 0;
  int nm_ = 0;

  std::vector<std::vector<double>> a_;
  std::vector<double> r_;
  std::vector<double> b_;
  std::vector<double> c_;
  std::vector<int> f_;
  std::vector<int> index_phase_;
  std::vector<double> cc_;

  std::vector<double> result_;
  double objective_function_ = 0.0;
  std::vector<double> result_sum_;
  int n_iter_ = 0;

  std::vector<int> is_basis_column_;
  std::vector<int> index_basis_column_;
  std::vector<std::vector<double>> changes_decomposition_;

  int max_iteration_count_ = kMaxIterationCount;
};

}  // namespace

EquilibriumOutput SolverCore::solve(const EquilibriumInput& input) {
  // Validate dense flattened input dimensions and basic thermodynamic arguments.
  if (input.rows <= 0 || input.cols <= 0) {
    throw std::invalid_argument("S must be non-empty");
  }
  if (static_cast<int>(input.s_matrix.size()) != input.rows * input.cols) {
    throw std::invalid_argument("flattened S shape mismatch");
  }
  if (static_cast<int>(input.phases.size()) != input.cols ||
      static_cast<int>(input.gibbs_energies.size()) != input.cols ||
      static_cast<int>(input.elements.size()) != input.rows) {
    throw std::invalid_argument("input vector dimensions mismatch");
  }
  if (input.temperature <= 0.0 || input.pressure <= 0.0) {
    throw std::invalid_argument("temperature and pressure must be positive");
  }

  const int rows = input.rows;
  const int cols = input.cols;

  // Map species index -> phase id.
  std::map<int, int> compounds;
  for (int i = 0; i < cols; ++i) {
    compounds[i] = input.phases[i];
  }

  // Count number of species in each phase.
  std::map<int, int> phase_counts;
  for (const auto& kv : compounds) {
    if (phase_counts.find(kv.second) != phase_counts.end()) {
      phase_counts[kv.second]++;
    } else {
      phase_counts[kv.second] = 1;
    }
  }

  // Reorder species so each phase is contiguous. Solver assumes this layout
  // for phase-block operations and index_phase_ construction.
  std::vector<std::vector<double>> s_sorted(rows, std::vector<double>(cols, 0.0));
  std::vector<int> clist_sorted;
  clist_sorted.reserve(static_cast<size_t>(cols));

  for (const auto& ph : phase_counts) {
    for (const auto& ci : compounds) {
      if (ci.second == ph.first) {
        clist_sorted.push_back(ci.first);
        const int i1 = static_cast<int>(clist_sorted.size()) - 1;
        for (int j = 0; j < rows; ++j) {
          s_sorted[j][i1] = input.s_matrix[static_cast<size_t>(j * cols + ci.first)];
        }
      }
    }
  }

  // Convert Gibbs energies to dimensionless form: -G/(R*T) with optional
  // pressure correction for selected phase model.
  constexpr double R = 8.314;
  std::vector<int> f;
  f.reserve(phase_counts.size());
  for (const auto& p : phase_counts) {
    f.push_back(p.second);
  }

  std::vector<bool> invariant_phase;
  invariant_phase.reserve(f.size());
  for (const int fi : f) {
    invariant_phase.push_back(fi == 1);
  }

  std::vector<double> G;
  G.reserve(clist_sorted.size());
  if (input.pressure == 1.0) {
    for (const int cs : clist_sorted) {
      G.push_back(-input.gibbs_energies[cs] / R / input.temperature);
    }
  } else {
    for (const int cs : clist_sorted) {
      G.push_back(-input.gibbs_energies[cs] / R / input.temperature -
                  (compounds[cs] == 1 ? std::log(input.pressure) : 0.0));
    }
  }

  // Normalize element totals to improve numerical conditioning.
  const double enorm = *std::max_element(input.elements.begin(), input.elements.end());
  if (!(enorm > 0.0) || !std::isfinite(enorm)) {
    throw std::invalid_argument("elements maximum must be positive and finite");
  }

  std::vector<double> elements_norm(rows, 0.0);
  for (int i = 0; i < rows; ++i) {
    elements_norm[i] = input.elements[i] / enorm;
  }

  Solver solver(s_sorted, elements_norm, G, f, invariant_phase);
  if (!solver.solve()) {
    throw std::runtime_error("Solution not found");
  }

  const std::vector<double> result = solver.result();
  if (static_cast<int>(result.size()) != static_cast<int>(compounds.size())) {
    throw std::runtime_error("Solution is incorrect");
  }

  // Revert from phase-sorted order back to original species indexing.
  std::vector<double> result_sorted(result.size(), 0.0);
  for (int i = 0; i < static_cast<int>(result.size()); ++i) {
    result_sorted[clist_sorted[i]] = result[i] * enorm;
  }

  EquilibriumOutput out;
  out.composition = std::move(result_sorted);
  out.converged = true;
  out.iteration_count = solver.iteration_count();
  out.error_code = 0;
  return out;
}

std::vector<double> calculate_equilibrium(const EquilibriumInput& input) {
  SolverCore solver;
  EquilibriumOutput out = solver.solve(input);
  return out.composition;
}

}  // namespace equilibrium
