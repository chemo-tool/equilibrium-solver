"""Ammonia Synthesis Equilibrium: Numerical Solution vs. Analytical Benchmark
==============================================================================

Reaction:   N2 + 3 H2  <=>  2 NH3     (ideal gas, stoichiometric feed)

This script demonstrates ``equilibrium_solver.calculate_equilibrium`` -- a
Gibbs-energy minimiser subject to linear element-balance constraints -- applied
to the ammonia synthesis equilibrium over a pressure range of 1-1000 atm at
T = 500 degC.

Correctness is verified by comparison with the exact closed-form solution
derived from the equilibrium-constant expression (see Analytical Solution
section below).

Thermodynamic data: standard Gibbs energies G(T) at 500 degC,
source: HSC Chemistry 9.0.
"""

import numpy as np
import matplotlib.pyplot as plt

from equilibrium_solver import calculate_equilibrium

# ---------------------------------------------------------------------------
# Physical constants and thermodynamic data
# ---------------------------------------------------------------------------

R = 8.314           # universal gas constant, J mol-1 K-1
T = 500.0 + 273.15  # temperature, K  (500 degC)

# Standard Gibbs energies G(T) at T = 500 degC, J mol-1  (HSC Chemistry 9.0)
G_N2  = -155_919.554
G_H2  = -108_636.545
G_NH3 = -205_546.813

# ---------------------------------------------------------------------------
# System definition
# ---------------------------------------------------------------------------

# Stoichiometric matrix A  (n_elements x n_species)
#   rows:    element N (row 0), element H (row 1)
#   columns: species N2 (col 0), H2 (col 1), NH3 (col 2)
#
#             N2    H2   NH3
A = np.array([[2.0, 0.0, 1.0],   # N atoms per molecule
              [0.0, 2.0, 3.0]])  # H atoms per molecule

phases = [1, 1, 1]   # all species belong to one gas phase (id = 1)

# Feed composition: 1 mol N2 + 3 mol H2  (stoichiometric 1:3 ratio)
n0 = np.array([1.0, 3.0, 0.0])   # initial molar amounts, mol
b  = A @ n0                       # element-balance RHS: [2.0, 6.0] mol

# Pressure sweep: 1-1000 atm, logarithmically spaced
P = np.logspace(0, 3, 60)   # atm

# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

# Reaction stoichiometric vector nu (products positive, reactants negative):
#   N2  +  3 H2  ->  2 NH3   =>   nu = (-1, -3, +2)
nu    = np.array([-1.0, -3.0, +2.0])
G_vec = np.array([G_N2, G_H2, G_NH3])

delta_G = nu @ G_vec                      # standard Gibbs energy of reaction, J mol-1
K_eq    = np.exp(-delta_G / (R * T))      # thermodynamic equilibrium constant

# For stoichiometric feed (1 mol N2 : 3 mol H2), let xi be the extent of
# reaction (moles of N2 converted).  The exact Kp expression is:
#
#   K_eq = 16 xi^2 (2 - xi)^2 / [27 (1 - xi)^4 P^2]           ... (1)
#
# Substituting the ansatz  xi = 1 - 1/sqrt(1 + beta)  with
# beta = (P/4) sqrt(27 K_eq)  into (1) reduces the RHS to K_eq identically,
# confirming that the following is an exact closed-form solution of (1):
#
#   xi(P) = 1 - 1 / sqrt(1 + (P/4) sqrt(27 K_eq))             ... (exact)

xi = 1.0 - 1.0 / np.sqrt(1.0 + (P / 4.0) * np.sqrt(27.0 * K_eq))

# Equilibrium molar amounts from the extent of reaction, mol
n_analyt = np.array([
    1.0 - xi,          # N2:  n0(N2) - xi
    3.0 * (1.0 - xi),  # H2:  n0(H2) - 3 xi
    2.0 * xi,          # NH3: n0(NH3) + 2 xi
])   # shape (3, n_P)

y_analyt = n_analyt / n_analyt.sum(axis=0) * 100.0   # mole percent

# ---------------------------------------------------------------------------
# Numerical solution via equilibrium_solver
# ---------------------------------------------------------------------------

# calculate_equilibrium minimises the total Gibbs free energy
#   G_tot = sum_i n_i [ G_i/(RT) + ln(n_i / n_phase) + ln P ]
# subject to the linear element-balance constraints  A n = b,  n_i >= 0.
# It returns equilibrium molar amounts in the column order of the matrix A.

gibbs_energies = np.array([G_N2, G_H2, G_NH3])   # J mol-1

n_num = np.empty((3, len(P)))
for j, p in enumerate(P):
    n_eq = calculate_equilibrium(
        S=A.tolist(),
        phases=phases,
        elements=b.tolist(),
        GibbsEnergies=gibbs_energies,
        T=T,
        Pa=float(p),
    )
    n_num[:, j] = n_eq

y_num = n_num / n_num.sum(axis=0) * 100.0   # mole percent

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

# Element-balance residual  ||A n - b||_inf / max(b)  at each pressure
elem_residuals = np.max(
    np.abs(A @ n_num - b[:, np.newaxis]) / np.abs(b).max(),
    axis=0,
)

# Deviation of numerical from analytical solution (mole percent)
diff    = y_num - y_analyt
rmse    = np.sqrt(np.mean(diff**2, axis=1))   # per species, mol %
max_dev = np.max(np.abs(diff), axis=1)         # per species, mol %

# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

species = ["N2", "H2", "NH3"]

print()
print("=" * 64)
print("  Ammonia Synthesis Equilibrium")
print(f"  T = 500 degC (773.15 K),  P = 1-1000 atm,  feed: 1 N2 + 3 H2")
print("=" * 64)
print(f"  delta_G_rxn = {delta_G / 1e3:+.2f} kJ mol-1     K_eq(T) = {K_eq:.4e}")
print()

for label, j in [("P =    1 atm", 0), ("P = 1000 atm", -1)]:
    print(f"  {label}")
    print(f"  {'Species':<6}  {'Analytical, mol%':>18}  {'Numerical, mol%':>16}  {'|dev|, mol%':>12}")
    print("  " + "-" * 58)
    for i, sp in enumerate(species):
        ya = y_analyt[i, j]
        yn = y_num[i, j]
        print(f"  {sp:<6}  {ya:>18.3f}  {yn:>16.3f}  {abs(yn - ya):>12.4f}")
    print()

print("  Aggregate deviations over full pressure sweep")
print(f"  {'Species':<6}  {'RMSE, mol%':>12}  {'Max |dev|, mol%':>16}")
print("  " + "-" * 38)
for i, sp in enumerate(species):
    print(f"  {sp:<6}  {rmse[i]:>12.5f}  {max_dev[i]:>16.5f}")
print()
print(f"  Max element-balance residual:  {elem_residuals.max():.2e}  (relative)")
print("=" * 64)
print()

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

COLORS = {"N2": "#1565C0", "H2": "#C62828", "NH3": "#2E7D32"}
LABELS = {
    "N2":  r"$\mathrm{N_2}$",
    "H2":  r"$\mathrm{H_2}$",
    "NH3": r"$\mathrm{NH_3}$",
}

fig, ax = plt.subplots(figsize=(8, 5))

for i, sp in enumerate(species):
    c = COLORS[sp]
    ax.plot(P, y_analyt[i], "-",  color=c, linewidth=1.8,
            label=f"{LABELS[sp]} analytical")
    ax.plot(P, y_num[i],    "o",  color=c, markersize=4,
            markerfacecolor="none", markeredgewidth=1.0,
            label=f"{LABELS[sp]} numerical")

ax.set_xscale("log")
ax.set_xlabel("Total pressure, atm", fontsize=11)
ax.set_ylabel("Mole fraction, mol %", fontsize=11)
ax.set_title(
    r"$\mathrm{N_2 + 3\,H_2 \rightleftharpoons 2\,NH_3}$"
    f"  --  $T$ = 500 $^\circ$C",
    fontsize=12,
)
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
ax.legend(fontsize=9, ncol=2, framealpha=0.9)

annotation = (
    f"$K_{{\\rm eq}}$ = {K_eq:.3e}\n"
    f"RMSE$_{{\\rm NH_3}}$ = {rmse[2]:.3f} mol %\n"
    f"max elem. balance res. = {elem_residuals.max():.1e}"
)
ax.text(
    0.98, 0.05, annotation,
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=8.5, family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#BDBDBD", alpha=0.85),
)

fig.tight_layout()
plt.show()
