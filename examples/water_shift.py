"""Water-Gas Shift: Numerical vs. Analytical Equilibrium
=======================================================

Reaction:
   CO(g) + H2O(g)  <=>  CO2(g) + H2(g)

This example demonstrates ``equilibrium_solver.calculate_equilibrium`` for a
homogeneous gas-phase equilibrium as a function of temperature (300-1475 K,
1 atm).

For this reaction, Delta_nu = 0, so (ideal gas):

   K(T) = exp(-DeltaG(T)/(R T)) = (y_CO2 * y_H2) / (y_CO * y_H2O)

The script compares analytical and numerical equilibrium compositions and
plots mole fractions of all components.
"""

import matplotlib.pyplot as plt
import numpy as np

from equilibrium_solver import calculate_equilibrium

# ---------------------------------------------------------------------------
# Thermodynamic data (J/mol), tabulated at T = 300..1475 K with step 25 K
# ---------------------------------------------------------------------------

G = {
    "CO": [
        -169953.2641168445, -174932.1180543994, -179967.30574472813,
        -185055.13096352003, -190192.3745402135, -195376.208017457,
        -200604.12678534118, -205873.89753993286, -211183.5164533516,
        -216531.17547528018, -221915.23489248808, -227334.20076518477,
        -232786.70620756724, -238271.49573051112, -243787.41204706763,
        -249333.3848764007, -254908.4213827158, -260511.59796203786,
        -266142.0531479982, -271798.9814527674, -277481.6279942908,
        -283189.2837884739, -288921.2816067251, -294676.9923166071,
        -300455.82163728145, -306257.20725268, -312080.61623449787,
        -317925.54273458436, -323791.50591247174, -329678.04806887166,
        -335584.73296020157, -341511.1442727386, -347456.8842379579,
        -353421.5723731138, -359404.84433322866, -365406.35086244706,
        -371425.7568342447, -377462.7403712805, -383516.99203680805,
        -389588.21409052203, -395676.11980255274, -401780.43282004213,
        -407900.8865813592, -414037.2237735652, -420189.1958292083,
        -426356.56245895254, -432539.0912169067, -438736.55709584535,
    ],
    "CO2": [
        -457942.3486287945, -463334.059335961, -468799.67821755266,
        -474335.8441330556, -479939.50834182784, -485607.89990335837,
        -491338.4927341056, -497128.97585932177, -502977.22722835816,
        -508881.29095683, -514839.3576629543, -520849.7475119176,
        -526910.8955921343, -533021.339282858, -539179.7073159148,
        -545384.710277174, -551635.1323323397, -557929.8239956509,
        -564267.6957890595, -570647.7126638524, -577068.889077073,
        -583530.2846320737, -590031.0002066555, -596570.1745039895,
        -603146.9809713099, -609760.6293990558, -616410.3750016887,
        -623095.5013637042, -629815.3152651639, -636569.1461850206,
        -643356.3457066584, -650176.2868574659, -657028.3634066387,
        -663911.9891395781, -670826.5971227366, -677771.6389693108,
        -684746.5841134915, -691750.9190989225, -698784.1468854125,
        -705845.7861767195, -712935.3707712699, -720052.4489369508,
        -727196.5828105503, -734367.3478220025, -741564.3321432711,
        -748787.1361614721, -756035.3719756573, -763308.6629165667,
    ],
    "H2": [
        -39230.063775039904, -42533.11757878159, -45891.82816360213,
        -49302.04531733515, -52760.24029859018, -56263.373574408426,
        -59808.79793415405, -63394.18584146621, -67017.47385225285,
        -70676.819337209, -74370.56626205362, -78097.21775879446,
        -81855.41387218845, -85643.91330787266, -89461.57831550547,
        -93307.3620571576, -97180.29796710763, -101079.49072300369,
        -105004.10853259345, -108953.37650338223, -112926.5709104889,
        -116923.01421471135, -120942.07071127213, -124983.14271196438,
        -129045.66718096474, -133129.11275852888, -137232.9771179574,
        -141356.7846102325, -145500.08415804288, -149662.4473668939,
        -153843.46682591317, -158042.75457502186, -162259.9407185142,
        -166494.67216790607, -170746.61149926975, -175015.4359122674,
        -179300.83627977286, -183602.5162784053, -187920.19159151352,
        -192253.58917719504, -196602.44659482752, -200966.5113843639,
        -205345.54049330708, -209739.299746859, -214147.56335724352,
        -218570.11346863612, -223006.7397345224, -227457.23892463796,
    ],
    "H2O": [
        -298675.5310122627, -303438.82114151155, -308267.0619556672,
        -313155.8729422053, -318101.5145509287, -323100.755187744,
        -328150.7729724403, -333249.0815349972, -338393.47287772223,
        -343581.97263332515, -348812.8045090256, -354084.3616593377,
        -359395.18336758116, -364743.93585241336, -370129.396320319,
        -375550.4396016303, -381006.02686424844, -386495.196014137,
        -392017.05347711226, -397570.7671208098, -403155.56012472487,
        -408770.7056439591, -414415.52214163705, -420089.36928796524,
        -425791.6443421037, -431521.77894753177, -437279.2362832408,
        -443063.5085225086, -448874.11455868103, -454710.5979636628,
        -460572.52514999313, -466459.48371165915, -472371.08092236763,
        -478306.9439733515, -484266.7214641888, -490250.074721552,
        -496256.67541431554, -502286.2051374972, -508338.35500735836,
        -514412.82526920433, -520509.32491889, -526627.5713386262,
        -532767.2899473691, -538928.2138658371, -545110.0835960221,
        -551312.6467149315, -557535.6575822007, -563778.8770611493,
    ],
}

T = np.arange(300.0, 1500.0, 25.0)  # K
R = 8.314  # J mol-1 K-1
P_TOTAL = 1.0  # atm

# ---------------------------------------------------------------------------
# System definition for equilibrium_solver
# ---------------------------------------------------------------------------

# Species order: [CO, H2O, CO2, H2]
# Element rows: [C, O, H]
S = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 2.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
])

# Equimolar CO/H2O feed without inert gas.
n0 = np.array([1.0, 1.3, 0.5, 0.0])
elements = S @ n0
phases = [0, 0, 0, 0]

# ---------------------------------------------------------------------------
# Analytical benchmark
# ---------------------------------------------------------------------------

g_co = np.array(G["CO"])
g_h2o = np.array(G["H2O"])
g_co2 = np.array(G["CO2"])
g_h2 = np.array(G["H2"])

delta_g = g_co2 + g_h2 - g_co - g_h2o  # J/mol for CO + H2O -> CO2 + H2
k_eq = np.exp(-delta_g / (R * T))


def solve_extent_from_keq(k_value, n_init):
    """Solve reaction extent from K for a + b <=> c + d."""
    a, b, c, d = n_init
    xi_min = max(-c, -d)
    xi_max = min(a, b)

    qa = k_value - 1.0
    qb = -(k_value * (a + b) + (c + d))
    qc = k_value * a * b - c * d

    if abs(qa) < 1e-14:
        # Near K=1, equation is effectively linear.
        xi = -qc / qb if abs(qb) > 1e-14 else 0.5 * (xi_min + xi_max)
        return float(np.clip(xi, xi_min, xi_max))

    roots = np.roots([qa, qb, qc])
    real_roots = roots[np.isreal(roots)].real

    in_bounds = real_roots[(real_roots >= xi_min - 1e-12) & (real_roots <= xi_max + 1e-12)]
    if len(in_bounds) == 0:
        # Fallback for tiny numerical excursions.
        xi = np.clip(real_roots[0], xi_min, xi_max)
        return float(xi)

    return float(in_bounds[0])


n_analyt = np.empty((len(T), 4))
for i, k_value in enumerate(k_eq):
    xi = solve_extent_from_keq(k_value, n0)
    n_analyt[i, :] = np.array([n0[0] - xi, n0[1] - xi, n0[2] + xi, n0[3] + xi])

y_analyt = n_analyt / n_analyt.sum(axis=1, keepdims=True)

# ---------------------------------------------------------------------------
# Numerical solution via equilibrium_solver
# ---------------------------------------------------------------------------

n_num = np.empty((len(T), 4))
balance_residual = np.empty_like(T)

for i, temp in enumerate(T):
    gibbs = np.array([g_co[i], g_h2o[i], g_co2[i], g_h2[i]])
    n_eq = np.array(
        calculate_equilibrium(
            S=S.tolist(),
            phases=phases,
            elements=elements.tolist(),
            GibbsEnergies=gibbs,
            T=float(temp),
            Pa=P_TOTAL,
        )
    )
    n_num[i, :] = n_eq
    balance_residual[i] = np.max(np.abs(S @ n_eq - elements))

y_num = n_num / n_num.sum(axis=1, keepdims=True)

# ---------------------------------------------------------------------------
# Quantitative comparison
# ---------------------------------------------------------------------------

species = ["CO", "H2O", "CO2", "H2"]
species_tex = {
    "CO": r"\mathrm{CO}",
    "H2O": r"\mathrm{H_2O}",
    "CO2": r"\mathrm{CO_2}",
    "H2": r"\mathrm{H_2}",
}
rmse_species = {
    sp: float(np.sqrt(np.mean((y_num[:, j] - y_analyt[:, j]) ** 2)))
    for j, sp in enumerate(species)
}
max_dev_species = {
    sp: float(np.max(np.abs(y_num[:, j] - y_analyt[:, j])))
    for j, sp in enumerate(species)
}

rmse_global = float(np.sqrt(np.mean((y_num - y_analyt) ** 2)))
max_dev_global = float(np.max(np.abs(y_num - y_analyt)))

sign_change_idx = np.where(np.diff(np.sign(delta_g)) != 0)[0]
if len(sign_change_idx) > 0:
    k = int(sign_change_idx[0])
    t0, t1 = T[k], T[k + 1]
    g0, g1 = delta_g[k], delta_g[k + 1]
    t_eq = t0 - g0 * (t1 - t0) / (g1 - g0)
else:
    t_eq = float("nan")

print()
print("=" * 78)
print("  Water-Gas Shift Equilibrium")
print("  Reaction: CO(g) + H2O(g) <=> CO2(g) + H2(g)")
print(f"  Pressure: {P_TOTAL:.1f} atm, Temperature grid: {T[0]:.0f}-{T[-1]:.0f} K")
print("=" * 78)
if np.isfinite(t_eq):
    print(f"  Temperature where DeltaG(T)=0 (linear estimate): {t_eq:.2f} K")
print(f"  Global RMSE(y_num vs y_analyt): {rmse_global:.5e}")
print(f"  Global max |y_num - y_analyt|:  {max_dev_global:.5e}")
for sp in species:
    print(
        f"  {sp:>3s}: RMSE = {rmse_species[sp]:.5e}, "
        f"max |dev| = {max_dev_species[sp]:.5e}"
    )
print(f"  Max element-balance residual: {balance_residual.max():.2e} mol")
print("=" * 78)
print()

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

colors = {
    "CO": "#1565C0",
    "H2O": "#2E7D32",
    "CO2": "#C62828",
    "H2": "#6A1B9A",
}

for j, sp in enumerate(species):
    ax.plot(
        T,
        y_analyt[:, j],
        "-",
        color=colors[sp],
        linewidth=2.0,
        label=rf"${species_tex[sp]}$ analytical",
    )
    ax.plot(
        T,
        y_num[:, j],
        "o",
        color=colors[sp],
        markersize=3.2,
        markerfacecolor="none",
        markeredgewidth=0.9,
        label=rf"${species_tex[sp]}$ numerical",
    )

ax.set_ylabel("Mole fraction", fontsize=11)
ax.set_ylim(-0.01, 0.6)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.legend(fontsize=8.2, ncol=2, framealpha=0.9)
ax.set_title(
    r"$\mathrm{CO} + \mathrm{H_2O} \rightleftharpoons \mathrm{CO_2} + \mathrm{H_2}$ at 1 atm: "
    r"mole fractions vs temperature",
    fontsize=12,
)



fig.tight_layout()
plt.show()
