"""Calcium Carbonate Decomposition: Numerical vs. Analytical Equilibrium
=========================================================================

Reaction:
   CaCO3(s)  <=>  CaO(s) + CO2(g)

This example demonstrates ``equilibrium_solver.calculate_equilibrium`` for a
heterogeneous equilibrium as a function of temperature (300-1475 K, 1 atm).

Correctness is established by comparison with the analytical criterion:
for pure solids, the decomposition equilibrium satisfies

   Kp(T) = exp(-DeltaG(T)/(R T)) = p_CO2,eq / p_ref.

In this formulation, the key observable is the equilibrium CO2 partial
pressure p_CO2,eq(T). For the ideal heterogeneous equilibrium,

   p_CO2,eq(T) = Kp(T)  (in atm, with p_ref = 1 atm),

until material/pressure limits are reached.
"""

import matplotlib.pyplot as plt
import numpy as np

from equilibrium_solver import calculate_equilibrium

# ---------------------------------------------------------------------------
# Thermodynamic data (J/mol), tabulated at T = 300..1475 K with step 25 K
# ---------------------------------------------------------------------------

G = {
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
   "CaCO3": [
      -1234939.2834208286, -1237333.0284817033, -1239896.1287110974,
      -1242622.7365377485, -1245506.869941677, -1248542.6465152593,
      -1251724.405965238, -1255046.7694529237, -1258504.6629278082,
      -1262093.3200264273, -1265808.273611315, -1269645.3412939725,
      -1273600.6081013312, -1277670.4081463509, -1281851.3063821138,
      -1286140.0810454134, -1290533.7071082157, -1295029.3408812075,
      -1299624.3058093379, -1304316.0794382251, -1309102.2814961288,
      -1313980.6630186192, -1318949.0964357457, -1324005.5665402943,
      -1329148.162258156, -1334375.0691462217, -1339684.562548624,
      -1345075.0013478997, -1350544.8222533972, -1356092.534574757,
      -1361716.7154334753, -1367416.0053702998, -1373189.104310551,
      -1379034.7678533848, -1384951.8038545295, -1390939.0692751925,
      -1396995.4672726563, -1403119.944510597, -1409311.4886694078,
      -1415569.1261388033, -1421891.9198767678, -1428278.967420491,
      -1434729.399036354, -1441242.3759972875, -1447817.088976946,
      -1454452.756551158, -1461148.6237979939, -1467903.9609886233,
   ],
   "CaO": [
      -646782.696426103, -647785.7856887843, -648873.368887047,
      -650041.7879642382, -651287.3940520716, -652606.657621112,
      -653996.2224563104, -655452.9283868563, -656973.8166353699,
      -658556.1256509826, -660197.2819468644, -661894.8885571053,
      -663646.7126223404, -665450.6729635205, -667304.8281176828,
      -669207.3650800821, -671156.5888608113, -673150.9128841227,
      -675188.850212934, -677269.0055560869, -679390.0680036441,
      -681550.8044308076, -683750.0535108133, -685986.7202794653,
      -688259.7711976147, -690568.2296621519, -692911.1719205071,
      -695287.723347991, -697697.0550514188, -700138.3807662637,
      -702610.9540180493, -705114.0655218392, -707647.040796481,
      -710209.2379727842, -712800.0457770487, -715418.881673338,
      -718065.1901496722, -720738.4411348609, -723438.128534097,
      -726163.7688726552, -728914.9000381303, -731691.0801126183,
      -734491.8862871084, -737316.9138511098, -740165.7752512279,
      -743038.0966978637, -745933.5095870416, -748851.6587208835,
   ],
   "N2": [0]*48, # inert reference species with zero Gibbs energy across all temperatures
}

T = np.arange(300.0, 1500.0, 25.0)  # K
R = 8.314  # J mol-1 K-1
P_TOTAL = 1.0  # atm

# ---------------------------------------------------------------------------
# System definition for equilibrium_solver
# ---------------------------------------------------------------------------

# Species order: [CaCO3, CaO, CO2, N2]
# Element rows: [Ca, C, O, N]
S = np.array([
   [1.0, 1.0, 0.0, 0.0],
   [1.0, 0.0, 1.0, 0.0],
   [3.0, 1.0, 2.0, 0.0],
   [0.0, 0.0, 0.0, 2.0],
])

# Initial state includes both solids so the equilibrium manifold is sampled
# across temperature; N2 is inert gas in the same gas phase as CO2.
n0 = np.array([100.0, 100.0, 0.0, 1.0])
elements = S @ n0

# Each condensed species is a separate pure phase; CO2 and N2 share one gas phase.
phases = [0, 1, 2, 2]

# ---------------------------------------------------------------------------
# Analytical benchmark
# ---------------------------------------------------------------------------

g_caco3 = np.array(G["CaCO3"])
g_cao = np.array(G["CaO"])
g_co2 = np.array(G["CO2"])
g_n2 = np.array(G["N2"])  # inert reference species (not involved in reaction)

delta_g = g_cao + g_co2 - g_caco3  # J/mol for CaCO3 -> CaO + CO2
k_p = np.exp(-delta_g / (R * T))

# Analytical p_CO2(T): ideal law p_CO2 = Kp, then clipped by total pressure
# and finite-inventory upper bound from max possible CO2 production.
p_co2_inventory_max = P_TOTAL * (n0[0] / (n0[0] + n0[3]))
p_co2_analyt = np.minimum(k_p, min(P_TOTAL, p_co2_inventory_max))

# ---------------------------------------------------------------------------
# Numerical solution via equilibrium_solver
# ---------------------------------------------------------------------------

p_co2_num = np.empty_like(T)
balance_residual = np.empty_like(T)

for i, temp in enumerate(T):
   gibbs = np.array([g_caco3[i], g_cao[i], g_co2[i], g_n2[i]])
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

   gas_total = n_eq[2] + n_eq[3]
   p_co2_num[i] = float(P_TOTAL * n_eq[2] / gas_total) if gas_total > 0.0 else 0.0
   balance_residual[i] = np.max(np.abs(S @ n_eq - elements))

# ---------------------------------------------------------------------------
# Quantitative comparison
# ---------------------------------------------------------------------------

abs_dev = np.abs(p_co2_num - p_co2_analyt)
rmse_pco2 = float(np.sqrt(np.mean((p_co2_num - p_co2_analyt) ** 2)))
max_dev_pco2 = float(np.max(abs_dev))

# Decomposition temperature estimate from DeltaG(T) = 0 (linear interpolation)
sign_change_idx = np.where(np.diff(np.sign(delta_g)) != 0)[0]
if len(sign_change_idx) > 0:
   k = int(sign_change_idx[0])
   t0, t1 = T[k], T[k + 1]
   g0, g1 = delta_g[k], delta_g[k + 1]
   t_eq = t0 - g0 * (t1 - t0) / (g1 - g0)
else:
   t_eq = float("nan")

print()
print("=" * 74)
print("  Calcium Carbonate Decomposition Equilibrium")
print("  Reaction: CaCO3(s) <=> CaO(s) + CO2(g)")
print(f"  Pressure: {P_TOTAL:.1f} atm, Temperature grid: {T[0]:.0f}-{T[-1]:.0f} K")
print("=" * 74)
if np.isfinite(t_eq):
   print(f"  Estimated decomposition temperature from DeltaG(T)=0: {t_eq:.2f} K")
print(f"  RMSE(p_CO2,num vs p_CO2,analyt): {rmse_pco2:.5e} atm")
print(f"  Max |p_CO2,num - p_CO2,analyt|:  {max_dev_pco2:.5e} atm")
print(f"  p_CO2 upper bound from finite inventory: {p_co2_inventory_max:.5f} atm")
print(f"  Max element-balance residual: {balance_residual.max():.2e} mol")
print("=" * 74)
print()

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)

ax1.plot(T, p_co2_analyt, "-", color="#1565C0", linewidth=2.0, label="Analytical $p_{CO_2}$")
ax1.plot(
   T,
   p_co2_num,
   "o",
   color="#C62828",
   markersize=4,
   markerfacecolor="none",
   markeredgewidth=1.0,
   label="Numerical $p_{CO_2}$ (equilibrium_solver)",
)
ax1.set_ylabel("$p_{CO_2}$, atm", fontsize=11)
ax1.set_ylim(-0.05, 1.05 * max(1.0, p_co2_analyt.max()))
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax1.legend(fontsize=9, framealpha=0.9)
ax1.set_title(
   "$CaCO_3$ decomposition at 1 atm: $p_{CO_2}(T)$ analytical vs numerical",
   fontsize=12,
)

ax2.plot(T, delta_g / 1000.0, color="#2E7D32", linewidth=1.8)
ax2.axhline(0.0, color="#616161", linestyle=":", linewidth=1.0)
if np.isfinite(t_eq):
   ax2.axvline(t_eq, color="#616161", linestyle=":", linewidth=1.0)
ax2.set_xlabel("Temperature, K", fontsize=11)
ax2.set_ylabel("DeltaG (kJ/mol)", fontsize=11)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

annotation = (
   f"RMSE($p_{{CO_2}}$) = {rmse_pco2:.3e} atm\n"
   f"max |dev| = {max_dev_pco2:.3e} atm\n"
   f"max balance res. = {balance_residual.max():.1e} mol"
)
ax1.text(
   0.98,
   0.05,
   annotation,
   transform=ax1.transAxes,
   ha="right",
   va="bottom",
   fontsize=8.5,
   family="monospace",
   bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BDBDBD", alpha=0.9),
)

fig.tight_layout()
plt.show()