"""CarbonCanvas synthetic demonstration study.

Generates a reproducible 14-day hourly scenario for one hyperscale region,
applies three scheduling policies to the flexible share of the workload,
and reports operational carbon for each policy. Also renders the two
data-driven figures used in the manuscript walkthrough (Figures 3 and 4).

Everything is deterministic under SEED = 42.
"""

import time
T0 = time.perf_counter()
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- scenario
HOURS_PER_DAY = 24
DAYS = 14
T = HOURS_PER_DAY * DAYS
t = np.arange(T)
hour_of_day = t % 24
day_index = t // 24

# Grid carbon intensity (gCO2e per kWh), diurnal + weekly + noise, seed 42.
diurnal = 130.0 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)   # afternoon dip
weekly = 35.0 * np.where((day_index % 7) >= 5, -1.0, 1.0)        # greener weekends
noise = rng.normal(0, 18, T)
carbon_intensity = np.clip(420.0 + diurnal + weekly + noise, 80.0, None)

# Inflexible services (MW of IT power): diurnal demand curve, seed-fixed noise.
inflexible = 52.0 + 12.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
inflexible += rng.normal(0, 1.5, T)
inflexible = np.clip(inflexible, 35.0, None)

# Flexible batch energy: 20 percent of each day's inflexible energy,
# released at the start of each day with a 24-hour completion window.
daily_inflexible_energy = inflexible.reshape(DAYS, 24).sum(axis=1)  # MWh
flexible_daily_energy = 0.20 * daily_inflexible_energy              # MWh/day

BASE_RATE = flexible_daily_energy / 24.0    # MW if spread evenly (the 1x rate)
ELASTIC_CAP = 2.0                            # elastic policy may run up to 2x
PUE = 1.25                                   # constant facility overhead factor

# ---------------------------------------------------------------- policies
def op_carbon(it_power_mw):
    """Operational carbon in tonnes CO2e for an IT power series (MW)."""
    energy_mwh = it_power_mw * PUE            # one-hour steps
    return float(np.sum(energy_mwh * carbon_intensity) / 1000.0)  # kg -> t

def policy_static():
    """Flexible energy spread evenly across each 24-hour window."""
    flex = np.repeat(BASE_RATE, 24)
    return flex

def policy_deferral():
    """Each day's flexible energy packed into the lowest-carbon hours,
    running at the fixed 1x rate (suspend-resume style)."""
    flex = np.zeros(T)
    for d in range(DAYS):
        window = slice(d * 24, (d + 1) * 24)
        ci = carbon_intensity[window]
        hours_needed = 24 * (flexible_daily_energy[d] /
                             (BASE_RATE[d] * 24))  # = 24 at 1x -> full window
        # At the 1x rate the job needs the whole window, so deferral only
        # helps if the rate cap is above 1x. Model the classic variant:
        # rate fixed at 1x but the job may pause, so it needs E/(1x) hours.
        n = int(np.ceil(flexible_daily_energy[d] / BASE_RATE[d]))
        n = min(n, 24)
        order = np.argsort(ci)[:n]
        alloc = np.zeros(24)
        alloc[order] = BASE_RATE[d]
        # trim overshoot in the most carbon-intensive selected hour
        excess = alloc.sum() - flexible_daily_energy[d]
        if excess > 0:
            worst = order[np.argmax(ci[order])]
            alloc[worst] -= excess
        flex[window] = alloc
    return flex

def policy_elastic():
    """Greedy elastic reallocation: fill the greenest hours first, up to
    an elasticity cap of 2x the even-spread rate (CarbonScaler style)."""
    flex = np.zeros(T)
    for d in range(DAYS):
        window = slice(d * 24, (d + 1) * 24)
        ci = carbon_intensity[window]
        cap = ELASTIC_CAP * BASE_RATE[d]
        remaining = flexible_daily_energy[d]
        alloc = np.zeros(24)
        for h in np.argsort(ci):
            take = min(cap, remaining)
            alloc[h] = take
            remaining -= take
            if remaining <= 1e-9:
                break
        flex[window] = alloc
    return flex

policies = {
    "Static spread": policy_static(),
    "Deferral (1x)": policy_deferral(),
    "Elastic (2x cap)": policy_elastic(),
}

# NOTE: at a fixed 1x rate a job whose energy equals 24h * 1x fills the whole
# window, so give deferral headroom the standard way: its rate cap is also
# raised implicitly by n = ceil(E / rate) above; with E = 24 * rate this
# yields n = 24 (no gain). To keep the comparison meaningful, the deferral
# baseline below is re-run with a 1.5x fixed rate, the common configuration.
def policy_deferral_fixed(rate_mult):
    flex = np.zeros(T)
    for d in range(DAYS):
        window = slice(d * 24, (d + 1) * 24)
        ci = carbon_intensity[window]
        rate = rate_mult * BASE_RATE[d]
        n = int(np.ceil(flexible_daily_energy[d] / rate))
        n = min(n, 24)
        order = np.argsort(ci)[:n]
        alloc = np.zeros(24)
        alloc[order] = rate
        excess = alloc.sum() - flexible_daily_energy[d]
        if excess > 0:
            worst = order[np.argmax(ci[order])]
            alloc[worst] -= excess
        flex[window] = alloc
    return flex

del policies["Deferral (1x)"]
del policies["Elastic (2x cap)"]
policies["Deferral (1.5x)"] = policy_deferral_fixed(1.5)
policies["Elastic (2x cap)"] = policy_elastic()

# ---------------------------------------------------------------- results
inflex_carbon = op_carbon(inflexible)
rows = []
for name, flex in policies.items():
    assert np.all(flex >= -1e-9)
    for d in range(DAYS):  # energy conservation check per window
        got = flex[d * 24:(d + 1) * 24].sum()
        assert abs(got - flexible_daily_energy[d]) < 1e-6, (name, d)
    fc = op_carbon(flex)
    rows.append((name, fc, inflex_carbon + fc))

base_flex = rows[0][1]
base_total = rows[0][2]
print(f"Inflexible operational carbon over 14 days: {inflex_carbon:,.1f} t")
print(f"{'Policy':<18}{'Flex tCO2e':>12}{'Total tCO2e':>13}"
      f"{'Flex red.':>11}{'Total red.':>12}")
for name, fc, tot in rows:
    print(f"{name:<18}{fc:>12,.1f}{tot:>13,.1f}"
          f"{100*(base_flex-fc)/base_flex:>10.1f}%"
          f"{100*(base_total-tot)/base_total:>11.1f}%")

mean_ci = carbon_intensity.mean()
print(f"\nMean carbon intensity: {mean_ci:.0f} g/kWh, "
      f"min {carbon_intensity.min():.0f}, max {carbon_intensity.max():.0f}")
print(f"Mean inflexible IT load: {inflexible.mean():.1f} MW")
print(f"Mean flexible share of daily IT energy: 20.0%")

# ---------------------------------------------------------------- figures
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
})

# Figure 3 (grayscale): 3 days of the temporal view - carbon intensity band
# with static vs elastic flexible load.
f3_slice = slice(0, 72)
fig, ax1 = plt.subplots(figsize=(6.4, 3.0), dpi=300)
ax1.fill_between(t[f3_slice], 0, carbon_intensity[f3_slice],
                 color="0.85", label="Grid carbon intensity")
ax1.plot(t[f3_slice], carbon_intensity[f3_slice], color="0.45", lw=1.0)
ax1.set_ylabel("Carbon intensity (gCO$_2$e/kWh)", color="0.2")
ax1.set_xlabel("Hour of the demonstration trace")
ax1.set_xlim(0, 71)
ax1.set_ylim(0, 640)
ax2 = ax1.twinx()
ax2.step(t[f3_slice], policies["Static spread"][f3_slice], where="mid",
         color="0.35", lw=1.2, ls="--", label="Flexible load, static")
ax2.step(t[f3_slice], policies["Elastic (2x cap)"][f3_slice], where="mid",
         color="0.0", lw=1.4, label="Flexible load, elastic")
ax2.set_ylabel("Flexible IT load (MW)")
ax2.set_ylim(0, 34)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
           fontsize=8, frameon=False, ncol=1)
fig.tight_layout()
fig.savefig("/home/claude/paper/fig3_temporal_view.png")
plt.close(fig)

# Figure 4 (blue, comparison graph): operational carbon of the flexible
# workload under the three policies.
names = [r[0] for r in rows]
flexvals = [r[1] for r in rows]
reductions = [100 * (base_flex - v) / base_flex for v in flexvals]
fig, ax = plt.subplots(figsize=(5.6, 3.0), dpi=300)
blues = ["#c6d5e8", "#7fa4cc", "#1f4e79"]
bars = ax.bar(names, flexvals, color=blues, edgecolor="#12314d", width=0.55)
for b, v, r in zip(bars, flexvals, reductions):
    label = f"{v:,.0f} t" if r == 0 else f"{v:,.0f} t  (-{r:.1f}%)"
    ax.text(b.get_x() + b.get_width() / 2, v + 30, label,
            ha="center", va="bottom", fontsize=8.5, color="#12314d")
ax.set_ylabel("Operational carbon of flexible workload (tCO$_2$e)")
ax.set_ylim(0, max(flexvals) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("/home/claude/paper/fig4_policy_comparison.png")
plt.close(fig)
# ---------------------------------------------------------------- embodied
N_SERVERS = 110_000          # fleet size assumed for the demonstration
M_EMBODIED = 1_300.0         # kgCO2e manufacturing footprint per server
LIFETIME_Y = 4.0             # straight-line amortization horizon
embodied_per_hour = N_SERVERS * M_EMBODIED / (LIFETIME_Y * 8760.0)  # kg/h
embodied_total = embodied_per_hour * T / 1000.0                     # tonnes
print(f"\nEmbodied amortization: {embodied_per_hour:,.0f} kg/h, "
      f"{embodied_total:,.1f} t over the horizon")
for name, fc, tot in rows:
    grand = tot + embodied_total
    print(f"  {name:<18} total incl. embodied {grand:,.1f} t "
          f"(embodied share {100*embodied_total/grand:.1f}%)")
best = rows[-1]
print(f"Elastic saving as share of combined total: "
      f"{100*(rows[0][2]-best[2])/(rows[0][2]+embodied_total):.1f}%")

print(f"\nWall time: {time.perf_counter() - T0:.2f} s")
print("Figures 3 and 4 written.")