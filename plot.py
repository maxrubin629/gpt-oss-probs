# Load mass.csv, compute summaries, and generate 4 plots (no seaborn; one chart per figure; no specific colors).
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from caas_jupyter_tools import display_dataframe_to_user

csv_path = Path("/mnt/data/mass.csv")
df = pd.read_csv(csv_path)

# Identify mass columns like "mass_k100", "mass_k128", "mass_k200"
mass_cols = [c for c in df.columns if c.startswith("mass_k")]
# Sort by the numeric k value
def k_of(col):
    try:
        return int(col.split("mass_k",1)[1])
    except Exception:
        return 10**9
mass_cols = sorted(mass_cols, key=k_of)

if "step" not in df.columns:
    # If step isn't present, synthesize it
    df["step"] = np.arange(len(df))

# Convert 0..1 masses to percentages for visualization
df_pct = df.copy()
for c in mass_cols:
    df_pct[c] = df_pct[c] * 100.0

# --- Summary stats (mean/median/min/p05/p95 and thresholds) ---
def summarize(series_pct: pd.Series):
    s = series_pct.dropna().values
    s_sorted = np.sort(s)
    n = len(s_sorted)
    if n == 0:
        return {}
    def q(p):
        # percentile by nearest-rank
        idx = max(0, min(n-1, int(round(p*(n-1)))))
        return float(s_sorted[idx])
    return {
        "count": int(n),
        "mean": float(np.mean(s_sorted)),
        "median": float(np.median(s_sorted)),
        "min": float(s_sorted[0]),
        "p05": q(0.05),
        "p95": q(0.95),
        "ge_90": float((s_sorted >= 90.0).mean()*100.0),
        "ge_95": float((s_sorted >= 95.0).mean()*100.0),
        "ge_99": float((s_sorted >= 99.0).mean()*100.0),
    }

summary_rows = []
for c in mass_cols:
    K = k_of(c)
    S = summarize(df_pct[c])
    summary_rows.append({
        "k": K,
        "steps": S.get("count", 0),
        "mean_%": S.get("mean", float("nan")),
        "median_%": S.get("median", float("nan")),
        "min_%": S.get("min", float("nan")),
        "p05_%": S.get("p05", float("nan")),
        "p95_%": S.get("p95", float("nan")),
        "pct_steps_ge_90%": S.get("ge_90", float("nan")),
        "pct_steps_ge_95%": S.get("ge_95", float("nan")),
        "pct_steps_ge_99%": S.get("ge_99", float("nan")),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("k").reset_index(drop=True)
summary_csv = Path("/mnt/data/mass_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# Show the raw CSV in a table for quick inspection
display_dataframe_to_user("Top‑k mass (per step)", df_pct)

# 1) Line plot: per-step mass vs step index, one line per k
fig1 = plt.figure()
for c in mass_cols:
    plt.plot(df_pct["step"], df_pct[c], label=c.replace("mass_k", "k="))
plt.axhline(90, linestyle="--")
plt.axhline(95, linestyle="--")
plt.axhline(99, linestyle="--")
plt.xlabel("Step index")
plt.ylabel("Top‑k mass (%)")
plt.title("Per‑step Top‑k Mass Across the Reasoning Trace")
plt.legend()
plt.tight_layout()
line_path = Path("/mnt/data/mass_line.png")
plt.savefig(line_path, dpi=150)
plt.close(fig1)

# 2) Histogram: distribution of mass for each k
fig2 = plt.figure()
bins = np.linspace(max(80.0, df_pct[mass_cols].min().min()), 100.0, 25)
for c in mass_cols:
    plt.hist(df_pct[c], bins=bins, alpha=0.5, label=c.replace("mass_k", "k="))
plt.axvline(90, linestyle="--")
plt.axvline(95, linestyle="--")
plt.axvline(99, linestyle="--")
plt.xlabel("Top‑k mass (%)")
plt.ylabel("Count of steps")
plt.title("Distribution of Top‑k Mass")
plt.legend()
plt.tight_layout()
hist_path = Path("/mnt/data/mass_hist.png")
plt.savefig(hist_path, dpi=150)
plt.close(fig2)

# 3) CDF (fraction of steps with mass ≥ threshold)
fig3 = plt.figure()
# Thresholds from min observed to 100
t_min = float(df_pct[mass_cols].min().min())
thresholds = np.linspace(max(80.0, t_min), 100.0, 400)
for c in mass_cols:
    vals = df_pct[c].dropna().values
    frac_ge = [(vals >= thr).mean() for thr in thresholds]
    plt.plot(thresholds, frac_ge, label=c.replace("mass_k", "k="))
plt.xlabel("Mass threshold (%)")
plt.ylabel("Fraction of steps ≥ threshold")
plt.title("Coverage CDF of Top‑k Mass")
plt.legend()
plt.tight_layout()
cdf_path = Path("/mnt/data/mass_cdf.png")
plt.savefig(cdf_path, dpi=150)
plt.close(fig3)

# 4) Outlier zoom (bottom 5% steps by k=100 mass) — compare all k on those steps
fig4 = plt.figure()
k_base = "mass_k100" if "mass_k100" in mass_cols else mass_cols[0]
n = len(df_pct)
m = max(1, int(math.ceil(0.05 * n)))
worst_idx = df_pct.nsmallest(m, k_base)["step"].astype(int).tolist()
df_worst = df_pct[df_pct["step"].isin(worst_idx)].sort_values("step")
for c in mass_cols:
    plt.plot(df_worst["step"], df_worst[c], marker="o", label=c.replace("mass_k", "k="))
plt.axhline(90, linestyle="--")
plt.axhline(95, linestyle="--")
plt.axhline(99, linestyle="--")
plt.xlabel("Step index (worst 5% by k=100 mass)")
plt.ylabel("Top‑k mass (%)")
plt.title("Low‑Mass Region (Bottom 5% of Steps)")
plt.legend()
plt.tight_layout()
outliers_path = Path("/mnt/data/mass_outliers.png")
plt.savefig(outliers_path, dpi=150)
plt.close(fig4)

print("Saved figures:")
print(str(line_path))
print(str(hist_path))
print(str(cdf_path))
print(str(outliers_path))
print("Summary CSV:", str(summary_csv))

summary_df
