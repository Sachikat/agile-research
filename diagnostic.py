"""
Quick diagnostic script — run this BEFORE retraining.
Identifies why R² is low without waiting for full training.

Checks:
  1. Raw feature-tz correlations (are muscles even correlated with yaw?)
  2. Effect of different tz normalisations on those correlations
  3. Variance of tz per individual and per species (z-score sanity check)
  4. Whether tz variance differs meaningfully across species
  5. A simple linear regression baseline R² (upper bound for a linear model)
  6. A quick MLP baseline R² with no autoencoder constraints
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Load raw data ─────────────────────────────────────────────────────────────

df = pd.read_csv("./agile-research/all10_big_wb.csv")

PHASE_COLS = ["lax", "lba", "lsa", "ldvm", "ldlm",
              "rdlm", "rdvm", "rsa", "rax", "rba"]
COUNT_COLS = ["lax_count", "lba_count", "lsa_count", "ldvm_count", "ldlm_count",
              "rdlm_count", "rdvm_count", "rsa_count", "rax_count", "rba_count"]
FEATURE_COLS = PHASE_COLS + COUNT_COLS
TARGET_COL   = "tz"

for col in FEATURE_COLS + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, "species", "moth", "wblen"]).copy()

print(f"Total rows after dropping NaN: {len(df)}")
print(f"Species: {df['species'].nunique()}")
print(f"Individuals (moths): {df['moth'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: Raw tz distribution
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CHECK 1: Raw tz distribution")
print("="*60)
print(df[TARGET_COL].describe().round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df[TARGET_COL], bins=60, edgecolor="none")
axes[0].set_title("Raw tz distribution (all wingbeats)")
axes[0].set_xlabel("tz")

# Per-species tz distributions
species_list = sorted(df["species"].unique())
for sp in species_list:
    axes[1].hist(df.loc[df["species"]==sp, TARGET_COL],
                 bins=30, alpha=0.4, label=sp, density=True)
axes[1].set_title("tz distribution per species (density)")
axes[1].set_xlabel("tz")
axes[1].legend(fontsize=6, bbox_to_anchor=(1.02,1))
plt.tight_layout()
plt.savefig("diag_tz_distribution.png", dpi=150)
plt.show()
print("  Saved diag_tz_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: tz variance per individual vs per species
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CHECK 2: tz variance per individual and per species")
print("="*60)

ind_stats = df.groupby("moth")[TARGET_COL].agg(["mean","std","count"])
sp_stats  = df.groupby("species")[TARGET_COL].agg(["mean","std","count"])

print("\nPer-individual tz std (first 20 individuals):")
print(ind_stats["std"].sort_values().head(20).round(4).to_string())

print(f"\nIndividuals with std < 0.01 (nearly constant tz): "
      f"{(ind_stats['std'] < 0.01).sum()}")
print(f"Individuals with std < 0.1:  {(ind_stats['std'] < 0.1).sum()}")

print("\nPer-species tz stats:")
print(sp_stats.round(4).to_string())

# After per-individual z-score, what does tz look like?
df["tz_ind_zscore"] = df.groupby("moth")[TARGET_COL].transform(
    lambda x: (x - x.mean()) / (x.std() if x.std() > 1e-6 else 1.0)
)
# After per-species z-score
df["tz_sp_zscore"] = df.groupby("species")[TARGET_COL].transform(
    lambda x: (x - x.mean()) / (x.std() if x.std() > 1e-6 else 1.0)
)
# Global z-score
scaler_g = StandardScaler()
df["tz_global_zscore"] = scaler_g.fit_transform(df[[TARGET_COL]]).ravel()

print("\nCorrelation between different tz normalisations and raw tz:")
for col in ["tz_ind_zscore","tz_sp_zscore","tz_global_zscore"]:
    r = df[TARGET_COL].corr(df[col])
    print(f"  raw tz ↔ {col}: r={r:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: Feature–tz correlations under each normalisation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CHECK 3: Feature–tz Pearson correlations under each normalisation")
print("="*60)

tz_variants = {
    "raw":            TARGET_COL,
    "ind_zscore":     "tz_ind_zscore",
    "species_zscore": "tz_sp_zscore",
    "global_zscore":  "tz_global_zscore",
}

corr_results = {}
for label, tz_col in tz_variants.items():
    corrs = df[FEATURE_COLS].corrwith(df[tz_col]).sort_values(key=abs, ascending=False)
    corr_results[label] = corrs
    print(f"\n  tz normalisation: {label}")
    print(corrs.round(3).to_string())

# Plot correlations side by side
fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
for ax, (label, corrs) in zip(axes, corr_results.items()):
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in corrs.values]
    ax.barh(corrs.index[::-1], corrs.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(f"tz = {label}")
    ax.set_xlabel("Pearson r with tz")
plt.suptitle("Feature–tz correlations under different tz normalisations", y=1.02)
plt.tight_layout()
plt.savefig("diag_feature_tz_correlations.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Saved diag_feature_tz_correlations.png")

# Key question: which normalisation gives the strongest correlations?
print("\n  Max |correlation| per normalisation:")
for label, corrs in corr_results.items():
    print(f"    {label:20s}: max |r| = {corrs.abs().max():.4f}  "
          f"(feature: {corrs.abs().idxmax()})")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: Simple linear baseline R² per normalisation
# Uses Ridge regression directly on raw features — no autoencoder bottleneck.
# This is the upper bound for a linear model.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CHECK 4: Ridge regression baseline R² (5-fold CV, no bottleneck)")
print("="*60)
print("  This is the ceiling for any linear model on these features.")
print("  If this is also ~0.08, the problem is in the data, not the model.\n")

# Normalise features the same way as the main model
df_feat = df[FEATURE_COLS].copy()
df_feat[COUNT_COLS]  = (df_feat[COUNT_COLS] / 10.0).clip(0, 1)
df_feat[PHASE_COLS]  = ((df_feat[PHASE_COLS] + 1.0) / 2.0).clip(0, 1)
X_all = df_feat.values

ridge = Ridge(alpha=1.0)
for label, tz_col in tz_variants.items():
    y_all = df[tz_col].values
    scores = cross_val_score(ridge, X_all, y_all, cv=5, scoring="r2")
    print(f"  Ridge R² ({label:20s}): "
          f"mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"per-fold={np.round(scores, 3)}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5: Quick MLP baseline — no autoencoder, no bottleneck
# If this is also low, the features genuinely don't predict tz well.
# If this is high, the bottleneck or normalisation is the problem.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CHECK 5: Quick MLP baseline R² (no bottleneck, no autoencoder)")
print("="*60)
print("  If MLP R² >> your autoencoder R², the latent dim=2 bottleneck")
print("  or the tz normalisation is the main problem.\n")

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                   random_state=0, early_stopping=True)
for label, tz_col in tz_variants.items():
    y_all  = df[tz_col].values
    scores = cross_val_score(mlp, X_all, y_all, cv=5, scoring="r2")
    print(f"  MLP R² ({label:20s}): "
          f"mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"per-fold={np.round(scores, 3)}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6: How many wingbeats actually have all 10 muscles?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CHECK 6: Data completeness per species")
print("="*60)

completeness = df.groupby("species")[FEATURE_COLS].apply(
    lambda g: g.notna().all(axis=1).sum()
).rename("complete_wingbeats")
total = df.groupby("species").size().rename("total_wingbeats")
summary = pd.concat([total, completeness], axis=1)
summary["pct_complete"] = (summary["complete_wingbeats"] / summary["total_wingbeats"] * 100).round(1)
print(summary.sort_values("complete_wingbeats", ascending=False).to_string())

print("\n" + "="*60)
print("INTERPRETATION GUIDE")
print("="*60)
print("""
  Ridge R² ~0.08  AND  MLP R² ~0.08:
    → Features genuinely don't predict tz well with this normalisation.
      Try a different tz normalisation (species zscore or global zscore).

  Ridge R² ~0.08  BUT  MLP R² >> 0.08:
    → Nonlinear interactions exist but linear models miss them.
      The interaction layer should help. Check tz normalisation too.

  MLP R² >> 0.08  AND  autoencoder R² ~0.08:
    → The latent dim=2 bottleneck is the problem.
      Try LATENT_DIM = 5 or 10.

  Max |feature-tz correlation| < 0.10 for ALL normalisations:
    → The signal in the data is very weak. Consider whether tz
      is the right target, or whether averaging left/right muscles
      first would reduce noise.

  Per-individual tz std is very small for many individuals:
    → Per-individual z-scoring is amplifying noise into signal.
      Switch to species-level or global z-score.
""")