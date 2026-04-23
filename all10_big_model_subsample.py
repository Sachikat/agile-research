import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("all10_big.csv")

clade_dict = {
    "Actias luna": "silkmoth",
    "Hyalophora cecropia": "silkmoth",
    "Automeris io": "silkmoth",
    "Saturnia walterorum": "silkmoth",
    "Syssphinx montana": "silkmoth",
    "Syssphinx hubbardi": "silkmoth",
    "Antheraea polyphemus": "silkmoth",
    "Citheronia regalis": "silkmoth",
    "Ceratomia amyntor": "hawkmoth",
    "Acherontia atropos": "hawkmoth",
    "Manduca sexta": "hawkmoth",
    "Proserpinus terlooii": "hawkmoth",
    "Hyles lineata": "hawkmoth",
    "Citheronia splendens": "silkmoth",
    "Hyalophora columbia": "silkmoth",
    "Automeris randa": "silkmoth",
    "Coloradia doris": "silkmoth",
    "Hemaris diffinis": "hawkmoth"
}
df["clade"] = df["species"].map(clade_dict)

phase_cols = ["lax", "lba", "lsa", "ldvm", "ldlm", "rdlm", "rdvm", "rsa", "rax", "rba"]
count_cols = [
    "lax_count", "lba_count", "lsa_count", "ldvm_count", "ldlm_count",
    "rdlm_count", "rdvm_count", "rsa_count", "rax_count", "rba_count"
]

feature_cols = phase_cols + count_cols
target_col = "tz"

required_cols = feature_cols + [target_col, "species", "moth", "wb", "wblen"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df_model = df.copy()
df_model[phase_cols] = df_model[phase_cols].apply(pd.to_numeric, errors="coerce")
df_model[count_cols] = df_model[count_cols].apply(pd.to_numeric, errors="coerce")
df_model[target_col] = pd.to_numeric(df_model[target_col], errors="coerce")

df_model = df_model.dropna(subset=[target_col, "species", "moth"]).copy()

min_wb_to_qualify = 15
clean_subsample_n = 10

rng_seed  = 42
wb_counts = df_model.groupby("species").size()
species_to_keep = wb_counts[wb_counts >= min_wb_to_qualify].index.tolist()
df_model  = df_model[df_model["species"].isin(species_to_keep)].copy()

df_model = (
    df_model.groupby("species", group_keys=False)
    .sample(n=clean_subsample_n, random_state=rng_seed)
    .reset_index(drop=True)
)
print(df_model["species"].value_counts().sort_index())
assert "species" in df_model.columns

ind_tz_stats = (
    df_model.groupby("moth")[target_col]
    .agg(["mean", "std"])
    .rename(columns={"mean": "ind_tz_mean", "std": "ind_tz_std"})
)
ind_tz_stats["ind_tz_std"] = ind_tz_stats["ind_tz_std"].fillna(1.0).replace(0.0, 1.0)
df_model = df_model.join(ind_tz_stats, on="moth")
df_model[target_col] = (df_model[target_col] - df_model["ind_tz_mean"]) / df_model["ind_tz_std"]
df_model = df_model.drop(columns=["ind_tz_mean", "ind_tz_std"])

print("\nPer-individual z-scored tz (should be ~mean=0, std=1 per moth):")
print(df_model.groupby("moth")[target_col].agg(["mean", "std"]).describe().round(3))

df_model[count_cols] = (df_model[count_cols] / 10.0).clip(0.0, 1.0)
df_model[phase_cols] = ((df_model[phase_cols] + 1.0) / 2.0).clip(0.0, 1.0)

individual_names = sorted(df_model["moth"].astype(str).unique())
individual_to_idx = {m: i for i, m in enumerate(individual_names)}
df_model["individual_idx"] = df_model["moth"].astype(str).map(individual_to_idx)
num_individuals = len(individual_names)
print(f"\nNumber of individuals (encoders): {num_individuals}")

X_df = df_model[feature_cols].copy()
y = df_model[target_col].values
individual_idx = df_model["individual_idx"].values
species_labels = df_model["species"].values
clade_labels = df_model["clade"].values
wbfreq_values = 1.0 / df_model["wblen"].values

(X_train_df, X_test_df,
 y_train,    y_test,
 ind_train,  ind_test,
 sp_train,   sp_test,
 cl_train,   cl_test,
 wf_train,   wf_test) = train_test_split(
    X_df, y, individual_idx,
    species_labels, clade_labels, wbfreq_values,
    test_size=0.2, random_state=42, stratify=species_labels
)

X_train = X_train_df.values.astype(np.float32)
X_test  = X_test_df.values.astype(np.float32)

y_scaler = StandardScaler()
y_train  = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test   = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

class MotorDataset(Dataset):
    def __init__(self, X, y, individual_idx,
                 species_labels=None, clade_labels=None, wbfreq=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.individual_idx = torch.tensor(individual_idx, dtype=torch.long)
        self.species_labels = np.array(species_labels) if species_labels is not None else None
        self.clade_labels = np.array(clade_labels) if clade_labels is not None else None
        self.wbfreq = np.array(wbfreq) if wbfreq is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.individual_idx[idx]

train_ds = MotorDataset(X_train, y_train, ind_train,
                        species_labels=sp_train, clade_labels=cl_train, wbfreq=wf_train)
test_ds = MotorDataset(X_test,  y_test,  ind_test,
                        species_labels=sp_test,  clade_labels=cl_test,  wbfreq=wf_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds,  batch_size=32, shuffle=False)

class MultiEncoderYawModel(nn.Module):
    def __init__(self, input_dim, latent_dim, num_individuals):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoders = nn.ModuleList([
            nn.Linear(input_dim, latent_dim) for _ in range(num_individuals)
        ])
        self.decoder_y = nn.Linear(latent_dim, 1)

    def forward(self, x, individual_idx):
        z = torch.zeros(x.shape[0], self.latent_dim, device=x.device)
        for ind in torch.unique(individual_idx):
            mask = (individual_idx == ind)
            z[mask] = self.encoders[ind.item()](x[mask])
        return self.decoder_y(z), z

def train_model(model, train_loader, test_loader,
                epochs=200, lr=1e-3, weight_decay=1e-2, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"train_yaw": [], "test_yaw": []}

    for epoch in range(epochs):
        model.train()
        total_train = 0.0
        for xb, yb, ib in train_loader:
            xb, yb, ib = xb.to(device), yb.to(device), ib.to(device)
            optimizer.zero_grad()
            y_hat, _ = model(xb, ib)
            loss = loss_fn(y_hat, yb)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * xb.size(0)

        avg_train = total_train / len(train_loader.dataset)
        history["train_yaw"].append(avg_train)

        model.eval()
        total_test = 0.0
        with torch.no_grad():
            for xb, yb, ib in test_loader:
                xb, yb, ib = xb.to(device), yb.to(device), ib.to(device)
                y_hat, _   = model(xb, ib)
                total_test += loss_fn(y_hat, yb).item() * xb.size(0)

        avg_test = total_test / len(test_loader.dataset)
        history["test_yaw"].append(avg_test)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Train {avg_train:.4f} | Test {avg_test:.4f}")

    return history

def evaluate_model(model, loader, y_scaler,
                   species_labels, clade_labels, wbfreq_values, device="cpu"):
    model.eval()
    y_true_all, y_pred_all, z_all, ind_all = [], [], [], []

    with torch.no_grad():
        for xb, yb, ib in loader:
            xb, yb, ib = xb.to(device), yb.to(device), ib.to(device)
            y_hat, z   = model(xb, ib)
            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(y_hat.cpu().numpy())
            z_all.append(z.cpu().numpy())
            ind_all.append(ib.cpu().numpy())

    y_true = np.vstack(y_true_all).ravel()
    y_pred = np.vstack(y_pred_all).ravel()
    Z  = np.vstack(z_all)
    ind_out = np.concatenate(ind_all)

    y_true_u = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_u = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    return {
        "y_true":         y_true_u,
        "y_pred":         y_pred_u,
        "Z":              Z,
        "individual_idx": ind_out,
        "species_labels": np.array(species_labels),
        "clade_labels":   np.array(clade_labels),
        "wbfreq":         np.array(wbfreq_values),
        "yaw_mse":        mean_squared_error(y_true_u, y_pred_u),
        "yaw_r2":         r2_score(y_true_u, y_pred_u),
    }

def get_effective_yaw_weights(model, feature_cols, individual_names):
    yaw_w = model.decoder_y.weight.detach().cpu().numpy().reshape(-1)
    out = {}
    for i, ind in enumerate(individual_names):
        E = model.encoders[i].weight.detach().cpu().numpy()
        eff = yaw_w @ E
        out[ind] = pd.Series(eff, index=feature_cols).sort_values(key=np.abs, ascending=False)
    return out

def aggregate_by_muscle(weight_series):
    muscle_scores = {}
    for feat, val in weight_series.items():
        muscle = feat.replace("_count", "")
        muscle_scores[muscle] = muscle_scores.get(muscle, 0.0) + abs(val)
    return pd.Series(muscle_scores).sort_values(ascending=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 2
print(f"\nUsing device: {device} | Latent dim: {latent_dim} | Individuals: {num_individuals}")

model = MultiEncoderYawModel(
    input_dim=X_train.shape[1],
    latent_dim=latent_dim,
    num_individuals=num_individuals
)

history = train_model(model, train_loader, test_loader,
                      epochs=200, lr=1e-3, weight_decay=1e-4, device=device)

eval_test = evaluate_model(model, test_loader, y_scaler,
                           sp_test, cl_test, wf_test, device=device)
print(f"\nFinal performance (test set):")
print(f"  Yaw MSE : {eval_test['yaw_mse']:.4f}")
print(f"  Yaw R²  : {eval_test['yaw_r2']:.4f}")
print(f"  Test pts: {len(eval_test['y_true'])}")

X_full = df_model[feature_cols].values.astype(np.float32)
y_full = y_scaler.transform(df_model[target_col].values.reshape(-1, 1)).ravel()
ind_full = df_model["individual_idx"].values
sp_full = df_model["species"].values
cl_full = df_model["clade"].values
wf_full = 1.0 / df_model["wblen"].values

full_ds = MotorDataset(X_full, y_full, ind_full,
                           species_labels=sp_full, clade_labels=cl_full, wbfreq=wf_full)
full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)

eval_full = evaluate_model(model, full_loader, y_scaler,
                           sp_full, cl_full, wf_full, device=device)

plt.figure(figsize=(7, 5))
plt.plot(history["train_yaw"], label="train yaw")
plt.plot(history["test_yaw"],  label="test yaw")
plt.xlabel("Epoch"); plt.ylabel("MSE loss")
plt.title(f"Yaw loss (latent_dim={latent_dim}, n={clean_subsample_n}/species)")
plt.legend(); plt.tight_layout()
plt.savefig("yaw_loss.png"); plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(eval_test["y_true"], eval_test["y_pred"], alpha=0.7, s=20)
mn = min(eval_test["y_true"].min(), eval_test["y_pred"].min())
mx = max(eval_test["y_true"].max(), eval_test["y_pred"].max())
plt.plot([mn, mx], [mn, mx], "--", color="gray")
plt.xlabel("True tz (within-ind z-score)"); plt.ylabel("Predicted tz")
plt.title(f"Predicted vs true yaw torque — test set  (R²={eval_test['yaw_r2']:.3f})")
plt.tight_layout(); plt.savefig("yaw_pred_vs_true.png"); plt.show()

Z = eval_full["Z"]
species_arr = eval_full["species_labels"]
clade_arr = eval_full["clade_labels"]
wbfreq_arr = eval_full["wbfreq"]
ind_arr = eval_full["individual_idx"].astype(int)
sp_unique = np.unique(species_arr)

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
cmap = plt.cm.get_cmap("tab20", len(sp_unique))

ax = axes[0]
for si, sp in enumerate(sp_unique):
    mask = species_arr == sp
    ax.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.75, color=cmap(si), label=sp)
ax.set_xlabel("Latent 1"); ax.set_ylabel("Latent 2")
ax.set_title(f"Latent space — by species (n={clean_subsample_n}/species)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)

ax = axes[1]
clade_colors = {"silkmoth": "#2196F3", "hawkmoth": "#FF5722"}
for clade in np.unique(clade_arr):
    mask = clade_arr == clade
    ax.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.75,
               color=clade_colors.get(clade, "grey"), label=clade)
ax.set_xlabel("Latent 1"); ax.set_ylabel("Latent 2")
ax.set_title("Latent space — by clade")
ax.legend()

plt.tight_layout(); plt.savefig("yaw_latent_species_clade.png"); plt.show()

# 4. Latent space — wingbeat frequency
plt.figure(figsize=(7, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=wbfreq_arr, s=20, alpha=0.75, cmap="viridis")
plt.xlabel("Latent 1"); plt.ylabel("Latent 2")
plt.title("Latent space — coloured by wingbeat frequency (full dataset)")
plt.colorbar(sc, label="Wingbeat frequency (Hz)")
plt.tight_layout(); plt.savefig("yaw_latent_wbfreq.png"); plt.show()

# ── Feature weights ───────────────────────────────────────────────────────────
eff_weights   = get_effective_yaw_weights(model, feature_cols, individual_names)
yaw_weight_df = pd.concat([eff_weights[i].rename(i) for i in individual_names], axis=1)
yaw_weight_df["mean_abs_weight"] = yaw_weight_df[individual_names].abs().mean(axis=1)

print("\nTop yaw-related features (averaged across individuals):")
print(yaw_weight_df["mean_abs_weight"].sort_values(ascending=False).head(20))

muscle_importance = aggregate_by_muscle(yaw_weight_df["mean_abs_weight"])
print("\nTop muscles (phase + count combined):")
print(muscle_importance.head(10))

yaw_weight_df.to_csv("yaw_feature_weights.csv")

pd.DataFrame([{
    "latent_dim":              latent_dim,
    "subsample_n_per_species": clean_subsample_n,
    "min_wb_to_qualify":       min_wb_to_qualify,
    "num_species":             len(sp_unique),
    "num_individuals":         num_individuals,
    "total_rows":              len(df_model),
    "yaw_mse":                 eval_test["yaw_mse"],
    "yaw_r2":                  eval_test["yaw_r2"],
    "tz_normalization":        "within_individual_zscore",
}]).to_csv("yaw_results_summary.csv", index=False)

df_model.to_csv("input_data_balanced_normalized.csv", index=False)

print("\nDone. Outputs saved:")
print("yaw_loss.png")
print("yaw_pred_vs_true.png(test set)")
print("yaw_latent_species_clade.png (full dataset)")
print("yaw_latent_wbfreq.png (full dataset)")
print("yaw_feature_weights.csv")
print("yaw_results_summary.csv")
print("input_data_balanced_normalized.csv")