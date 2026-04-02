import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("preprocessedCache.csv")

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


def base_muscle(m):
    m = str(m)

    if m.endswith("_L") or m.endswith("_R"):
        return m[:-2]

    if "left" in m.lower():
        return m.lower().replace("left", "").strip()

    if "right" in m.lower():
        return m.lower().replace("right", "").strip()

    return m


df["muscle_base"] = df["muscle"].apply(base_muscle)

n_spikes_keep = 10
min_valid_wingbeats_per_species = 20
wingbeat_keys = ["species", "moth", "trial", "wb"]
group_cols = wingbeat_keys + ["muscle_base"]

all_muscles = sorted(df["muscle_base"].dropna().unique())
all_muscle_set = set(all_muscles)

print("All muscles in dataset:")
print(all_muscles)
print("Number of muscles:", len(all_muscles))

wingbeat_muscles = (
    df.groupby(wingbeat_keys)["muscle_base"]
    .unique()
    .reset_index()
)

wingbeat_muscles["has_all_muscles"] = wingbeat_muscles["muscle_base"].apply(
    lambda x: set(x) == all_muscle_set
)

valid_wingbeats = wingbeat_muscles[wingbeat_muscles["has_all_muscles"]].copy()

print("\nTotal wingbeats:", len(wingbeat_muscles))
print("Wingbeats with all muscles present:", len(valid_wingbeats))

if len(valid_wingbeats) == 0:
    raise ValueError("No wingbeats contain all muscles at least once.")

df_valid = df.merge(
    valid_wingbeats[wingbeat_keys],
    on=wingbeat_keys,
    how="inner"
).copy()

print("Rows after valid-wingbeat filter:", len(df_valid))

df_valid = df_valid.sort_values(group_cols + ["phase"]).copy()
df_valid["spike_idx"] = df_valid.groupby(group_cols).cumcount() + 1

df_firstN = df_valid[df_valid["spike_idx"] <= n_spikes_keep].copy()

print(f"Rows after taking first {n_spikes_keep} spikes:", len(df_firstN))

X_phase = df_firstN.pivot_table(
    index=wingbeat_keys,
    columns=["muscle_base", "spike_idx"],
    values="phase",
    aggfunc="first"
)

X_phase.columns = [f"{muscle}_spike{int(spike)}" for muscle, spike in X_phase.columns]
X_phase = X_phase.reset_index()

print("Wingbeats after pivot:", len(X_phase))
print("Timing columns after pivot:", len([c for c in X_phase.columns if "_spike" in c]))

spike_counts = (
    df_valid.groupby(group_cols)
    .size()
    .clip(upper=n_spikes_keep)
    .reset_index(name="n_spikes")
)

X_counts = spike_counts.pivot_table(
    index=wingbeat_keys,
    columns="muscle_base",
    values="n_spikes",
    aggfunc="first"
)

X_counts.columns = [f"{muscle}_n_spikes" for muscle in X_counts.columns]
X_counts = X_counts.reset_index()

print("Spike-count columns:", len([c for c in X_counts.columns if c.endswith("_n_spikes")]))

meta = df_valid.groupby(wingbeat_keys).agg({
    "wbfreq": "mean",
    "clade": "first"
}).reset_index()

target = df_valid.groupby(wingbeat_keys).agg({
    "tz": "mean"
}).reset_index()

X = X_phase.merge(X_counts, on=wingbeat_keys, how="left")
X = X.merge(meta, on=wingbeat_keys, how="left")
X = X.merge(target, on=wingbeat_keys, how="left")

print("Merged dataset shape:", X.shape)

non_features = ["species", "moth", "trial", "wb", "wbfreq", "clade", "tz"]

timing_feature_cols = [c for c in X.columns if "_spike" in c and not c.endswith("_n_spikes")]
count_feature_cols = [c for c in X.columns if c.endswith("_n_spikes")]

print("Timing feature count:", len(timing_feature_cols))
print("Count feature count:", len(count_feature_cols))

mask_df = X[timing_feature_cols].isna().astype(float)
mask_df.columns = [f"{c}_missing" for c in timing_feature_cols]
mask_feature_cols = list(mask_df.columns)

print("Mask feature count:", len(mask_feature_cols))

species_counts = X["species"].value_counts()
valid_species = species_counts[species_counts >= min_valid_wingbeats_per_species].index.tolist()

X = X[X["species"].isin(valid_species)].copy()
mask_df = mask_df.loc[X.index].copy()

print("\nSpecies retained:")
print(X["species"].value_counts())

if len(X) == 0:
    raise ValueError("No species remain after filtering for minimum wingbeat count.")

# Remove subsampling
# min_wb = X["species"].value_counts().min()

# X_bal = (
#     X.groupby("species", group_keys=False)
#     .sample(min_wb, random_state=0)
#     .sort_index()
#     .copy()
# )

X_bal = X.copy()

mask_bal = mask_df.loc[X_bal.index].copy()

print("\nWingbeats per species after balancing:", min_wb)
print("Balanced dataset shape:", X_bal.shape)

species_names = sorted(X_bal["species"].unique())
species_to_idx = {s: i for i, s in enumerate(species_names)}
idx_to_species = {i: s for s, i in species_to_idx.items()}

X_bal["species_idx"] = X_bal["species"].map(species_to_idx)
num_species = len(species_names)

timing_values = X_bal[timing_feature_cols].copy()

all_nan_timing_cols = timing_values.columns[timing_values.isna().all()].tolist()
if len(all_nan_timing_cols) > 0:
    print("\nDropping all-NaN timing columns:")
    print(all_nan_timing_cols)

timing_values = timing_values.drop(columns=all_nan_timing_cols)
timing_feature_cols = [c for c in timing_feature_cols if c not in all_nan_timing_cols]

# rebuild masks to match kept timing columns
mask_feature_cols = [f"{c}_missing" for c in timing_feature_cols]
mask_bal = mask_bal[mask_feature_cols].copy()

# impute remaining NaNs in timings with column means
timing_values = timing_values.fillna(timing_values.mean())

count_values = X_bal[count_feature_cols].copy().fillna(0)

print("\nRemaining NaNs after imputation:")
print("timing:", timing_values.isna().sum().sum())
print("counts:", count_values.isna().sum().sum())
print("masks:", mask_bal.isna().sum().sum())

X_model = pd.concat(
    [
        X_bal[wingbeat_keys + ["wbfreq", "clade", "tz", "species_idx"]].reset_index(drop=True),
        timing_values.reset_index(drop=True),
        count_values.reset_index(drop=True),
        mask_bal.reset_index(drop=True)
    ],
    axis=1
)

model_feature_cols = timing_feature_cols + count_feature_cols + mask_feature_cols

X_train_df, X_test_df, y_train, y_test, species_train, species_test = train_test_split(
    X_model[model_feature_cols],
    X_model["tz"].values,
    X_model["species_idx"].values,
    test_size=0.2,
    random_state=42,
    stratify=X_model["species_idx"].values
)

numeric_feature_cols = timing_feature_cols + count_feature_cols

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_numeric = x_scaler.fit_transform(X_train_df[numeric_feature_cols])
X_test_numeric = x_scaler.transform(X_test_df[numeric_feature_cols])

X_train_masks = X_train_df[mask_feature_cols].values.astype(np.float32)
X_test_masks = X_test_df[mask_feature_cols].values.astype(np.float32)

X_train = np.concatenate([X_train_numeric, X_train_masks], axis=1)
X_test = np.concatenate([X_test_numeric, X_test_masks], axis=1)

y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

class MotorDataset(Dataset):
    def __init__(self, X, y, species_idx):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.species_idx = torch.tensor(species_idx, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.species_idx[idx]


train_ds = MotorDataset(X_train, y_train, species_train)
test_ds = MotorDataset(X_test, y_test, species_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

class MultiEncoderYawModel(nn.Module):
    def __init__(self, input_dim, latent_dim, num_species):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleList([
            nn.Linear(input_dim, latent_dim) for _ in range(num_species)
        ])

        self.decoder_y = nn.Linear(latent_dim, 1)

    def forward(self, x, species_idx):
        z = torch.zeros(x.shape[0], self.latent_dim, device=x.device)

        for s in torch.unique(species_idx):
            mask = (species_idx == s)
            z[mask] = self.encoders[s.item()](x[mask])

        y_hat = self.decoder_y(z)
        return y_hat, z

def train_model(model, train_loader, test_loader, epochs=200, lr=1e-3, weight_decay=1e-4, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {
        "train_yaw": [],
        "test_yaw": []
    }

    for epoch in range(epochs):
        model.train()
        total_train = 0.0

        for xb, yb, sb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)

            optimizer.zero_grad()
            y_hat, z = model(xb, sb)
            loss = loss_fn(y_hat, yb)
            loss.backward()
            optimizer.step()

            total_train += loss.item() * xb.size(0)

        avg_train = total_train / len(train_loader.dataset)
        history["train_yaw"].append(avg_train)

        model.eval()
        total_test = 0.0

        with torch.no_grad():
            for xb, yb, sb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                sb = sb.to(device)

                y_hat, z = model(xb, sb)
                loss = loss_fn(y_hat, yb)
                total_test += loss.item() * xb.size(0)

        avg_test = total_test / len(test_loader.dataset)
        history["test_yaw"].append(avg_test)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Train yaw {avg_train:.4f} | Test yaw {avg_test:.4f}")

    return history

def evaluate_model(model, loader, y_scaler, device="cpu"):
    model.eval()

    y_true_all = []
    y_pred_all = []
    z_all = []
    species_all = []

    with torch.no_grad():
        for xb, yb, sb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)

            y_hat, z = model(xb, sb)

            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(y_hat.cpu().numpy())
            z_all.append(z.cpu().numpy())
            species_all.append(sb.cpu().numpy())

    y_true = np.vstack(y_true_all).ravel()
    y_pred = np.vstack(y_pred_all).ravel()
    Z = np.vstack(z_all)
    species_idx_out = np.concatenate(species_all)

    y_true_unscaled = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_unscaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    yaw_mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    yaw_r2 = r2_score(y_true_unscaled, y_pred_unscaled)

    return {
        "y_true": y_true_unscaled,
        "y_pred": y_pred_unscaled,
        "Z": Z,
        "species_idx": species_idx_out,
        "yaw_mse": yaw_mse,
        "yaw_r2": yaw_r2
    }

def get_effective_yaw_weights(model, feature_cols, species_names):
    yaw_w = model.decoder_y.weight.detach().cpu().numpy().reshape(-1)
    out = {}

    for i, species in enumerate(species_names):
        E = model.encoders[i].weight.detach().cpu().numpy()
        eff = yaw_w @ E
        out[species] = pd.Series(eff, index=feature_cols).sort_values(key=np.abs, ascending=False)

    return out

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing device:", device)

results = []

latent_dim = 2

print("\n======================================")
print(f"MULTI-ENCODER YAW MODEL | latent_dim = {latent_dim}")
print("======================================")

model = MultiEncoderYawModel(
    input_dim=X_train.shape[1],
    latent_dim=latent_dim,
    num_species=num_species
)

history = train_model(
    model,
    train_loader,
    test_loader,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    device=device
)

eval_out = evaluate_model(
    model,
    test_loader,
    y_scaler,
    device=device
)

print("Yaw MSE:", eval_out["yaw_mse"])
print("Yaw R2 :", eval_out["yaw_r2"])

results.append({
    "latent_dim": latent_dim,
    "yaw_mse": eval_out["yaw_mse"],
    "yaw_r2": eval_out["yaw_r2"]
})

plt.figure(figsize=(7, 5))
plt.plot(history["train_yaw"], label="train yaw")
plt.plot(history["test_yaw"], label="test yaw")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title(f"Yaw loss (latent_dim={latent_dim})")
plt.legend()
plt.tight_layout()
plt.savefig(f"yaw_loss_dim_{latent_dim}.png")
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(eval_out["y_true"], eval_out["y_pred"], alpha=0.6, s=12)
mn = min(eval_out["y_true"].min(), eval_out["y_pred"].min())
mx = max(eval_out["y_true"].max(), eval_out["y_pred"].max())
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("True tz")
plt.ylabel("Predicted tz")
plt.title(f"Predicted vs true yaw torque (latent_dim={latent_dim})")
plt.tight_layout()
plt.savefig(f"yaw_pred_vs_true_dim_{latent_dim}.png")
plt.show()

if latent_dim >= 2:
    plt.figure(figsize=(9, 7))
    for s in np.unique(eval_out["species_idx"]):
        mask = eval_out["species_idx"] == s
        plt.scatter(
            eval_out["Z"][mask, 0],
            eval_out["Z"][mask, 1],
            s=10,
            alpha=0.6,
            label=idx_to_species[s]
        )

    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.title(f"Shared latent space (latent_dim={latent_dim})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"yaw_shared_latent_dim_{latent_dim}.png")
    plt.show()

# interpretation for this model
effective_yaw_weights = get_effective_yaw_weights(model, model_feature_cols, species_names)

print("\nTop yaw-related features by species:")
for species in species_names:
    print("\n" + "=" * 60)
    print(species)
    print("=" * 60)
    print(effective_yaw_weights[species].head(15))

# save per-latent-dim feature weights
yaw_weight_df = pd.concat(
    [effective_yaw_weights[s].rename(s) for s in species_names],
    axis=1
)
yaw_weight_df.to_csv(f"yaw_feature_weights_dim_{latent_dim}.csv")

results_df = pd.DataFrame(results)
print("\nFinal results:")
print(results_df)

results_df.to_csv("multi_encoder_yaw_results.csv", index=False)
X_model.to_csv("motor_program_track2_inputs.csv", index=False)