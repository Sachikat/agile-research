import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from analysis import get_encoder_weight_df, overall_latent_importance, aggregate_by_muscle, get_effective_yaw_weights, get_raw_feature_variance

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

group_cols = ["species", "moth", "trial", "wb", "muscle_base"]

df_sorted = df.sort_values(group_cols + ["phase"]).copy()
df_sorted["spike_idx"] = df_sorted.groupby(group_cols).cumcount() + 1

df_first10 = df_sorted[df_sorted["spike_idx"] <= 10].copy()

X_phase = df_first10.pivot_table(
    index=["species", "moth", "trial", "wb"],
    columns=["muscle_base", "spike_idx"],
    values="phase",
    aggfunc="first"
)

X_phase.columns = [f"{muscle}_spike{int(spike)}" for muscle, spike in X_phase.columns]
X_phase = X_phase.reset_index()

reference_muscles = []
for ref in reference_muscles:
    for col in X_phase.columns:
        if "_spike" not in col:
            continue

        muscle_name = col.split("_spike")[0]
        spike_num = col.split("_spike")[1]
        ref_col = f"{ref}_spike{spike_num}"

        if ref_col in X_phase.columns and muscle_name != ref:
            new_col = f"{muscle_name}_minus_{ref}_spike{spike_num}"
            X_phase[new_col] = X_phase[col] - X_phase[ref_col]

meta = df.groupby(["species", "moth", "trial", "wb"]).agg({
    "wbfreq": "mean",
    "clade": "first"
}).reset_index()

target = df.groupby(["species", "moth", "trial", "wb"]).agg({
    "tz": "mean"
}).reset_index()

X = X_phase.merge(meta, on=["species", "moth", "trial", "wb"], how="left")
X = X.merge(target, on=["species", "moth", "trial", "wb"], how="left")

non_features = ["species", "moth", "trial", "wb", "wbfreq", "clade", "tz"]
feature_cols = [c for c in X.columns if c not in non_features]

X[feature_cols] = X[feature_cols].fillna(0)

min_wb = X["species"].value_counts().min()

X_bal = (
    X.groupby("species", group_keys=False)
    .sample(min_wb, random_state=0)
    .reset_index(drop=True)
)

print("Wingbeats per species:", min_wb)
print("Final dataset shape:", X_bal.shape)
print("Number of features:", len(feature_cols))


species_names = sorted(X_bal["species"].unique())
species_idx = {s: i for i, s in enumerate(species_names)}
idx_species = {i: s for s, i in species_idx.items()}

X_bal["species_idx"] = X_bal["species"].map(species_idx)
num_species = len(species_names)


X_input = X_bal[feature_cols].values
y_target = X_bal["tz"].values
species_idx = X_bal["species_idx"].values

X_train, X_test, y_train, y_test, species_train, species_test = train_test_split(
    X_input,
    y_target,
    species_idx,
    test_size=0.2,
    random_state=42,
    stratify=species_idx
)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

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


class MultiEncoderLinearAEYaw(nn.Module):
    def __init__(self, input_dim, latent_dim, num_species):
        super().__init__()
        self.latent_dim = latent_dim

        # multiple encoders - linear
        self.encoders = nn.ModuleList([
            nn.Linear(input_dim, latent_dim) for _ in range(num_species)
        ])

        # universal decoder - linear 
        self.decoder_x = nn.Linear(latent_dim, input_dim)
        self.decoder_y = nn.Linear(latent_dim, 1)

    def forward(self, x, species_idx):
        z = torch.zeros(x.shape[0], self.latent_dim, device=x.device)

        for s in torch.unique(species_idx):
            mask = (species_idx == s)
            z[mask] = self.encoders[s.item()](x[mask])

        x_hat = self.decoder_x(z)
        y_hat = self.decoder_y(z)
        return x_hat, y_hat, z

def train_model(
    model,
    train_loader,
    test_loader,
    alpha_recon=1.0,
    beta_yaw=1.0,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu"
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {
        "train_total": [],
        "test_total": [],
        "train_recon": [],
        "test_recon": [],
        "train_yaw": [],
        "test_yaw": []
    }

    for epoch in range(epochs):
        model.train()

        total_train = 0.0
        total_train_recon = 0.0
        total_train_yaw = 0.0

        for xb, yb, sb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)

            optimizer.zero_grad()

            x_hat, y_hat, z = model(xb, sb)

            recon_loss = loss_fn(x_hat, xb)
            yaw_loss = loss_fn(y_hat, yb)
            loss = alpha_recon * recon_loss + beta_yaw * yaw_loss

            loss.backward()
            optimizer.step()

            total_train += loss.item() * xb.size(0)
            total_train_recon += recon_loss.item() * xb.size(0)
            total_train_yaw += yaw_loss.item() * xb.size(0)

        avg_train = total_train / len(train_loader.dataset)
        avg_train_recon = total_train_recon / len(train_loader.dataset)
        avg_train_yaw = total_train_yaw / len(train_loader.dataset)

        history["train_total"].append(avg_train)
        history["train_recon"].append(avg_train_recon)
        history["train_yaw"].append(avg_train_yaw)

        model.eval()

        total_test = 0.0
        total_test_recon = 0.0
        total_test_yaw = 0.0

        with torch.no_grad():
            for xb, yb, sb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                sb = sb.to(device)

                x_hat, y_hat, z = model(xb, sb)

                recon_loss = loss_fn(x_hat, xb)
                yaw_loss = loss_fn(y_hat, yb)
                loss = alpha_recon * recon_loss + beta_yaw * yaw_loss

                total_test += loss.item() * xb.size(0)
                total_test_recon += recon_loss.item() * xb.size(0)
                total_test_yaw += yaw_loss.item() * xb.size(0)

        avg_test = total_test / len(test_loader.dataset)
        avg_test_recon = total_test_recon / len(test_loader.dataset)
        avg_test_yaw = total_test_yaw / len(test_loader.dataset)

        history["test_total"].append(avg_test)
        history["test_recon"].append(avg_test_recon)
        history["test_yaw"].append(avg_test_yaw)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train total {avg_train:.4f} | Test total {avg_test:.4f} | "
                f"Train recon {avg_train_recon:.4f} | Test recon {avg_test_recon:.4f} | "
                f"Train yaw {avg_train_yaw:.4f} | Test yaw {avg_test_yaw:.4f}"
            )

    return history

def evaluate_model(model, loader, x_scaler, y_scaler, device="cpu"):
    model.eval()

    x_true_all = []
    x_hat_all = []
    y_true_all = []
    y_pred_all = []
    z_all = []
    species_all = []

    with torch.no_grad():
        for xb, yb, sb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)

            x_hat, y_hat, z = model(xb, sb)

            x_true_all.append(xb.cpu().numpy())
            x_hat_all.append(x_hat.cpu().numpy())
            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(y_hat.cpu().numpy())
            z_all.append(z.cpu().numpy())
            species_all.append(sb.cpu().numpy())

    x_true = np.vstack(x_true_all)
    x_hat = np.vstack(x_hat_all)
    y_true = np.vstack(y_true_all).ravel()
    y_pred = np.vstack(y_pred_all).ravel()
    Z = np.vstack(z_all)
    species_out = np.concatenate(species_all)

    x_true_unscaled = x_scaler.inverse_transform(x_true)
    x_hat_unscaled = x_scaler.inverse_transform(x_hat)

    y_true_unscaled = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_unscaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    recon_mse = mean_squared_error(x_true_unscaled.ravel(), x_hat_unscaled.ravel())
    yaw_mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    yaw_r2 = r2_score(y_true_unscaled, y_pred_unscaled)

    return {
        "x_true": x_true_unscaled,
        "x_hat": x_hat_unscaled,
        "y_true": y_true_unscaled,
        "y_pred": y_pred_unscaled,
        "Z": Z,
        "species_idx": species_out,
        "recon_mse": recon_mse,
        "yaw_mse": yaw_mse,
        "yaw_r2": yaw_r2
    }

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

results = []

for latent_dim in [16, 8, 4, 2]:

    model = MultiEncoderLinearAEYaw(
        input_dim=len(feature_cols),
        latent_dim=latent_dim,
        num_species=num_species
    )

    history = train_model(
        model,
        train_loader,
        test_loader,
        alpha_recon=1.0,
        beta_yaw=1.0,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-4,
        device=device
    )

    eval_out = evaluate_model(
        model,
        test_loader,
        x_scaler,
        y_scaler,
        device=device
    )

    print("Reconstruction MSE:", eval_out["recon_mse"])
    print("Yaw MSE:", eval_out["yaw_mse"])
    print("Yaw R2:", eval_out["yaw_r2"])

    results.append({
        "latent_dim": latent_dim,
        "recon_mse": eval_out["recon_mse"],
        "yaw_mse": eval_out["yaw_mse"],
        "yaw_r2": eval_out["yaw_r2"]
    })

    # total loss
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_total"], label="train total")
    plt.plot(history["test_total"], label="test total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Total loss (latent_dim={latent_dim})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"multi_encoder_total_loss_dim_{latent_dim}.png")
    plt.show()

    # yaw pred
    plt.figure(figsize=(6, 6))
    plt.scatter(eval_out["y_true"], eval_out["y_pred"], alpha=0.6, s=12)
    mn = min(eval_out["y_true"].min(), eval_out["y_pred"].min())
    mx = max(eval_out["y_true"].max(), eval_out["y_pred"].max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("True tz")
    plt.ylabel("Predicted tz")
    plt.title(f"Predicted vs true yaw torque (latent_dim={latent_dim})")
    plt.tight_layout()
    plt.savefig(f"multi_encoder_pred_vs_true_dim_{latent_dim}.png")
    plt.show()

    # latent plot
    if latent_dim >= 2:
        plt.figure(figsize=(9, 7))
        for s in np.unique(eval_out["species_idx"]):
            mask = eval_out["species_idx"] == s
            plt.scatter(
                eval_out["Z"][mask, 0],
                eval_out["Z"][mask, 1],
                s=10,
                alpha=0.6,
                label=idx_species[s]
            )

        plt.xlabel("Latent 1")
        plt.ylabel("Latent 2")
        plt.title(f"Shared latent space (latent_dim={latent_dim})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"multi_encoder_latent_space_dim_{latent_dim}.png")
        plt.show()

results_df = pd.DataFrame(results)
print("\nFinal results:")
print(results_df)

results_df.to_csv("multi_encoder_linear_ae_yaw_results.csv", index=False)
X_bal.to_csv("motor_program_first10_balanced.csv", index=False)


# ANALYSIS - print what muscles affected yaw torque based on spikes

encoder_tables = get_encoder_weight_df(model, feature_cols, species_names)

latent_feature_importance = {}
latent_muscle_importance = {}
effective_yaw_weights = get_effective_yaw_weights(model, feature_cols, species_names)
effective_yaw_abs_importance = {}
effective_yaw_muscle_importance = {}

for species in species_names:
    feat_imp = overall_latent_importance(encoder_tables[species])
    latent_feature_importance[species] = feat_imp
    latent_muscle_importance[species] = aggregate_by_muscle(feat_imp)

    yaw_abs = effective_yaw_weights[species].abs().sort_values(ascending=False)
    effective_yaw_abs_importance[species] = yaw_abs
    effective_yaw_muscle_importance[species] = aggregate_by_muscle(yaw_abs)

raw_feature_variance = get_raw_feature_variance(X_bal, feature_cols)

TOP_N = 15

for species in species_names:
    print("\n" + "=" * 70)
    print(f"SPECIES: {species}")
    print("=" * 70)

    print("\nTop latent-space spike features:")
    print(latent_feature_importance[species].head(TOP_N))

    print("\nTop latent-space muscles:")
    print(latent_muscle_importance[species].head(TOP_N))

    print("\nTop yaw-related spike features (signed):")
    top_yaw_signed = effective_yaw_weights[species].reindex(
        effective_yaw_weights[species].abs().sort_values(ascending=False).head(TOP_N).index
    )
    print(top_yaw_signed)

    print("\nTop yaw-related muscles:")
    print(effective_yaw_muscle_importance[species].head(TOP_N))


latent_feat_df = pd.concat(
    [latent_feature_importance[s].rename(s) for s in species_names],
    axis=1
)
latent_feat_df["mean_importance"] = latent_feat_df.mean(axis=1)
latent_feat_df["std_importance"] = latent_feat_df[species_names].std(axis=1)

latent_muscle_df = pd.concat(
    [latent_muscle_importance[s].rename(s) for s in species_names],
    axis=1
).fillna(0)
latent_muscle_df["mean_importance"] = latent_muscle_df.mean(axis=1)
latent_muscle_df["std_importance"] = latent_muscle_df[species_names].std(axis=1)

yaw_feat_df = pd.concat(
    [effective_yaw_abs_importance[s].rename(s) for s in species_names],
    axis=1
)
yaw_feat_df["mean_importance"] = yaw_feat_df.mean(axis=1)
yaw_feat_df["std_importance"] = yaw_feat_df[species_names].std(axis=1)

yaw_muscle_df = pd.concat(
    [effective_yaw_muscle_importance[s].rename(s) for s in species_names],
    axis=1
).fillna(0)
yaw_muscle_df["mean_importance"] = yaw_muscle_df.mean(axis=1)
yaw_muscle_df["std_importance"] = yaw_muscle_df[species_names].std(axis=1)


print("\n" + "=" * 70)
print("TOP LATENT-SPACE FEATURES ACROSS SPECIES")
print("=" * 70)
print(latent_feat_df["mean_importance"].sort_values(ascending=False).head(25))

print("\n" + "=" * 70)
print("TOP LATENT-SPACE MUSCLES ACROSS SPECIES")
print("=" * 70)
print(latent_muscle_df["mean_importance"].sort_values(ascending=False).head(25))

print("\n" + "=" * 70)
print("TOP YAW-RELATED FEATURES ACROSS SPECIES")
print("=" * 70)
print(yaw_feat_df["mean_importance"].sort_values(ascending=False).head(25))

print("\n" + "=" * 70)
print("TOP YAW-RELATED MUSCLES ACROSS SPECIES")
print("=" * 70)
print(yaw_muscle_df["mean_importance"].sort_values(ascending=False).head(25))

print("\n" + "=" * 70)
print("TOP RAW-VARIANCE FEATURES IN DATASET")
print("=" * 70)
print(raw_feature_variance.head(25))


species_to_inspect = species_names[0]   # change this if you want
W_df = encoder_tables[species_to_inspect]

print("\n" + "=" * 70)
print(f"TOP FEATURES DEFINING z1 and z2 FOR {species_to_inspect}")
print("=" * 70)

for z_name in W_df.index:
    top = W_df.loc[z_name].abs().sort_values(ascending=False).head(15)
    print(f"\nTop features for {z_name}:")
    for feat in top.index:
        print(f"{feat}: {W_df.loc[z_name, feat]:.4f}")

# 1) top latent-space features across species
top_latent_feats = latent_feat_df["mean_importance"].sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
top_latent_feats.sort_values().plot(kind="barh")
plt.xlabel("Mean latent importance across species")
plt.title("Top spike features shaping latent space")
plt.tight_layout()
plt.show()

# 2) top latent-space muscles across species
top_latent_muscles = latent_muscle_df["mean_importance"].sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
top_latent_muscles.sort_values().plot(kind="barh")
plt.xlabel("Mean latent importance across species")
plt.title("Top muscles shaping latent space")
plt.tight_layout()
plt.show()

# 3) top yaw-related features across species
top_yaw_feats = yaw_feat_df["mean_importance"].sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
top_yaw_feats.sort_values().plot(kind="barh")
plt.xlabel("Mean |effective yaw weight| across species")
plt.title("Top spike features affecting predicted yaw")
plt.tight_layout()
plt.show()

# 4) top yaw-related muscles across species
top_yaw_muscles = yaw_muscle_df["mean_importance"].sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
top_yaw_muscles.sort_values().plot(kind="barh")
plt.xlabel("Mean |effective yaw weight| across species")
plt.title("Top muscles affecting predicted yaw")
plt.tight_layout()
plt.show()

latent_feat_df.to_csv("latent_feature_importance_by_species.csv")
latent_muscle_df.to_csv("latent_muscle_importance_by_species.csv")
yaw_feat_df.to_csv("yaw_feature_importance_by_species.csv")
yaw_muscle_df.to_csv("yaw_muscle_importance_by_species.csv")
raw_feature_variance.to_csv("raw_feature_variance.csv")

print("\nSaved:")
print("- latent_feature_importance_by_species.csv")
print("- latent_muscle_importance_by_species.csv")
print("- yaw_feature_importance_by_species.csv")
print("- yaw_muscle_importance_by_species.csv")
print("- raw_feature_variance.csv")