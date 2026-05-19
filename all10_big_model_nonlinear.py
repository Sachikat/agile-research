import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("all10_big_wb.csv")

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
    "Hemaris diffinis": "hawkmoth",
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

for col in feature_cols + [target_col]:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

df_model = df_model.dropna(subset=feature_cols + [target_col, "species", "moth", "wblen"]).copy()

min_wb_to_qualify = 100
clean_subsample_n = 150

wb_counts = df_model.groupby("species").size()
species_to_keep = wb_counts[wb_counts >= min_wb_to_qualify].index.tolist()
df_model = df_model[df_model["species"].isin(species_to_keep)].copy()

df_model = (
    df_model.groupby("species", group_keys=False)
    .sample(n=clean_subsample_n)
    .reset_index(drop=True)
)

print("\nRows per species after balanced subsampling:")
print(df_model["species"].value_counts().sort_index())

# Within-individual z-score for yaw torque.
# This keeps yaw comparable across individuals while still using species encoders.
ind_tz_stats = (
    df_model.groupby("moth")[target_col]
    .agg(["mean", "std"])
    .rename(columns={"mean": "ind_tz_mean", "std": "ind_tz_std"})
)
ind_tz_stats["ind_tz_std"] = ind_tz_stats["ind_tz_std"].fillna(1.0).replace(0.0, 1.0)
df_model = df_model.join(ind_tz_stats, on="moth")
df_model[target_col] = (df_model[target_col] - df_model["ind_tz_mean"]) / df_model["ind_tz_std"]
df_model = df_model.drop(columns=["ind_tz_mean", "ind_tz_std"])

print("\nPer-individual z-scored tz check:")
print(df_model.groupby("moth")[target_col].agg(["mean", "std"]).describe().round(3))

df_model[count_cols] = (df_model[count_cols] / 10.0).clip(0.0, 1.0)
df_model[phase_cols] = ((df_model[phase_cols] + 1.0) / 2.0).clip(0.0, 1.0)

# species encoder

species_names = sorted(df_model["species"].astype(str).unique())
species_to_idx = {sp: i for i, sp in enumerate(species_names)}
df_model["species_idx"] = df_model["species"].astype(str).map(species_to_idx)
num_species = len(species_names)

print(f"\nNumber of species encoders: {num_species}")
print("Species encoder mapping:")
for sp, idx in species_to_idx.items():
    print(f"  {idx}: {sp}")

X_df = df_model[feature_cols].copy()
y = df_model[target_col].values.astype(np.float32)
species_idx = df_model["species_idx"].values.astype(np.int64)
species_labels = df_model["species"].values
clade_labels = df_model["clade"].values
wbfreq_values = 1.0 / df_model["wblen"].values

(
    X_train_df, X_test_df,
    y_train_raw, y_test_raw,
    spidx_train, spidx_test,
    splab_train, splab_test,
    cl_train, cl_test,
    wf_train, wf_test,
) = train_test_split(
    X_df,
    y,
    species_idx,
    species_labels,
    clade_labels,
    wbfreq_values,
    test_size=0.2,
    stratify=species_labels,
)

X_train = X_train_df.values.astype(np.float32)
X_test = X_test_df.values.astype(np.float32)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel().astype(np.float32)
y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel().astype(np.float32)

class MotorDataset(Dataset):
    def __init__(self, X, y, species_idx):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.species_idx = torch.tensor(species_idx, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.species_idx[idx]


train_ds = MotorDataset(X_train, y_train, spidx_train)
test_ds = MotorDataset(X_test, y_test, spidx_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


# species-specific linear encoder + nonlinear decoder

class SpeciesEncoderAutoencoderYaw(nn.Module):
    """
    Architecture:

    X muscles/features:
        species-specific LINEAR encoder
        latent motor program z
        nonlinear decoder_x reconstructs X
        linear decoder_y predicts yaw

    The encoder is  linear.
    The nonlinearity is only added in decoder_x through ReLU layers.
    This encourages z to capture muscle-feature variance, not just yaw prediction.
    """

    def __init__(self, input_dim, latent_dim, num_species, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # One linear encoder per species, not per individual.
        self.encoders = nn.ModuleList([
            nn.Linear(input_dim, latent_dim) for _ in range(num_species)
        ])

        # Nonlinear muscle reconstruction decoder.
        # This is where nonlinearity is added.
        self.decoder_x = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Linear yaw decoder for interpretability.
        self.decoder_y = nn.Linear(latent_dim, 1)

    def forward(self, x, species_idx):
        z = torch.zeros(x.shape[0], self.latent_dim, device=x.device)

        for sp in torch.unique(species_idx):
            mask = species_idx == sp
            z[mask] = self.encoders[sp.item()](x[mask])

        x_hat = self.decoder_x(z)
        y_hat = self.decoder_y(z)

        return x_hat, y_hat, z


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=300,
    lr=1e-3,
    weight_decay=1e-4,
    yaw_loss_weight=1.0,
    device="cpu",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {
        "train_total": [],
        "train_recon": [],
        "train_yaw": [],
        "test_total": [],
        "test_recon": [],
        "test_yaw": [],
    }

    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_yaw = 0.0

        for xb, yb, sb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)

            optimizer.zero_grad()

            x_hat, y_hat, _ = model(xb, sb)

            recon_loss = loss_fn(x_hat, xb)
            yaw_loss = loss_fn(y_hat, yb)
            loss = recon_loss + yaw_loss_weight * yaw_loss

            loss.backward()
            optimizer.step()

            n = xb.size(0)
            train_total += loss.item() * n
            train_recon += recon_loss.item() * n
            train_yaw += yaw_loss.item() * n

        train_total /= len(train_loader.dataset)
        train_recon /= len(train_loader.dataset)
        train_yaw /= len(train_loader.dataset)

        model.eval()
        test_total = 0.0
        test_recon = 0.0
        test_yaw = 0.0

        with torch.no_grad():
            for xb, yb, sb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                sb = sb.to(device)

                x_hat, y_hat, _ = model(xb, sb)

                recon_loss = loss_fn(x_hat, xb)
                yaw_loss = loss_fn(y_hat, yb)
                loss = recon_loss + yaw_loss_weight * yaw_loss

                n = xb.size(0)
                test_total += loss.item() * n
                test_recon += recon_loss.item() * n
                test_yaw += yaw_loss.item() * n

        test_total /= len(test_loader.dataset)
        test_recon /= len(test_loader.dataset)
        test_yaw /= len(test_loader.dataset)

        history["train_total"].append(train_total)
        history["train_recon"].append(train_recon)
        history["train_yaw"].append(train_yaw)
        history["test_total"].append(test_total)
        history["test_recon"].append(test_recon)
        history["test_yaw"].append(test_yaw)

        if epoch % 25 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train total {train_total:.4f} | recon {train_recon:.4f} | yaw {train_yaw:.4f} || "
                f"Test total {test_total:.4f} | recon {test_recon:.4f} | yaw {test_yaw:.4f}"
            )

    return history


def evaluate_model(model, loader, y_scaler, device="cpu"):
    model.eval()
    y_true_all = []
    y_pred_all = []
    x_true_all = []
    x_recon_all = []
    z_all = []
    species_idx_all = []

    with torch.no_grad():
        for xb, yb, sb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            sb = sb.to(device)

            x_hat, y_hat, z = model(xb, sb)

            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(y_hat.cpu().numpy())
            x_true_all.append(xb.cpu().numpy())
            x_recon_all.append(x_hat.cpu().numpy())
            z_all.append(z.cpu().numpy())
            species_idx_all.append(sb.cpu().numpy())

    y_true_scaled = np.vstack(y_true_all).ravel()
    y_pred_scaled = np.vstack(y_pred_all).ravel()

    y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    X_true = np.vstack(x_true_all)
    X_recon = np.vstack(x_recon_all)
    Z = np.vstack(z_all)
    species_idx_out = np.concatenate(species_idx_all)

    yaw_mse = mean_squared_error(y_true, y_pred)
    yaw_r2 = r2_score(y_true, y_pred)
    recon_mse = mean_squared_error(X_true, X_recon)
    recon_r2 = r2_score(X_true, X_recon, multioutput="variance_weighted")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "X_true": X_true,
        "X_recon": X_recon,
        "Z": Z,
        "species_idx": species_idx_out,
        "yaw_mse": yaw_mse,
        "yaw_r2": yaw_r2,
        "recon_mse": recon_mse,
        "recon_r2": recon_r2,
    }


def save_species_encoder_weights(model, feature_cols, species_names, latent_dim):
    rows = []

    for sp_idx, sp_name in enumerate(species_names):
        W = model.encoders[sp_idx].weight.detach().cpu().numpy()
        b = model.encoders[sp_idx].bias.detach().cpu().numpy()

        for latent_i in range(latent_dim):
            for feat_i, feat in enumerate(feature_cols):
                rows.append({
                    "species": sp_name,
                    "latent_dim": f"latent_{latent_i + 1}",
                    "feature": feat,
                    "encoder_weight": W[latent_i, feat_i],
                    "encoder_bias": b[latent_i],
                })

    encoder_df = pd.DataFrame(rows)
    encoder_df.to_csv("species_encoder_weights.csv", index=False)
    return encoder_df


def save_yaw_latent_weights(model, latent_dim):
    yaw_w = model.decoder_y.weight.detach().cpu().numpy().reshape(-1)
    yaw_b = model.decoder_y.bias.detach().cpu().numpy().reshape(-1)[0]

    yaw_latent_df = pd.DataFrame({
        "latent_dim": [f"latent_{i + 1}" for i in range(latent_dim)],
        "yaw_weight": yaw_w,
        "yaw_bias": yaw_b,
    })

    yaw_latent_df.to_csv("latent_to_yaw_weights.csv", index=False)
    return yaw_latent_df


def save_effective_feature_to_yaw_weights(model, feature_cols, species_names, latent_dim):
    """
    Because encoder is linear and yaw decoder is linear, this gives an interpretable
    feature -> latent -> yaw effective weight for each species.

    Note: this does NOT include the nonlinear reconstruction decoder.
    It only explains the linear yaw path.
    """
    yaw_w = model.decoder_y.weight.detach().cpu().numpy().reshape(-1)
    rows = []

    for sp_idx, sp_name in enumerate(species_names):
        E = model.encoders[sp_idx].weight.detach().cpu().numpy()
        effective = yaw_w @ E

        for feat, weight in zip(feature_cols, effective):
            rows.append({
                "species": sp_name,
                "feature": feat,
                "effective_feature_to_yaw_weight": weight,
                "abs_weight": abs(weight),
            })

    effective_df = pd.DataFrame(rows)
    effective_df.to_csv("effective_feature_to_yaw_weights_by_species.csv", index=False)

    mean_df = (
        effective_df
        .groupby("feature", as_index=False)["abs_weight"]
        .mean()
        .sort_values("abs_weight", ascending=False)
    )
    mean_df.to_csv("mean_abs_feature_to_yaw_weights.csv", index=False)

    return effective_df, mean_df


def save_decoder_x_weights(model):
    """
    Saves nonlinear decoder weights layer by layer.
    These are less directly interpretable than the linear yaw path because ReLU is nonlinear.
    """
    for name, param in model.decoder_x.named_parameters():
        arr = param.detach().cpu().numpy()
        clean_name = name.replace(".", "_")
        pd.DataFrame(arr).to_csv(f"decoder_x_{clean_name}.csv", index=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 8
hidden_dim = 32
yaw_loss_weight = 2.0

print(f"\nUsing device: {device}")
print(f"Latent dim: {latent_dim}")
print(f"Hidden dim for nonlinear decoder: {hidden_dim}")
print(f"Yaw loss weight: {yaw_loss_weight}")

model = SpeciesEncoderAutoencoderYaw(
    input_dim=X_train.shape[1],
    latent_dim=latent_dim,
    num_species=num_species,
    hidden_dim=hidden_dim,
)

history = train_model(
    model,
    train_loader,
    test_loader,
    epochs=300,
    lr=1e-3,
    weight_decay=1e-4,
    yaw_loss_weight=yaw_loss_weight,
    device=device,
)

eval_test = evaluate_model(model, test_loader, y_scaler, device=device)

print("\n================ FINAL TEST PERFORMANCE ================")
print(f"Yaw MSE:            {eval_test['yaw_mse']:.4f}")
print(f"Yaw R2 score:       {eval_test['yaw_r2']:.4f}")
print(f"Reconstruction MSE: {eval_test['recon_mse']:.4f}")
print(f"Reconstruction R2:  {eval_test['recon_r2']:.4f}")
print("========================================================\n")

X_full = df_model[feature_cols].values.astype(np.float32)
y_full = y_scaler.transform(df_model[target_col].values.reshape(-1, 1)).ravel().astype(np.float32)
spidx_full = df_model["species_idx"].values.astype(np.int64)

full_ds = MotorDataset(X_full, y_full, spidx_full)
full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)
eval_full = evaluate_model(model, full_loader, y_scaler, device=device)

Z = eval_full["Z"]
species_arr = df_model["species"].values
clade_arr = df_model["clade"].values
wbfreq_arr = 1.0 / df_model["wblen"].values
sp_unique = np.array(species_names)

plt.figure(figsize=(7, 5))
plt.plot(history["train_total"], label="train total")
plt.plot(history["test_total"], label="test total")
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.title(f"Total loss: reconstruction + {yaw_loss_weight} * yaw")
plt.legend()
plt.tight_layout()
plt.savefig("loss_total.png")
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(history["train_recon"], label="train reconstruction")
plt.plot(history["test_recon"], label="test reconstruction")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction MSE")
plt.title("Muscle reconstruction loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_reconstruction.png")
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(history["train_yaw"], label="train yaw")
plt.plot(history["test_yaw"], label="test yaw")
plt.xlabel("Epoch")
plt.ylabel("Yaw MSE")
plt.title("Yaw prediction loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_yaw.png")
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(eval_test["y_true"], eval_test["y_pred"], alpha=0.7, s=20)
mn = min(eval_test["y_true"].min(), eval_test["y_pred"].min())
mx = max(eval_test["y_true"].max(), eval_test["y_pred"].max())
plt.plot([mn, mx], [mn, mx], "--", color="gray")
plt.xlabel("True tz")
plt.ylabel("Predicted tz")
plt.title(f"Predicted vs true yaw torque, test set, R2={eval_test['yaw_r2']:.3f}")
plt.tight_layout()
plt.savefig("yaw_pred_vs_true_species_ae.png")
plt.show()

if latent_dim >= 2:
    plt.figure(figsize=(12, 7))
    cmap = plt.cm.get_cmap("tab20", len(sp_unique))

    for i, sp in enumerate(sp_unique):
        mask = species_arr == sp
        plt.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.75, color=cmap(i), label=sp)

    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.title("Latent motor-program space by species")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig("latent_space_by_species.png")
    plt.show()

    plt.figure(figsize=(7, 6))
    for clade in np.unique(clade_arr):
        mask = clade_arr == clade
        plt.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.75, label=clade)

    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.title("Latent motor-program space by clade")
    plt.legend()
    plt.tight_layout()
    plt.savefig("latent_space_by_clade.png")
    plt.show()

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=wbfreq_arr, s=20, alpha=0.75, cmap="viridis")
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.title("Latent motor-program space by wingbeat frequency")
    plt.colorbar(sc, label="Wingbeat frequency")
    plt.tight_layout()
    plt.savefig("latent_space_wbfreq.png")
    plt.show()

encoder_df = save_species_encoder_weights(model, feature_cols, species_names, latent_dim)
yaw_latent_df = save_yaw_latent_weights(model, latent_dim)
effective_df, mean_effective_df = save_effective_feature_to_yaw_weights(
    model,
    feature_cols,
    species_names,
    latent_dim,
)
save_decoder_x_weights(model)

results_summary = pd.DataFrame([{
    "latent_dim": latent_dim,
    "hidden_dim_decoder_x": hidden_dim,
    "yaw_loss_weight": yaw_loss_weight,
    "subsample_n_per_species": clean_subsample_n,
    "min_wb_to_qualify": min_wb_to_qualify,
    "num_species": num_species,
    "total_rows": len(df_model),
    "yaw_mse_test": eval_test["yaw_mse"],
    "yaw_r2_test": eval_test["yaw_r2"],
    "reconstruction_mse_test": eval_test["recon_mse"],
    "reconstruction_r2_test": eval_test["recon_r2"],
    "tz_normalization": "within_individual_zscore",
    "encoder_type": "species_specific_linear_encoder",
    "decoder_x_type": "nonlinear_decoder_with_relu",
    "decoder_y_type": "linear_latent_to_yaw_decoder",
}])
results_summary.to_csv("species_ae_yaw_results_summary.csv", index=False)

df_model.to_csv("input_data_species_balanced_normalized.csv", index=False)

print("\nTop feature-to-yaw weights averaged across species:")
print(mean_effective_df.head(20))

print("\nSaved outputs:")
print("loss_total.png")
print("loss_reconstruction.png")
print("loss_yaw.png")
print("yaw_pred_vs_true_species_ae.png")
print("latent_space_by_species.png")
print("latent_space_by_clade.png")
print("latent_space_wbfreq.png")
print("species_encoder_weights.csv")
print("latent_to_yaw_weights.csv")
print("effective_feature_to_yaw_weights_by_species.csv")
print("mean_abs_feature_to_yaw_weights.csv")
print("decoder_x_*.csv")
print("species_ae_yaw_results_summary.csv")
print("input_data_species_balanced_normalized.csv")
