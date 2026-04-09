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
target_col = "tz" # yaw torque

required_cols = feature_cols + [target_col, "species", "moth", "wb", "wblen"]
missing_needed = [c for c in required_cols if c not in df.columns]
if missing_needed:
    raise ValueError(f"Missing required columns: {missing_needed}")

df_model = df.copy()

df_model[phase_cols] = df_model[phase_cols].apply(pd.to_numeric, errors="coerce")
df_model[count_cols] = df_model[count_cols].apply(pd.to_numeric, errors="coerce")
df_model[target_col] = pd.to_numeric(df_model[target_col], errors="coerce")

df_model = df_model.dropna(subset=[target_col, "species", "moth"]).copy()

# get all wingbeats with 10 muscles represented
complete_mask = np.ones(len(df_model), dtype=bool)
for p_col, c_col in zip(phase_cols, count_cols):
    complete_mask &= df_model[p_col].notna()
    complete_mask &= df_model[c_col].fillna(0) > 0

df_model = df_model[complete_mask].copy()

print("\nRows remaining after requiring all 10 muscles present:", len(df_model))

# fill in phase values and count values if missing
df_model[phase_cols] = df_model[phase_cols].fillna(df_model[phase_cols].mean())
df_model[count_cols] = df_model[count_cols].fillna(0)

print("\nRemaining NaNs:")
print(df_model[feature_cols + [target_col]].isna().sum())

wb_counts = (
    df_model.groupby("species")
    .size()
    .sort_values()
)

# can also pick to do static value like 20
clean_subsample_n = 10

species_to_keep = wb_counts[wb_counts >= clean_subsample_n].index.tolist()
df_model = df_model[df_model["species"].isin(species_to_keep)].copy()

rng_seed = 42
df_model = (
    df_model.groupby("species", group_keys=False)
    .sample(n=clean_subsample_n, random_state=rng_seed)
    .reset_index(drop=True)
)

print(df_model["species"].value_counts().sort_index())

assert "species" in df_model.columns, "species column lost after subsampling"

# normalize counts and timing
df_model[count_cols] = df_model[count_cols] / 10.0
df_model[count_cols] = df_model[count_cols].clip(lower=0.0, upper=1.0)

df_model[phase_cols] = (df_model[phase_cols] + 1.0) / 2.0
df_model[phase_cols] = df_model[phase_cols].clip(lower=0.0, upper=1.0)

# one hot encode individual
individual_names = sorted(df_model["moth"].astype(str).unique())
individual_to_idx = {m: i for i, m in enumerate(individual_names)}
idx_to_individual = {i: m for m, i in individual_to_idx.items()}

df_model["individual_idx"] = df_model["moth"].astype(str).map(individual_to_idx)
num_individuals = len(individual_names)

X_df = df_model[feature_cols].copy()
y = df_model[target_col].values
individual_idx = df_model["individual_idx"].values

species_labels = df_model["species"].values
clade_labels = df_model["clade"].values
wbfreq_values = (1.0 / df_model["wblen"].values) if "wblen" in df_model.columns else np.full(len(df_model), np.nan)

X_train_df, X_test_df, y_train, y_test, ind_train, ind_test, species_train_labels, species_test_labels, clade_train, clade_test, wbfreq_train, wbfreq_test = train_test_split(
    X_df,
    y,
    individual_idx,
    species_labels,
    clade_labels,
    wbfreq_values,
    test_size=0.2,
    random_state=42,
    stratify=species_labels
)

# removed normalization of X
X_train = X_train_df.values.astype(np.float32)
X_test = X_test_df.values.astype(np.float32)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

class MotorDataset(Dataset):
    def __init__(self, X, y, individual_idx, species_labels=None, clade_labels=None, wbfreq=None):
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

train_ds = MotorDataset(
    X_train, y_train, ind_train,
    species_labels=species_train_labels,
    clade_labels=clade_train,
    wbfreq=wbfreq_train
)

test_ds = MotorDataset(
    X_test, y_test, ind_test,
    species_labels=species_test_labels,
    clade_labels=clade_test,
    wbfreq=wbfreq_test
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

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

        for xb, yb, ib in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ib = ib.to(device)

            optimizer.zero_grad()
            y_hat, z = model(xb, ib)
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
                xb = xb.to(device)
                yb = yb.to(device)
                ib = ib.to(device)

                y_hat, z = model(xb, ib)
                loss = loss_fn(y_hat, yb)
                total_test += loss.item() * xb.size(0)

        avg_test = total_test / len(test_loader.dataset)
        history["test_yaw"].append(avg_test)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Train yaw {avg_train:.4f} | Test yaw {avg_test:.4f}")

    return history

def evaluate_model(model, loader, y_scaler, species_labels, clade_labels, wbfreq_values, device="cpu"):
    model.eval()

    y_true_all = []
    y_pred_all = []
    z_all = []
    individual_all = []

    with torch.no_grad():
        for xb, yb, ib in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ib = ib.to(device)

            y_hat, z = model(xb, ib)

            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(y_hat.cpu().numpy())
            z_all.append(z.cpu().numpy())
            individual_all.append(ib.cpu().numpy())

    y_true = np.vstack(y_true_all).ravel()
    y_pred = np.vstack(y_pred_all).ravel()
    Z = np.vstack(z_all)
    individual_idx_out = np.concatenate(individual_all)

    y_true_unscaled = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_unscaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    yaw_mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    yaw_r2 = r2_score(y_true_unscaled, y_pred_unscaled)

    return {
        "y_true": y_true_unscaled,
        "y_pred": y_pred_unscaled,
        "Z": Z,
        "individual_idx": individual_idx_out,
        "species_labels": np.array(species_labels),
        "clade_labels": np.array(clade_labels),
        "wbfreq": np.array(wbfreq_values),
        "yaw_mse": yaw_mse,
        "yaw_r2": yaw_r2
    }

def get_effective_yaw_weights(model, feature_cols, individual_names):
    yaw_w = model.decoder_y.weight.detach().cpu().numpy().reshape(-1)
    out = {}

    for i, individual in enumerate(individual_names):
        E = model.encoders[i].weight.detach().cpu().numpy()
        eff = yaw_w @ E
        out[individual] = pd.Series(eff, index=feature_cols).sort_values(key=np.abs, ascending=False)

    return out

def aggregate_by_muscle(weight_series):
    muscle_scores = {}
    for feat, val in weight_series.items():
        muscle = feat.replace("_count", "")
        muscle_scores[muscle] = muscle_scores.get(muscle, 0.0) + abs(val)
    return pd.Series(muscle_scores).sort_values(ascending=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing device:", device)

latent_dim = 2

model = MultiEncoderYawModel(
    input_dim=X_train.shape[1],
    latent_dim=latent_dim,
    num_individuals=num_individuals
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
    species_test_labels,
    clade_test,
    wbfreq_test,
    device=device
)

print("\nFinal performance:")
print("Yaw MSE:", eval_out["yaw_mse"])
print("Yaw R2 :", eval_out["yaw_r2"])

# loss plot
plt.figure(figsize=(7, 5))
plt.plot(history["train_yaw"], label="train yaw")
plt.plot(history["test_yaw"], label="test yaw")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Yaw loss (latent_dim=2)")
plt.legend()
plt.tight_layout()
plt.savefig("subsampled_all10_big_yaw_loss_dim_2.png")
plt.show()

# pred vs true
plt.figure(figsize=(6, 6))
plt.scatter(eval_out["y_true"], eval_out["y_pred"], alpha=0.6, s=12)
mn = min(eval_out["y_true"].min(), eval_out["y_pred"].min())
mx = max(eval_out["y_true"].max(), eval_out["y_pred"].max())
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("True tz")
plt.ylabel("Predicted tz")
plt.title("Predicted vs true yaw torque (latent_dim=2)")
plt.tight_layout()
plt.savefig("subsampled_all10_big_yaw_pred_vs_true_dim_2.png")
plt.show()

# plot by species
plt.figure(figsize=(10, 8))

species_unique = np.unique(eval_out["species_labels"])

for species in species_unique:
    mask = eval_out["species_labels"] == species
    
    plt.scatter(
        eval_out["Z"][mask, 0],
        eval_out["Z"][mask, 1],
        s=16,
        alpha=0.7,
        label=species
    )

plt.xlabel("Latent 1")
plt.ylabel("Latent 2")
plt.title("Shared latent space colored by species")

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=7
)

plt.tight_layout()
plt.savefig("subsampled_all10_big_yaw_shared_latent_by_species_dim_2.png")
plt.show()

# plot by clade
plt.figure(figsize=(8, 6))
for clade in np.unique(eval_out["clade_labels"]):
    mask = eval_out["clade_labels"] == clade
    plt.scatter(
        eval_out["Z"][mask, 0],
        eval_out["Z"][mask, 1],
        s=16,
        alpha=0.65,
        label=clade
    )

plt.xlabel("Latent 1")
plt.ylabel("Latent 2")
plt.title("Shared latent space colored by clade")
plt.legend()
plt.tight_layout()
plt.savefig("subsampled_all10_big_yaw_shared_latent_by_clade_dim_2.png")
plt.show()

# plot by wbfreq
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    eval_out["Z"][:, 0],
    eval_out["Z"][:, 1],
    c=eval_out["wbfreq"],
    s=16,
    alpha=0.7
)
plt.xlabel("Latent 1")
plt.ylabel("Latent 2")
plt.title("Shared latent space colored by wingbeat frequency")
plt.colorbar(sc, label="Wingbeat frequency")
plt.tight_layout()
plt.savefig("subsampled_all10_big_yaw_shared_latent_by_wbfreq_dim_2.png")
plt.show()

effective_yaw_weights = get_effective_yaw_weights(model, feature_cols, individual_names)

print("\nTop yaw-related features by individual:")
for individual in individual_names:
    print("\n" + "=" * 60)
    print(individual)
    print("=" * 60)
    print(effective_yaw_weights[individual].head(20))

yaw_weight_df = pd.concat(
    [effective_yaw_weights[i].rename(i) for i in individual_names],
    axis=1
)
yaw_weight_df["mean_abs_weight"] = yaw_weight_df[individual_names].abs().mean(axis=1)

print("\nTop yaw-related features across individuals:")
print(yaw_weight_df["mean_abs_weight"].sort_values(ascending=False).head(20))

muscle_importance = aggregate_by_muscle(yaw_weight_df["mean_abs_weight"])
print("\nTop muscles across individuals:")
print(muscle_importance.head(20))

yaw_weight_df.to_csv("all10_big_yaw_feature_weights_dim_2.csv")

summary_df = pd.DataFrame([{
    "latent_dim": 2,
    "yaw_mse": eval_out["yaw_mse"],
    "yaw_r2": eval_out["yaw_r2"],
    "subsample_n_per_species": clean_subsample_n,
    "num_individuals": num_individuals
}])
summary_df.to_csv("subsampled_all10_big_multi_encoder_yaw_results_dim_2.csv", index=False)

df_model.to_csv("subsampled_all10_big_input_data_balanced_normalized.csv", index=False)