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

phase_cols = ["lax", "lba", "lsa", "ldvm", "ldlm", "rdlm", "rdvm", "rsa", "rax", "rba"] # muscle names
count_cols = [
    "lax_count", "lba_count", "lsa_count", "ldvm_count", "ldlm_count",
    "rdlm_count", "rdvm_count", "rsa_count", "rax_count", "rba_count"
] # muscle spike count

feature_cols = phase_cols + count_cols
target_col = "tz"

meta_cols = ["moth", "wb", "species", "wblen", "tz"]

missing_needed = [c for c in feature_cols + [target_col, "species"] if c not in df.columns]
if missing_needed:
    raise ValueError(f"Missing required columns: {missing_needed}")

# for missing values
df_model = df.copy()

df_model[phase_cols] = df_model[phase_cols].apply(pd.to_numeric, errors="coerce")
df_model[count_cols] = df_model[count_cols].apply(pd.to_numeric, errors="coerce")
df_model[target_col] = pd.to_numeric(df_model[target_col], errors="coerce")

# drop rows with missing target/species
df_model = df_model.dropna(subset=[target_col, "species"]).copy()

df_model[phase_cols] = df_model[phase_cols].fillna(df_model[phase_cols].mean())
df_model[count_cols] = df_model[count_cols].fillna(0)

print("\nRemaining NaNs:")
print(df_model[feature_cols + [target_col]].isna().sum())

# drop columns with only one value for training
min_value = 2
species_count = df_model["species"].value_counts()
species_to_keep = species_count[species_count >= min_value].index.tolist()
df_model = df_model[df_model["species"].isin(species_to_keep)].copy()

# one hot encoding
species_names = sorted(df_model["species"].unique())
species_to_idx = {s: i for i, s in enumerate(species_names)}
idx_to_species = {i: s for s, i in species_to_idx.items()}

df_model["species_idx"] = df_model["species"].map(species_to_idx)
num_species = len(species_names)

X_df = df_model[feature_cols].copy()
y = df_model[target_col].values
species_idx = df_model["species_idx"].values

X_train_df, X_test_df, y_train, y_test, species_train, species_test = train_test_split(
    X_df,
    y,
    species_idx,
    test_size=0.2,
    random_state=42,
    stratify=species_idx
)

# z score normalize yaw
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train_df)
X_test = x_scaler.transform(X_test_df)

# z-score yaw
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

print("\nTrain shape:", X_train.shape)
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

print("\nFinal performance:")
print("Yaw MSE:", eval_out["yaw_mse"])
print("Yaw R2 :", eval_out["yaw_r2"])


plt.figure(figsize=(7, 5))
plt.plot(history["train_yaw"], label="train yaw")
plt.plot(history["test_yaw"], label="test yaw")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Yaw loss (latent_dim=2)")
plt.legend()
plt.tight_layout()
plt.savefig("all10_big_yaw_loss_dim_2.png")
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(eval_out["y_true"], eval_out["y_pred"], alpha=0.6, s=12)
mn = min(eval_out["y_true"].min(), eval_out["y_pred"].min())
mx = max(eval_out["y_true"].max(), eval_out["y_pred"].max())
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("True tz")
plt.ylabel("Predicted tz")
plt.title("Predicted vs true yaw torque (latent_dim=2)")
plt.tight_layout()
plt.savefig("all10_big_yaw_pred_vs_true_dim_2.png")
plt.show()

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
plt.title("Shared latent space (latent_dim=2)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("all10_big_yaw_shared_latent_dim_2.png")
plt.show()

effective_yaw_weights = get_effective_yaw_weights(model, feature_cols, species_names)

print("\nTop yaw-related features by species:")
for species in species_names:
    print("\n" + "=" * 60)
    print(species)
    print("=" * 60)
    print(effective_yaw_weights[species].head(20))

yaw_weight_df = pd.concat(
    [effective_yaw_weights[s].rename(s) for s in species_names],
    axis=1
)
yaw_weight_df["mean_abs_weight"] = yaw_weight_df[species_names].abs().mean(axis=1)

print("\nTop yaw-related features across species:")
print(yaw_weight_df["mean_abs_weight"].sort_values(ascending=False).head(20))

muscle_importance = aggregate_by_muscle(yaw_weight_df["mean_abs_weight"])
print("\nTop muscles across species:")
print(muscle_importance.head(20))

yaw_weight_df.to_csv("all10_big_yaw_feature_weights_dim_2.csv")

summary_df = pd.DataFrame([{
    "latent_dim": 2,
    "yaw_mse": eval_out["yaw_mse"],
    "yaw_r2": eval_out["yaw_r2"]
}])
summary_df.to_csv("all10_big_multi_encoder_yaw_results_dim_2.csv", index=False)

df_model.to_csv("all10_big_input_data.csv", index=False)