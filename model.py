"""
New Architecture:
  Shared linear encoder
    Single weight matrix readable across all species
    One-hot columns give per-species latent mean offset (interpretable)

  Shared latent space

  Tiny interaction layer 

  Linear yaw head

  Nonlinear decoder_x 

  Though decoder is nonlinear, there is intepretablity by using SHAP

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print(
        "WARNING: shap not installed. Run  pip install shap  to enable SHAP "
        "analysis. The model will still train and all other outputs will be saved."
    )

LATENT_DIM          = 10
INTERACTION_UNITS   = 12     # slightly wider — more latent signal available now
RECON_HIDDEN_DIM    = 64     # nonlinear decoder_x hidden size
EPOCHS              = 400
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
BATCH_SIZE          = 256    # larger batch fine with full dataset (55k rows)
TEST_SIZE           = 0.2

MIN_WB_TO_QUALIFY   = 100    
SUBSAMPLE_N         = None  

YAW_WEIGHT_UPPER    = 50.0
YAW_WEIGHT_BALANCED = 2.0

PHASE_COLS = ["lax", "lba", "lsa", "ldvm", "ldlm",
              "rdlm", "rdvm", "rsa", "rax", "rba"]
COUNT_COLS = ["lax_count", "lba_count", "lsa_count", "ldvm_count", "ldlm_count",
              "rdlm_count", "rdvm_count", "rsa_count", "rax_count", "rba_count"]
FEATURE_COLS = PHASE_COLS + COUNT_COLS
TARGET_COL   = "tz"

CLADE_DICT = {
    "Actias luna": "silkmoth",        "Hyalophora cecropia": "silkmoth",
    "Automeris io": "silkmoth",       "Saturnia walterorum": "silkmoth",
    "Syssphinx montana": "silkmoth",  "Syssphinx hubbardi": "silkmoth",
    "Antheraea polyphemus": "silkmoth","Citheronia regalis": "silkmoth",
    "Ceratomia amyntor": "hawkmoth",  "Acherontia atropos": "hawkmoth",
    "Manduca sexta": "hawkmoth",      "Proserpinus terlooii": "hawkmoth",
    "Hyles lineata": "hawkmoth",      "Citheronia splendens": "silkmoth",
    "Hyalophora columbia": "silkmoth","Automeris randa": "silkmoth",
    "Coloradia doris": "silkmoth",    "Hemaris diffinis": "hawkmoth",
}


def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    df["clade"] = df["species"].map(CLADE_DICT)

    required = FEATURE_COLS + [TARGET_COL, "species", "moth", "wb", "wblen"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, "species", "moth", "wblen"]).copy()

    # ── Quick correlation check: individual muscle vs tz ─────────────────
    print("\n── Feature–tz Pearson correlations (raw, before normalisation) ──")
    corrs = (
        df[FEATURE_COLS + [TARGET_COL]]
        .corr()[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(key=abs, ascending=False)
    )
    print(corrs.round(3).to_string())
    print(
        "\nIf most correlations are weak (<0.15) but R² ends up high, the "
        "interaction layer is doing real work capturing muscle synergies.\n"
    )

    counts = df.groupby("species").size()
    keep   = counts[counts >= MIN_WB_TO_QUALIFY].index.tolist()
    df     = df[df["species"].isin(keep)].copy()
    if SUBSAMPLE_N is not None:
        df = (
            df.groupby("species", group_keys=False)
            .sample(n=SUBSAMPLE_N, replace=False)
            .reset_index(drop=True)
        )
    else:
        df = df.reset_index(drop=True)
    print("\nRows per species after filtering:")
    print(df["species"].value_counts().sort_index())


    tz_mean = df[TARGET_COL].mean()
    tz_std  = df[TARGET_COL].std()
    if tz_std == 0.0:
        tz_std = 1.0
    df[TARGET_COL] = (df[TARGET_COL] - tz_mean) / tz_std
    print(f"\nGlobal tz z-score: mean={tz_mean:.4f}, std={tz_std:.4f}")
    print(f"  After normalisation — mean: {df[TARGET_COL].mean():.4f}, "
          f"std: {df[TARGET_COL].std():.4f}")

    df[COUNT_COLS]  = (df[COUNT_COLS] / 10.0).clip(0.0, 1.0)
    df[PHASE_COLS]  = ((df[PHASE_COLS] + 1.0) / 2.0).clip(0.0, 1.0)

    species_names  = sorted(df["species"].astype(str).unique())
    species_to_idx = {sp: i for i, sp in enumerate(species_names)}
    df["species_idx"] = df["species"].astype(str).map(species_to_idx)

    print(f"\nSpecies ({len(species_names)}):")
    for sp, idx in species_to_idx.items():
        print(f"  {idx}: {sp}")

    return df, species_names, species_to_idx


def build_splits(df):
    X     = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[TARGET_COL].values.astype(np.float32)
    sp    = df["species_idx"].values.astype(np.int64)
    splab = df["species"].values
    clade = df["clade"].values
    wbf   = (1.0 / df["wblen"].values).astype(np.float32)

    (X_tr, X_te,
     y_tr_r, y_te_r,
     sp_tr, sp_te,
     spl_tr, spl_te,
     cl_tr, cl_te,
     wf_tr, wf_te) = train_test_split(
        X, y_raw, sp, splab, clade, wbf,
        test_size=TEST_SIZE, stratify=splab,
    )

    y_scaler = StandardScaler()
    y_tr = y_scaler.fit_transform(y_tr_r.reshape(-1, 1)).ravel().astype(np.float32)
    y_te = y_scaler.transform(y_te_r.reshape(-1, 1)).ravel().astype(np.float32)

    return (X_tr, X_te,
            y_tr, y_te,
            sp_tr, sp_te,
            spl_tr, spl_te,
            cl_tr, cl_te,
            wf_tr, wf_te,
            y_scaler)


class MotorDataset(Dataset):
    def __init__(self, X, y, sp_idx):
        self.X  = torch.tensor(X,      dtype=torch.float32)
        self.y  = torch.tensor(y,      dtype=torch.float32).view(-1, 1)
        self.sp = torch.tensor(sp_idx, dtype=torch.long)

    def __len__(self):              return len(self.X)
    def __getitem__(self, i):       return self.X[i], self.y[i], self.sp[i]


class MotorProgramModel(nn.Module):

    def __init__(self, input_dim, latent_dim, num_species,
                 interaction_units=12, recon_hidden=64):
        super().__init__()
        self.latent_dim   = latent_dim
        self.input_dim    = input_dim
        self.num_species  = num_species

        self.encoder = nn.Linear(input_dim + num_species, latent_dim)

        self.interaction = nn.Sequential(
            nn.Linear(latent_dim, interaction_units),
            nn.ReLU(),
        )

        self.yaw_head = nn.Linear(interaction_units, 1)

        self.decoder_x = nn.Sequential(
            nn.Linear(latent_dim, recon_hidden),
            nn.ReLU(),
            nn.Linear(recon_hidden, recon_hidden),
            nn.ReLU(),
            nn.Linear(recon_hidden, input_dim),
        )

    def _make_one_hot(self, sp_idx, device):
        oh = torch.zeros(sp_idx.shape[0], self.num_species, device=device)
        oh.scatter_(1, sp_idx.view(-1, 1), 1.0)
        return oh

    def encode(self, x, sp_idx):
        """Concatenate one-hot species indicator and run shared linear encoder."""
        oh = self._make_one_hot(sp_idx, x.device)
        return self.encoder(torch.cat([x, oh], dim=1))

    def predict_yaw(self, z):
        """Latent → interaction layer → yaw. Used separately for SHAP."""
        return self.yaw_head(self.interaction(z))

    def forward(self, x, sp_idx):
        z     = self.encode(x, sp_idx)
        y_hat = self.predict_yaw(z)
        x_hat = self.decoder_x(z)
        return x_hat, y_hat, z

def train_model(model, train_loader, test_loader,
                epochs, lr, weight_decay, yaw_loss_weight, device="cpu"):
    model.to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    hist    = {k: [] for k in
               ["train_total","train_recon","train_yaw",
                "test_total", "test_recon", "test_yaw"]}

    for epoch in range(epochs):
        model.train()
        tr = dict(total=0., recon=0., yaw=0.)
        for xb, yb, sb in train_loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            opt.zero_grad()
            x_hat, y_hat, _ = model(xb, sb)
            rl = loss_fn(x_hat, xb)
            yl = loss_fn(y_hat, yb)
            loss = rl + yaw_loss_weight * yl
            loss.backward()
            opt.step()
            n = xb.size(0)
            tr["total"] += loss.item() * n
            tr["recon"] += rl.item()   * n
            tr["yaw"]   += yl.item()   * n

        N_tr = len(train_loader.dataset)
        for k in tr: tr[k] /= N_tr

        model.eval()
        te = dict(total=0., recon=0., yaw=0.)
        with torch.no_grad():
            for xb, yb, sb in test_loader:
                xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
                x_hat, y_hat, _ = model(xb, sb)
                rl = loss_fn(x_hat, xb)
                yl = loss_fn(y_hat, yb)
                loss = rl + yaw_loss_weight * yl
                n = xb.size(0)
                te["total"] += loss.item() * n
                te["recon"] += rl.item()   * n
                te["yaw"]   += yl.item()   * n

        N_te = len(test_loader.dataset)
        for k in te: te[k] /= N_te

        for k in ["total","recon","yaw"]:
            hist[f"train_{k}"].append(tr[k])
            hist[f"test_{k}"].append(te[k])

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch:3d} | "
                f"Train total={tr['total']:.4f} recon={tr['recon']:.4f} "
                f"yaw={tr['yaw']:.4f} || "
                f"Test  total={te['total']:.4f} recon={te['recon']:.4f} "
                f"yaw={te['yaw']:.4f}"
            )
    return hist

def evaluate_model(model, loader, y_scaler, device="cpu"):
    model.eval()
    yt, yp, xt, xr, zs, sps = [], [], [], [], [], []
    with torch.no_grad():
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            x_hat, y_hat, z = model(xb, sb)
            yt.append(yb.cpu().numpy());   yp.append(y_hat.cpu().numpy())
            xt.append(xb.cpu().numpy());   xr.append(x_hat.cpu().numpy())
            zs.append(z.cpu().numpy());    sps.append(sb.cpu().numpy())

    y_true_s = np.vstack(yt).ravel()
    y_pred_s = np.vstack(yp).ravel()
    y_true   = y_scaler.inverse_transform(y_true_s.reshape(-1,1)).ravel()
    y_pred   = y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
    X_true   = np.vstack(xt);  X_recon = np.vstack(xr)
    Z        = np.vstack(zs);  sp_out  = np.concatenate(sps)

    return dict(
        y_true=y_true, y_pred=y_pred,
        X_true=X_true, X_recon=X_recon,
        Z=Z, species_idx=sp_out,
        yaw_mse=mean_squared_error(y_true, y_pred),
        yaw_r2 =r2_score(y_true, y_pred),
        recon_mse=mean_squared_error(X_true, X_recon),
        recon_r2 =r2_score(X_true, X_recon, multioutput="variance_weighted"),
    )

def run_shap_analysis(model, Z_all, sp_idx_all, species_arr,
                      species_names, tag="", device="cpu"):

    if not SHAP_AVAILABLE:
        print("  Skipping SHAP (not installed).")
        return None, None, None

    print("\n  Running SHAP on interaction_layer + yaw_head …")

    model.eval()
    model.to("cpu")

    def yaw_from_latent(z_np):
        with torch.no_grad():
            z_t = torch.tensor(z_np, dtype=torch.float32)
            h   = model.interaction(z_t)
            y   = model.yaw_head(h)
        return y.numpy()

    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(Z_all), size=min(100, len(Z_all)), replace=False)
    background = Z_all[idx]

    explainer   = shap.KernelExplainer(yaw_from_latent, background)
    shap_values = explainer.shap_values(Z_all, nsamples=200)

    latent_shap_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_latent_{i+1}" for i in range(LATENT_DIM)]
    )
    latent_shap_df["species"] = species_arr
    latent_shap_df.to_csv(f"shap_latent{tag}.csv", index=False)
    print(f"    Saved shap_latent{tag}.csv")

    W_full = model.encoder.weight.detach().cpu().numpy()  
    W_mus  = W_full[:, :len(FEATURE_COLS)] 


    muscle_shap_df = pd.DataFrame(muscle_shap_matrix, columns=FEATURE_COLS)
    muscle_shap_df["species"] = species_arr
    muscle_shap_df.to_csv(f"shap_muscle_per_wingbeat{tag}.csv", index=False)
    print(f"    Saved shap_muscle_per_wingbeat{tag}.csv")

    mean_muscle_shap = (
        muscle_shap_df[FEATURE_COLS]
        .abs()
        .mean()
        .reset_index()
        .rename(columns={"index": "feature", 0: "mean_abs_shap"})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    mean_muscle_shap.to_csv(f"shap_mean_abs_muscle{tag}.csv", index=False)
    print(f"    Saved shap_mean_abs_muscle{tag}.csv")

    return latent_shap_df, muscle_shap_df, mean_muscle_shap

def save_encoder_weights(model, species_names, tag=""):
    """
    Save shared encoder weights split into:
      - muscle_weights.csv : W_enc[:, :n_muscle_feats]  — universal mapping
      - species_offsets.csv: W_enc[:, n_muscle_feats:]  — per-species latent offsets

    This replaces the per-species encoder CSVs from the old architecture.
    The muscle_weights block is identical for all species; species_offsets
    show how each species shifts the latent mean.
    """
    W_full = model.encoder.weight.detach().cpu().numpy()  
    b      = model.encoder.bias.detach().cpu().numpy()  
    n_mus  = len(FEATURE_COLS)
    n_sp   = len(species_names)

    W_mus = W_full[:, :n_mus]   
    rows = []
    for li in range(LATENT_DIM):
        for fi, feat in enumerate(FEATURE_COLS):
            rows.append(dict(
                latent_dim=f"latent_{li+1}",
                feature=feat,
                encoder_weight=W_mus[li, fi],
                encoder_bias=b[li],
            ))
    mus_df = pd.DataFrame(rows)
    mus_df.to_csv(f"encoder_muscle_weights{tag}.csv", index=False)
    print(f"    Saved encoder_muscle_weights{tag}.csv  (universal muscle→latent)")

    W_sp = W_full[:, n_mus:]  
    sp_offset_df = pd.DataFrame(
        W_sp.T,
        index=species_names,
        columns=[f"latent_{li+1}" for li in range(LATENT_DIM)],
    )
    sp_offset_df.index.name = "species"
    sp_offset_df.to_csv(f"encoder_species_offsets{tag}.csv")
    print(f"    Saved encoder_species_offsets{tag}.csv  (per-species latent offset)")

    return mus_df, sp_offset_df


def save_interaction_weights(model, tag=""):
    """Save interaction layer and yaw head weights for reference."""
    w_int = model.interaction[0].weight.detach().cpu().numpy()
    b_int = model.interaction[0].bias.detach().cpu().numpy()
    w_yaw = model.yaw_head.weight.detach().cpu().numpy().reshape(-1)
    b_yaw = model.yaw_head.bias.detach().cpu().numpy().item()

    int_df = pd.DataFrame(
        w_int,
        index=[f"unit_{i+1}" for i in range(w_int.shape[0])],
        columns=[f"latent_{j+1}" for j in range(w_int.shape[1])],
    )
    int_df.to_csv(f"interaction_layer_weights{tag}.csv")

    yaw_df = pd.DataFrame({
        "unit": [f"unit_{i+1}" for i in range(len(w_yaw))],
        "yaw_weight": w_yaw,
        "yaw_bias": b_yaw,
    })
    yaw_df.to_csv(f"yaw_head_weights{tag}.csv", index=False)
    print(f"    Saved interaction_layer_weights{tag}.csv, yaw_head_weights{tag}.csv")
    return int_df, yaw_df


def plot_losses(hist, yaw_weight, tag="", title_prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, lbl in zip(axes,
                             ["total","recon","yaw"],
                             ["Total loss","Reconstruction MSE","Yaw MSE"]):
        ax.plot(hist[f"train_{key}"], label="train")
        ax.plot(hist[f"test_{key}"],  label="test")
        ax.set_xlabel("Epoch"); ax.set_ylabel(lbl); ax.legend()
    axes[0].set_title(f"{title_prefix}Total (recon + {yaw_weight}×yaw)")
    axes[1].set_title(f"{title_prefix}Muscle reconstruction")
    axes[2].set_title(f"{title_prefix}Yaw prediction")
    plt.tight_layout()
    plt.savefig(f"loss_curves{tag}.png", dpi=150); plt.show()
    print(f"    Saved loss_curves{tag}.png")


def plot_pred_vs_true(res, tag="", title_extra=""):
    r2   = res["yaw_r2"]
    yt, yp = res["y_true"], res["y_pred"]
    mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    plt.figure(figsize=(5,5))
    plt.scatter(yt, yp, alpha=0.55, s=16)
    plt.plot([mn,mx],[mn,mx],"--",color="gray")
    plt.xlabel("True tz"); plt.ylabel("Predicted tz")
    plt.title(f"{title_extra}\nR²={r2:.3f}")
    plt.tight_layout()
    plt.savefig(f"yaw_pred_vs_true{tag}.png", dpi=150); plt.show()
    print(f"    Saved yaw_pred_vs_true{tag}.png")


def plot_latent_space(Z, sp_arr, clade_arr, wbf_arr, species_names,
                      tag="", title_prefix=""):
    cmap = cm.get_cmap("tab20", len(species_names))

    # By species
    plt.figure(figsize=(10,6))
    for i, sp in enumerate(species_names):
        mask = sp_arr == sp
        if mask.any():
            plt.scatter(Z[mask,0], Z[mask,1], s=16, alpha=0.75,
                        color=cmap(i), label=sp)
    plt.xlabel("Latent 1"); plt.ylabel("Latent 2")
    plt.title(f"{title_prefix}Latent space — by species")
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"latent_species{tag}.png", dpi=150); plt.show()

    # By clade
    plt.figure(figsize=(6,5))
    for clade in np.unique(clade_arr):
        mask = clade_arr == clade
        plt.scatter(Z[mask,0], Z[mask,1], s=16, alpha=0.75, label=clade)
    plt.xlabel("Latent 1"); plt.ylabel("Latent 2")
    plt.title(f"{title_prefix}Latent space — by clade")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"latent_clade{tag}.png", dpi=150); plt.show()

    # By wingbeat frequency
    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z[:,0], Z[:,1], c=wbf_arr, s=16, alpha=0.75, cmap="viridis")
    plt.colorbar(sc, label="Wingbeat frequency (Hz)")
    plt.xlabel("Latent 1"); plt.ylabel("Latent 2")
    plt.title(f"{title_prefix}Latent space — by wingbeat frequency")
    plt.tight_layout()
    plt.savefig(f"latent_wbfreq{tag}.png", dpi=150); plt.show()
    print(f"    Saved latent space plots (tag={tag!r})")


def plot_shap_summary(mean_muscle_shap, tag="", title_prefix=""):
    if mean_muscle_shap is None:
        return
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(
        mean_muscle_shap["feature"][::-1],
        mean_muscle_shap["mean_abs_shap"][::-1],
    )
    ax.set_xlabel("Mean |SHAP| (chained muscle → yaw)")
    ax.set_title(f"{title_prefix}Muscle importance for yaw torque (SHAP)")
    plt.tight_layout()
    plt.savefig(f"shap_muscle_importance{tag}.png", dpi=150); plt.show()
    print(f"    Saved shap_muscle_importance{tag}.png")


def plot_shap_by_species(muscle_shap_df, species_names, tag="", title_prefix=""):
    """Heatmap: mean |SHAP| per muscle per species."""
    if muscle_shap_df is None:
        return
    feat_cols = [c for c in muscle_shap_df.columns if c != "species"]
    heat = (
        muscle_shap_df.groupby("species")[feat_cols]
        .apply(lambda df: df.abs().mean())
        .reindex(species_names)
    )
    fig, ax = plt.subplots(figsize=(14, max(5, len(species_names) * 0.5)))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(feat_cols)));  ax.set_xticklabels(feat_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(species_names))); ax.set_yticklabels(species_names)
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    ax.set_title(f"{title_prefix}Muscle → yaw SHAP by species")
    plt.tight_layout()
    plt.savefig(f"shap_by_species{tag}.png", dpi=150); plt.show()
    print(f"    Saved shap_by_species{tag}.png")


def plot_shap_latent(latent_shap_df, tag="", title_prefix=""):
    """Scatter of SHAP latent 1 vs SHAP latent 2 coloured by species."""
    if latent_shap_df is None:
        return
    species_list = latent_shap_df["species"].values
    unique_sp    = sorted(set(species_list))
    cmap         = cm.get_cmap("tab20", len(unique_sp))
    sp_color_map = {sp: cmap(i) for i, sp in enumerate(unique_sp)}

    plt.figure(figsize=(7,6))
    for sp in unique_sp:
        mask = species_list == sp
        plt.scatter(
            latent_shap_df.loc[mask,"shap_latent_1"],
            latent_shap_df.loc[mask,"shap_latent_2"],
            s=14, alpha=0.6, color=sp_color_map[sp], label=sp,
        )
    plt.xlabel("SHAP value — Latent 1")
    plt.ylabel("SHAP value — Latent 2")
    plt.title(f"{title_prefix}Latent SHAP values by species")
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"shap_latent_scatter{tag}.png", dpi=150); plt.show()
    print(f"    Saved shap_latent_scatter{tag}.png")


def plot_r2_comparison(r2_ub, r2_bal):
    labels = [
        f"Upper bound\n(λ={YAW_WEIGHT_UPPER})",
        f"Balanced\n(λ={YAW_WEIGHT_BALANCED})",
    ]
    plt.figure(figsize=(5,4))
    bars = plt.bar(labels, [r2_ub, r2_bal], color=["#2196F3","#4CAF50"], width=0.4)
    for bar, v in zip(bars, [r2_ub, r2_bal]):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11)
    plt.ylim(0, 1.05)
    plt.ylabel("Yaw R² (test set)")
    plt.title("R² ceiling vs balanced loss")
    plt.tight_layout()
    plt.savefig("r2_comparison.png", dpi=150); plt.show()
    print("    Saved r2_comparison.png")


def run_phase(tag, title, yaw_weight,
              X_tr, X_te, y_tr, y_te,
              sp_tr, sp_te, spl_tr, spl_te,
              cl_tr, cl_te, wf_tr, wf_te,
              y_scaler, species_names, df_model, device):

    print(f"\n{'='*64}")
    print(f"  {title}  (yaw_loss_weight={yaw_weight})")
    print(f"{'='*64}")

    model = MotorProgramModel(
        input_dim=X_tr.shape[1],
        latent_dim=LATENT_DIM,
        num_species=len(species_names),
        interaction_units=INTERACTION_UNITS,
        recon_hidden=RECON_HIDDEN_DIM,
    )

    tr_ds = MotorDataset(X_tr, y_tr, sp_tr)
    te_ds = MotorDataset(X_te, y_te, sp_te)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    hist = train_model(model, tr_ld, te_ld,
                       epochs=EPOCHS, lr=LR,
                       weight_decay=WEIGHT_DECAY,
                       yaw_loss_weight=yaw_weight,
                       device=device)

    eval_test = evaluate_model(model, te_ld, y_scaler, device=device)
    print(f"\n── {title} TEST RESULTS ──")
    print(f"  Yaw R²:               {eval_test['yaw_r2']:.4f}")
    print(f"  Yaw MSE:              {eval_test['yaw_mse']:.4f}")
    print(f"  Reconstruction MSE:   {eval_test['recon_mse']:.4f}")
    print(f"  Reconstruction R²:    {eval_test['recon_r2']:.4f}")

    # Full-dataset pass for latent space & SHAP
    X_full    = df_model[FEATURE_COLS].values.astype(np.float32)
    y_full_s  = y_scaler.transform(
        df_model[TARGET_COL].values.reshape(-1,1)
    ).ravel().astype(np.float32)
    sp_full   = df_model["species_idx"].values.astype(np.int64)

    full_ds = MotorDataset(X_full, y_full_s, sp_full)
    full_ld = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False)
    eval_full = evaluate_model(model, full_ld, y_scaler, device=device)

    Z         = eval_full["Z"]
    sp_arr    = df_model["species"].values
    clade_arr = df_model["clade"].values
    wbf_arr   = 1.0 / df_model["wblen"].values
    sp_idx_f  = df_model["species_idx"].values.astype(np.int64)

    # Plots
    plot_losses(hist, yaw_weight, tag=tag, title_prefix=title+" | ")
    plot_pred_vs_true(eval_test, tag=tag,  title_extra=title)
    plot_latent_space(Z, sp_arr, clade_arr, wbf_arr,
                      species_names, tag=tag, title_prefix=title+" | ")

    # Weights
    save_encoder_weights(model, species_names, tag=tag)
    save_interaction_weights(model, tag=tag)

    # SHAP (full dataset)
    lat_shap, mus_shap, mean_mus_shap = run_shap_analysis(
        model, Z, sp_idx_f, sp_arr, species_names, tag=tag, device=device
    )
    plot_shap_summary(mean_mus_shap,  tag=tag, title_prefix=title+" | ")
    plot_shap_by_species(mus_shap,    tag=tag, title_prefix=title+" | ")
    plot_shap_latent(lat_shap,        tag=tag, title_prefix=title+" | ")

    return eval_test, mean_mus_shap, model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Latent dim: {LATENT_DIM}  |  Interaction units: {INTERACTION_UNITS}  "
          f"|  Recon hidden: {RECON_HIDDEN_DIM}  |  Epochs: {EPOCHS}")

    df_model, species_names, _ = load_and_preprocess("./agile-research/all10_big_wb.csv")

    (X_tr, X_te, y_tr, y_te,
     sp_tr, sp_te, spl_tr, spl_te,
     cl_tr, cl_te, wf_tr, wf_te,
     y_scaler) = build_splits(df_model)

    shared = dict(
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
        sp_tr=sp_tr, sp_te=sp_te, spl_tr=spl_tr, spl_te=spl_te,
        cl_tr=cl_tr, cl_te=cl_te, wf_tr=wf_tr, wf_te=wf_te,
        y_scaler=y_scaler, species_names=species_names,
        df_model=df_model, device=device,
    )

    eval_ub, shap_ub, model_ub = run_phase(
        tag="_upper_bound",
        title="Phase 1 — Upper bound",
        yaw_weight=YAW_WEIGHT_UPPER,
        **shared,
    )

    eval_bal, shap_bal, model_bal = run_phase(
        tag="_balanced",
        title="Phase 2 — Balanced",
        yaw_weight=YAW_WEIGHT_BALANCED,
        **shared,
    )

    print("\n" + "="*64)
    print("  SUMMARY")
    print("="*64)
    print(f"  Upper bound yaw R²:  {eval_ub['yaw_r2']:.4f}  (λ={YAW_WEIGHT_UPPER})")
    print(f"  Balanced    yaw R²:  {eval_bal['yaw_r2']:.4f}  (λ={YAW_WEIGHT_BALANCED})")
    print(f"  R² cost of balanced loss: "
          f"{eval_ub['yaw_r2'] - eval_bal['yaw_r2']:.4f}")
    if SHAP_AVAILABLE and shap_ub is not None and shap_bal is not None:
        print("\n  Top muscles by mean |SHAP| (upper bound):")
        print(shap_ub.head(10).to_string(index=False))
        print("\n  Top muscles by mean |SHAP| (balanced):")
        print(shap_bal.head(10).to_string(index=False))
    print("="*64)

    plot_r2_comparison(eval_ub["yaw_r2"], eval_bal["yaw_r2"])

    pd.DataFrame([
        dict(phase="upper_bound", yaw_loss_weight=YAW_WEIGHT_UPPER,
             yaw_r2_test=eval_ub["yaw_r2"],  yaw_mse_test=eval_ub["yaw_mse"],
             recon_r2_test=eval_ub["recon_r2"]),
        dict(phase="balanced",   yaw_loss_weight=YAW_WEIGHT_BALANCED,
             yaw_r2_test=eval_bal["yaw_r2"], yaw_mse_test=eval_bal["yaw_mse"],
             recon_r2_test=eval_bal["recon_r2"]),
    ]).to_csv("results_summary.csv", index=False)

    df_model.to_csv("input_data_balanced_normalised.csv", index=False)
    print("\nAll outputs saved.")


if __name__ == "__main__":
    main()