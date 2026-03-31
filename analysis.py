import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_encoder_weight_df(model, feature_cols, species_names):
    encoder_tables = {}

    for i, species in enumerate(species_names):
        W = model.encoders[i].weight.detach().cpu().numpy()   # [latent_dim, input_dim]
        df_w = pd.DataFrame(
            W,
            index=[f"z{k+1}" for k in range(W.shape[0])],
            columns=feature_cols
        )
        encoder_tables[species] = df_w

    return encoder_tables

def overall_latent_importance(weight_df):
    # L2 norm across rows (latent dims) for each feature
    scores = np.sqrt((weight_df.values ** 2).sum(axis=0))
    return pd.Series(scores, index=weight_df.columns).sort_values(ascending=False)

def muscle_name_from_feature(feat):
    if "_minus_" in feat:
        # example: DLM_minus_DVM_spike1
        return feat.split("_minus_")[0]
    return feat.split("_spike")[0]


def aggregate_by_muscle(importance_series):
    muscle_scores = {}

    for feat, score in importance_series.items():
        muscle = muscle_name_from_feature(feat)
        muscle_scores[muscle] = muscle_scores.get(muscle, 0.0) + float(score)

    return pd.Series(muscle_scores).sort_values(ascending=False)


def get_effective_yaw_weights(model, feature_cols, species_names):
    # decoder_y: [1, latent_dim]
    yaw_w = model.decoder_y.weight.detach().cpu().numpy().reshape(-1)  # [latent_dim]
    out = {}

    for i, species in enumerate(species_names):
        E = model.encoders[i].weight.detach().cpu().numpy()  # [latent_dim, input_dim]
        eff = yaw_w @ E   # [input_dim]
        out[species] = pd.Series(eff, index=feature_cols)

    return out



def get_raw_feature_variance(X_bal, feature_cols):
    return pd.Series(
        X_bal[feature_cols].var().values,
        index=feature_cols
    ).sort_values(ascending=False)