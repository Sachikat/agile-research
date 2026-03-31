import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap


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
        return m.lower().replace("left","").strip()

    if "right" in m.lower():
        return m.lower().replace("right","").strip()

    return m


df["muscle_base"] = df["muscle"].apply(base_muscle)


group_cols = [
    "species",
    "moth",
    "trial",
    "wb",
    "muscle_base"
]

feat_long = df.groupby(group_cols).agg({

    "phase":[
        "mean",
        "std",
        "min",
        "max"
    ]

}).reset_index()


feat_long.columns = [

    "_".join(col).strip("_")
    for col in feat_long.columns

]


feat_long["phase_width"] = (
    feat_long["phase_max"] -
    feat_long["phase_min"]
)


value_cols = [
    "phase_mean",
    "phase_std",
    "phase_min",
    "phase_max",
    "phase_width"
]

X = feat_long.pivot_table(

    index=[
        "species",
        "moth",
        "trial",
        "wb"
    ],

    columns="muscle_base",
    values=value_cols

)

X.columns = [
    f"{feat}_{muscle}"
    for feat,muscle in X.columns
]

X = X.reset_index()


meta = df.groupby(

    ["species","moth","trial","wb"]

).agg({

    "wbfreq":"mean",
    "clade":"first"

}).reset_index()


X = X.merge(
    meta,
    on=["species","moth","trial","wb"],
    how="left"
)


non_features = [
    "species",
    "moth",
    "trial",
    "wb",
    "wbfreq",
    "clade"
]

feature_cols = [
    c for c in X.columns
    if c not in non_features
]

X[feature_cols] = X[feature_cols].fillna(0)


min_wb = X["species"].value_counts().min()

X_bal = (
    X.groupby("species")
    .sample(min_wb, random_state=0)
)

print("Wingbeats per species:",min_wb)


Xs = StandardScaler().fit_transform(
    X_bal[feature_cols]
)


pca = PCA()

Z = pca.fit_transform(Xs)


X_bal["PC1"] = Z[:,0]
X_bal["PC2"] = Z[:,1]
X_bal["PC3"] = Z[:,2]


print("Explained variance ratio:")
print(pca.explained_variance_ratio_[:10])

loadings = pd.DataFrame(
    pca.components_,
    columns=feature_cols
)

loadings.index = [f"PC{i+1}" for i in range(len(loadings))]

print(loadings)

for i in range(3):

    pc = loadings.iloc[i]

    print(f"\nTop contributing features for PC{i+1}:")

    top = pc.abs().sort_values(ascending=False).head(10)

    for feat in top.index:
        print(f"{feat}: {pc[feat]:.4f}")

for i, var in enumerate(pca.explained_variance_ratio_[:10]):
    print(f"PC{i+1}: {var:.3f} variance explained")


cum_var = np.cumsum(
    pca.explained_variance_ratio_
)

plt.figure(figsize=(7,5))

plt.plot(
    range(1,len(cum_var)+1),
    cum_var
)

plt.xlabel("PC count")
plt.ylabel("Cumulative variance explained")
plt.title("Variance explained vs PC count")

plt.show()


plt.figure(figsize=(8,6))

for cl in X_bal["clade"].unique():

    sub = X_bal[X_bal["clade"]==cl]

    plt.scatter(
        sub["PC1"],
        sub["PC2"],
        s=10,
        alpha=0.6,
        label=cl
    )

plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Motor program PCA")

plt.show()

plt.figure(figsize=(8,6))

sc = plt.scatter(

    X_bal["PC1"],
    X_bal["PC2"],
    c=X_bal["wbfreq"],
    s=10,
    alpha=0.7

)

plt.colorbar(sc,label="wbfreq")

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Motor program PCA colored by wbfreq")

plt.savefig("cummlative_pca.png")

plt.show()


reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    random_state=0
)

U = reducer.fit_transform(Xs)

X_bal["UMAP1"] = U[:,0]
X_bal["UMAP2"] = U[:,1]


plt.figure(figsize=(8,6))

for cl in X_bal["clade"].unique():

    sub = X_bal[X_bal["clade"]==cl]

    plt.scatter(
        sub["UMAP1"],
        sub["UMAP2"],
        s=10,
        alpha=0.6,
        label=cl
    )

plt.legend()
plt.title("UMAP motor programs")

plt.savefig("umap.png")
plt.show()

plt.figure(figsize=(8,6))
sc = plt.scatter(X_bal["UMAP1"], X_bal["UMAP2"], c=X_bal["wbfreq"], s=10, alpha=0.7)

plt.colorbar(sc, label="wbfreq")
plt.title("UMAP motor programs (colored by wbfreq)")
plt.tight_layout()
plt.savefig("umap_wbfreq.png")
plt.show()


print("Final dataset shape:",X_bal.shape)
print("Number of features:",len(feature_cols))