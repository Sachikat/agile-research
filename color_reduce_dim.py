import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA

df = pd.read_csv('preprocessedCache.csv')

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
df['clade'] = df['species'].map(clade_dict)

group_cols = ['species','moth','trial','wb','muscle']

feat_long = df.groupby(group_cols).agg({
    'phase': ['mean', 'std', 'min', 'max'],
    'fz':    ['mean', 'std', 'min', 'max'],
    'tz':    ['mean', 'std', 'min', 'max'],  # yaw torque often useful
    'wbfreq': 'mean'
}).reset_index()

feat_long.columns = ['_'.join(col).strip('_') for col in feat_long.columns]

value_cols = [
    'phase_mean','phase_std','phase_min','phase_max',
    'fz_mean','fz_std','fz_min','fz_max',
    'tz_mean','tz_std','tz_min','tz_max'
]

X = feat_long.pivot_table(
    index=['species','moth','trial','wb'],
    columns='muscle',
    values=value_cols
)

X.columns = [f"{feat}_{muscle}" for feat, muscle in X.columns]
X = X.reset_index()

meta = df.groupby(['species','moth','trial','wb']).agg({
    'wbfreq': 'mean',
    'clade':  'first'
}).reset_index()

X = X.merge(meta, on=['species','moth','trial','wb'], how='left')

non_feature = ['species','moth','trial','wb','wbfreq','clade']
feature_cols = [c for c in X.columns if c not in non_feature]

X[feature_cols] = X[feature_cols].fillna(0)

Xs = StandardScaler().fit_transform(X[feature_cols])

#PCA
pca = PCA(n_components=3)
Z = pca.fit_transform(Xs)

X['PC1'] = Z[:,0]
X['PC2'] = Z[:,1]
X['PC3'] = Z[:,2]

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained (PC1+PC2):",
      pca.explained_variance_ratio_[:2].sum())

plt.figure(figsize=(9,7))
for sp in X['species'].unique():
    sub = X[X['species'] == sp]
    plt.scatter(sub['PC1'], sub['PC2'], s=10, alpha=0.45, label=sp)

plt.legend(fontsize=8, bbox_to_anchor=(1.02,1), loc="upper left")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of motor program features (colored by species)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,7))
for cl in sorted(X['clade'].dropna().unique()):
    sub = X[X['clade'] == cl]
    plt.scatter(sub['PC1'], sub['PC2'], s=10, alpha=0.6, label=cl)

plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of motor program features (colored by clade)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,7))
sc = plt.scatter(X['PC1'], X['PC2'], c=X['wbfreq'], s=10, alpha=0.65)
plt.colorbar(sc, label='wbfreq')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of motor program features (colored by wingbeat frequency)")
plt.tight_layout()
plt.show()

#UMAP
# reducer = umap.UMAP(
#     n_neighbors=30,
#     min_dist=0.1,
#     random_state=0
# )
# U = reducer.fit_transform(Xs)

# X['UMAP1'] = U[:,0]
# X['UMAP2'] = U[:,1]

# plt.figure(figsize=(9,7))
# for sp in X['species'].unique():
#     sub = X[X['species'] == sp]
#     plt.scatter(sub['UMAP1'], sub['UMAP2'], s=10, alpha=0.45, label=sp)

# plt.legend(fontsize=8, bbox_to_anchor=(1.02,1), loc="upper left")
# plt.title("UMAP of motor program features (colored by species)")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(9,7))
# for cl in sorted(X['clade'].dropna().unique()):
#     sub = X[X['clade'] == cl]
#     plt.scatter(sub['UMAP1'], sub['UMAP2'], s=10, alpha=0.6, label=cl)

# plt.legend()
# plt.title("UMAP of motor program features (colored by clade)")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(9,7))
# sc = plt.scatter(X['UMAP1'], X['UMAP2'], c=X['wbfreq'], s=10, alpha=0.65)
# plt.colorbar(sc, label='wbfreq')
# plt.title("UMAP of motor program features (colored by wingbeat frequency)")
# plt.tight_layout()
# plt.show()

# print("X shape (wingbeats x features):", X.shape)
# print("Num feature cols:", len(feature_cols))
# print("Clade counts:\n", X['clade'].value_counts(dropna=False))