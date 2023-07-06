"""Explore data using UMAP."""
import pandas as pd
import umap
from sklearn.preprocessing import MinMaxScaler

from source.utils.data import load_data, remove_nan
from source.utils.plotting import plot_3d

X_train, y_train, X_valid, y_valid = load_data()

X_train, y_train = remove_nan(X_train, y_train)
X_valid, y_valid = remove_nan(X_valid, y_valid)

X = pd.concat([X_train, X_valid])
y = pd.concat([y_train, y_valid])

scaler = MinMaxScaler()

X_norm = scaler.fit_transform(X)

umap_model = umap.UMAP(n_components=3)
umap_result = umap_model.fit_transform(X_norm)

umap_df = pd.DataFrame(umap_result, columns=["UMAP 1", "UMAP 2", "UMAP 3"])

plot_3d(umap_df, y, "UMAP.png")
