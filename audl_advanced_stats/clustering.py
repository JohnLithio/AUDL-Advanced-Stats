"""Functions for clustering players based on their statistics."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from .constants import *
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from yellowbrick.cluster import (
    KElbowVisualizer,
    InterclusterDistance,
    SilhouetteVisualizer,
)


def cluster_players(
    df,
    columns,
    pca=True,
    pca_ncomponents=3,
    cluster_algorithm="kmeans",
    cluster_algorithm_kwargs=dict(n_clusters=5, random_state=0),
    diagnostics=False,
):
    """Cluster players into different roles/groups based on their statistics.

    Args:
        df (dataframe): Dataframe with player stats.
        columns (list): List of columns to use in clustering.
        pca (bool, optional): Whether or not to use PCA. Defaults to True.
        pca_ncomponents (int, optional): If PCA is True, the number of components to keep. Defaults to 3.
        cluster_algorithm (str, optional): The clustering algorithm to use. Defaults to "kmeans".
        cluster_algorithm_kwargs (dict, optional): Dict of args to be passed to the clustering algorithm.
            Defaults to 5 clusters.
        diagnostics (bool, optional): Whether or not to display some diagnostic plots for the clustering.

    Returns:
        A dataframe with the player stats and their clusters labels.

    """
    # Convert inf values to nan to be imputed
    df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan)

    # Trim ratios to constrain values to the middle 95%
    for col in ["yx_ratio_throwing", "yx_ratio_receiving"]:
        if col in columns:
            df[col] = df[col].clip(
                lower=df[col].quantile(0.025), upper=df[col].quantile(0.975)
            )

    df_temp = df[columns].copy()

    # Build pipeline
    pipe_components = [
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
    ]
    if pca:
        pipe_components.append(("pca", PCA(n_components=pca_ncomponents)))

    if diagnostics:
        pipe_diagnostics = Pipeline(pipe_components)
        if pca:
            if "label" in list(df):
                df = df.drop(columns=["label"])
            dfpca = pd.concat(
                [df, pd.DataFrame(pipe_diagnostics.fit_transform(df_temp)),], axis=1,
            )

            # Correlations between PCA components and input stats
            display(
                dfpca[list(range(pca_ncomponents)) + columns]
                .corr()[list(range(pca_ncomponents))]
                .iloc[pca_ncomponents:]
                .style.background_gradient(cmap="coolwarm", axis=1)
            )

            # Explained variance
            dft = Pipeline(pipe_components[:-1]).fit_transform(df_temp)
            pca = PCA().fit(dft)
            ax = sns.lineplot(
                x=[i + 1 for i, _ in enumerate(pca.explained_variance_ratio_)],
                y=np.cumsum(pca.explained_variance_ratio_),
            )
            ax.set_title("PCA Explained Variance By Components")
            ax.set_xlabel("Number of Components")
            ax.set_ylabel("Cumulative Explained Variance")
            plt.show()

        if cluster_algorithm.lower() == "kmeans":
            # Create elbow plot
            cluster_elbow = KMeans(random_state=0)
            visualizer_elbow = KElbowVisualizer(cluster_elbow, k=(2, 15), timings=False)

            visualizer_elbow.fit(pipe_diagnostics.fit_transform(df_temp))
            visualizer_elbow.show()

            # Create silhouette plot
            cluster = KMeans(**cluster_algorithm_kwargs)
            visualizer_sil = SilhouetteVisualizer(cluster)

            visualizer_sil.fit(pipe_diagnostics.fit_transform(df_temp))
            visualizer_sil.show()

            # Create cluster distance map
            visualizer_map = InterclusterDistance(cluster)

            visualizer_map.fit(pipe_diagnostics.fit_transform(df_temp))
            visualizer_map.show()

    if cluster_algorithm.lower() == "kmeans":
        pipe_components.append((cluster_algorithm, KMeans(**cluster_algorithm_kwargs)))
    elif cluster_algorithm.lower() == "agglomerativeclustering":
        pipe_components.append(
            (cluster_algorithm, AgglomerativeClustering(**cluster_algorithm_kwargs))
        )
    pipe = Pipeline(pipe_components)

    # Run data through pipeline
    df["label"] = pipe.fit_predict(df_temp)
    df["label"] = df["label"].astype(str)

    if diagnostics:
        # Correlations between cluster labels and input stats
        display(
            pd.concat(
                [
                    df.groupby("label").size().rename("cnt"),
                    df.groupby("label")[columns].mean(),
                ],
                axis=1,
            )
            .T.style.format("{:.2f}")
            .background_gradient(cmap="coolwarm", axis=1)
        )

    return df
