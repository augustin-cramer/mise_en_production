"""Functions to perform the spectral clustering of a selectd 
corpus, also using the functions in [`src/Clustering/`]"""

import os
import pandas as pd
import numpy as np

from ..Axes.projection_functions import txt_to_model_sentences
from ..Clustering.clustering_spectral import *


def get_quantiles(data, percentiles):
    """
    Calculates quantiles for specified percentiles from given data.

    Parameters:
    - data (np.array): The data from which to calculate quantiles.
    - percentiles (list of int): The specific percentiles to calculate.

    Returns:
    - np.array: The quantiles for the specified percentiles.
    """
    return np.percentile(data, percentiles)


def cluster_words(
    data_loader,
    year,
    axis,
    left_threshold=None,
    right_threshold=None,
    head=None,
    tail=None,
    with_parliament=True,
    percentiles=None,
    company=None,
):
    """
    Clusters words from sentence embeddings and displays variation and clustering results.

    Parameters:
    - year (int): The year of the dataset to process.
    - axis (int): The axis on which to project the data.
    - left_threshold, right_threshold (float, optional):
    Thresholds for filtering cosine axes values.
    - head, tail (int, optional): Number of items to take from the head or tail of the dataset.
    - with_parliament (bool): Flag to use parliament data.
    - percentiles (list of int, optional): Percentiles for further filtering.
    - company (str, optional): Specific company to focus on.

    Displays:
    - Plots the silhouette and SSE plots, and asks for number
    of clusters to further visualize clustering.
    """
    if year > 2019:
        year = year + 18090
    i = eval(str(year)[-1:])

    str_parliament = "with" if with_parliament else "without"
    model_sentences = txt_to_model_sentences(
        data_loader,
        f"{str_parliament}_parliament/sentence_embeddings/sentence_embeddings_"
        + str(year)
        + ".txt"
        )
    df_BT = data_loader.read_csv(f"{str_parliament}_parliament/current_dataframes/df_BT.csv", index_col=[0]).reset_index()

    df_t = df_BT.loc[df_BT["year"] == year]

    if company:
        df_t = df_t.loc[df_t["class"] == company]

    if company:
        df_t = df_t.loc[df_t["class"] == company]

    if left_threshold or right_threshold:
        df_t = df_t.loc[
            (df_t[f"cos axe {axis}"] < left_threshold)
            | (df_t[f"cos axe {axis}"] > right_threshold)
        ]

    if head or tail:
        df1 = df_t.head(head)
        df2 = df_t.tail(tail)
        df_t = pd.concat([df1, df2])

    if percentiles:
        quantiles = get_quantiles(
            df_t[f"cos axe {axis}"], percentiles=percentiles
        )
        df_t = df_t.loc[
            (df_t[f"cos axe {axis}"] < quantiles[0])
            | (df_t[f"cos axe {axis}"] > quantiles[1])
        ]

    embeds_list = [model_sentences[sentence] for sentence in df_t["text"]]
    data = np.array(embeds_list)

    fig_1, fig_2 = plot_silhouette_and_sse(11, data)
    fig_1.show()
    fig_2.show()

    # Ask the user for the number of clusters
    try:
        n_clusters = int(
            input("Enter the number of clusters you want to use: ")
        )
    except ValueError:
        print("Invalid number, using a default value of 5 clusters.")
        n_clusters = 5  # Default value if the user input is not valid

    plot_clusters_on_pc_spectral_3d(n_clusters, data, marker_size=1.4).show()
    visualize_main_words_in_clusters_TFIDF(n_clusters, data, df_t)


def cluster_words_intermediate(
    data_loader,
    year,
    axis,
    left_threshold=None,
    right_threshold=None,
    head=None,
    tail=None,
    with_parliament=True,
    percentiles=None,
    company=None,
):
    """
    Prepares data and intermediate outputs for detailed cluster analysis.

    Parameters:
    - year (int), axis (int): Parameters defining the subset of data to use.
    - left_threshold, right_threshold (float, optional): Thresholds for filtering data.
    - head, tail (int, optional): Parameters to select data extremes.
    - with_parliament (bool): Whether to include parliament-related data.
    - percentiles (list of int, optional): Percentiles for quantile calculation.
    - company (str, optional): Company-specific data to include.

    Returns:
    - tuple: Returns silhouette and SSE plots, along with the
    processed data and DataFrame for further analysis.
    """
    if year > 2019:
        year = year + 18090
    i = eval(str(year)[-1:])

    str_parliament = "with" if with_parliament else "without"
    model_sentences = txt_to_model_sentences(
        data_loader,
        f"{str_parliament}_parliament/sentence_embeddings/sentence_embeddings_"
        + str(year)
        + ".txt"
        )
    df_BT = data_loader.read_csv(f"{str_parliament}_parliament/current_dataframes/df_BT.csv", index_col=[0]).reset_index()


    df_t = df_BT.loc[df_BT["year"] == year]

    if company:
        df_t = df_t.loc[df_t["class"] == company]

    if company:
        df_t = df_t.loc[df_t["class"] == company]

    if left_threshold or right_threshold:
        df_t = df_t.loc[
            (df_t[f"cos axe {axis}"] < left_threshold)
            | (df_t[f"cos axe {axis}"] > right_threshold)
        ]

    if head or tail:
        df1 = df_t.head(head)
        df2 = df_t.tail(tail)
        df_t = pd.concat([df1, df2])

    if percentiles:
        quantiles = get_quantiles(
            df_t[f"cos axe {axis}"], percentiles=percentiles
        )
        df_t = df_t.loc[
            (df_t[f"cos axe {axis}"] < quantiles[0])
            | (df_t[f"cos axe {axis}"] > quantiles[1])
        ]

    embeds_list = [model_sentences[sentence] for sentence in df_t["text"]]
    data = np.array(embeds_list)

    # Ask the user for the number of clusters
    """try:
        n_clusters = int(
            input("Enter the number of clusters you want to use: ")
        )
    except ValueError:
        print("Invalid number, using a default value of 5 clusters.")
        n_clusters = 5  # Default value if the user input is not valid"""

    fig_1, fig_2 = plot_silhouette_and_sse(11, data)

    return fig_1, fig_2, data, df_t


def display_clusters(n_clusters, data, df_t):
    """
    Displays clustering results and visualizations for the specified number of clusters.

    Parameters:
    - n_clusters (int): Number of clusters to form and visualize.
    - data (array): Data used for clustering.
    - df_t (DataFrame): DataFrame containing text data corresponding to the `data` for analysis.

    Returns:
    - tuple: Tuple containing figures for clustering and text analysis visualizations.
    """
    return (
        plot_clusters_on_pc_spectral_3d(n_clusters, data, marker_size=1.4),
        visualize_main_words_in_clusters_TFIDF_streamlit(
            n_clusters, data, df_t
        ),
    )
