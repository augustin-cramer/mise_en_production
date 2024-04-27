from ..GloVe.weights import *
from ..Axes.projection_functions import *
from ..Clustering.clustering_spectral import *
import matplotlib.pyplot as plt
import os

df_BT_p = pd.read_csv(
    "data/with parliament/current_dataframes/df_BT.csv", index_col=[0]
).reset_index()
df_BT_wp = pd.read_csv(
    "data/without parliament/current_dataframes/df_BT.csv", index_col=[0]
).reset_index()

print(os.getcwd())


def get_quantiles(data, percentiles):
    """
    Compute quantiles for a given dataset and percentiles.

    :param data: Numerical data from which to calculate quantiles.
    :param percentiles: A list of percentiles to calculate for the data.
    :return: An array of quantiles corresponding to the specified percentiles.
    """
    return np.percentile(data, percentiles)


def cluster_words(
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
    if year > 2019:
        year = year + 18090
    i = eval(str(year)[-1:])

    if with_parliament:
        df_BT = df_BT_p
        model_sentences = txt_to_model_sentences(
            "data/with parliament/sentence_embeddings/sentence_embeddings_"
            + str(year)
            + ".txt"
        )
    if not with_parliament:
        df_BT = df_BT_wp
        model_sentences = txt_to_model_sentences(
            "data/without parliament/sentence_embeddings/sentence_embeddings_"
            + str(year)
            + ".txt"
        )

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
        quantiles = get_quantiles(df_t[f"cos axe {axis}"], percentiles=percentiles)
        df_t = df_t.loc[
            (df_t[f"cos axe {axis}"] < quantiles[0])
            | (df_t[f"cos axe {axis}"] > quantiles[1])
        ]

    embeds_list = [model_sentences[sentence] for sentence in df_t["text"]]
    data = np.array(embeds_list)

    plot_silhouette_and_sse(11, data)

    # Ask the user for the number of clusters
    try:
        n_clusters = int(input("Enter the number of clusters you want to use: "))
    except ValueError:
        print("Invalid number, using a default value of 5 clusters.")
        n_clusters = 5  # Default value if the user input is not valid

    plot_clusters_on_pc_spectral_3d(n_clusters, data, marker_size=1.4)
    visualize_main_words_in_clusters_TFIDF(n_clusters, data, df_t)


def cluster_words_intermediate(
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
    if year > 2019:
        year = year + 18090
    i = eval(str(year)[-1:])

    if with_parliament:
        df_BT = df_BT_p
        model_sentences = txt_to_model_sentences(
            "data/with parliament/sentence_embeddings/sentence_embeddings_"
            + str(year)
            + ".txt"
        )
    if not with_parliament:
        df_BT = df_BT_wp
        model_sentences = txt_to_model_sentences(
            "data/without parliament/sentence_embeddings/sentence_embeddings_"
            + str(year)
            + ".txt"
        )

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
        quantiles = get_quantiles(df_t[f"cos axe {axis}"], percentiles=percentiles)
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

    return  fig_1, fig_2, data, df_t


def display_clusters(n_clusters, data, df_t):
    return (
        plot_clusters_on_pc_spectral_3d(n_clusters, data, marker_size=1.4),
        visualize_main_words_in_clusters_TFIDF_streamlit(n_clusters, data, df_t),
    )
