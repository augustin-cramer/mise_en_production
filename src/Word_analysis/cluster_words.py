from GloVe.weights import *
from Axes.projection_functions import *
from Clustering.clustering_spectral import *
import matplotlib.pyplot as plt
import os
from ..Load_Data.load_data import *

os.chdir("../")

ssp_cloud = False

df_BT_p = load_csv_index_col("/with_parliament/current_dataframes/df_BT.csv")
df_BT_p = df_BT_p.reset_index()
df_BT_wp = load_csv_index_col("/without_parliament/current_dataframes/df_BT.csv")
df_BT_wp = df_BT_wp.reset_index()

os.chdir(r"src")

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
    ssp_cloud=False,
    fs=None,
    bucket=None
):
    if year > 2019:
        year = year + 18090
    i = eval(str(year)[-1:])

    if with_parliament:
        df_BT = df_BT_p
        model_sentences = load_txt_model_sentence(
            "/with_parliament/sentence_embeddings/sentence_embeddings_"+str(year)+".txt",
            ssp_cloud, fs, bucket
        )
    if not with_parliament:
        df_BT = df_BT_wp
        model_sentences = load_txt_model_sentence(
            "/without_parliament/sentence_embeddings/sentence_embeddings_"+str(year)+".txt",
            ssp_cloud, fs, bucket
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
        quantiles = get_quantiles(
            df_t[f"cos axe {axis}"], percentiles=percentiles
        )
        df_t = df_t.loc[
            (df_t[f"cos axe {axis}"] < quantiles[0])
            | (df_t[f"cos axe {axis}"] > quantiles[1])
        ]

    embeds_list = [model_sentences[sentence] for sentence in df_t["text"]]
    data = np.array(embeds_list)

    plot_silhouette_and_sse(11, data)

    # Ask the user for the number of clusters
    try:
        n_clusters = int(
            input("Enter the number of clusters you want to use: ")
        )
    except ValueError:
        print("Invalid number, using a default value of 5 clusters.")
        n_clusters = 5  # Default value if the user input is not valid

    plot_clusters_on_pc_spectral_3d(n_clusters, data, marker_size=1.4)
    visualize_main_words_in_clusters_TFIDF(n_clusters, data, df_t)
