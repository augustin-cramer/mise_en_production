"""Functions to look at the words in the poles which are
the most responsible for the movement of the corpus towards
their respective pole"""

import json
import os
import warnings
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd

from ..Axes.axes_definition import *
from ..Axes.filter_words import *
from ..GloVe.weights import get_weights_word2vec
from ..Axes.projection_functions import axis_vector, cosine_with_axis
from ..Axes.models import *
from ..Load_Data.load_data import *


warnings.filterwarnings("ignore")


events_keywordskeywords = list(set(clean(events_keywords, "unigram")))
new_topics = list(set(clean(new_topics, "unigram")))


def process_year_data(year, model_words_year, with_parliament=True, ssp_cloud=False, fs=None, bucket=None):
    """
    Processes and computes word embeddings and their cosine
    similarities with axes for a specified year.

    Args:
        year (int): The year of data to process.
        model_words_year (dict): Dictionary of word embeddings for the specified year.
        with_parliament (bool): Whether to include parliament session data.

    Returns:
        pd.DataFrame: A DataFrame containing words, their embeddings,
        and cosine similarities with predefined axes.
    """
    words_year = load_words_year(with_parliament, year, ssp_cloud=False, fs=None, bucket=None)

    weights_year = get_weights_word2vec(words_year, a=1e-3)

    vocab_year = load_vocab_year(with_parliament, year, ssp_cloud=False, fs=None, bucket=None)

    vocab_embed_year = [
        weights_year.get(word, 0) * model_words_year.get(word, 0)
        for word in vocab_year
    ]

    df_words_year = pd.DataFrame(
        zip(vocab_year, vocab_embed_year), columns=["text", "embedding"]
    )

    axis_v1 = axis_vector(pos_1, neg_1, model_words_year)
    axis_v2 = axis_vector(pos_2, neg_2, model_words_year)

    df_words_year["cos axe 1"] = df_words_year["text"].apply(
        cosine_with_axis, axis_v=axis_v1, model_sentences=model_words_year
    )
    df_words_year["cos axe 2"] = df_words_year["text"].apply(
        cosine_with_axis, axis_v=axis_v2, model_sentences=model_words_year
    )

    df_words_year["year"] = year if year <= 2019 else year - 18090

    return df_words_year


def var_embed_real(word: str, df1, df2, cos_axe: str):
    """
    Calculates the variation in embeddings between two dataframes
    for a given word along a specified axis.

    Args:
        word (str): The word to compute variations for.
        df1 (pd.DataFrame): DataFrame containing earlier year data.
        df2 (pd.DataFrame): DataFrame containing later year data.
        cos_axe (str): The axis ('cos axe 1' or 'cos axe 2') to
        compute the variation.

    Returns:
        float: The difference in cosine axis value for the word between two
          years or None if word not found.
    """
    try:
        return (
            df2.loc[df2["text"] == word][cos_axe].values[0]
            - df1.loc[df1["text"] == word][cos_axe].values[0]
        )
    except:
        return None


def is_in_keywords(word):
    """
    Determines if a word is in predefined keyword lists.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is in either 'new_topics' or
        'events_keywords', False otherwise.
    """
    if word in new_topics:
        return True
    if word in events_keywords:
        return True
    return False


def process_yearly_data(df, year, with_parliament=True, ssp_cloud=False, fs=None, bucket=None):
    """
    Processes yearly data by loading specific word data and
    applying keyword filters.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        year (int): Year of data to process.
        with_parliament (bool): Whether to process data including
        parliament data.

    Returns:
        pd.DataFrame: The processed DataFrame filtered by keywords
        and word counts.
    """
    # Load the words from the file
    words = load_finalwords(with_parliament, year, ssp_cloud, fs, bucket)

    # Calculate word counts
    word_counts = Counter(words)

    # Apply the word count to the dataframe
    df["word count"] = df["text"].apply(lambda word: word_counts.get(word, 0))

    # Filter rows where 'word count' is greater than 100
    df_filtered = df[df["word count"] > 100]

    # Apply the check for 'in keywords'
    df_filtered["in keywords"] = df_filtered["text"].apply(is_in_keywords)

    # Filter by 'in keywords'
    df_keywords = df_filtered[df_filtered["in keywords"]]

    return df_keywords


def get_top_variations(df_keywords, axis, number):
    """
    Retrieves the top variations for words along a specified axis.

    Args:
        df_keywords (pd.DataFrame): DataFrame containing keyword data.
        axis (int): The axis to sort data by.
        number (int): Number of top variations to retrieve.

    Returns:
        tuple: Two DataFrames containing the top increasing
        and decreasing variations.
    """
    var_up = df_keywords.sort_values(
        by=[f"var cos axe {axis}"], ascending=False
    ).head(number)[["text", "year", f"var cos axe {axis}"]]
    var_down = df_keywords.sort_values(
        by=[f"var cos axe {axis}"], ascending=True
    ).head(number)[["text", "year", f"var cos axe {axis}"]]
    return var_up, var_down


def visualize_top_variations(
    df_keywords,
    axis_1,
    axis_2=None,
    variation_1="up",
    variation_2="down",
    with_parliament=True,
    number=20,
):
    """
    Visualizes top variations of word embeddings on specified
    axes using bar charts.

    Args:
        df_keywords (pd.DataFrame): DataFrame containing keyword and variation data.
        axis_1 (int): Primary axis for visualization.
        axis_2 (int, optional): Secondary axis for visualization.
        variation_1 (str): Direction of variation on the primary axis ('up' or 'down').
        variation_2 (str): Direction of variation on the secondary axis ('up' or 'down').
        with_parliament (bool): Whether to include parliament data in the analysis.
        number (int): Number of top words to visualize.

    Returns:
        plotly.graph_objects.Figure: A plotly figure with subplots showing the top variations.
    """
    # Data fetching logic remains the same
    var_up_1, var_down_1 = get_top_variations(df_keywords, axis_1, number)

    if axis_2:
        var_up_2, var_down_2 = get_top_variations(df_keywords, axis_2, number)
    else:
        var_up_2, var_down_2 = var_up_1, var_down_1
        axis_2 = axis_1

    if variation_1 == "down":
        var_up_1 = var_down_1
    if variation_2 == "up":
        var_down_2 = var_up_2

    # Initialize Plotly figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.6,
        subplot_titles=[
            f"Increasing Variation on Axis {axis_1}",
            f"Decreasing Variation on Axis {axis_2}",
        ],
    )

    # Adding bar charts for increasing and decreasing variations
    fig.add_trace(
        go.Bar(
            x=var_up_1["text"],
            y=var_up_1[f"var cos axe {axis_1}"],
            name="Increasing",
            marker_color="skyblue",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=var_down_2["text"],
            y=var_down_2[f"var cos axe {axis_2}"],
            name="Decreasing",
            marker_color="lightgreen",
        ),
        row=2,
        col=1,
    )

    # Update layout and axes properties
    fig.update_layout(
        title=f"Extreme Embedding Variation on Axis {axis_1} and {axis_2}",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
        },
    )
    fig.update_xaxes(
        tickangle=45, tickmode="array", tickvals=var_up_1["text"], row=1, col=1
    )
    fig.update_xaxes(
        tickangle=45,
        tickmode="array",
        tickvals=var_down_2["text"],
        row=2,
        col=1,
    )

    # Display the figure
    return fig


def word_variations(
    year,
    axis_1=1,
    axis_2=1,
    variation_1="up",
    variation_2="down",
    with_parliament=True,
    number=20,
    ssp_cloud=False,
    fs=None,
    bucket=None
):
    """
    Processes and visualizes the top word variations between
    two consecutive years on specified axes.

    Args:
        year (int): Base year for analysis.
        axis_1 (int): The primary axis for variation analysis.
        axis_2 (int, optional): The secondary axis for variation
        analysis, defaults to the same as axis_1.
        variation_1 (str): The type of variation on the primary
        axis ('up' or 'down').
        variation_2 (str, optional): The type of variation on the
        secondary axis ('up' or 'down').
        with_parliament (bool): Whether to include parliament session data in the
        analysis.
        number (int): Number of words to analyze for variations.

    Returns:
        plotly.graph_objects.Figure: A figure illustrating the variations in word embeddings across specified axes.
    """
    if year > 2019:
        year += 18090  # Adjusting the year by adding 18090 to it if it's above 2019
    i = year % 10  # Getting the last digit of the year

    path_1 = f"word analysis values/processed yearly data ; year = {year}, model = {i}, with parliament = {with_parliament}"
    if not os.path.exists(path_1):
        st.write((f"processing year {year}"))
        print(f"processing year {year}")
        current_df = process_year_data(year, models_w[i], with_parliament, ssp_cloud, fs, bucket)
        current_df.to_csv(path_1, index=False)
    else:
        st.write(f"{year} already processed")
        print(f"{year} already processed")
        current_df = pd.read_csv(path_1)

    path_2 = f"word analysis values/processed yearly data ; year = {year-1}, model = {i-1}, with parliament = {with_parliament}"
    if not os.path.exists(path_2):
        st.write(f"processing year {year-1}")
        print(f"processing year {year-1}")
        previous_df = process_year_data(
            year - 1, models_w[i - 1], with_parliament, ssp_cloud, fs, bucket
        )
        previous_df.to_csv(path_2, index=False)
    else:
        st.write(f"{year-1} already processed")
        print(f"{year-1} already processed")
        previous_df = pd.read_csv(path_2)

    path_3 = f"word analysis values/var embed real ; current year = {year}, previous year = {year-1}"
    if not os.path.exists(path_3):
        st.write("computing...")
        for cos_axe in ["cos axe 1", "cos axe 2"]:
            var_column_name = f"var {cos_axe}"
            print(f"comuting var embed for {cos_axe}")
            current_df[var_column_name] = current_df["text"].apply(
                var_embed_real,
                df1=previous_df,
                df2=current_df,
                cos_axe=cos_axe,
            )
        current_df.to_csv(
            f"word analysis values/var embed real ; current year = {year}, previous year = {year-1}",
            index=False,
        )
    else:
        st.write("All already computed..")
        current_df = pd.read_csv(path_3)

    current_df = process_yearly_data(current_df, year, with_parliament, ssp_cloud, fs, bucket)

    return visualize_top_variations(
        current_df,
        axis_1,
        axis_2,
        variation_1,
        variation_2,
        with_parliament=with_parliament,
        number=number,
    )
