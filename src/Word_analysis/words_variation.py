import json
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ..Processing.text_cleaning import *
from ..GloVe.weights import *
from collections import Counter
import warnings
import streamlit as st

warnings.filterwarnings("ignore")
from ..Axes.projection_functions import *
from ..Axes.models import *
from ..Axes.filter_words import *
from ..Processing.preprocess_parliament import *
import os
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

from ..Processing.text_cleaning import *
from ..GloVe.weights import *
from ..Axes.projection_functions import *
from ..Axes.models import *
from ..Axes.filter_words import *
from ..Processing.preprocess_parliament import *


events_keywordskeywords = list(set(clean(events_keywords, "unigram")))
new_topics = list(set(clean(new_topics, "unigram")))


def process_year_data(year, model_words_year, with_parliament=True):
    """
    Processes embedding data for a given year and word model, adjusting for parliament session data.

    Args:
        year (int): The year to process data for.
        model_words_year (Dict[str, float]): A dictionary mapping words to their model weights.
        with_parliament (bool): Flag to determine whether to use data from parliament sessions or not.

    Returns:
        pd.DataFrame: A DataFrame containing words, their embeddings, and cosine similarities with two axes.
    """
     
    if with_parliament:
        with open(f"data/with parliament/words/Finalwords_{year}.json") as f:
            words_year = json.load(f)
    else:
        with open(f"data/without parliament/words/Finalwords_{year}_WP.json") as f:
            words_year = json.load(f)

    weights_year = get_weights_word2vec(words_year, a=1e-3)

    if with_parliament:
        with open(f"data/with parliament/vocabs/vocab_{year}.json") as f:
            vocab_year = json.load(f)
    else:
        with open(f"data/without parliament/vocabs/vocab_{year}_WP.json") as f:
            vocab_year = json.load(f)

    vocab_embed_year = [weights_year.get(word, 0) * model_words_year.get(word, 0) for word in vocab_year]

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
    try:
        return (
            df2.loc[df2["text"] == word][cos_axe].values[0]
            - df1.loc[df1["text"] == word][cos_axe].values[0]
        )
    except:
        return None


def is_in_keywords(word):
    if word in new_topics:
        return True
    if word in events_keywords:
        return True
    return False


def process_yearly_data(df, year, with_parliament=True):
    # Load the words from the file
    if with_parliament:
        with open(f"data/with parliament/words/Finalwords_{year}.json") as f:
            words = json.load(f)
    if not with_parliament:
        with open(f"data/without parliament/words/Finalwords_{year}_WP.json") as f:
            words = json.load(f)

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
    """Sorts the dataframe by the specified axis and gets the top number variations."""
    var_up = df_keywords.sort_values(by=[f"var cos axe {axis}"], ascending=False).head(
        number
    )[["text", "year", f"var cos axe {axis}"]]
    var_down = df_keywords.sort_values(by=[f"var cos axe {axis}"], ascending=True).head(
        number
    )[["text", "year", f"var cos axe {axis}"]]
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
):
    if year > 2019:
        year = year + 18090
    i = eval(str(year)[-1:])

    path_1 = f"word analysis values/processed yearly data ; year = {year}, model = {i}, with parliament = {with_parliament}"
    if not os.path.exists(path_1):
        st.write((f"processing year {year}"))
        st.write((f"processing year {year}"))
        print(f"processing year {year}")
        current_df = process_year_data(year, models_w[i], with_parliament)
        current_df.to_csv(path_1, index=False)
    else:
        st.write(f"{year} already processed")
        st.write(f"{year} already processed")
        print(f"{year} already processed")
        current_df = pd.read_csv(path_1)

    path_2 = f"word analysis values/processed yearly data ; year = {year-1}, model = {i-1}, with parliament = {with_parliament}"
    if not os.path.exists(path_2):
        st.write(f"processing year {year-1}")
        st.write(f"processing year {year-1}")
        print(f"processing year {year-1}")
        previous_df = process_year_data(year - 1, models_w[i - 1], with_parliament)
        previous_df.to_csv(path_2, index=False)
    else:
        st.write(f"{year-1} already processed")
        print(f"{year-1} already processed")
        previous_df = pd.read_csv(path_2)

    path_3 = f"word analysis values/var embed real ; current year = {year}, previous year = {year-1}"
    if not os.path.exists(path_3):
        st.write('computing...')
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
        st.write('All already computed..')
        current_df = pd.read_csv(path_3)

    current_df = process_yearly_data(current_df, year, with_parliament)

    return visualize_top_variations(
        current_df,
        axis_1,
        axis_2,
        variation_1,
        variation_2,
        with_parliament=with_parliament,
        number=number,
    )
