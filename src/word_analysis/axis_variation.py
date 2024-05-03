"""Functions to look at the biggest variations for words 
in embedding between two years"""

import warnings
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from ..glove.weights import standard_opening
from ..axes.projection_functions import filter_model, barycentre
from ..processing.text_cleaning import clean
from ..axes.axes_definition import (
    tech,
    reg,
    pos,
    neg,
    pos_1,
    neg_1,
    pos_2,
    neg_2,
)
from ..axes.models import instatiate_models_w

warnings.filterwarnings("ignore")


def process_embeddings(data_loader, file_path):
    """
    Processes embedding data from a specified file, normalizing by source counts.

    Parameters:
    - file_path (str): The path to the file containing the embeddings data.

    Returns:
    - DataFrame: Processed DataFrame with normalized sentence embeddings.
    """
    # Load the data
    df = standard_opening(data_loader, file_path, False)
    # Transform the 'sentence_embedding' column
    df["sentence_embedding"] = (
        df["sentence_embedding"].apply(eval).apply(np.array, args=(float,))
    )
    df["count"] = 1
    # Select specific columns and group by 'source'
    df = (
        df[["sentence_embedding", "source", "count"]]
        .groupby(by="source", as_index=False)
        .sum()
    )
    # Normalize the sentence embeddings by the count using vectorized operation
    df["sentence_embedding"] = df.apply(
        lambda x: x["sentence_embedding"] / x["count"], axis=1
    )
    return df


def give_embed_anyway(word, model_word, list_of_words):
    """
    Retrieves the embedding for a given word or returns a zero vector if the word is not present.

    Parameters:
    - word (str): The word to find the embedding for.
    - model_word (dict): The model containing word embeddings.
    - list_of_words (list): A list of words to filter from the model.

    Returns:
    - np.array: The embedding of the word or a zero vector.
    """
    if word in filter_model(list_of_words, model_word):
        return model_word[word]
    else:
        return np.array([0 for i in range(50)], dtype=float)


def see_variation_on_axis(year: int, df, source=None):
    """
    Computes and sorts word variations based on embeddings from a specific year and source.

    Parameters:
    - year (int): The year to filter the dataframe on.
    - df (DataFrame): The dataframe containing word embedding data.
    - source (str, optional): Specific source to filter the dataframe.

    Returns:
    - tuple: A tuple of sorted dictionaries containing word variations for different categories.
    """
    st.write("Computing...")

    if source:
        df = df.loc[df["source"] == source]
    df = df.loc[df["year"] == year]

    l = []
    for word in clean(tech, "unigram"):
        try:
            l.append(df[word].tolist()[0])
        except:
            print(f"{word} not in the model")
    var_tech = dict(zip(clean(tech, "unigram"), l))
    sorted_var_tech = sorted(
        var_tech.items(), key=lambda x: x[1], reverse=True
    )

    l = []
    for word in clean(reg, "unigram"):
        try:
            l.append(df[word].tolist()[0])
        except:
            print(word)
    var_reg = dict(zip(clean(reg, "unigram"), l))
    sorted_var_reg = sorted(var_reg.items(), key=lambda x: x[1], reverse=True)

    l = []
    for word in clean(pos, "unigram"):
        try:
            l.append(df[word].tolist()[0])
        except:
            print(word)
    var_pos = dict(zip(clean(pos, "unigram"), l))
    sorted_var_pos = sorted(var_pos.items(), key=lambda x: x[1], reverse=True)

    l = []
    for word in clean(neg, "unigram"):
        try:
            l.append(df[word].tolist()[0])
        except:
            print(word)
    var_neg = dict(zip(clean(neg, "unigram"), l))
    sorted_var_neg = sorted(var_neg.items(), key=lambda x: x[1], reverse=True)

    return (sorted_var_tech, sorted_var_reg, sorted_var_pos, sorted_var_neg)


def project_variation_on_axis(
    axis, source: str, year: int, df, number_of_words, with_parliament
):
    """
    Projects and visualizes the top word variations along a specified axis.

    Parameters:
    - axis (int): The axis index for projection.
    - source (str): Source to filter the data.
    - year (int): Year of the data.
    - df (DataFrame): The dataframe containing word variations.
    - number_of_words (int): Number of top words to display.
    - with_parliament (bool): Flag to include parliament data.

    Returns:
    - plotly.graph_objs._figure.Figure: A Plotly figure object with the visualization.
    """
    # Fetching data for the plots

    if axis == 1:
        data_for_var_up = dict(
            see_variation_on_axis(year, df, source)[0][:number_of_words]
        )
        data_for_var_down = dict(
            see_variation_on_axis(year, df, source)[1][:number_of_words]
        )

    if axis == 2:
        data_for_var_up = dict(
            see_variation_on_axis(year, df, source)[2][:number_of_words]
        )
        data_for_var_down = dict(
            see_variation_on_axis(year, df, source)[3][:number_of_words]
        )

    # Axis titles based on the 'axis' variable
    labels = [
        "Tech" if axis == 1 else "Positive",
        "Reg" if axis == 1 else "Negative",
    ]
    colors = ["#1f77b4", "#ff7f0e"]

    # Create a subplot figure with 2 vertical plots
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.3,
        subplot_titles=[
            f"Variation in {labels[0]}",
            f"Variation in {labels[1]}",
        ],
    )

    # Add bar charts to the subplots
    fig.add_trace(
        go.Bar(
            x=list(data_for_var_up.keys()),
            y=list(data_for_var_up.values()),
            name=f"Variation in {labels[0]}",
            marker_color=colors[0],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=list(data_for_var_down.keys()),
            y=list(data_for_var_down.values()),
            name=f"Variation in {labels[1]}",
            marker_color=colors[1],
        ),
        row=2,
        col=1,
    )

    # Customize layout
    fig.update_layout(
        title_text=f"{number_of_words} words most responsible for the move of {source} towards the respective poles between year {year} and {year + 1}; axis = {axis}",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
        },
    )

    # Customize x-axis properties
    fig.update_xaxes(tickangle=45, tickfont={"size": 12}, row=1, col=1)
    fig.update_xaxes(tickangle=45, tickfont={"size": 12}, row=2, col=1)

    # Customize y-axis properties for clarity
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # Add grid lines
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="gray")

    # Display the figure
    return fig


axes_words = clean(tech + reg + pos + neg, "unigram")


def axis_variation(
    data_loader,
    axis,
    source=None,
    year=2013,
    number_of_words=30,
    with_parliament=True,
):
    """
    Computes and visualizes the words that define the poles of the axes most responsible for
    attracting the corpus towards them between two consecutive years.

    Parameters:
    - axis (int): The axis to analyze.
    - source (str, optional): Source filter for the data.
    - year (int): Starting year for the analysis.
    - number_of_words (int): Number of words to analyze.
    - with_parliament (bool): Flag to include parliament data.

    Returns:
    - plotly.graph_objs._figure.Figure: A figure visualizing the word variations along the specified axis.
    """
    print("computing")

    if year > 2019:
        year = year + 18090
    i = eval(str(year)[-1:])

    year_plus1 = year
    year = year_plus1 - 1

    if year_plus1 == 20110:
        year = 2019

    str_parliament = "with" if with_parliament else "without"
    file_path_1 = f"{str_parliament}_parliament/sentence_embeddings/sentence_embeddings_{year}.csv"
    file_path_2 = f"{str_parliament}_parliament/sentence_embeddings/sentence_embeddings_{year_plus1}.csv"

    dataframes = []
    dataframes.append(process_embeddings(data_loader, file_path_1))
    dataframes.append(process_embeddings(data_loader, file_path_2))

    models_w = instatiate_models_w(data_loader)
    axes_words_embeddings = [
        [
            give_embed_anyway(word, models_w[i], axes_words)
            for word in axes_words
        ],
        [
            give_embed_anyway(word, models_w[i + 1], axes_words)
            for word in axes_words
        ],
    ]

    df_axes = pd.DataFrame(
        zip(axes_words, *axes_words_embeddings),
        columns=["text", f"embedding 201{i}", f"embedding 201{i+1}"],
    )

    poles = []

    pos_a = filter_model(pos_1, models_w[i])
    neg_a = filter_model(neg_1, models_w[i])

    pos_b = filter_model(pos_2, models_w[i])
    neg_b = filter_model(neg_2, models_w[i])

    b1 = barycentre(pos_a, models_w[i]) - barycentre(neg_a, models_w[i])
    b2 = barycentre(pos_b, models_w[i]) - barycentre(neg_b, models_w[i])

    poles = [b1, b2]

    for i in df_axes.index:
        word = df_axes[df_axes.columns[1]][i]
        var = []
        for j in dataframes[0].index:
            diff = (
                dataframes[1]["sentence_embedding"][j]
                / (np.linalg.norm(dataframes[1]["sentence_embedding"][j]))
            ) - (
                dataframes[0]["sentence_embedding"][j]
                / (np.linalg.norm(dataframes[0]["sentence_embedding"][j]))
            )
            var.append(np.dot(diff, word) / (np.linalg.norm(poles[axis - 1])))
        dataframes[0][str(df_axes["text"][i])] = var

    dataframes[0]["year"] = year
    dataframes[1]["year"] = year_plus1

    df = pd.concat([dataframes[0], dataframes[1]])

    return project_variation_on_axis(
        axis=axis,
        source=source,
        year=year,
        df=df,
        number_of_words=number_of_words,
        with_parliament=with_parliament,
    )
