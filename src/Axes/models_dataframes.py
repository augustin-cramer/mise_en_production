import os
import pandas as pd
from GloVe.weights import *
import warnings

warnings.filterwarnings("ignore")
from Axes.projection_functions import *
from Axes.axes_definition import *
from Axes.models import *

# DataFrames Preparation and Processing

dfs = []
# Loop through a specified range to load and project sentence embeddings for each year
for i in range(14):
    dfs.append(
        open_to_project(
            f"data/without parliament/sentence_embeddings/sentence_embeddings_201{i}.csv",
            eval(f"201{i}"),
        )
    )

# Remove sources with fewer than 50 articles per year for cleaner analysis
# Specifically targets "DM" and "TE" sources in various years
sources_to_remove = ["DM", "TE"]
for i, df in enumerate(dfs[:6]):
    for source in sources_to_remove:
        dfs[i] = dfs[i][~dfs[i]["source"].isin([source] if i == 0 else ["TE"])]

# Projections by Year


def both_cosines(df, pos_1, neg_1, pos_2, neg_2, model_words, model_sentences):
    """
    Computes cosine similarities for two axes and updates the dataframe with these values.

    Parameters:
    - df (DataFrame): The dataframe containing sentence embeddings.
    - pos_1, neg_1, pos_2, neg_2 (str): Positive and negative words defining two axes.
    - model_words, model_sentences (Model): Pre-trained word and sentence models.

    Returns:
    - DataFrame: Updated dataframe with added columns for cosine similarity on both axes.
    """
    axis_v1 = axis_vector(pos_1, neg_1, model_words)
    df["cos axe 1"] = df["text"].apply(
        cosine_with_axis,
        args=(axis_v1,),
        model_sentences=model_sentences,
    )
    axis_v2 = axis_vector(pos_2, neg_2, model_words)
    df["cos axe 2"] = df["text"].apply(
        cosine_with_axis,
        args=(axis_v2,),
        model_sentences=model_sentences,
    )
    return df


# Apply the cosine similarity calculations for all prepared DataFrames
for i in range(14):
    dfs[i] = both_cosines(dfs[i], pos_1, neg_1, pos_2, neg_2, models_w[i], models_s[i])

# Combine all DataFrames into one for further analysis
df = pd.concat(dfs)

# BigTech DataFrames Creation


def tostring(list):
    """
    Converts a list to a string representation.

    Parameters:
    - list (List): The list to be converted.

    Returns:
    - str: A string representation of the input list.
    """
    return str(list)


# Copying the main DataFrame to isolate Big Tech related entries
df_BT = df.copy()
df_BT.reset_index(drop=True, inplace=True)
df_BT["keywords"] = df_BT["keywords"].apply(eval)
# Applying thematic analysis and conversion to string for keyword matching
df_BT["theme"] = df_BT["keywords"].apply(theme)
df_BT["theme"] = df_BT["theme"].apply(tostring)

# Filtering entries by Big Tech companies based on theme presence
big_tech_companies = ["amazon", "facebook", "apple", "google", "microsoft"]
dfs_big_tech = {
    company: df_BT[df_BT["theme"].str.contains(company)]
    for company in big_tech_companies
}

# Assigning a unique class identifier for each Big Tech company
class_map = {
    "amazon": "am",
    "facebook": "fb",
    "apple": "ap",
    "google": "go",
    "microsoft": "mi",
}
for company, dfi in dfs_big_tech.items():
    dfi["class"] = class_map[company]

# Combining all Big Tech DataFrames into one
df_BT = pd.concat(dfs_big_tech.values())

# Saving the processed DataFrames for future use
df.to_csv("data/without parliament/current_dataframes/df.csv", index=False)
df_BT.to_csv("data/without parliament/current_dataframes/df_BT.csv", index=False)
