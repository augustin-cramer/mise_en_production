"""Contains the function plotting the polarization long with 
the cosine similarity when restricted to an axis."""

import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from ..Polarization.polarization_plots import choose_pol
from ..Axes.bootstraping import bootstrap


def draw_cos_pol(
    left_side,
    right_side,
    curves_by_company=None,
    axis=None,
    percentiles=[10, 90],
    print_random_pol=True,
    force_i_lim=None,
    with_parliament=True,
):
    """
    Generates a plotly visualization comparing polarization and
    cosine similarity metrics between two political or thematic
    sides over time.

    This function evaluates and plots the real and random
    polarization between the specified groups,
    alongside their cosine similarities to a specified semantic
    axis. It handles data organization,
    calculates necessary statistics, and dynamically builds a
    comprehensive visualization.

    Parameters:
    - left_side (list of str): Identifiers for the sources or
    parties on the left side of the analysis.
    - right_side (list of str): Identifiers for the sources or
    parties on the right side.
    - curves_by_company (bool, optional): If True, individual
    company curves will be considered. Not implemented yet.
    - axis (int): The axis index for which the cosine similarities
    are calculated.
    - percentiles (list of int): Percentile thresholds for
    filtering significant cosine values.
    - print_random_pol (bool): Flag to determine whether random
    polarization values are printed.
    - force_i_lim (tuple, optional): Forces the plot's x-axis
    limits if specified.
    - with_parliament (bool): Flag to indicate whether the
    analysis includes parliament data.

    Returns:
    - plotly.graph_objs._figure.Figure: A Plotly figure object
    containing the generated visualization of polarization
    and cosine similarities.

    Raises:
    - ValueError: If 'axis' is not specified or if curves by
    company are requested but not implemented.

    The function reads and writes intermediate CSV files to
    cache polarization calculations, and retrieves them if
    already computed, optimizing reprocessing for repeated analyses.
    """
    if curves_by_company:
        raise ValueError("Not implemented with company curves yet")

    if not axis:
        raise ValueError("It only works on an axis")

    companies = "all"

    sources = left_side + right_side

    if not os.path.exists(
        f"polarization values/Polarization between {left_side} VS {right_side} ; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}.csv"
    ):
        st.write("computing polarization...")
        choose_pol(
            left_side=left_side,
            right_side=right_side,
            curves_by_company=None,
            axis=axis,
            percentiles=percentiles,
            print_random_pol=print_random_pol,
            force_i_lim=force_i_lim,
            with_parliament=with_parliament,
            return_fig=False,
        )

    else:
        st.write("polarization already computed...")
        print("polarization already computed...")

    if with_parliament:
        df_proj = pd.read_csv("data/with parliament/current_dataframes/df.csv")
    if not with_parliament:
        df_proj = pd.read_csv(
            "data/without parliament/current_dataframes/df.csv"
        )
        df_proj["party"], df_proj["Speaker"] = 0, 0

    df_par = df_proj.loc[
        df_proj["source"].isin(sources) | df_proj["party"].isin(sources)
    ]

    # Further refine the DataFrame structure for analysis
    df_par = df_par[
        [
            "year",
            "party",
            "text",
            "source",
            "keywords",
            "Speaker",
            "cos axe 1",
            "cos axe 2",
        ]
    ]

    # Split the data into two DataFrames based on a specific source criterion
    df1 = df_par[df_par["source"] == "par"]
    df2 = df_par[df_par["source"] != "par"]

    # Define a function to translate newspaper source to party
    def translate_party(newspaper):
        """
        Translates newspaper sources to their corresponding political party.

        :param newspaper: The source to be translated.
        :return: The political party corresponding to the source.
        """
        if newspaper in left_side:
            return "Lab"
        if newspaper in right_side:
            return "Con"

    # Apply the translation function to assign parties based on sources
    df2["party"] = df2["source"].apply(translate_party)
    df2["Speaker"] = range(len(df2))

    # Combine the two DataFrames and reset index for continuity
    df_par = pd.concat([df1, df2]).reset_index(drop=True)

    df_par_grouped = df_par[["party", "year", f"cos axe {axis}"]]
    df_par_grouped = df_par_grouped.groupby(["party", "year"]).mean()
    df_par_grouped = df_par_grouped.reset_index()

    def change_year(old_year):
        if int(old_year) == 20110:
            return 2020
        if int(old_year) == 20111:
            return 2021
        if int(old_year) == 20112:
            return 2022
        if int(old_year) == 20113:
            return 2023
        else:
            return int(old_year)

    df_par["year"] = df_par["year"].apply(change_year)
    df_par_grouped["year"] = df_par_grouped["year"].apply(change_year)

    df_par_grouped = bootstrap(
        df_par_grouped, df_par, source_column="party", axis=axis
    )

    df_par_grouped["cos axe"] = df_par_grouped[f"cos axe {axis}"]

    df_pol = pd.read_csv(
        f"polarization values/Polarization between {left_side} VS {right_side} ; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}.csv"
    )

    real_pol = np.array(df_pol["real_pol"])
    CI_lows_real = np.array(df_pol["CI_lows_real"])
    CI_high_real = np.array(df_pol["CI_high_real"])

    Con_cos = np.array(
        df_par_grouped[df_par_grouped["party"] == "Con"]["cos axe"]
    )
    Con_CI_low = np.array(
        df_par_grouped[df_par_grouped["party"] == "Con"][f"CI_{axis}_inf"],
        dtype="float",
    )
    Con_CI_high = np.array(
        df_par_grouped[df_par_grouped["party"] == "Con"][f"CI_{axis}_sup"],
        dtype="float",
    )

    Lab_cos = np.array(
        df_par_grouped[df_par_grouped["party"] == "Lab"]["cos axe"]
    )
    Lab_CI_low = np.array(
        df_par_grouped[df_par_grouped["party"] == "Lab"][f"CI_{axis}_inf"],
        dtype="float",
    )
    Lab_CI_high = np.array(
        df_par_grouped[df_par_grouped["party"] == "Lab"][f"CI_{axis}_sup"],
        dtype="float",
    )

    x = [2010 + i for i in range(len(real_pol))]
    len_x = len(x)
    len_y = len(Con_cos)

    if len_y < len_x:
        x = x[len_x - len_y :]
        real_pol = real_pol[len_x - len_y :]
        CI_lows_real = CI_lows_real[len_x - len_y :]
        CI_high_real = CI_high_real[len_x - len_y :]

    # Initialize Plotly figure
    fig = go.Figure()

    # Add first set of data (real polarization) to the primary y-axis
    fig.add_trace(
        go.Scatter(
            x=x,
            y=real_pol,
            name="Polarisation réelle",
            line={"color": "blue", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=CI_high_real,
            line={"width": 0},
            showlegend=False,
            fill=None,
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=CI_lows_real,
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.1)",
            line={"width": 0},
            showlegend=False,
        )
    )

    # Add second set of data (cosine similarity of left side) to the secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Con_cos,
            name=f"Cosine similarity of {left_side}",
            line={"color": "green", "dash": "dash", "width": 2},
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Con_CI_high,
            line={"width": 0},
            showlegend=False,
            fill=None,
            mode="lines",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Con_CI_low,
            fill="tonexty",
            fillcolor="rgba(0, 128, 0, 0.1)",
            line={"width": 0},
            showlegend=False,
            yaxis="y2",
        )
    )

    # Add third set of data (cosine similarity of right side) to the secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Lab_cos,
            name=f"Cosine similarity of {right_side}",
            line={"color": "red", "dash": "dash", "width": 2},
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Lab_CI_high,
            line={"width": 0},
            showlegend=False,
            fill=None,
            mode="lines",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Lab_CI_low,
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.1)",
            line={"width": 0},
            showlegend=False,
            yaxis="y2",
        )
    )

    # Update layout for dual y-axes
    fig.update_layout(
        title=f"Polarization and cosine similarity between {left_side} vs {right_side}; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}",
        xaxis_title="Année",
        yaxis={
            "title": "Polarisation",
            "side": "left",
            "rangemode": "nonnegative",
        },
        yaxis2={
            "title": "Cosine similarity",
            "side": "right",
            "overlaying": "y",
            "rangemode": "tozero",
            "showgrid": False,
        },
        legend={
            "x": 0.01,
            "y": 0.99,
            "bordercolor": "Black",
            "borderwidth": 1,
        },
        width=1200,
        height=700,
    )

    # Add vertical lines for each year
    for year in x:
        fig.add_vline(
            x=year,
            line={"color": "gray", "dash": "dash", "width": 1},
            line_width=0.5,
        )

    # Display the figure
    return fig
