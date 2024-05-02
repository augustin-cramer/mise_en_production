"""This script defines the important function computing and 
plotting polarization values given parties and multiple variables, 
and storing the values and the plots in the `data` folder."""
import time
import os
import datetime

from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


from ..Polarization.polarization_functions import compute_polarization_and_CI
from ..Axes.projection_functions import df_BT
from ..GloVe.weights import standard_opening


def print_with_timestamp(message):
    """
    Prints a message with a timestamp.

    :param message: The message to print.
    """
    # Get the current time
    now = datetime.datetime.now()

    # Format the timestamp as desired
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Print the message with the timestamp
    print(f"{timestamp}: {message}")


def get_quantiles(data, percentiles):
    """
    Compute quantiles for a given dataset and percentiles.

    :param data: Numerical data from which to calculate quantiles.
    :param percentiles: A list of percentiles to calculate for the data.
    :return: An array of quantiles corresponding to the specified percentiles.
    """
    return np.percentile(data, percentiles)


def choose_pol(
    left_side,
    right_side,
    data_loader,
    curves_by_company=None,
    axis=None,
    percentiles=[10, 90],
    print_random_pol=True,
    force_i_lim=None,
    with_parliament=True,
    return_fig=True,
):
    """
    Analyzes and visualizes polarization based on political
    alignment, optionally segmented by company and axis.

    :param left_side: List of sources considered to be on the left side of the political spectrum.
    :param right_side: List of sources considered to be on the right side of the political spectrum.
    :param curves_by_company: Flag to indicate whether to analyze data by company.
    :param axis: Specifies the axis for projection if any.
    :param percentiles: Percentiles to use for filtering data based on 'cos axe' values.
    :param print_random_pol: Whether to print random polarization values.
    :param force_i_lim: Overrides the default limit for iterations if specified.
    """
    str_parliament = "with" if with_parliament else "without"

    # Merge left and right sources into a single list for easier handling
    sources = left_side + right_side

    # Define political parties (or groups) for analysis
    party_1 = "Lab"
    party_2 = "Con"

    # Set default iteration limits based on whether certain keywords are in sources
    if "Lab" in sources or "Con" in sources or "par" in sources:
        i_limit = 10
    else:
        i_limit = 14

    # Allow for manual override of iteration limits
    if force_i_lim:
        i_limit = force_i_lim

    # Initialize storage for results by company
    values_by_company = {}

    # Determine companies to analyze based on whether curves_by_company is set
    if curves_by_company:
        companies = ["am", "fb", "ap", "go", "mi"]
    else:
        companies = ["all"]

    for company in companies:
        if not data_loader.exists(
            f"polarization values/Polarization between {left_side} VS {right_side} ; axis = {axis}, companies = {company}, percentiles = {percentiles}, with parliament = {with_parliament}.csv"
        ):
            break

        st.markdown("Polarization values already computed...")

        fig = go.Figure()

        # Convert the company's metrics into a DataFrame and save to CSV
        df_pol = data_loader.read_csv(
            f"polarization values/Polarization between {left_side} VS {right_side} ; axis = {axis}, companies = {company}, percentiles = {percentiles}, with parliament = {with_parliament}.csv"
        )

        # Extract polarization metrics for plotting
        real_pol = np.array(df_pol["real_pol"])
        random_pol = np.array(df_pol["random_pol"])
        CI_lows_real = np.array(df_pol["CI_lows_real"])
        CI_high_real = np.array(df_pol["CI_high_real"])
        CI_lows_random = np.array(df_pol["CI_lows_random"])
        CI_high_random = np.array(df_pol["CI_high_random"])
        x = [2010 + i for i in range(len(real_pol))]

        # Plot real polarization with confidence intervals
        fig.add_trace(
            go.Scatter(
                x=x,
                y=real_pol,
                mode="lines",
                name=f"Real Polarization - {company}",
                line={"color": "blue"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=CI_high_real,
                mode="lines",
                name="Upper CI Real",
                line={"width": 0},
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=CI_lows_real,
                mode="lines",
                fill="tonexty",
                name="CI Real",
                line={"width": 0},
                fillcolor="rgba(0, 0, 255, 0.2)",
                showlegend=False,
            )
        )

        # Optionally plot random polarization with confidence intervals
        if print_random_pol:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=random_pol,
                    mode="lines",
                    name=f"Random Polarization - {company}",
                    line={"color": "red", "dash": "dash"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=CI_high_random,
                    mode="lines",
                    name="Upper CI Random",
                    line={"width": 0},
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=CI_lows_random,
                    mode="lines",
                    fill="tonexty",
                    name="CI Random",
                    line={"width": 0},
                    fillcolor="rgba(255, 165, 0, 0.2)",
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f"Polarization between {left_side} vs {right_side}; Axis = {axis}, Companies = {companies}, Percentiles = {percentiles}, With Parliament = {with_parliament}",
            xaxis_title="Year",
            yaxis_title="Polarization",
            legend_title="Metric",
            template="plotly_white",
        )

        # Show the figure
        return fig

    st.markdown("Polarization values not computed, starting computation...")
    # Load projection data if an axis is specified
    if axis:
        df_proj = data_loader.read_csv(f"{str_parliament}_parliament/current_dataframes/df.csv")

    # Main loop over each company (or all companies together)
    for company in companies:
        # Initialize data structures for storing various metrics
        values_by_company[company] = {
            "real_pol": [],
            "random_pol": [],
            "CI_lows_real": [],
            "CI_high_real": [],
            "CI_lows_random": [],
            "CI_high_random": [],
        }

        progress_bar = st.progress(0)
        start_time = time.time()
        status_text = st.empty()

        for i in tqdm(range(i_limit)):  # Progress bar for iteration
            if i < 10:
                year = 2010 + i
            else:
                year = 20110 + (i - 10)  # Dynamically generate year

            # Load data for the current year, with preprocessing
            if with_parliament:
                df = standard_opening(
                        data_loader,
                        f"{str_parliament}_parliament/FinalDataframes/FilteredFinalDataFrame_201{i}.csv",
                        True,
                    ).reset_index()
            if not with_parliament:
                df = standard_opening(
                        data_loader,
                        f"{str_parliament}_parliament/FinalDataframes/FilteredFinalDataFrame_201{i}_WP.csv",
                        True,
                    ).reset_index()
                df["party"], df["Speaker"] = 0, 0

            # Project data onto specified axis if applicable
            if axis:

                def to_phrase(list_of_words):
                    text = ""
                    for word in list_of_words:
                        text += word + "_"
                    return text

                df_proj_year = df_proj[df_proj["year"] == year].reset_index()

                df["to join"] = df["text"].apply(to_phrase)
                df_proj_year["to join"] = df_proj_year["text"]

                df_proj_year = df_proj_year[
                    ["cos axe 1", "cos axe 2", "to join"]
                ]

                df = pd.merge(df_proj_year, df, on="to join", how="inner")
                df["cos axe"] = df[f"cos axe {axis}"]

            else:
                df["cos axe"] = 0  # Default to 0 if no axis specified

            # Filter data based on sources and party
            df = df.loc[df["source"].isin(sources) | df["party"].isin(sources)]

            # Additional processing for company-specific data
            if curves_by_company:
                df = df_BT(
                    df
                )  # Presumably filters or processes data by company
                df = df[df["class"] == company]
            else:
                df["class"] = 0  # Default class if not processing by company

            # Further refine the DataFrame structure for analysis
            df = df[
                [
                    "year",
                    "party",
                    "text",
                    "source",
                    "keywords",
                    "Speaker",
                    "class",
                    "cos axe",
                ]
            ]

            # Split the data into two DataFrames based on a specific source criterion
            df1 = df[df["source"] == "par"]
            df2 = df[df["source"] != "par"]

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
            df = pd.concat([df1, df2]).reset_index(drop=True)

            # Filter data based on quantiles if axis and percentiles are specified
            if axis is not None:
                quantiles = get_quantiles(df["cos axe"], percentiles)
                df = df[
                    (df["cos axe"] < quantiles[0])
                    | (df["cos axe"] > quantiles[1])
                ]

            df = df[["year", "party", "text", "Speaker"]]

            # Compute polarization and confidence intervals
            values = compute_polarization_and_CI(
                df, year, party_1, party_2, with_parliament=with_parliament, data_loader=data_loader
            )

            # Store computed metrics in the respective lists within values_by_company
            metrics = [
                "real_pol",
                "random_pol",
                "CI_lows_real",
                "CI_high_real",
                "CI_lows_random",
                "CI_high_random",
            ]
            for metric, value in zip(metrics, values):
                values_by_company[company][metric].append(value)

            # Informative print statement indicating completion of the current year's computation
            progress = (i + 1) / i_limit
            progress_bar.progress(progress)
            st.markdown(f"Year 201{i} computed")

            elapsed_time = time.time() - start_time
            iterations_left = i_limit - (i + 1)
            if i > 0:
                avg_time_per_iteration = elapsed_time / (i + 1)
                estimated_time_left = avg_time_per_iteration * iterations_left
                time_left_display = f"Approximately {estimated_time_left:.2f} seconds remaining"
            else:
                time_left_display = "Calculating time remaining..."

            status_text.text(
                f"Year 201{i} computed, {iterations_left} iterations remaining. {time_left_display}"
            )

    status_text.empty()
    progress_bar.empty()

    # Initialize a dictionary to store polarization DataFrames by company
    df_pol_BT = {}

    # Set up the plotting figure with a specified size
    fig = go.Figure()

    # Loop over each company to plot polarization metrics
    for company in companies:
        # Convert the company's metrics into a DataFrame and save to CSV
        df_pol_BT[company] = pd.DataFrame(values_by_company[company])
        df_pol_BT[company].to_csv(
            f"polarization values/Polarization between {left_side} VS {right_side} ; axis = {axis}, companies = {company}, percentiles = {percentiles}, with parliament = {with_parliament}.csv",
            index=False,
        )

        # Extract polarization metrics for plotting
        real_pol = np.array(values_by_company[company]["real_pol"])
        random_pol = np.array(values_by_company[company]["random_pol"])
        CI_lows_real = np.array(values_by_company[company]["CI_lows_real"])
        CI_high_real = np.array(values_by_company[company]["CI_high_real"])
        CI_lows_random = np.array(values_by_company[company]["CI_lows_random"])
        CI_high_random = np.array(values_by_company[company]["CI_high_random"])
        x = [2010 + i for i in range(len(real_pol))]

        # Plot real polarization with confidence intervals
        fig.add_trace(
            go.Scatter(
                x=x,
                y=real_pol,
                mode="lines",
                name=f"Real Polarization - {company}",
                line={"color": "blue"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=CI_high_real,
                mode="lines",
                name="Upper CI Real",
                line={"width": 0},
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=CI_lows_real,
                mode="lines",
                fill="tonexty",
                name="CI Real",
                line={"width": 0},
                fillcolor="rgba(0, 0, 255, 0.2)",
                showlegend=False,
            )
        )

        # Optionally plot random polarization with confidence intervals
        if print_random_pol:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=random_pol,
                    mode="lines",
                    name=f"Random Polarization - {company}",
                    line={"color": "red", "dash": "dash"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=CI_high_random,
                    mode="lines",
                    name="Upper CI Random",
                    line={"width": 0},
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=CI_lows_random,
                    mode="lines",
                    fill="tonexty",
                    name="CI Random",
                    line={"width": 0},
                    fillcolor="rgba(255, 165, 0, 0.2)",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"Polarization between {left_side} vs {right_side}; Axis = {axis}, Companies = {companies}, Percentiles = {percentiles}, With Parliament = {with_parliament}",
        xaxis_title="Year",
        yaxis_title="Polarization",
        legend_title="Metric",
        template="plotly_white",
    )
    if return_fig is True:
        return fig
