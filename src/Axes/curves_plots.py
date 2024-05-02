"""Builds the main function to plot the cosine between selected sources and axis, with multple parameters available."""

import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from ..S3_Bucket_utils.read_data import read_csv_bucket
from ..Axes.projection_functions import *
from ..Axes.bootstraping import *


def format_year(integer):
    """
    Format a year integer to ensure it is a valid four-digit year.

    If the integer has more than four digits, it combines all but the
    second-to-last digit with the last digit. Otherwise, it returns the integer unchanged.

    Parameters:
    - integer (int): The year integer to format.

    Returns:
    - int: A four-digit year integer.
    """
    if integer > 20109:
        return integer - 18090
    else:
        return integer


def plot_from_sources(df_all_grouped, axis, sources):
    """
    Plots cosine similarity values from different sources over years,
    including confidence intervals.

    Parameters:
    - df_all_grouped (DataFrame): A pandas DataFrame containing the grouped data.
    - axis (int): The axis number to plot the cosine similarity for.
    - sources (list): A list of sources to include in the plot.
    """
    # Initialize dictionaries to store data
    cos = {source: [] for source in sources}
    CI_low = {source: [] for source in sources}
    CI_high = {source: [] for source in sources}

    # Loop through sources to gather data
    for source in sources:
        cos[source] = np.array(
            df_all_grouped[df_all_grouped["source"] == source][
                "cos axe " + str(axis)
            ]
        )
        CI_low[source] = np.array(
            df_all_grouped[df_all_grouped["source"] == source][
                "CI_" + str(axis) + "_inf"
            ],
            dtype=float,
        )
        CI_high[source] = np.array(
            df_all_grouped[df_all_grouped["source"] == source][
                "CI_" + str(axis) + "_sup"
            ],
            dtype=float,
        )

    # Plot setup
    plt.figure(figsize=(12, 6))

    # Plot each source's data
    for source in sources:
        x = df_all_grouped[df_all_grouped["source"] == source]["year"]
        plt.plot(
            x,
            cos[source],
            label="Cosine similarity on axis "
            + str(axis)
            + " of "
            + str(source),
            linewidth=2,
        )
        plt.fill_between(x, CI_low[source], CI_high[source], alpha=0.2)

    # Customize plot appearance
    plt.title("On axis " + str(axis), fontsize=16)
    plt.xlabel("Année", fontsize=14)
    plt.ylabel("Similarité cosinus", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()  # Adjust layout
    plt.show()


def plot_from_sources_plotly(df_all_grouped, axis, sources):
    """
    Plots cosine similarity values from different sources over years,
    including confidence intervals, using Plotly.

    Parameters:
    - df_all_grouped (DataFrame): A pandas DataFrame containing the grouped data.
    - axis (int): The axis number to plot the cosine similarity for.
    - sources (list): A list of sources to include in the plot.
    """
    # Initialize the Plotly figure
    fig = go.Figure()

    # Loop through each source to add to the plot
    for source in sources:
        # Filter data for the current source
        df_source = df_all_grouped[df_all_grouped["source"] == source]

        # Extract data
        x = df_source["year"]
        y = df_source[f"cos axe {axis}"]
        ci_low = df_source[f"CI_{axis}_inf"]
        ci_high = df_source[f"CI_{axis}_sup"]

        # Add the main line plot
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Cosine similarity on axis {axis} of {source}",
                line={"width": 2},
            )
        )

        # Add confidence interval area
        fig.add_trace(
            go.Scatter(
                x=list(x) + list(x)[::-1],  # x, then x reversed
                y=list(ci_high) + list(ci_low)[::-1],
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line={"color": "rgba(255,255,255,0)"},
                showlegend=False,
                name=f"Confidence Interval {source}",
            )
        )

    # Update plot layout
    fig.update_layout(
        title=f"Cosine Similarity on Axis {axis}",
        xaxis_title="Year",
        yaxis_title="Cosine Similarity",
        legend_title="Source",
        template="plotly_white",
    )

    # Show the figure
    return fig


def choose_projection_cos(
    axis=1,
    sources=["par", "GUA", "TE", "DM", "DE", "MET"],
    focus_on_companies=None,
    curves_by_company=None,
    with_parliament=True,
    data_loader=None,
):
    """
    Chooses the projection of cosine similarity to plot, validates inputs,
    and manages dataframes for plotting.

    Parameters:
    - axis (int): The axis to use for the cosine similarity projection.
    - sources (list): A list of initial sources to include in the analysis.
    - focus_on_companies (list, optional): Specific companies to focus on.
    - curves_by_company (list, optional): Specific companies to generate curves for.
    - ssp_cloud (bool) : boolean to decide if we retrieve data from SSP Cloud.
    - fs (bucket object) : s3fs object linked to the S3 bucket, used is ssp_cloud=True.
    - bucket (str): name of the bucket where data is stored

    Raises:
    - ValueError: If an invalid source or company is provided.
    """
    # Data loading
    parliament_str = "with" if with_parliament else "without"
    # if connection["fs"] is not None:
        
    #     if with_parliament:
    #         df = read_csv_bucket(
    #             fs, bucket + "/with_parliament/current_dataframes/df.csv"
    #         )
    #         df_BT = read_csv_bucket(
    #             fs, bucket + "/with_parliament/current_dataframes/df_BT.csv"
    #         )

    #     if not with_parliament:
    #         df = read_csv_bucket(
    #             fs, bucket + "/without_parliament/current_dataframes/df.csv"
    #         )
    #         df_BT = read_csv_bucket(
    #             fs, bucket + "/without_parliament/current_dataframes/df_BT.csv"
    #         )

    # if not ssp_cloud:
    #     if with_parliament:
    #         df = pd.read_csv("data/with parliament/current_dataframes/df.csv")
    #         df_BT = pd.read_csv(
    #             "data/with parliament/current_dataframes/df_BT.csv"
    #         )

    #     if not with_parliament:
    #         df = pd.read_csv(
    #             "data/without parliament/current_dataframes/df.csv"
    #         )
    #         df_BT = pd.read_csv(
    #             "data/without parliament/current_dataframes/df_BT.csv"
    #         )
    df = data_loader.read_csv(f"{parliament_str}_parliament/current_dataframes/df.csv")
    df_BT = data_loader.read_csv(f"{parliament_str}_parliament/current_dataframes/df_BT.csv")
    
    # Validate the sources parameter
    if sources is not None:
        for source in sources:
            if source not in [
                "par",
                "Con",
                "Lab",
                "GUA",
                "TE",
                "DM",
                "DE",
                "MET",
            ]:
                raise ValueError(
                    "source parameter must be one of ['par', 'Con', 'Lab', 'GUA', 'TE', 'DM', 'DE', 'MET']"
                )

    # Focus on specific companies if specified
    if focus_on_companies is not None:
        for company in focus_on_companies:
            if company not in ["am", "fb", "go", "ap", "mi"]:
                raise ValueError(
                    "company parameter must be one of ['am', 'fb', 'go', 'ap', 'mi']"
                )
        df = df_BT[df_BT["class"].isin(focus_on_companies)]

    df_cbc = None
    if curves_by_company is not None:
        for company in curves_by_company:
            if company not in ["am", "fb", "go", "ap", "mi"]:
                raise ValueError(
                    "company parameter must be one of ['am', 'fb', 'go', 'ap', 'mi']"
                )

        # Preparing data for company-specific curves
        df_inter = df_BT[df_BT["class"].isin(curves_by_company)]
        df_inter["source"] = df_inter["class"]  # Assigning company as source
        df_cbc = pd.concat((df_inter, df))  # Combining data

    # Data transformation for party sources
    df_inter = df[df["source"] == "par"]
    df_inter["source"] = df_inter["party"]  # Relabeling party data
    df = pd.concat((df, df_inter, df_cbc))  # Consolidating data

    # Filtering and grouping the data
    df_all_grouped = df[["source", "year", "cos axe 1", "cos axe 2"]]
    if sources is not None and curves_by_company is not None:
        sources += curves_by_company
    if sources is None and curves_by_company is not None:
        sources = curves_by_company
    df_all_grouped = df_all_grouped[df_all_grouped["source"].isin(sources)]

    # Additional data preparation based on source type
    if "par" in sources or "Con" in sources or "Lab" in sources:
        df_all_grouped = df_all_grouped[
            df_all_grouped["year"] < 2020
        ]  # Filter by year
    else:
        df_all_grouped["year"] = df_all_grouped["year"].apply(
            format_year
        )  # Format years
        df["year"] = df["year"].apply(format_year)

    # Grouping and averaging the data
    df_all_grouped = (
        df_all_grouped.groupby(["source", "year"]).mean().reset_index()
    )

    # Bootstrapping for confidence intervals
    df_all_grouped = bootstrap(
        df_all_grouped, df, source_column="source", axis=axis
    )

    # Plotting the final visualization
    return plot_from_sources_plotly(df_all_grouped, axis, sources)
