from ..Polarization.polarization_plots import *
from ..Axes.bootstraping import *
import os
import plotly.graph_objects as go
import streamlit as st


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
    if curves_by_company:
        raise ValueError("Not implemented with company curves yet")

    if not axis:
        raise ValueError("It only works on an axis")

    companies = "all"

    sources = left_side + right_side

    if not os.path.exists(
        f"polarization values/Polarization between {left_side} VS {right_side} ; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}.csv"
    ):
        print("computin polarization...")
        st.write("computin polarization...")
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
    random_pol = np.array(df_pol["random_pol"])
    CI_lows_real = np.array(df_pol["CI_lows_real"])
    CI_high_real = np.array(df_pol["CI_high_real"])
    CI_lows_random = np.array(df_pol["CI_lows_random"])
    CI_high_random = np.array(df_pol["CI_high_random"])

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
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=CI_high_real,
            line=dict(width=0),
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
            line=dict(width=0),
            showlegend=False,
        )
    )

    # Add second set of data (cosine similarity of left side) to the secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Con_cos,
            name=f"Cosine similarity of {left_side}",
            line=dict(color="green", dash="dash", width=2),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Con_CI_high,
            line=dict(width=0),
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
            line=dict(width=0),
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
            line=dict(color="red", dash="dash", width=2),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Lab_CI_high,
            line=dict(width=0),
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
            line=dict(width=0),
            showlegend=False,
            yaxis="y2",
        )
    )

    # Update layout for dual y-axes
    fig.update_layout(
        title=f"Polarization and cosine similarity between {left_side} vs {right_side}; axis = {axis}, companies = {companies}, percentiles = {percentiles}, with parliament = {with_parliament}",
        xaxis_title="Année",
        yaxis=dict(title="Polarisation", side="left", rangemode="nonnegative"),
        yaxis2=dict(
            title="Cosine similarity",
            side="right",
            overlaying="y",
            rangemode="tozero",
            showgrid=False,
        ),
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        width=1200,
        height=700,
    )

    # Add vertical lines for each year
    for year in x:
        fig.add_vline(
            x=year,
            line=dict(color="gray", dash="dash", width=1),
            line_width=0.5,
        )

    # Display the figure
    return fig
