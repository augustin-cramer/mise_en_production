import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import time
import os
from src.Axes.curves_plots import choose_projection_cos
from src.Polarization.polarization_plots import choose_pol
from src.Polarization.cos_pol import draw_cos_pol
from src.Word_analysis.words_variation import word_variations
from src.Word_analysis.axis_variation import axis_variation
from src.Word_analysis.cluster_words import *

# Style improvements
st.markdown("""
<style>
div.stButton > button {
    font-size: 16px;
    height: 2.5em;
    width: 100%;
    margin: 0.5em 0;
}
div.st-cm {
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
analysis_type = st.sidebar.radio(
    "Choose Analysis Type",
    ('Home', 'Curves Analysis', 'Word Analysis')
)

if analysis_type == 'Home':
    st.title("Welcome to Our Data Visualization App")
    st.markdown("""
    This app allows you to visualize and analyze data with interactive graphs and filters.
    Explore data by selecting various analysis types from the sidebar. Enjoy a customized
    visualization experience with powerful graphing tools.
    """)

# Curves Analysis Section
elif analysis_type == 'Curves Analysis':
    st.title("Curves Analysis")
    type_of_curve = st.multiselect(
        "Select the type of curve you want",
        [
            "Cosine similarity",
            "Polarization",
            "Comparison of cosine and polarization on an axis",
        ],
        max_selections=1,
    )

    # Cos
    ###################################################################################################
    if type_of_curve == ["Cosine similarity"]:
        with_parliament = st.checkbox("With Parliament")
        sources = st.multiselect(
            "Select the sources",
            ["par", "Lab", "Con", "GUA", "TE", "DM", "DE", "MET"],
        )
        st.markdown(
            'To select "par", "Lab" or "Con", you need to take the database with the parliament speeches. If no sources selected, all will be taken into account.'
        )

        axis = st.selectbox("Axis", (1, 2))

        focus_on_companies = st.multiselect(
            "Select the companies to focus on",
            ["fb", "am", "go", "ap", "mi"],
            default=None,
        )
        st.write(
            "Does not work for polarization. If none, all will be taken into account."
        )
        curves_by_company = st.multiselect(
            "Select the companies to plot",
            ["fb", "am", "go", "ap", "mi"],
            default=None,
        )
        st.write("If none, we don't plot a company.")

        if st.button("Generate graph"):
            if sources == []:
                sources = ["par", "GUA", "TE", "DE", "DM", "MET"]

            tab1, tab2 = st.tabs(["Cosine similarity", "Word analysis ?"])
            with tab1:
                fig = choose_projection_cos(
                    axis,
                    sources,
                    focus_on_companies,
                    curves_by_company,
                    with_parliament,
                )
                st.plotly_chart(fig, use_container_width=True)

    # Pol
    ###################################################################################################
    if type_of_curve == ["Polarization"]:
        with_parliament = st.checkbox("With Parliament")

        st.markdown(
            "To compute polarization, you have to separate sources in two parts :"
        )
        first_part = st.multiselect(
            "Select the sources for the first side",
            ["par", "Lab", "Con", "GUA", "TE", "DM", "DE", "MET"],
        )

        second_part = st.multiselect(
            "Select the sources for the second side",
            ["par", "Lab", "Con", "GUA", "TE", "DM", "DE", "MET"],
        )
        st.markdown("The two sides should not overlap.")

        axis = st.selectbox("Axis", (1, 2, None))

        percentiles = [10, 90]

        if axis:
            st.markdown(
                "When we compute polarization on an axis, we select the relevant texts on this axis by taking texts with cosine values with the axis under and above two percentiles. Default is [10; 90], you can modify if here if you want."
            )
            lower_bound = st.number_input(
                "Enter a lower percentile",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                value=10.0,
            )
            st.write("You entered:", lower_bound)
            upper_bound = st.number_input(
                "Enter an upper percentile",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                value=90.0,
            )
            st.write("You entered:", upper_bound)
            percentiles = [int(lower_bound), int(upper_bound)]

        curves_by_company = st.multiselect(
            "Do you want to plot by company ?", ["Yes"]
        )
        st.markdown("If none, we don't plot by company.")
        if not curves_by_company:
            curves_by_company = None
        else:
            curves_by_company = True

        if st.button("Generate graph"):
            fig = choose_pol(
                first_part,
                second_part,
                curves_by_company=curves_by_company,
                axis=axis,
                percentiles=percentiles,
                with_parliament=with_parliament,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Cos Pol
    ###################################################################################################
    if type_of_curve == ["Comparison of cosine and polarization on an axis"]:
        with_parliament = st.checkbox("With Parliament")

        st.markdown(
            "To compute polarization, you have to separate sources in two parts :"
        )
        first_part = st.multiselect(
            "Select the sources for the first side",
            ["par", "Lab", "Con", "GUA", "TE", "DM", "DE", "MET"],
        )

        second_part = st.multiselect(
            "Select the sources for the second side",
            ["par", "Lab", "Con", "GUA", "TE", "DM", "DE", "MET"],
        )
        st.markdown("The two sides should not overlap.")

        axis = st.selectbox("Axis", (1, 2))

        percentiles = [10, 90]

        if axis:
            st.markdown(
                "When we compute polarization on an axis, we select the relevant texts on this axis by taking texts with cosine values with the axis under and above two percentiles. Default is [10; 90], you can modify if here if you want."
            )
            lower_bound = st.number_input(
                "Enter a lower percentile",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                value=10.0,
            )
            st.write("You entered:", lower_bound)
            upper_bound = st.number_input(
                "Enter an upper percentile",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                value=90.0,
            )
            st.write("You entered:", upper_bound)
            percentiles = [int(lower_bound), int(upper_bound)]

        if st.button("Generate graph"):
            fig = draw_cos_pol(
                first_part,
                second_part,
                axis=axis,
                percentiles=percentiles,
                with_parliament=with_parliament,
            )
            st.plotly_chart(fig, use_container_width=True)

# Word Analysis Section
elif analysis_type == 'Word Analysis':

    # Word analysis part
    ###################################################################################################
    st.title("Word Analysis")

    type_of_analysis = st.multiselect(
        "Select the type of words analysis you want",
        ["Embeddings variation", "Axis variation", "Spectral Clustering"],
        max_selections=1,
    )

    # Embeddings variation
    ###################################################################################################
    if type_of_analysis == ["Embeddings variation"]:
        with_parliament = st.checkbox("With Parliament")

        with st.expander("Embeddings Variation", expanded=True):
            # Input for year with a slider
            year = st.slider(
                "Select the year you want to study. You will get the variation between the year selected and the previous year.",
                min_value=2010,
                max_value=2023,
                value=2022,
                step=1,
            )

        variation_map = {"positive": "up", "negative": "down"}

        first_axis = st.selectbox(
            "Select the axis for the first variation analysis.",
            options=[1, 2],
            index=0,
        )
        variation_1 = st.radio(
            "For the first axis, select the direction of embedding variation:",
            options=["positive", "negative"],
            key="first_variation",
        )
        variation_1 = variation_map[variation_1]

        second_axis = st.selectbox(
            "Select the axis for the second variation analysis.",
            options=[1, 2],
            index=0,
        )
        variation_2 = st.radio(
            "For the second axis, select the direction of embedding variation:",
            options=["positive", "negative"],
            key="second_variation",
        )
        variation_2 = variation_map[variation_2]

        number = st.number_input(
            "Enter the number of words you want to display.",
            min_value=10,
            max_value=100,
            value=20,
        )

        if st.button("Generate graph"):
            fig = word_variations(
                year=year,
                axis_1=first_axis,
                axis_2=second_axis,
                variation_1=variation_1,
                variation_2=variation_2,
                with_parliament=with_parliament,
                number=number,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Axis variation
    ###################################################################################################
    if type_of_analysis == ["Axis variation"]:
        with_parliament = st.checkbox("With Parliament")

        with st.expander("Embeddings Variation", expanded=True):
            # Input for year with a slider
            year = st.slider(
                "Select the year you want to study. You will get the variation between the year selected and the previous year.",
                min_value=2010,
                max_value=2023,
                value=2022,
                step=1,
            )

        variation_map = {"positive": "up", "negative": "down"}

        axis = st.radio(
            "Select the axis for which you want to what the words in the poles attract the corpus towards them.",
            options=[1, 2],
        )

        source = st.multiselect(
            "For which source do you want to do the analysis ?",
            ["par", "GUA", "TE", "DM", "DE", "MET"],
            default="GUA",
            max_selections=1,
        )

        number = st.number_input(
            "Enter the number of words you want to display.",
            min_value=10,
            max_value=100,
            value=20,
        )

        if st.button("Generate graph"):
            fig = axis_variation(
                axis=axis,
                source=source[0],
                year=year,
                number_of_words=number,
                with_parliament=with_parliament,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Spectral clustering
    ###################################################################################################
    if type_of_analysis == ["Spectral Clustering"]:
        with_parliament = st.checkbox("With Parliament")

        with st.expander("Embeddings Variation", expanded=True):
            # Input for year with a slider
            year = st.slider(
                "Select the year you want to study. You will get the variation between the year selected and the previous year.",
                min_value=2010,
                max_value=2023,
                value=2022,
                step=1,
            )

        axis = st.radio(
            "Select the axis for which you want to do the analysis.", options=[1, 2]
        )

        method = st.radio(
            "To select the restricted corpus on which you want to perform the clustering, what method do you want to use ?",
            options=["percentiles", "thresholding", "head-tail"],
        )

        left_threshold = None
        right_threshold = None
        head = None
        tail = None
        percentiles = None

        if method == "percentiles":
            st.markdown(
                "We can select the relevant texts on this axis by taking texts with cosine values with the axis under and above two percentiles. Default is [10, 90], you can modify if here if you want."
            )
            lower_bound = st.number_input(
                "Enter a lower percentile",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                value=10.0,
            )
            st.write("You entered:", lower_bound)
            upper_bound = st.number_input(
                "Enter an upper percentile",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                value=90.0,
            )
            st.write("You entered:", upper_bound)
            percentiles = [int(lower_bound), int(upper_bound)]

        if method == "thresholding":
            st.markdown(
                "We can select the relevant texts on this axis by taking texts with cosine values with the axis under and above two thresholds. Default is [-0.3, 0.3], you can modify if here if you want."
            )
            lower_bound = st.number_input(
                "Enter a lower threshold",
                min_value=-1.0,
                max_value=1.0,
                step=0.1,
                value=-0.3,
            )
            st.write("You entered:", lower_bound)
            upper_bound = st.number_input(
                "Enter an upper threshold",
                min_value=-1.0,
                max_value=1.0,
                step=0.1,
                value=0.3,
            )
            st.write("You entered:", upper_bound)

            left_threshold, right_threshold = lower_bound, upper_bound

        if method == "head-tail":
            st.markdown(
                "We can select the relevant texts on this axis by taking texts with the n highest and lowest cosine values with the axis. Default is 100 highest values and 100 lowest values, you can modify if here if you want."
            )
            lower_bound = st.number_input(
                "Enter a number for the tail",
                min_value=1,
                max_value=5000,
                step=50,
                value=100,
            )
            st.write("You entered:", lower_bound)
            upper_bound = st.number_input(
                "Enter a number for the head",
                min_value=1,
                max_value=5000,
                step=50,
                value=100,
            )
            st.write("You entered:", upper_bound)
            bounds = [lower_bound, upper_bound]

            tail, head = lower_bound, upper_bound

        company = st.multiselect(
            "Do you want to focus on a company ?",
            [None, "fb", "am", "go", "ap", "mi"],
            default=None,
        )[0]

        if st.button("Generate graph"):
            fig_1, fig_2, data, df_t = cluster_words_intermediate(
                year=year,
                axis=1,
                percentiles=percentiles,
                with_parliament=with_parliament,
                company=company,
                right_threshold=right_threshold,
                left_threshold=left_threshold,
                head=head,
                tail=tail,
            )

            st.plotly_chart(fig_1, use_container_width=True)
            st.plotly_chart(fig_2, use_container_width=True)

            n_clusters = st.number_input(
                "Enter the number of clusters you want to use:", min_value=1, step=1
            )
            st.write(n_clusters)

            if st.button("View clusters"):
                fig = display_clusters(n_clusters, data, df_t)
                st.plotly_chart(fig, use_container_width=True)
