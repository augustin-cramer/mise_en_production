import streamlit as st
from src.Word_analysis.words_variation import word_variations
from src.Word_analysis.axis_variation import axis_variation
from src.Word_analysis.cluster_words import (
    cluster_words_intermediate,
    display_clusters,
)


def display_word_analysis():
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
            "Select the axis for which you want to do the analysis.",
            options=[1, 2],
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

        if "data" not in st.session_state or "df_t" not in st.session_state:
            st.session_state["data"] = None
            st.session_state["df_t"] = None

        # Button to generate silhouette and sse
        if st.button("Generate silhouette and sse"):
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

            st.session_state["data"] = data
            st.session_state["df_t"] = df_t

            st.plotly_chart(fig_1, use_container_width=True)
            st.plotly_chart(fig_2, use_container_width=True)

        # Input for number of clusters
        n_clusters = st.number_input(
            "Enter the number of clusters you want to use:",
            min_value=1,
            step=1,
            value=5,
        )
        st.write(f"You chose {n_clusters} clusters.")

        if st.button("View clusters"):
            if (
                st.session_state.get("data") is not None
                and st.session_state.get("df_t") is not None
            ):
                fig_1, figures = display_clusters(
                    n_clusters,
                    st.session_state["data"],
                    st.session_state["df_t"],
                )
                # Store figures in session state to avoid losing them on rerun
                st.session_state["figures"] = figures
                st.session_state["fig_1"] = fig_1
            else:
                st.error(
                    "Please generate the data first by clicking 'Generate silhouette and sse'"
                )

        # Tabs outside the button check to preserve the state
        if "figures" in st.session_state and "fig_1" in st.session_state:
            tab1, tab2 = st.tabs(
                [
                    "Most important words in clusters",
                    "Plotting on the 3 first pc",
                ]
            )

            with tab1:
                # Default selection is the first cluster if available
                cluster_number = st.multiselect(
                    "Select Cluster Number:", range(n_clusters), default=[0]
                )
                # Display plots for selected clusters
                if cluster_number:
                    for cn in cluster_number:
                        fig = st.session_state["figures"][cn]
                        st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.plotly_chart(
                    st.session_state["fig_1"], use_container_width=True
                )
