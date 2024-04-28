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
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Sidebar for navigation

analysis_type = st.sidebar.radio(
    "Directory", ("Home", "Description", "Curves Analysis", "Word Analysis")
)

if analysis_type == "Home":
    st.title("Tools for the anaylis of public opinion with a GloVe model")

    st.markdown(
        """
    ## Project Overview

    This project explores techniques described in recent scholarly papers and adapts them to analyze public opinion about BigTech companies in the UK. Our aim is to develop robust analysis tools tailored to this specific context.

    ## Objective

    The primary goal is to devise methods that can:
    - Track positions of newspapers and political parties on specific issues.
    - Identify words and phrases that are most indicative of their stance on BigTechs.

    ## Data Sources

    Our analysis tools are built upon a dense and balanced database of relevant texts, which includes:
    - Speeches from the House of Commons related to BigTechs, spanning from 2010 to 2019.
    - Articles from five major British newspapers covering the same theme from 2010 to 2023:
        - *The Guardian*
        - *The Telegraph*
        - *The Daily Mail*
        - *The Daily Express*
        - *Metro*

    ## Approach

    We adapt methodologies from referenced papers to suit the topic of public opinion on BigTechs in the UK, enhancing our ability to derive insightful analytics from textual data.
                
    We trained a GloVe model on the database, and we defined simple and relevant axes in the embeddings space. The axes defined aim to have a global feel of the opinion from the media and the politicians regarding tech companies. 
                
    The two axes we work with are :
    - Axis 1 : a positive pole formed of words describing economic technologic laisser-faire, and a negative pole formed of words describing more regulation. 
    - Axis 2 : a positive pole with positive words, and a negative pole with negative words.
                
    ## The two main parts

    #### Curves analysis 

    In this part we can visualize, given some filtering arguments, 2 types of curves :
    - The curves of cosine similarity of the embeddings of different corpuses with the axes.
    - The curves of polarization between two different corpuses. 

    We can also visualize them both on the same graphs, for the same sources and axis.
                
    #### Words analysis

    In this part we can visualize, given some filtering arguments, 3 types of measures :
    - The words with projected embeddings on an axis that are varying the most between two years.
    - The words that define the poles of the axes that are the most responsible for attracting the corpus towards them between two years.
    - A spectral clustering of the words embeddings of words in filtered corpuses.
    
                
    """,
        unsafe_allow_html=True,
    )

# Parameters section
elif analysis_type == "Description":
    st.title("Description of the tools")

    st.markdown(
        """
        We provide a description of the different tools we developped, as well as a guide of the different parameters that are used in the tools. 

        #### `with_parliament`
        There are actually two GloVe models that have been trained. One with a corpus of texts from the House of Commons and the texts from th media, and another one only on the corpus from the media.
        Therefore, before using a tool, you must indicate if you want to take the model trained on the corpus with the parliament, or not. 
        - The `with_parliament` parameter indicates if you want to work with the model trained on the corpus with the speeches from the parliament (ending in 2019), or only the corpus of newspapers (from 2010 to 2023). The parameter is either `True` or `False`.

        #### Abreviations :
        - `GUA` for *The Guardian*, `TE` for *The Telegraph*, `DM` for *The Daily Mail*, `DE` for The *Daily Express*, `par` for the whole Parliament, `Lab` for only the Labour party, `Con` for only the Conservative party.
        - `am` for Amazon, `fb` for Facebook, `ap` for Apple, `go` for Google, `mi` for Microsoft.
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Curves analysis", "Words analysis"])

    with tab1:
        st.markdown(
            """
        
        <details>
        <summary><strong>Cosine similarity curves</strong></summary>
        <br>
                    
        #### `axis` 
        - The `axis` parameter designates the currentlty defined axes of study. For now, `axis=1` corresponds to a positive pole of laisser-faire, and a negative pole of regulation. `axis=2` is a simple positive and negative axis. 
        For example :
        
                    axis = 1

        #### `sources` 
        - The `sources` parameter is a list of the newpapers, or parties considered in the computations. The liste must contain a subset of `['par', 'Lab', 'Con', 'GUA', 'TE', 'DM', 'DE', 'MET']`. It's a list of the sources we decided to aggregate the texts by, and display on the graph. 
        For example :
                    
                    sources = ["GUA", "TE", "DM"]

        #### `focus_on_companies` 
        - The `focus_on_companies` parameter allows to consider the curves from the different defined sources but restricting their corpuses to the companies given. The parameter is a sublist of `['fb', 'am', 'go', 'ap', 'mi']`. Leave it to `None` if you want to consider all the companies.
        For example :
                    
                    focus_on_companies = ['am', 'fb']

        #### `curves_by_company`
        - The `curves_by_company` method aggregates the entire corpus by the company mentionned in each text, then proceeds with these aggregates. This parameter is also a sublist of `['fb', 'am', 'go', 'ap', 'mi']`. You will thus have one curve by company selected. Leave it to `None` if you want to consider all the companies. For example :
                    
                    curves_by_company = ['am', 'fb']

        </details>
                    
        <details>
        <summary><strong>Polarization curves</strong></summary>
        <br>
                    
        #### `left` party and `right` party
        - To compute polarization in a corpus you need to split it into two separate parties, a `left` party and a `right` party. Each of these parameters is a sublist of `['par', 'Lab', 'Con', 'GUA', 'TE', 'DM', 'DE', 'MET']`. They cannot overlap. For example :
                    
                left = ["Lab"]
            and 
                    
                right = ["Con"]
                    
        #### `axis` 
        - The `axis` parameter allows to compute the polarization while considering texts close to the poles of the axis selected. This parameter goes hand in hand with the `percentiles` parameter. Given these parameters, we only keep the texts with a cosine with the `axis` outside of the quantiles given by the `percentiles`. Doing so, we only compute polarization on the texts relevant to the subjects defined by the axis. We leave them both to `None` if we don't want to consider this method. 
        For example :
        
                    axis = 1

            and       
                    
                    percentiles = [10, 90]


        #### `curves_by_company`
        - The `curves_by_company` method aggregates the entire corpus by the company mentionned in each text, then proceeds with these aggregates. This parameter is also a sublist of `['fb', 'am', 'go', 'ap', 'mi']`. You will thus have one curve by company selected. Leave it to `None` if you want to consider all the companies. For example :
                    
                    curves_by_company = ['am', 'fb']

        #### `print_random_pol`
        -The `print_random_pol` parameter is a boolean indicating if you want to plot the polarization computed on random labels. (It should be relatively close to 0.5). It is set to `True` by default.
                    
                    print_random_pol = True

        </details>
        
        """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            """
        
        <details>
        <summary><strong>Words variations</strong></summary>
        <br>
                    
        This tool allows to visualize, given different parameters, the words with the biggest embedding variations, when projected on an axis, between two years. It gives a first example of the terms driving the cosine similarity toward a pole of this axis.
                    
       #### `year` 
        - The year considered. For example :
                    
                    year = 2016

        #### Axes
        - The axes considered for each of the subplots. For example :
                    
                    first_axis = 1

            and 
                    
                    second_axis = 1

        #### Variations
        - For each subplot, towards which pole we are looking at the variations. Either positive or negative. 
                
        #### Number
        - The number of words you want to see :
                    
                    number = 20

        </details>
                    
        <details>
        <summary><strong>Axis variations</strong></summary>
        <br>
                    
        This tool allows to visualize, given different parameters, the words defining the axes poles that are the most responsible for attracting the corpus towards their respective pole. It gives another example of the terms driving the cosine similarity toward a pole of this axis.
                    
        #### `year` 
        - The year considered. For example :
                    
                    year = 2016

        #### `source`
        - The `source` parameter allows us to restrict our corpus to a specific source. The `source` must be one of `['par', 'GUA', 'TE', 'DM', 'DE', 'MET']`, or leave it to `None` if you want to consider all sources. For example :
                    
                    source = "GUA"

        #### Number
        - The number of words you want to see :
                    
                    number = 20

        </details>
                    
        <details>
        <summary><strong>Spectral clustering</strong></summary>
        <br>
                    
        This tool first selects texts positionned close to the axes poles and then performs spectral clustering on the words embeddings. This allows to build clusters of similar notions, and try to decipher explanations for the variations of the curves. 
                    
        #### `year` 
        - The year considered. For example :
                    
                    year = 2016

        #### `axis`
        - The `axis` considered :
                    
                    axis = 1
                        
        #### 1 - Percentiles method
        - We can select the relevant texts on this axis by taking texts with cosine values with the axis under and above two percentiles. Default is [10, 90] :
                    
                    percentiles = [10, 90]

        #### 2 - Thresholding method 
        - We can select the relevant texts on this axis by taking texts with cosine values with the axis under and above two thresholds. Default is [-0.3, 0.3]. For example :
                    
                    left_threshold = -0.3

            and 
                    

                    right_threshold = 0.3

        #### 3 - Head-tail method 
        - We can select the relevant texts on this axis by taking texts with the n highest and lowest cosine values with the axis. Default is 100 highest values and 100 lowest values. For example :
                    
                    head = 100

            and 
                    tail = 200

        #### **⚠️ For now, these 3 methods cannot be combined !**
                    
        #### `company`
        - If you want to focus on a company. The parameter is a sublist of `['fb', 'am', 'go', 'ap', 'mi']`. Leave it to `None` if you want to consider all the companies. For example :
                    
                    company = 'am'

        </details>
        
        """,
            unsafe_allow_html=True,
        )

# Curves Analysis Section
elif analysis_type == "Curves Analysis":
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
elif analysis_type == "Word Analysis":
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
