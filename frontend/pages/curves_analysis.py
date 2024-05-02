import streamlit as st

# from src.Word_analysis.cluster_words import *
from src.Polarization.polarization_plots import choose_pol
from src.Axes.curves_plots import choose_projection_cos
from src.Polarization.cos_pol import draw_cos_pol


def display_curves_analysis(data_loader):
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
                data_loader,
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
                data_loader=data_loader,
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
