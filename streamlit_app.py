"""Streamlit app"""

import streamlit as st

from frontend.pages.home import display_home
from frontend.pages.description import display_description
from frontend.pages.curves_analysis import display_curves_analysis
from frontend.pages.word_analysis import display_word_analysis
from frontend.static.style import improve_style


def main():
    improve_style()
    # Sidebar for navigation
    analysis_type = st.sidebar.radio(
        "Directory",
        ("Home", "Description", "Curves Analysis", "Word Analysis"),
    )

    if analysis_type == "Home":
        display_home()

    # Parameters section
    elif analysis_type == "Description":
        display_description()

    # Curves Analysis Section
    elif analysis_type == "Curves Analysis":
        display_curves_analysis()

    # Word Analysis Section
    elif analysis_type == "Word Analysis":
        display_word_analysis()


if __name__ == "__main__":
    main()
