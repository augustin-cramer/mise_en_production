import streamlit as st


def improve_style():
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
