import streamlit as st
import numpy as np
import plotly.figure_factory as ff
from src.Axes.curves_plots import choose_projection_cos
import os
import s3fs

st.text(os.environ.get("AWS_ACCESS_KEY_ID", "NO AWS_ACCESS_KEY_ID"))
st.text(os.environ.get("AWS_S3_ENDPOINT", "NO ENDPOINT"))

S3_ENDPOINT_URL = os.environ.get("AWS_S3_ENDPOINT")
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})
st.text(fs.ls("."))

with_parliament = st.checkbox("With Parliament")


axis = st.selectbox("axis", (1, 2))

sources = st.multiselect(
    "Select the sources", ["par", "Lab", "Con", "GUA", "TE", "DM", "DE", "MET"]
)

focus_on_companies = st.multiselect(
    "Select the companies to focus on", ["fb", "am", "go", "ap", "mi"]
)

curves_by_company = st.multiselect(
    "Select the companies to plot", ["fb", "am", "go", "ap", "mi"]
)

if st.button("Generate graph"):
    # hist_data = []
    # # Add histogram data
    # for i in range(option1):
    #     hist_data.append(np.random.randn(option2) + 2 * (i - option1/2))

    # group_labels = [f'Group {i+1}' for i in range(option1)]
    # # Create distplot with custom bin_size
    # fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])

    tab1, tab2 = st.tabs(["Cosine similarity", "Plotly native theme"])

    with tab1:
        # Plot!
        if focus_on_companies == []:
            focus_on_companies = None
        if curves_by_company == []:
            curves_by_company = None
        fig = choose_projection_cos(
            axis,
            sources,
            focus_on_companies,
            curves_by_company,
            with_parliament,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
    # with tab2:
    #     # Plot!
    #     st.plotly_chart(fig, use_container_width=True, theme="streamlit")
