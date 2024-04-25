import streamlit as st
from src.Axes.curves_plots import choose_projection_cos
import s3fs
import yaml

s3_config = yaml.safe_load(open("S3_config.yml"))
ssp_cloud = True
fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": s3_config["endpoint_url"]},
    key=s3_config["key"],
    secret=s3_config["secret"],
    token=s3_config["token"],
)
bucket = "nfarhan/diffusion/mise_en_production"

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
    tab1, tab2 = st.tabs(["Cosine similarity", "Euclidean distance"])

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
            ssp_cloud=ssp_cloud,
            fs=fs,
            bucket=bucket,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
