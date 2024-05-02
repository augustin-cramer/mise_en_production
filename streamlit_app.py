"""Streamlit app"""

import streamlit as st
import yaml
import s3fs

from src.S3_Bucket_utils.read_data import DataLoader

from frontend.pages.home import display_home
from frontend.pages.description import display_description
from frontend.pages.curves_analysis import display_curves_analysis
from frontend.pages.word_analysis import display_word_analysis
from frontend.static.style import improve_style

S3_CONFIG = yaml.safe_load(open("S3_config.yml"))
BUCKET = "nfarhan/diffusion/mise_en_production/"

try:
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": S3_CONFIG["endpoint_url"]},
        key=S3_CONFIG["key"],
        secret=S3_CONFIG["secret"],
        token=S3_CONFIG["token"],
    )
    fs.ls(BUCKET)
    connection = {"fs": fs, "bucket": BUCKET}
except:
    connection = None

data_loader = DataLoader(connection)


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
        display_curves_analysis(data_loader)

    # Word Analysis Section
    elif analysis_type == "Word Analysis":
        display_word_analysis(data_loader)


if __name__ == "__main__":
    main()
