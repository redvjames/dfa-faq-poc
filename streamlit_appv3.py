__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from datetime import datetime, timezone, timedelta
import time
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path
import os

# Create columns for the title and logo
col1, col2 = st.columns([3.5, 1])  # Adjust the ratio as needed

# Title in the first column
with col1:
    st.title("🤖 SIGLA Proof of Concept")
    st.write(
        "This app screens if student is stunted or wasted"
        " based on the student's image."
    )
# Logo and "Developed by E-CAIR" text in the second column
with col2:
    st.image("images/CAIR_cropped.png", use_column_width=True)
    st.markdown(
        """
        <div style="text-align: center; margin-top: -10px;">
            Developed by CAIR
        </div>
        """,
        unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload image")

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    image = Image.open('uploaded_file')
    st.image(image)




