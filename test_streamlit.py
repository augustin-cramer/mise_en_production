import streamlit as st
import numpy as np
import plotly.figure_factory as ff

option1 = int(st.selectbox(
    'enter the number of groups',
    ('1', '2', '3')))

option2 = st.selectbox(
    'enter the number of samples',
    (100, 200, 1000))

hist_data = []
# Add histogram data
for i in range(option1):
    hist_data.append(np.random.randn(option2) + 2 * (i - option1/2))


group_labels = [f'Group {i+1}' for i in range(option1)]

# Create distplot with custom bin_size
fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.plotly_chart(fig, use_container_width=True)

