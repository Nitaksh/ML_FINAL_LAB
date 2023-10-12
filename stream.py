import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('train.csv')

# Create a sidebar for feature selection
features = st.sidebar.multiselect('Select features to visualize', df.columns)

# Create a dropdown menu for visualization type selection
viz_type = st.sidebar.selectbox('Select visualization type', ['Histogram', 'Scatterplot'])

# Create the visualizations based on user selections
if viz_type == 'Histogram':
    for feature in features:
        plt.hist(df[feature])
        st.write(feature)
        st.pyplot()
elif viz_type == 'Scatterplot':
    sns.pairplot(df[features])
    st.pyplot()