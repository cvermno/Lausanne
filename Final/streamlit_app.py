import streamlit as st
import pandas as pd
import plotly.express as px

training = pd.read_csv("https://raw.githubusercontent.com/cvermno/ML-Project/main/Datasets/training_data.csv")
test = pd.read_csv("https://raw.githubusercontent.com/cvermno/ML-Project/main/Datasets/unlabelled_test_data.csv")

# Streamlit app
st.title('Petit Prof')
st.write(training)
