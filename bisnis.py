import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

st.set_page_config(
    page_title="Klasifikasi Penerima PIP dan KIP",
    page_icon='blood.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.write("""<h1>Aplikasi Klasifikasi Penerima PIP dan KIP</h1>""", unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = st.selectbox(
            'Menu',
            ['Preprocessing', 'Modeling'],
            index=0
        )

    if selected == 'Preprocessing':
        st.subheader('Normalisasi Data')
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

        st.subheader('Data Asli')
        st.dataframe(df, width=600)

        X = df.drop(columns=['Status'])
        y = df['Status'].values

        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X.values)  # Convert DataFrame to NumPy array
        scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

        st.subheader('Data Setelah Normalisasi')
        st.dataframe(scaled_df, width=600)
