import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
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
            ['Oke' 'Implementation'],
            index=0
        )

    if selected == 'Oke':
        st.subheader('Normalisasi Data')

    elif selected == 'Implementation':
        st.subheader('Implementasi Prediksi Penerima PIP dan KIP')
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

        X = df[['Nama', 'Jenis_Tinggal', 'Jenis_Pendidikan_Ortu_Wali', 'Pekerjaan_Ortu_Wali', 'Penghasilan_Ortu_Wali']]
        y = df['Status'].values

        # One-hot encoding pada atribut kategorikal
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X).toarray()
        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X_encoded)
        scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=1)

        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)

        # Input data untuk prediksi
        input_data = st.text_input('Masukkan data untuk prediksi (pisahkan dengan koma):')
        input_data = list(map(str.strip, input_data.split(',')))

        # Mengecek jumlah input yang sesuai dengan jumlah kolom pada dataset
        if len(input_data) != len(X.columns):
            st.warning('Jumlah input tidak sesuai. Masukkan {} nilai.'.format(len(X.columns)))
        else:
            # Mengubah input menjadi dataframe
            input_df = pd.DataFrame([input_data], columns=X.columns)
            
            # Melakukan one-hot encoding pada input data
            input_encoded = encoder.transform(input_df).toarray()
            
            # Melakukan normalisasi pada input data
            scaled_input = scaler.transform(input_encoded)
            
            # Melakukan prediksi menggunakan model Gaussian Naive Bayes
            prediction = gaussian.predict(scaled_input)
            
            # Mengembalikan hasil prediksi ke label asli menggunakan inverse transform
            predicted_label = encoder.inverse_transform(prediction)[0]
            
            st.subheader('Hasil Prediksi')
            st.write('Prediksi: ', predicted_label)

