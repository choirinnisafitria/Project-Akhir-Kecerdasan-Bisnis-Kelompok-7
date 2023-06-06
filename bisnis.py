import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt

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
            ['Preprocessing', 'Modeling', 'Implementation'],
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
        scaled_X = scaler.fit_transform(X)
        scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

        st.subheader('Data Setelah Normalisasi')
        st.dataframe(scaled_df, width=600)

    elif selected == 'Modeling':
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

        X = df.drop(columns=['Status'])
        y = df['Status'].values

        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
        scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=1)

        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)
        y_pred = gaussian.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader('Hasil Modeling dengan Naive Bayes')
        st.write('Akurasi: {:.2f}%'.format(accuracy * 100))
    elif selected == "Implementation":
        with st.form("Implementation"):
            # Read Dataset
            df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

            # Preprocessing data
            # Mendefinisikan Variable X dan Y
            X = df.drop(columns=['Status'])
            y = df['Status'].values

            # NORMALISASI NILAI X
            scaler = MinMaxScaler()
            scaled_X = scaler.fit_transform(X)
            scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=1)

            # Gaussian Naive Bayes
            gaussian = GaussianNB()
            gaussian.fit(X_train, y_train)

            st.subheader("Implementasi Prediksi Penyakit Diabetes")
            nama = st.number_input('Masukkan nama:')
            jenist_inggal = st.number_input('Masukkan jenis jinggal:')
            jenjang_pendidikan_ortu_wali = st.number_input('Masukkan jenis pendidikan ortu atau wali:')
            pekerjaan_ortu_wali = st.number_input('Masukkan pekerjaan ortu atau wali:')
            penghasilan_ortu_wali = st.number_input('Masukkan penghasilan ortu atau wali:')
            model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                                ('Gaussian Naive Bayes'))

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    nama,
                    jenist_inggal,
                    jenjang_pendidikan_ortu_wali,
                    pekerjaan_ortu_wali,
                    penghasilan_ortu_wali
                ])

                input_norm = scaler.transform([inputs])

                if model == 'Gaussian Naive Bayes':
                    y_pred = gaussian.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan:', model)

                if y_pred == 1:
                    st.error('Positive')
                else:
                    st.success('Negative')
