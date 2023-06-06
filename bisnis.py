import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
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
            ['Implementation'],
            index=0
        )

    if selected == "Implementation":
        # Read Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

        # Preprocessing data
        # Mendefinisikan Variable X dan Y
        X = df[['Nama', 'Jenis_Tinggal', 'Jenis_Pendidikan_Ortu_Wali', 'Pekerjaan_Ortu_Wali', 'Penghasilan_Ortu_Wali']]
        y = df['Status'].values

        # One-hot encoding pada atribut kategorikal
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X.astype(str)).toarray()
        features_names = encoder.get_feature_names_out(input_features=X.columns)
        scaled_features = pd.DataFrame(X_encoded, columns=features_names)

        # Split Data 
        training, test = train_test_split(scaled_features, test_size=0.2, random_state=1)  # Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)  # Nilai Y training dan Nilai Y testing

        # Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)
        probas = gaussian.predict_proba(test)
        probas = probas[:, 1]
        probas = probas.round()

        st.subheader("Implementasi Prediksi Penyakit Diabetes")
        nama = st.text_input('Masukkan nama:')
        jenis_tinggal = st.text_input('Masukkan jenis tinggal:')
        jenis_pendidikan_ortu_wali = st.text_input('Masukkan jenis pendidikan ortu atau wali:')
        pekerjaan_ortu_wali = st.text_input('Masukkan pekerjaan ortu atau wali:')
        penghasilan_ortu_wali = st.text_input('Masukkan penghasilan ortu atau wali:')
        model = 'Gaussian Naive Bayes'  # Menggunakan model Gaussian Naive Bayes secara langsung

        inputs = np.array([
            nama,
            jenis_tinggal,
            jenis_pendidikan_ortu_wali,
            pekerjaan_ortu_wali,
            penghasilan_ortu_wali
        ]).reshape(1, -1)

        # Ubah input menjadi tipe data string
        inputs = inputs.astype(str)

        # Transformasi one-hot encoding pada input data
        inputs_encoded = encoder.transform(inputs).toarray()

        if model == 'Gaussian Naive Bayes':
            mod = gaussian
            akurasi = round(100 * accuracy_score(test_label, probas))

        input_pred = mod.predict(inputs_encoded)

        st.subheader('Hasil Prediksi')
        st.write('Menggunakan Pemodelan:', model)
        st.write('Akurasi: {0:0.0f}'.format(akurasi), '%')

        if input_pred == 1:
            st.error('PIP')
        else:
            st.success('KIP')
