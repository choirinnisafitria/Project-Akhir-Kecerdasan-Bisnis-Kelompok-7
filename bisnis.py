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

    elif selected == 'Implementation':
        st.subheader('Implementasi Prediksi Penerima PIP dan KIP')
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/dataset.csv')

        X = df.drop(columns=['Status'])
        y = df['Status'].values

        # Mengubah DataFrame menjadi array numerik
        X_values = X.values
        scaled_X = scaler.fit_transform(X_values)

        scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=1)

        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)

        st.subheader('Implementasi Prediksi Penerima PIP dan KIP')

        nama = st.text_input('Masukkan nama:')
        jenis_tinggal = st.selectbox('Masukkan jenis tinggal:', ['Rumah Pribadi', 'Rumah Sewa'])
        jenjang_pendidikan_ortu_wali = st.selectbox('Masukkan jenjang pendidikan ortu atau wali:', ['SD', 'SMP', 'SMA', 'D3', 'S1', 'S2', 'S3'])
        pekerjaan_ortu_wali = st.selectbox('Masukkan pekerjaan ortu atau wali:', ['PNS', 'Swasta', 'Wiraswasta', 'Tidak Bekerja'])
        penghasilan_ortu_wali = st.selectbox('Masukkan penghasilan ortu atau wali:', ['< 1 Juta', '1-3 Juta', '3-5 Juta', '> 5 Juta'])

        input_data = np.array([
            nama,
            jenis_tinggal,
            jenjang_pendidikan_ortu_wali,
            pekerjaan_ortu_wali,
            penghasilan_ortu_wali
        ]).reshape(1, -1)

        # Mengubah data input menjadi data numerik
        input_data[0][1] = 1 if input_data[0][1] == 'Rumah Sewa' else 0
        input_data[0][2] = label_encoder.transform([input_data[0][2]])[0]
        input_data[0][3] = label_encoder.transform([input_data[0][3]])[0]
        input_data[0][4] = label_encoder.transform([input_data[0][4]])[0]

        input_data_scaled = scaler.transform(input_data)

        prediction = gaussian.predict(input_data_scaled)

        # Menerjemahkan kembali label numerik menjadi label teks
        predicted_label = label_encoder.inverse_transform(prediction)

        st.subheader('Hasil Prediksi')
        st.write('Klasifikasi:', predicted_label[0])
