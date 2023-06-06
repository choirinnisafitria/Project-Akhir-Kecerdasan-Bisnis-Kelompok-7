import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
        df = pd.read_csv('data.csv')  # Ganti dengan nama file dataset Anda
        
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
        df = pd.read_csv('data.csv')  # Ganti dengan nama file dataset Anda

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
        st.subheader('Implementasi Prediksi Penyakit Diabetes')
        df = pd.read_csv('data.csv')  # Ganti dengan nama file dataset Anda

        X = df.drop(columns=['Status'])
        y = df['Status'].values

        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
        scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=1)

        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)

        nama = st.number_input('Masukkan nama:')
        jenis_tinggal = st.number_input('Masukkan jenis tinggal:')
        jenjang_pendidikan_ortu_wali = st.number_input('Masukkan jenjang pendidikan ortu atau wali:')
        pekerjaan_ortu_wali = st.number_input('Masukkan pekerjaan ortu atau wali:')
        penghasilan_ortu_wali = st.number_input('Masukkan penghasilan ortu atau wali:')
        
        input_data = np.array([
            nama,
            jenis_tinggal,
            jenjang_pendidikan_ortu_wali,
            pekerjaan_ortu_wali,
            penghasilan_ortu_wali
        ]).reshape(1, -1)

        input_data_scaled = scaler.transform(input_data)

        prediction = gaussian.predict(input_data_scaled)

        st.subheader('Hasil Prediksi')
        st.write('Klasifikasi:', prediction[0])
