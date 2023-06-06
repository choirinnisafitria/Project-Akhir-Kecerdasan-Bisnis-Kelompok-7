import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
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
st.write("""<h1>Aplikasi Klasifikasi Penerima PIP dan KIP</h1>""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://lh3.googleusercontent.com/a/ALm5wu2PukBXPMX88VuehLVmYvtTCLj1-XFDgkoky1-JBg=s192-c-rg-br100" width="90" height="90"><br> Kecerdasan Bisnis <p> Kelompok 7 </p></h3>""",unsafe_allow_html=True), 
        ["Prepocessing", "Modeling", "Implementation"], 
            icons=['gear', 'arrow-down-square', 'check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#FF4B4B"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#FF4B4B"}
            }
        )

    if selected == "Prepocessing":
        st.subheader("""Normalisasi Data""")
        st.write("""Rumus Normalisasi Data :""")
        st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/data.csv')
        st.markdown("""
        Dimana :
        - X = data yang akan dinormalisasi atau data asli
        - min = nilai minimum semua data asli
        - max = nilai maksimum semua data asli
        """)
        #Mendefinisikan Varible X dan Y
        X = df.drop(columns=['Status'])
        y = df['Status'].values
        df_min = X.min()
        df_max = X.max()
        
        #NORMALISASI NILAI X
        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        #features_names.remove('label')
        scaled_features = pd.DataFrame(scaled, columns=features_names)

        st.subheader('Hasil Normalisasi Data')
        st.dataframe(scaled_features, width=600)

        st.subheader('Target Label')
        dumies = pd.get_dummies(df.Result).columns.values.tolist()
        dumies = np.array(dumies)

        labels = pd.DataFrame({
            'Positive' : [dumies[1]],
            'Negative' : [dumies[0]]
        })

        st.write(labels)

    elif selected == "Modeling":
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/data.csv')

        X = df.drop(columns=['Result'])
        y = df['Result'].values
        
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        scaled_features = pd.DataFrame(scaled, columns=features_names)

        training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)
        training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)

        with st.form("modeling"):
            st.subheader('Modeling')
            st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
            naive = st.checkbox('Gaussian Naive Bayes')
            submitted = st.form_submit_button("Submit")

            gaussian = GaussianNB()
            gaussian = gaussian.fit(training, training_label)
            probas = gaussian.predict_proba(test)
            probas = probas[:,1]
            probas = probas.round()
            gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

            if submitted :
                if naive :
                    st.write('Model Naive Bayes accuracy score: {0:0.0f}'. format(gaussian_akurasi),'%')
            grafik = st.form_submit_button("Grafik akurasi semua model")
            if grafik:
                data = pd.DataFrame({
                    'Akurasi' : [gaussian_akurasi],
                    'Model' : ['Gaussian Naive Bayes'],
                })

                bar_chart = px.bar(data, 
                    x='Model', 
                    y='Akurasi',
                    text='Akurasi',
                    color_discrete_sequence =['#FF4B4B']*len(data),
                    width=680)
                bar_chart

    elif selected == "Implementation":
        with st.form("Implementation"):
            df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/data.csv')

            X = df.drop(columns=['Status'])
            y = df['Status'].values
            
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(X)
            features_names = X.columns.copy()
            scaled_features = pd.DataFrame(scaled, columns=features_names)

            training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
            training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing

            gaussian = GaussianNB()
            gaussian = gaussian.fit(training, training_label)
            probas = gaussian.predict_proba(test)
            probas = probas[:,1]
            probas = probas.round()


            st.subheader("Implementasi Prediksi Penyakit Diabetes")
            nama = st.number_input('Masukkan nama :')
            jenist_inggal = st.number_input('Maukkan jenis jinggal :')
            jenjang_pendidikan_ortu_wali = st.number_input('Masukkan jenis pendidikan ortu atau wali :')
            pekerjaan_ortu_wali = st.number_input('Masukkan pekerjaan ortu atau wali :')
            penghasilan_ortu_wali = st.number_input('Masukkan penghasilan ortu atau wali :')
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
                
                df_min = X.min()
                df_max = X.max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                if model == 'Gaussian Naive Bayes':
                    mod = gaussian
                    akurasi = round(100 * accuracy_score(test_label,probas))

                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan :',model)
                st.write('Akurasi: {0:0.0f}'. format(akurasi),'%')
