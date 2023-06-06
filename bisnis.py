from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Klasifikasi",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998664.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">ANALISIS SENTIMEN PADA WISATA DIENG DENGAN ALGORITMA K-NEAREST NEIGHBOR (K-NN)</h2></center>
""",unsafe_allow_html=True)
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/1998/1998664.png" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home","Dataset", "Implementation", "Tentang Kami"], 
            icons=['house', 'bar-chart','check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://bareng-bjn.desa.id/desa/upload/artikel/sedang_1554884848_e.jpg" width="500" height="300">
        </h3>""",unsafe_allow_html=True)

    elif selected == "Dataset":
        st.write("#### Deskripsi Dataset")
        st.write(""" <p style = "text-align: justify;">dataset tentang ulasan terhadap wisata dieng dari website tripadvisor. Selanjutnya data ulasan tersebut akan diklasifikasikan ke dalam dua kategori sentimen yaitu negatif dan positif kemudian dilakukan penerapan algoritma k-nearest neighbor (K-NN) untuk mengetahui nilai akurasinya.</p>""",unsafe_allow_html=True)
        st.write("#### Preprocessing Dataset")
        st.write(""" <p style = "text-align: justify;">Preprocessing data merupakan proses dalam mengganti teks tidak teratur supaya teratur yang nantinya dapat membantu pada proses pengolahan data.</p>""",unsafe_allow_html=True)
        
        st.write("#### Dataset")
        # Read Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/datasetpipkip.csv')
        st.write(df)

    elif selected == "Implementation":
        # Read Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/datasetpipkip.csv')
        st.write(df)

        # Preprocessing data
        # Mendefinisikan Variable X dan Y
        X = df[['Jenis_Tinggal', 'Jenis_Pendidikan_Ortu_Wali', 'Pekerjaan_Ortu_Wali', 'Penghasilan_Ortu_Wali']]
        y = df['Status'].values

        # One-hot encoding pada atribut kategorikal
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X.astype(str)).toarray()
        feature_names = encoder.get_feature_names_out(input_features=X.columns)
        scaled_features = pd.DataFrame(X_encoded, columns=feature_names)

        # Label encoding pada target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split Data
        training, test, training_label, test_label = train_test_split(scaled_features, y_encoded, test_size=0.2, random_state=42)

        # Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian.fit(training, training_label)
        probas = gaussian.predict_proba(test)
        probas = probas[:, 1]
        probas = probas.round().astype(int)

        st.subheader("Implementasi Penerima bantuan PIP dan KIP")
        jenis_tinggal = st.selectbox('Masukkan jenis tinggal:', ['Bersama orang tua', 'Wali'])
        jenis_pendidikan_ortu_wali = st.selectbox('Masukkan jenis pendidikan ortu atau wali:', ['Tidak sekolah', 'SD sederajat', 'SMP sederajat', 'SMA sederajat', 'D2', 'S1'])
        pekerjaan_ortu_wali = st.selectbox('Masukkan pekerjaan ortu atau wali:', ['Sudah Meninggal', 'Petani', 'Pedagang Kecil', 'Karyawan Swasta', 'Wiraswasta'])
        penghasilan_ortu_wali = st.selectbox('Pilih penghasilan ortu atau wali:', ['Tidak Berpenghasilan', 'Kurang dari 1.000.000', '500,000 - 999,999', '1,000,000 - 1,999,999'])
        model = 'Gaussian Naive Bayes'  # Menggunakan model Gaussian Naive Bayes secara langsung

        if st.button('Submit'):
            inputs = np.array([
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

            input_pred = mod.predict(inputs_encoded)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan:', model)

            if len(test_label) > 0:
                test_label = test_label.astype(int)
                probas = probas.round().astype(int)
                akurasi = round(100 * accuracy_score(test_label, probas))
                st.write('Akurasi: {0:0.0f}'.format(akurasi), '%')

            if input_pred == 1:
                st.error('PIP')
            else:
                st.success('KIP')

    elif selected == "Tentang Kami":
        st.write("##### Mata Kuliah = Pemrosesan Bahasa Alami -A") 
        st.write('##### Kelompok 5')
        st.write("1. Hambali Fitrianto - 200411100074")
        st.write("2. Firdatul Fitriyah - 200411100020")
        st.write("3. Choirinnisa’ Fitria - 200411100149")
        
