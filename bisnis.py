from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
<center><h3 style = "text-align: justify;">KLASIFIKASI PENERIMA BANTUAN PIP DAN KIP SD NEGERI LOMBANG DAJAH 1 MENGGUNAKAN METODE NAIVE BAYES</h3></center>
""",unsafe_allow_html=True)
st.write("### Dosen Pengampu : Eka Mala Sari Rochman, S.Kom., M.Kom.",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="data:image/jpeg;base64,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" width="120" height="120"></h3>""",unsafe_allow_html=True), 
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
        st.write(""" <p style = "text-align: justify;">Program Indonesia Pintar (PIP) dan Kartu Indonesia Pintar (KIP) adalah program bantuan sosial yang ditujukan untuk meningkatkan akses dan kualitas pendidikan bagi anak-anak di Indonesia. Program ini didesain untuk membantu meringankan beban biaya pendidikan bagi keluarga yang berpenghasilan rendah.</p>""",unsafe_allow_html=True)
        st.write(""" <p style = "text-align: justify;">PIP adalah program bantuan sosial yang menyediakan bantuan pendidikan kepada siswa dari keluarga miskin atau kurang mampu. Program ini bertujuan untuk memastikan bahwa anak-anak dari keluarga kurang mampu dapat mengakses pendidikan yang layak dan berkualitas. Bantuan yang diberikan meliputi biaya pendidikan seperti uang sekolah, uang buku, seragam, dan biaya kegiatan sekolah lainnya. PIP membantu mengurangi beban keuangan keluarga dalam memenuhi kebutuhan pendidikan anak-anak mereka.</p>""",unsafe_allow_html=True)
        st.write(""" <p style = "text-align: justify;">KIP adalah kartu elektronik yang diberikan kepada siswa dari keluarga miskin atau kurang mampu. Kartu ini berfungsi sebagai sarana identifikasi dan verifikasi penerima bantuan pendidikan. Melalui KIP, siswa dapat memperoleh akses ke berbagai program bantuan pendidikan, seperti bantuan biaya sekolah, beasiswa, atau subsidi lainnya. KIP membantu meningkatkan aksesibilitas siswa ke pendidikan formal dan membantu mereka dalam memenuhi kebutuhan pendidikan.</p>""",unsafe_allow_html=True)
        st.write(""" <p style = "text-align: justify;">Dengan adanya program PIP dan KIP, diharapkan anak-anak dari keluarga kurang mampu dapat tetap melanjutkan pendidikan mereka tanpa terkendala oleh faktor finansial. Program ini berperan penting dalam menciptakan kesetaraan dan kesempatan yang adil dalam pendidikan, serta meningkatkan tingkat partisipasi dan kelulusan siswa dari keluarga kurang mampu.</p>""",unsafe_allow_html=True)
        
        st.write("#### Dataset")
        # Read Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/datasetpipkip.csv')
        st.write(df)

    elif selected == "Implementation":
        # Read Dataset
        df = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/Project-Akhir-Kecerdasan-Bisnis-Kelompok-7/main/datasetpipkip.csv')

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
        training, test, training_label, test_label = train_test_split(scaled_features, y_encoded, test_size=0.2, random_state=50)

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
        st.write("##### Mata Kuliah = Kecerdasan Bisnis -A") 
        st.write('##### Kelompok 7')
        st.write("1. Hambali Fitrianto - 200411100074")
        st.write("2. Firdatul Fitriyah - 200411100020")
        st.write("3. Choirinnisa’ Fitria - 200411100149")
