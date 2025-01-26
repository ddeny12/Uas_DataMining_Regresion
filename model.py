import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import io

st.title('Prediksi Biaya Asuransi Kesehatan Menggunakan Dataset Pelanggan')

# Membaca dataset
with st.expander('Dataset'):
    data = pd.read_csv('Regression.csv')
    st.write(data)

    st.success('Informasi Dataset')
    data1 = pd.DataFrame(data)
    buffer = io.StringIO()
    data1.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.success('Analisa Univariat')
    deskriptif = data.describe()
    st.write(deskriptif)


# Fungsi untuk memplot outliers
def plot_outlier(data, column):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    sns.boxplot(data[column], ax=axes[0], color='orange')
    axes[0].set_title(f'{column} - Box Plot')

    # Histogram
    sns.histplot(data[column], kde=True, ax=axes[1], color='purple')
    axes[1].set_title(f'{column} - Histogram')

    st.pyplot(fig)


# Pilih kolom untuk memplot outliers
with st.expander('Visualisasi Outliers'):
    st.info('Visualisasi Outliers')

    # Daftar kolom numerik
    columns = ['age', 'bmi', 'children', 'charges']

    # Pilihan kolom dari dropdown
    selected_column = st.selectbox("Pilih kolom untuk melihat outliers:", columns)

    # Plot outliers berdasarkan kolom yang dipilih
    if selected_column:
        plot_outlier(data, selected_column)


# Fungsi untuk menghapus outliers menggunakan metode IQR
def remove_outliers_iqr(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Hanya menyimpan data yang berada dalam rentang IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data


# Menghapus outliers
with st.expander('Hapus Outliers'):
    st.info('Hapus Outliers Menggunakan Metode IQR')

    # Kolom numerik untuk deteksi outliers
    numeric_columns = ['age', 'bmi', 'children', 'charges']

    # Hapus outliers
    data_cleaned = remove_outliers_iqr(data, numeric_columns)

    # Informasi jumlah data yang dihapus
    st.write(f"Jumlah data sebelum penghapusan: {len(data)}")
    st.write(f"Jumlah data setelah penghapusan: {len(data_cleaned)}")
    st.write(f"Jumlah data yang dihapus: {len(data) - len(data_cleaned)}")


with st.expander('Modeling'):
    st.title('Modeling: Prediksi Biaya Asuransi')

    # Preprocessing data
    st.info("Preprocessing Data")
    # Mengonversi kolom kategoris menjadi dummy variables
    X = pd.get_dummies(data.drop(columns=['charges']), drop_first=True)
    y = data['charges']

    # Split data menjadi training dan testing dengan ukuran tetap
    test_size = 0.2  # Ukuran data testing tetap 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Inisialisasi dan pelatihan model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    st.subheader("Evaluasi Model")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")

    # Input manual untuk prediksi melalui sidebar
    st.sidebar.header("Input Manual untuk Memprediksi")
    st.sidebar.info("Masukkan nilai untuk setiap kolom:")

    age = st.sidebar.slider('Umur (age):', min_value=18, max_value=64, value=19, step=1)
    sex = st.sidebar.radio('Jenis Kelamin (sex):', ['male', 'female'])
    bmi = st.sidebar.slider('Indeks Massa Tubuh (bmi):', min_value=15.0, max_value=54.0, value=25.0, step=0.1)
    children = st.sidebar.slider('Jumlah Anak (children):', min_value=0, max_value=5, value=0, step=1)
    smoker = st.sidebar.radio('Apakah Merokok (smoker):', ['yes', 'no'])
    region = st.sidebar.selectbox('Wilayah (region):', ['northeast', 'northwest', 'southeast', 'southwest'])

    # Membuat data input menjadi format sesuai dengan data training
    input_data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }

    # Kolom yang tidak dipilih diatur ke 0
    for col in X.columns:
        if col not in input_data:
            input_data[col] = 0

    # Konversi input data menjadi DataFrame
    input_df = pd.DataFrame([input_data])

    st.write("**Data Input untuk Prediksi:**")
    st.write(input_df)

    # Prediksi berdasarkan input pengguna
    if st.sidebar.button("Prediksi"):
        prediction = model.predict(input_df)
        st.sidebar.success(f"Hasil Prediksi Biaya Asuransi: ${prediction[0]:,.2f}")
