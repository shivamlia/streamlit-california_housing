import streamlit as st

from web_functions import predict

def app(df, x, y):

    st.title("Halaman prediksi Harga Rumah")

    col1, col2 = st.columns(2)

    with col1:
        longitude      = st.text_input("input nilai longitude (garis bujur)")     
        latitude       = st.text_input("input nilai latitude (garis lintang)")          
        housing_median_age = st.text_input("input nilai house median age(rata-rata umur rumah)")
        total_rooms        = st.text_input("input nilai total rooms (jumlah ruangan)")
        total_bedrooms     = st.text_input("input nilai total bedrooms (jumlah kamar)")
    
    with col2:
        population         = st.text_input("input nilai population (jumlah penduduk)")
        households         = st.text_input("input nilai household (jumlah rumah tangga yang tinggal)")
        median_income      = st.text_input("input nilai median income (rata-rata pendapatan)")
        median_house_value = st.text_input("input nilai median value (rata-rata nila rumah)")

    features = [longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value]

    # tombol prediksi
    if st.button("Prediksi"):
        prediction, score = predict(x,y,features)
        score = score
        st.info("Prediksi Sukses...")

        if(prediction == 1):
            st.warning("Pembeli membeli rumah tidak dekat laut")
        else:
            st.success("Pembeli membeli rumah dekat dengan laut")

        st.write("Model yang digunakan memiliki tingkat akurasi", (score*100),"%")