# MindScope — Student Depression Risk Predictor

Aplikasi prediksi risiko depresi mahasiswa berbasis machine learning, dibuat untuk tugas praktikum Data Mining.

**Live app:** https://dataminingpratikum-kelompok7.streamlit.app/

## Kelompok 7 — Kelas I231B

- Muhammad Akmal Hafiz (312310184)
- Pranaja
- eza

## Tentang Aplikasi

Aplikasi ini menganalisis faktor akademik, psikologis, dan gaya hidup mahasiswa untuk memprediksi risiko depresi, memakai dataset [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) dari Kaggle. Target kolom: `Depression` (0 = tidak depresi, 1 = depresi).

Dua algoritma tersedia dan bisa dibandingkan langsung di aplikasi:

- **Decision Tree** — mudah dibaca, dilengkapi feature importance
- **Naive Bayes** — probabilistik, cepat dan stabil untuk data kategorikal

## Fitur

- Dashboard dengan ringkasan metrik utama
- Upload dan eksplorasi dataset CSV, termasuk statistik deskriptif dan deteksi missing values
- Visualisasi interaktif (distribusi gender, tekanan akademik, durasi tidur, heatmap korelasi, dan lainnya)
- Training model dengan parameter yang bisa diatur, classification report, dan confusion matrix
- Prediksi risiko untuk profil mahasiswa baru, lengkap dengan gauge dan persentase probabilitas
- Tutorial step-by-step untuk pengguna baru
- Info tooltip di setiap field yang berpotensi membingungkan

## Teknologi

Python, Streamlit, scikit-learn, Plotly, pandas.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```
