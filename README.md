# Machine Learning for Student Performance

Aplikasi prediksi kelulusan siswa menggunakan berbagai algoritma Machine Learning yang dibangun dengan Streamlit.

## Deskripsi

Project ini merupakan aplikasi web interaktif untuk memprediksi kelulusan siswa berdasarkan berbagai faktor seperti nilai, demografi, perilaku, dan dukungan belajar. Aplikasi ini menggunakan beberapa algoritma Machine Learning dan memberikan visualisasi lengkap untuk analisis performa model.

## Fitur

### Mode Prediksi

- Early Prediction: Prediksi tanpa nilai G1 & G2 (cocok untuk awal semester)
- Mid/End Semester: Prediksi dengan nilai G1 & G2 (untuk pertengahan/akhir semester)

### Algoritma Machine Learning

- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes

### Visualisasi

- Confusion Matrix
- Grafik perbandingan metrik evaluasi
- Distribusi data aktual vs prediksi
- Feature importance/koefisien model

### Metrik Evaluasi

- Accuracy
- Precision
- Recall
- F1-Score

## Requirements

- Python 3.8 atau lebih tinggi
- pip (Python package manager)

## Instalasi

1. Clone repository ini:

```bash
git clone https://github.com/Jailbirdss/Machine-Learning-for-Student-Peformance.git
cd Machine-Learning-for-Student-Peformance
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

Jalankan aplikasi dengan perintah:

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser dengan alamat `http://localhost:8501`

## Dataset

Dataset yang digunakan adalah `student_data.csv` yang berisi informasi siswa meliputi:

- Nilai: G1, G2, G3 (nilai akhir)
- Akademik: studytime, failures, absences, higher, schoolsup, paid
- Dukungan: famsup, internet
- Pendidikan Orangtua: Medu (ibu), Fedu (ayah)
- Perilaku: goout, Dalc (konsumsi alkohol hari kerja), Walc (konsumsi alkohol akhir pekan), freetime
- Demografi: age, health

Definisi Label:

- Lulus: Nilai G3 >= 10
- Tidak Lulus: Nilai G3 < 10

## Cara Penggunaan

1. Pilih Mode Prediksi di sidebar:

   - Early Prediction: Untuk siswa yang belum memiliki nilai G1 & G2
   - Mid/End Semester: Untuk siswa yang sudah memiliki nilai G1 & G2

2. Pilih Algoritma yang ingin digunakan

3. Lihat Hasil:
   - Statistik data siswa
   - Metrik evaluasi model
   - Visualisasi confusion matrix dan distribusi
   - Feature importance/koefisien
   - Kesimpulan performa model

## Struktur Project

```
.
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Dependencies Python
├── student_data.csv       # Dataset siswa
└── README.md              # Dokumentasi project
```

## Teknologi yang Digunakan

- Streamlit: Framework web app
- Pandas: Manipulasi dan analisis data
- NumPy: Komputasi numerik
- Scikit-learn: Algoritma Machine Learning
- Matplotlib & Seaborn: Visualisasi data

## Performa Model

Model dilatih dengan pembagian:

- Training Set: 80% data
- Testing Set: 20% data

Fitur yang digunakan disesuaikan dengan mode prediksi yang dipilih:

- Early Prediction: 16 fitur (tanpa G1 & G2)
- Mid/End Semester: 18 fitur (dengan G1 & G2)

## License

Project ini dibuat untuk keperluan pembelajaran Matakuliah Fundamen Sains Data.
