# Eksperimen SML — Miftah Al Ghifari

Project Bootcamp Sistem Machine Learning — Eksperimen dan Preprocessing Bank Marketing Dataset.

## 📋 Deskripsi

Repository ini berisi eksperimen machine learning untuk dataset **Bank Marketing** dari UCI ML Repository. Proyek ini mencakup tahapan lengkap dari eksplorasi data (EDA) hingga otomatisasi preprocessing menggunakan GitHub Actions.

## 📁 Struktur Folder

```
Eksperimen_SML_Miftah Al Ghifari-Siswa/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions untuk auto-preprocessing
├── bankmarketing_raw/
│   └── bank-additional-full.csv       # Dataset mentah
├── preprocessing/
│   ├── Eksperimen_Miftah Al Ghifari-Siswa.ipynb  # Notebook eksperimen
│   ├── automate_Miftah Al Ghifari-Siswa.py       # Script otomatisasi
│   └── bankmarketing_preprocessing/     # Hasil preprocessing
│       ├── bankmarketing_preprocessed.csv
│       ├── train.csv
│       └── test.csv
└── README.md
```

## 🔬 Dataset

- **Nama**: Bank Marketing Dataset
- **Sumber**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Jumlah Sampel**: 1,599
- **Fitur**: 11 fitur numerik (fixed acidity, volatile acidity, citric acid, dll.)
- **Target**: quality (3-8) → dikategorikan menjadi low, medium, high

## 🔄 Tahapan Preprocessing

1. **Data Loading** — Memuat dataset dari CSV
2. **Handling Missing Values** — Cek dan tangani data kosong
3. **Removing Duplicates** — Hapus baris duplikat
4. **Outlier Handling** — Capping menggunakan metode IQR
5. **Target Encoding** — Encoding quality menjadi kategori (low/medium/high)
6. **Feature Scaling** — Normalisasi menggunakan StandardScaler
7. **Train-Test Split** — Pembagian data 80:20 dengan stratify

## ⚙️ Cara Menjalankan

### Manual
```bash
cd preprocessing
python "automate_Miftah Al Ghifari-Siswa.py"
```

### Otomatis (GitHub Actions)
Workflow akan berjalan otomatis ketika:
- Push ke branch `main` (perubahan pada dataset raw atau script preprocessing)
- Manual trigger via GitHub Actions UI

## 📦 Dependencies

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
