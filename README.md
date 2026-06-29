# Projek Computer Vision - Analisis & Klasifikasi Ketebalan Tinta Tulisan Tangan

Aplikasi web berbasis **Flask** dan **Computer Vision** (OpenCV) untuk melakukan analisis citra tulisan tangan, ekstraksi fitur ketebalan stroke (*Distance Transform*), analisis kepadatan piksel, dan klasifikasi jenis tinta (Bold/Tebal vs Normal/Tipis).

Proyek ini dibuat sebagai syarat pemenuhan **Final Proyek Semester (Pertemuan 15)**.

---

## 🚀 Fitur Utama

- **Pre-processing Citra**: Normalisasi background, adaptive thresholding, pembersihan garis horizontal, dan pembersihan noise (Connected Components).
- **Ekstraksi Fitur Fisik**:
  - Ketebalan stroke asli (rata-rata, minimal, maksimal) menggunakan L2 Distance Transform.
  - Kepadatan piksel (*pixel density*).
- **Visualisasi Citra**:
  - Citra hasil pre-processing 128x128.
  - Heatmap Distance Transform untuk melihat sebaran ketebalan stroke secara visual.
- **Klasifikasi Otomatis**: Menentukan kategori pena (Spidol/Gel vs Ballpoint/Pensil) beserta tingkat keyakinan (confidence score).

---

## 🛠️ Teknologi & Library

- **Python 3**
- **Flask** (Framework Web Backend)
- **OpenCV (`opencv-python-headless`)** (Pengolahan Citra Digital)
- **NumPy** & **scikit-image** (Ekstraksi & Manipulasi Data Matrix)
- **Gunicorn** (WSGI Server untuk Production)
- **HTML5 & CSS3** (Frontend UI)

---

## 📂 Struktur Direktori

```text
├── app.py                      # Server Flask utama & Logika Pengolahan Citra
├── Procfile                    # Konfigurasi Gunicorn untuk Cloud Deployment
├── requirements.txt            # Daftar dependensi Python
├── .gitignore                  # Berkas pengecualian Git
├── templates/
│   └── index.html              # Tampilan Antarmuka Web
├── model_handwriting_cv.pkl    # Model Machine Learning Trained
├── scaler_handwriting_cv.pkl   # Scaler Fitur
└── mapping_handwriting_cv.pkl  # Mapping Label Class
```

---

## 🔧 Cara Menjalankan Secara Lokal

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/stefanycrista/projek-computer-vision.git
   cd projek-computer-vision
   ```

2. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan server Flask:**
   ```bash
   python app.py
   ```

4. Buka browser dan akses `http://localhost:5000`.
