# Laporan Proyek Prediksi Harga Sewa Apartemen

## Domain Proyek
### Latar Belakang
Pasar properti, sebagai sektor yang penting dan dinamis, berperan besar dalam perekonomian, dengan harga menjadi faktor kunci yang mempengaruhi berbagai pemangku kepentingan. Prediksi harga properti sebelumnya didasarkan pada model statistik dan matematis tradisional, seperti regresi dan penilaian harga hedonis, yang menjadi dasar untuk memahami faktor-faktor yang mempengaruhi penilaian properti. Namun, seiring dengan kemajuan daya komputasi dan munculnya teknik machine learning (ML), model-model tradisional ini kini dilengkapi dengan metode yang lebih canggih. Machine learning, terutama teknik seperti neural networks, deep learning, dan gradient boosting, telah terbukti meningkatkan akurasi prediksi, terutama dengan ketersediaan dataset yang lebih besar. Selain itu, penggunaan data time-series yang memperhitungkan fluktuasi harga dari waktu ke waktu menandai kemajuan signifikan dalam prediksi harga properti. Oleh karena itu, pengembangan sistem prediksi harga sewa apartemen menggunakan machine learning menjadi sangat penting untuk membantu pemilik properti dan penyewa membuat keputusan yang lebih baik dan lebih efisien.

### Mengapa Masalah Ini Harus Diselesaikan?
- Penetapan harga sewa yang akurat sangat penting untuk mencegah kerugian bagi kedua belah pihak, baik pemilik properti maupun penyewa.
- Memiliki sistem prediksi harga yang tepat dapat meningkatkan efektivitas pasar penyewaan properti dengan mempermudah proses penentuan harga yang sesuai.

### Referensi Terkait
- [MACHINE LEARNING FOR PROPERTY PRICE PREDICTION AND PRICE VALUATION: A SYSTEMATIC LITERATURE REVIEW](https://www.researchgate.net/publication/355373964_MACHINE_LEARNING_FOR_PROPERTY_PRICE_PREDICTION_AND_PRICE_VALUATION_A_SYSTEMATIC_LITERATURE_REVIEW)
- [Machine Learning for Housing Price Prediction](https://www.researchgate.net/publication/367317216_Machine_Learning_for_Housing_Price_Prediction)


## Business Understanding
### Problem Statements
- Bagaimana cara memprediksi harga sewa apartemen berdasarkan fitur-fitur seperti jumlah kamar mandi, kamar tidur, luas apartemen, dsb?
- Fitur-fitur apa saja yang paling berpengaruh dalam menentukan harga sewa apartemen?
- Bagaimana cara meningkatkan akurasi prediksi harga sewa dengan menggunakan teknik machine learning yang canggih seperti K-Nearest Neighbors, Random Forest, atau Boosting?

### Goals
- Mengembangkan model machine learning yang mampu memprediksi harga sewa apartemen dengan tingkat akurasi yang tinggi, berdasarkan fitur-fitur yang relevan.
- Menentukan fitur-fitur yang memiliki pengaruh terbesar dalam penentuan harga sewa.
- Meningkatkan akurasi model prediksi dengan melakukan penyesuaian hyperparameter dan menggunakan teknik machine learning yang sesuai.

### Solution Statements
- Menerapkan berbagai algoritma machine learning seperti KNN, Random Forest, dan AdaBoost untuk memprediksi harga sewa.
- Membandingkan kinerja model dan memilih model terbaik berdasarkan metrik evaluasi seperti Mean Squared Error (MSE).

## Data Understanding

### Dataset
Dataset yang digunakan diperoleh dari Kaggle dan berisi informasi mengenai apartemen yang disewakan, termasuk fitur-fitur seperti jumlah kamar mandi, kamar tidur, luas apartemen, dan harga sewa. Dataset ini memiliki 100.000 baris dan 22 kolom, namun hanya 25.000 baris yang digunakan dalam analisis.

| Column         | Dtype    |
|----------------|----------|
| id             | int64    |
| category       | object   |
| title          | object   |
| body           | object   |
| amenities      | object   |
| bathrooms      | float64  |
| bedrooms       | float64  |
| currency       | object   |
| fee            | object   |
| has_photo      | object   |
| pets_allowed   | object   |
| price          | int64    |
| price_display  | object   |
| price_type     | object   |
| square_feet    | int64    |
| address        | object   |
| cityname       | object   |
| state          | object   |
| latitude       | float64  |
| longitude      | float64  |
| source         | object   |
| time           | int64    |


### Missing Values
| Tipe Data      | Jumlah Missing Value |
|----------------|----------------------|
| amenities      | 4778                 |
| bathrooms      | 37                   |
| bedrooms       | 20                   |
| pets_allowed   | 13239                |
| address        | 18320                |
| cityname       | 77                   |
| state          | 77                   |
| latitude       | 10                   |
| longitude      | 10                   |

### Outliers
Dataset mengandung nilai outlier pada fitur-fitur numerik, terutama pada fitur `price` dan `square_feet`.
![boxplot](./images/outputcode.png)

### Variabel yang ada pada dataset dijelaskan sebagai berikut:
1. id : Identifikasi unik untuk listing apartemen
2. category : Kategori dari iklan apartemen
3. title : Nama apartemen
4. body : Deskripsi atau informasi tambahan tentang apartemen
5. amenities*: Fasilitas yang disediakan, seperti AC, lapangan basket, kabel TV, gym, akses internet, kolam renang, lemari es, dll.
6. bathrooms : Jumlah kamar mandi yang tersedia
7. bedrooms : Jumlah kamar tidur yang tersedia
8. currency : Mata uang yang digunakan untuk harga apartemen
9.  fee : Biaya tambahan yang mungkin dikenakan
10. has_photo : Menunjukkan apakah apartemen memiliki foto
11. pets_allowed : Jenis hewan peliharaan yang diizinkan, seperti anjing/kucing, dll.
12. price : Harga sewa apartemen
13. price_display : Harga yang ditampilkan untuk pembaca
14. price_type : Harga yang tertera dalam USD
15. square_feet : Ukuran atau luas apartemen dalam satuan kaki persegi
16. address : Alamat lokasi apartemen
17. cityname : Nama kota tempat apartemen berada
18. state : Nama provinsi tempat apartemen berada
19. latitude : Koordinat lintang lokasi apartemen
20. longitude : Koordinat bujur lokasi apartemen
21. source : Sumber iklan apartemen
22. time : Waktu saat iklan dibuat


### Exploratory Data Analysis (EDA)
- Membuat heatmap korelasi untuk menganalisis hubungan antar fitur numerik
 ![Korelasi heatmap](./images/outputcode2.png)
    * Hubungan antara Luas Bangunan dan Fitur Lain: Fitur square_feet (luas bangunan) menunjukkan hubungan yang cukup kuat dengan bathrooms dan bedrooms, yang menunjukkan bahwa luas bangunan dapat menjadi indikator penting dalam menentukan ukuran dan fasilitas apartemen.
    * Harga Sewa: Korelasi antara harga sewa (price) dan fitur lainnya (kamar mandi dan kamar tidur) relatif rendah. Ini menunjukkan bahwa harga sewa mungkin dipengaruhi oleh faktor-faktor lain seperti lokasi, fasilitas tambahan, atau kondisi pasar yang tidak tercakup dalam fitur yang dianalisis.

- Membuat histogram untuk menganalisis distribusi data numerik
 ![Histogram](./images/outputcode3.png)
    * Kamar Mandi (Bathrooms):
        * Mayoritas properti memiliki 1 kamar mandi, dengan beberapa properti memiliki 1.5 dan 2.5 kamar mandi.
        * Kamar mandi seharusnya tidak dalam format float, sehingga entri dengan nilai kamar mandi pecahan akan dihapus (drop).
    * Kamar Tidur (Bedrooms):
        * Mayoritas properti memiliki 1 atau 2 kamar tidur.
    * Harga Sewa (Price):
        * Harga sewa apartemen tersebar dengan puncak di sekitar 700-1000 USD.
        * Distribusi harga menunjukkan pola normal dengan sedikit skew ke kanan.
    * Luas (Square Feet):
        * Luas apartemen bervariasi dengan puncak sekitar 500-800 kaki persegi.
        * Distribusi luas juga menunjukkan pola normal dengan skew ke kanan.

## Data Preparation

### Teknik Data Preparation
- **Menangani Nilai yang Hilang**: Mengimputasi atau menghapus nilai yang hilang dalam dataset.
- **Menghapus Outlier**: Menghapus data yang memiliki nilai outlier.
- **Pengkodean Variabel Kategorikal**: Mengonversi variabel kategorikal menjadi numerik dengan menggunakan teknik one-hot encoding.
- **Pembagian Dataset**: Membagi data menjadi data latih dan data uji dengan rasio 80:20.
- **Skalasi Fitur**: Melakukan standarisasi pada fitur numerik.

### Proses Persiapan Data

- Fitur yang memiliki jumlah missing value < 100 akan dihapus (drop).
- Fitur yang memiliki jumlah missing value > 1000 akan diimputasi.
- Outlier ditangani menggunakan metode IQR.
- Fitur seperti `id`, `latitude`, `longitude`, dan `time` tidak memberikan nilai tambah, sehingga dilakukan penghapusan (drop).
- Fitur seperti `category`, `currency`, `fee`, dan `price_type` memiliki nilai yang sama untuk seluruh dataset, sehingga dihapus (drop).
- Fitur kategorikal diubah menjadi numerik menggunakan teknik one-hot encoding.
- Pembagian dataset menjadi train dan test dengan rasio 80:20.
  | Jumlah         |             |
  |----------------|-------------|
  | Whole Dataset  | 21473       |
  | Train          | 17178       |
  | Test           | 4295        |

- Skalasi fitur dilakukan menggunakan StandardScaler pada data latih dan data uji.

### Alasan Dilakukannya Data Preparation
- Menangani missing values untuk menghindari masalah yang dapat muncul saat proses pelatihan model.
- Menghapus outlier untuk meningkatkan akurasi model dengan mengeliminasi data yang dapat mempengaruhi kinerja model.
- Menghapus fitur yang tidak memberikan kontribusi signifikan untuk menghemat sumber daya komputasi.
- Melakukan encoding pada variabel kategorikal agar model machine learning dapat memproses data tersebut.
- Menggunakan metode pembagian 80:20 karena jumlah dataset yang tidak terlalu besar, sehingga pembagian antara data latih dan data uji cukup seimbang.
- Melakukan feature scaling untuk memastikan model tidak bias terhadap fitur dengan skala yang lebih besar.

## Modeling
### Tahap Modeling

- **Menyiapkan DataFrame untuk Analisis Masing-Masing Model**
    ```python
    models = pd.DataFrame(index=['train_mse', 'test_mse'], columns=['KNN', 'RandomForest', 'Boosting'])
    ```
    Pada langkah ini, DataFrame bernama `models` disiapkan untuk menyimpan nilai Mean Squared Error (MSE) pada data latih dan uji untuk setiap model yang akan diuji, yaitu KNN, Random Forest, dan Boosting.

- **Melatih Model KNN**
    ```python
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train, y_train)
    ```
    Model K-Nearest Neighbors dilatih menggunakan parameter `n_neighbors=10`. Model ini dilatih dengan data latih `X_train` dan `y_train`, kemudian nilai MSE pada data latih disimpan dalam DataFrame `models`.

- **Melatih Model Random Forest**
    ```python
    RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
    RF.fit(X_train, y_train)
    ```
    Menggunakan `RandomForestRegressor` dengan parameter `n_estimators=50`, `max_depth=16`, dan `random_state=55`. Model ini dilatih dengan data latih yang sama dan nilai MSE pada data latih disimpan dalam DataFrame `models`.

- **Melatih Model AdaBoost**
    ```python
    boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
    boosting.fit(X_train, y_train)
    ```
    Menggunakan `AdaBoostRegressor` dengan parameter `learning_rate=0.05` dan `random_state=55`. Model ini juga dilatih dengan data yang sama dan nilai MSE pada data latih disimpan dalam DataFrame `models`.

### Tahapan dan Parameter yang Digunakan

Pada tahap modeling, tiga algoritma yang berbeda digunakan untuk memprediksi harga sewa apartemen:

- **K-Nearest Neighbors (KNN):**
    - Parameter: `n_neighbors=10`
    - Deskripsi: Algoritma KNN mencari 10 tetangga terdekat untuk membuat prediksi berdasarkan rata-rata nilai target dari tetangga tersebut.

- **Random Forest:**
    - Parameter: `n_estimators=50`, `max_depth=16`, `random_state=55`
    - Deskripsi: Random Forest adalah ensemble yang terdiri dari beberapa pohon keputusan yang dilatih pada subset data yang berbeda untuk meningkatkan akurasi prediksi dan mengurangi overfitting.

- **AdaBoost:**
    - Parameter: `learning_rate=0.05`, `random_state=55`
    - Deskripsi: AdaBoost adalah algoritma boosting yang meningkatkan akurasi prediksi dengan menggabungkan beberapa model sederhana (biasanya pohon keputusan dengan kedalaman 1) yang dilatih secara berurutan, dengan memberikan bobot lebih pada kesalahan yang dibuat oleh model sebelumnya.

### Kelebihan dan Kekurangan Setiap Algoritma

- **K-Nearest Neighbors (KNN):**
    - **Kelebihan:**
        - Sederhana dan mudah diimplementasikan.
        - Tidak ada asumsi yang kuat mengenai distribusi data.
    - **Kekurangan:**
        - Sensitif terhadap outliers dan noise.
        - Tidak efisien untuk dataset besar karena kompleksitas komputasi yang tinggi.

- **Random Forest:**
    - **Kelebihan:**
        - Dapat menangani data kompleks dengan baik.
        - Tahan terhadap overfitting karena menggunakan banyak pohon keputusan.
        - Mampu menangani missing values dan bekerja dengan baik pada dataset besar.
    - **Kekurangan:**
        - Interpretasi model lebih sulit dibandingkan dengan model yang lebih sederhana.
        - Membutuhkan lebih banyak sumber daya komputasi dan memori.

- **AdaBoost:**
    - **Kelebihan:**
        - Dapat meningkatkan akurasi dengan menggabungkan beberapa model sederhana.
        - Fokus pada kesalahan sebelumnya dapat meningkatkan performa model secara iteratif.
    - **Kekurangan:**
        - Rentan terhadap outliers karena memberi bobot lebih pada kesalahan.
        - Kinerja menurun jika data sangat bising (noisy).

### Memilih Model Terbaik Sebagai Solusi

Berdasarkan evaluasi nilai MSE pada data uji, model terbaik dipilih sebagai solusi. Sebagai contoh, jika model Random Forest menunjukkan MSE terendah pada data uji dibandingka


## Evaluation

### Metrik Evaluasi
- **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat dari kesalahan prediksi. MSE yang lebih rendah menunjukkan performa model yang lebih baik.

### Hasil Proyek

| Model          | Train MSE | Test MSE |
|----------------|-----------|----------|
| KNN            | 78.1      | 97.8     |
| Random Forest  | 59.2      | 93.7     |
| AdaBoost       | 148.3     | 156.2    |

### Kesimpulan
- Model Random Forest terbukti menjadi model terbaik dengan MSE terendah.
- Hyperparameter tuning memainkan peran penting dalam meningkatkan performa model.
- Solusi yang diimplementasikan berhasil memenuhi problem statement dan goals yang ditetapkan.

