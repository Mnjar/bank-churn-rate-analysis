# Laporan Proyek Machine Learning - Muhamad Fajar Faturohman

## Domain Proyek

Pentingnya mempertahankan pelanggan dalam industri perbankan sangat krusial, mengingat churn atau hilangnya pelanggan dapat menyebabkan penurunan pendapatan dan biaya tinggi untuk menarik pelanggan baru. Oleh karena itu, memprediksi churn dan memahami perilaku pelanggan menjadi kunci dalam mengambil langkah pencegahan.

Masalah churn perlu diatasi karena dampaknya yang signifikan terhadap profitabilitas. Penelitian menunjukkan bahwa meningkatkan retensi pelanggan sebesar 5% dapat meningkatkan profitabilitas perusahaan antara 25% hingga 95% (Harvard Business Review). Teknologi machine learning dapat membantu bank menganalisis data pelanggan seperti usia, saldo akun, dan riwayat transaksi untuk memprediksi churn dan melakukan intervensi yang tepat.

Studi terkait dari industri telekomunikasi menunjukkan bahwa penggunaan analitik prediktif membantu mengurangi churn dengan mengintegrasikan model prediksi dalam sistem manajemen hubungan pelanggan. Namun, perusahaan harus memperhatikan pertimbangan etis terkait privasi dan keamanan data pelanggan serta tantangan seperti kualitas data dan akurasi model.

Dengan demikian, penggunaan analitik prediktif dapat memberikan keuntungan kompetitif bagi bank dalam mempertahankan pelanggan dan meningkatkan loyalitas mereka.

Referensi:
[[1] Harvard Business Review: The Value of Keeping the Right Customers.](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)
[[2]A Study on Use of Predictive Analytics to Reduce Customer Churn in Telecommunication Industry](https://www.sdmimd.ac.in/marketingconference2024/papers/IMC2467.pdf)

## Business Understanding

### Probel Statements

Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi "churn" pelanggan pada pelanggan bank . Berikut adalah beberapa pernyataan masalah yang muncul:

1. Bagaimana cara memprediksi pelanggan yang berpotensi meninggalkan layanan bank berdasarkan data demografis dan aktivitas mereka?

2. Faktor-faktor apa saja yang dapat memengaruhi keputusan pelanggan untuk tetap atau berhenti menggunakan layanan?

3. Bagaimana cara memproses dan menyiapkan data yang sangat bervariasi, seperti data numerik, kategorikal, untuk digunakan dalam model prediksi churn?

### Goals

1. Mengembangkan model prediksi yang dapat memisahkan pelanggan yang berisiko meninggalkan layanan (churn) dan pelanggan yang tetap setia dengan menggunakan variabel yang relevan.

2. Mengidentifikasi variabel penting yang berkontribusi pada churn pelanggan.

3. Melakukan tahapan pra-pemrosesan data yang efektif, seperti encoding variabel kategorikal, normalisasi fitur numerik, dan penanganan data yang hilang, agar model dapat memproses informasi dengan baik.

### Solution Statement

Untuk mencapai tujuan-tujuan tersebut, berikut beberapa solusi yang berbasis pada model machine learning yang diterapkan:

1. Menggunakan model regresi logistik dan decision trees untuk membandingkan akurasi prediksi churn pelanggan.
2. Mengimplementasikan PCA untuk reduksi dimensi dan SMOTE untuk menangani class imbalance
3. Melakukan tuning hyperparameter pada model seperti Random Forest dan XGBoost untuk meningkatkan performa model dalam memprediksi churn pelanggan.

Model-model ini akan diuji menggunakan metrik evaluasi yang relevan seperti akurasi, precision, recall, dan F1 score.

## Data Understanding

Data yang digunakan dalam proyek ini adalah [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) yang berisi informasi mengenai demografi dan informasi yang berhubungan lainnya. Dataset ini berisi sekitar 10.000 sampel dan mencakup beberapa fitur yang relevan untuk prediksi churn.

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada data Bank Customer Churn Prediction

- CustomerId: ID unik untuk setiap pelanggan.

- Surname: Nama belakang pelanggan.

- Skor kredit pelanggan. Skor ini menggambarkan kelayakan kredit seseorang. Nilai yang lebih tinggi menunjukkan riwayat kredit yang lebih baik, sementara nilai yang lebih rendah menunjukkan risiko kredit yang lebih tinggi.

- Geography: Negara asal pelanggan.

- Gender: Jenis kelamin pelanggan

- Age: Umur pelanggan.

- Tenure: Lama waktu pelanggan telah menggunakan layanan (dalam tahun).

- Balance: Saldo akun pelanggan. Ini adalah jumlah uang yang pelanggan miliki di akun mereka. Fitur ini bisa menjadi indikator kekayaan pelanggan atau potensi transaksi.

- NumOfProducts: Jumlah produk yang digunakan pelanggan. Kolom ini mencerminkan berapa banyak layanan atau produk yang pelanggan gunakan dari perusahaan, yang bisa menunjukkan loyalitas pelanggan.

- HasCrCard: Status apakah pelanggan memiliki kartu kredit atau tidak (1 = Ya, 0 = Tidak).

- IsActiveMember: Status apakah pelanggan adalah anggota yang aktif (1 = Aktif, 0 = Tidak Aktif).

- EstimatedSalary: Estimasi gaji atau penghasilan tahunan pelanggan.

- Exited: Ini adalah target atau label yang menunjukkan apakah pelanggan telah keluar atau tidak (1 = Keluar, 0 = Tidak Keluar).

### Explanatory Data Analaysis

#### Univariate Analysis

Pada dataset ini, label `Not Churn` lebih banyak dibandingkan label `Churn`. Ini berarti data untuk kolom `Exited` memiliki distribusi yang tidak seimbang, distribusi label dapat dilihat pada gambar di bawah ini.

![Distribusi churn](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/univariate-analysis/churn-vs-not-churn.png?raw=true)
Maka, pada saat pemodelan, resampling techniques seperti SMOTE atau undersampling akan digunakan agar model tidak overfit atau bias.

Selanjutnya, berikut visualisasi histogram dan kde plot dari fitur numerik.

![Histogram](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/univariate-analysis/numerical-feature-histogram.png?raw=true)
![KDE plot](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/univariate-analysis/kde-plot-numerical-features.png?raw=true)
Berikut poin-poin penting yang dapat diambil dari 2 plot di atas.

- Credit score pelanggan berada di range 400 hingga 800 dan miring ke kiri (left skewed).
- Rentang umur pengguna berada di sekitar 20 hingga 80 tahun, dan miring ke kanan (right skewed). Usia pelanggan dominan berada di rentang 20 hingga 40 tahun.
- Lama waktu pelanggan menggunakan layanan (Tenure) dari rentang 1 hingga 9 tahun hampir berjumlah sama rata.
- Produk atau layanan yang digunakan pelanggan paling banyak 1 hingga 2 layanan, hanya sedikit pelanggan yang menggunakan lebih dari 2 layanan.
- 70.5% pelanggan memiliki kartu kredit, sisanya tidak.
- Perbedaan proporsi pelanggan yang aktif dan tidak aktif cukup sedikit, hanya berbeda 3% dengan dengan pelanggan aktif yang terbanyak.
- Distribusi saldo pelanggan menunjukkan dua puncak (bimodal). Ada kelompok pelanggan dengan saldo mendekati nol, dan ada kelompok lain dengan saldo lebih tinggi, berkisar di sekitar 100k hingga 150k. Hal ini menunjukkan bahwa ada dua tipe utama pelanggan: mereka yang cenderung tidak menyimpan saldo di akun mereka dan mereka yang menyimpan saldo dalam jumlah signifikan.
- Distribusi Estimated Salary tersebar merata di seluruh rentang, dari 0 hingga sekitar 200k. Ini menunjukkan bahwa tidak ada kelompok pendapatan yang dominan di antara pelanggan; nasabah tersebar secara merata di berbagai tingkatan gaji.

Distribusi pelanggan yang memiliki kartu kredit, status keaktifan pelanggan dapat dilihat pada gambar berikut.

![Has CrCard & Membership](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/univariate-analysis/pie-hscard-activemember.png?raw=true)
Sebanyak 70.5% pelanggan memiliki credit card, dan distribusi member aktif dan tidak aktif hampir seimbang.

Selanjutnya, lihat distibusi geografi dan gender berikut.

![Pie Geography](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/univariate-analysis/pie-geo.png?raw=true)
![Pie Gender](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/univariate-analysis/pie-gender.png?raw=true)
Pelanggan terbanyak berasal dari France yaitu sebesar 50%, diikuti Germany dan Spain. Lalu, sebanyak 54.6% pelanggan adalah pria, sisanya wanita

### Multivariate Analysis

Berikut ini adalah cross-tabulation untuk gender dan geografi terhadap churn rate dalam bentuk pie chart.

![Gender vs Churn](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/multivariate-analysis/churn-gender.png?raw=true)

![Geography vs Churn](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/multivariate-analysis/churn-geography.png?raw=true)
Berdasarkan heatmap dan pie chart, churn lebih banyak untuk gender `male` sebanyak 25%, dan negara `Germany` sebanyak 34%. Sebagian banyak pelanggan berhenti melanjutkan atau menggunaakn layanan (not churn) dengan persentase rata-rata di atas 70%, baik dari kategori gender maupun geografi.

Sebelum melihat korelasi antar fitur numerik, dilakukan pengecekan outlier dan lakukan pembersihan terhadapt outlier.

![Credit Score outlier](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/outliers/credit-score.png?raw=true)
![Age outlier](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/outliers/age.png?raw=true)
![Num of credit outlier](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/outliers/image.png?raw=true)
Pada boxplot di atas terdapat beberapa outliers pada fitur numerik. Outlier pada data `Age` akan ditangani karena analisis akan berfokus pada pelanggan dengan usia 20 sampai 60 tahun, dan pelanggan di atas rentang tersebut cukup sedikit, bisa dilihat pada plot KDE di atas. Outlier juga terdapat pada variabel `CreditScore`, dan satu titik outlier terdapat pada variabel `NumOfProducts`.

Metode Interquartile Range (IQR) digunakan untuk mendeteksi dan menangani outlier dalam data. IQR adalah rentang antara kuartil pertama (Q1) dan kuartil ketiga (Q3). Perhitungannya dapat dilihat pada kode berikut.

```bash
# Menghitung Q1 (kuartil pertama) dan Q3 (kuartil ketiga)
Q1 = data[numerical_features].quantile(0.25)
Q3 = data[numerical_features].quantile(0.75)

# Menghitung IQR (Interquartile Range)
IQR = Q3 - Q1

# Menentukan batas bawah dan batas atas untuk mendeteksi outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Menyaring data yang bukan outliers
data = data[~((data[numerical_features] < lower_bound) | (data[numerical_features] > upper_bound)).any(axis=1)]
```

#### Korelasi Fitur Numerik

Pengamatan hubungan antar fitur numerik dilakukan menggunakan correlation matrix menggunakan metode Pearson correlation. Pearson correlation mengukur kekuatan dan arah hubungan linier antara dua variabel.

![Correlation Matrix](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/multivariate-analysis/corr-matrix.png?raw=true)
Kesimpulan dari correlation matrix di atas adalah:
Variabel `Age`, `Balance`, `NumOfProducts`, `Tenure`, dan `IsActiveMember` adalah variabel yang menunjukkan korelasi paling kuat dengan churn, meskipun keempatnya masih dalam kategori moderat hingga lemah.
Variabel lain seperti `CreditScore`, `EstimatedSalary`, dan `HasCrCard` tidak memiliki korelasi yang signifikan dengan churn dalam dataset ini.

## Data Preparation

Pada tahap Data Preparation, beberapa teknik yang diterapkan dalam proyek ini meliputi:

1. Scaling dan Encoding
Scaling diterapkan pada fitur numerik menggunakan StandardScaler. Proses ini memastikan bahwa semua fitur numerik memiliki distribusi yang seragam sehingga model machine learning tidak terpengaruh oleh perbedaan skala antar fitur.
One-Hot Encoding diterapkan pada fitur kategorikal menggunakan OneHotEncoder. Proses ini mengubah variabel kategorikal menjadi representasi numerik yang dapat digunakan oleh model.

2. Dimensionality Reduction dengan PCA
Principal Component Analysis (PCA) digunakan untuk mereduksi dimensi dari fitur numerik. Setelah di-scaling, fitur-fitur numerik dikurangi menjadi dua komponen utama.
![PCA]()
Dua komponen ini menjelaskan 100% dari variansi data, yang berarti seluruh informasi yang relevan dari fitur asli dipertahankan dalam dua dimensi ini. Hal ini penting untuk mempercepat proses pelatihan model dan mengurangi risiko overfitting pada data yang memiliki banyak fitur.
Teknik ini dipilih karena hasil analisis variansi kumulatif menunjukkan bahwa dua komponen utama dapat menjelaskan semua variansi dalam data, sehingga merupakan pilihan optimal untuk menjaga kesederhanaan tanpa kehilangan informasi yang penting.

3. Handling Imbalance Data dengan SMOTE:
Setelah preprocessing, diterapkan teknik Synthetic Minority Over-sampling Technique (SMOTE) untuk menangani masalah ketidakseimbangan kelas pada target variabel Exited. SMOTE membuat contoh sintetik dari kelas minoritas (Churned) untuk meningkatkan proporsi data minoritas sehingga model dapat lebih seimbang dalam belajar dari kedua kelas. Ini dilakukan untuk meningkatkan kinerja model klasifikasi terutama dalam mengidentifikasi kelas minoritas dengan lebih baik.

**Berikut Alasan mengapa tahapan ini diperlukan**:

- Scaling diperlukan agar model tidak bias terhadap fitur dengan skala besar. Fitur yang berada pada skala yang sama membantu algoritma seperti Gradient Boosting dan SVM menghasilkan performa lebih baik.
- Encoding diperlukan agar fitur kategorikal dapat diproses oleh algoritma machine learning yang umumnya bekerja dengan data numerik.
- PCA membantu menyederhanakan data dan mengurangi dimensi, yang tidak hanya mempercepat waktu pelatihan tetapi juga dapat meningkatkan generalisasi model.
- SMOTE penting dalam situasi di mana data tidak seimbang, karena model machine learning cenderung bias terhadap kelas mayoritas jika ketidakseimbangan tidak diatasi.
**Rubrik/Kriteria Tambahan (Opsional)**:
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Pada proyek ini, saya menggunakan beberapa algoritma machine learning untuk menyelesaikan permasalahan prediksi churn pada dataset nasabah. Algoritma yang dipilih memiliki karakteristik yang berbeda-beda, sehingga memungkinkan perbandingan kinerja dari berbagai perspektif. Berikut adalah penjelasan mengenai model-model yang digunakan serta kelebihan, kekurangan, dan parameter utama yang digunakan dalam proses pemodelan.

1. Logistic Regression
**Kelebihan**: Logistic Regression sederhana dan mudah diinterpretasikan. Algoritma ini bekerja dengan baik ketika hubungan antara fitur dan target linear. Selain itu, Logistic Regression tidak memerlukan tuning yang rumit dan dapat diimplementasikan dengan cepat.
**Kekurangan**: Logistic Regression tidak bekerja dengan baik pada data yang tidak linear dan sensitif terhadap multikolinearitas. Model ini juga mungkin kurang akurat jika dataset terlalu kompleks.
2. Random Forest Classifier
**Kelebihan**: Random Forest sangat baik dalam menangani dataset dengan fitur yang banyak serta secara alami mengatasi overfitting karena melakukan averaging dari beberapa decision trees. Algoritma ini juga robust terhadap data yang tidak seimbang.
**Kekurangan**: Random Forest dapat menjadi lambat pada dataset yang sangat besar, terutama jika menggunakan banyak pohon. Selain itu, interpretasi dari model ini tidak sesederhana Logistic Regression.
3. Gradient Boosting Classifier
**Kelebihan**: Gradient Boosting adalah salah satu metode ensemble yang sangat powerful, karena menggabungkan beberapa weak learners untuk menghasilkan model yang kuat. Algoritma ini biasanya bekerja sangat baik pada data yang tidak seimbang dan dapat mencapai performa tinggi.
**Kekurangan**: Gradient Boosting rentan terhadap overfitting jika tidak dikontrol dengan baik. Selain itu, waktu komputasi yang diperlukan relatif lebih lama dibandingkan model lain, terutama pada dataset besar.
4. Support Vector Classifier (SVC)
**Kelebihan**: SVC bekerja sangat baik pada dataset dengan batas-batas kelas yang jelas. Algoritma ini optimal untuk dataset yang tidak seimbang dan mampu menemukan hyperplane yang memaksimalkan margin antara dua kelas.
**Kekurangan**: SVC tidak skalabel dengan baik pada dataset yang sangat besar, karena waktu komputasinya tinggi. Selain itu, pemilihan kernel yang salah bisa menyebabkan performa yang buruk.
5. K-Nearest Neighbors Classifier (KNN)
**Kelebihan**: KNN adalah algoritma yang sederhana dan tidak memerlukan pelatihan. KNN sangat cocok untuk data yang tidak linier dan mudah diimplementasikan.
**Kekurangan**: KNN sangat sensitif terhadap jumlah tetangga yang dipilih dan dapat lambat pada dataset yang besar. Algoritma ini juga rentan terhadap noise dalam data.
6. XGBoost Classifier
**Kelebihan**: XGBoost adalah versi optimasi dari Gradient Boosting, yang terkenal dengan performa tinggi dan kemampuan menangani data yang tidak seimbang dengan baik. Algoritma ini lebih efisien dan cepat dibandingkan Gradient Boosting biasa.
**Kekurangan**: Sama seperti Gradient Boosting, XGBoost dapat overfit jika tidak diatur dengan baik. Selain itu, tuning XGBoost memerlukan waktu dan pemahaman yang lebih mendalam.
7. MLPClassifier (Multilayer Perceptron)
**Kelebihan**: MLP adalah jaringan saraf tiruan yang mampu menangkap pola non-linear dan interaksi kompleks antar fitur. Algoritma ini seringkali menghasilkan performa yang tinggi ketika dataset memiliki hubungan yang rumit antar fitur.
**Kekurangan**: MLP cenderung memerlukan tuning yang lebih rumit dan lebih banyak waktu komputasi. Selain itu, MLP rawan overfitting jika jaringan terlalu kompleks.

Model terbaik yang dipilih pada kasus ini adalah Gradient Boosting Classifier. Penjelasan detail mengapa model ini dipilih dapat dilihat pada section selanjutnya

## Evaluation

Dalam proyek ini, beberapa metrik evaluasi untuk menilai kinerja model dalam mendeteksi churn. Metrik yang dipilih adalah Precision, Recall, F1 Score, ROC-AUC, dan Akurasi. Pemilihan metrik juga cocok dengan permasalahan yang dihadapi, yaitu ketidakseimbangan data antara kelas positif (nasabah yang churn) dan kelas negatif (nasabah yang tidak churn).
Metrik evaluasi yang digunakan adalah sebagai berikut:

1. Precision

    Precision mengukur proporsi prediksi positif yang benar dari seluruh prediksi positif yang dilakukan oleh model.
    $$\text{PRECISSION} = \frac{TP}{TP+FP}$$
    dimana TP adalah True Postivie dan FP adalah false positives.
    Metrik ini menunjukkan seberapa banyak dari prediksi churn yang benar-benar merupakan churn. Precision yang tinggi berarti model dapat meminimalisir false positive.

2. Recall

    Recall mengukur proporsi prediksi positif yang benar dari seluruh data aktual yang positif.
    $$\text{RECALL} = \frac{TP}{TP+FN}$$
    di mana FN adalah false negatives.
    Recall memberikan gambaran seberapa baik model dalam mendeteksi semua nasabah yang churn. Recall yang tinggi menunjukkan bahwa model mampu menangkap banyak kasus churn meskipun terdapat beberapa kesalahan.

3. F1 Score

    F1 Score adalah rata-rata harmonis dari Precision dan Recall, memberikan keseimbangan antara keduanya.
    $$\text{F1-Score} = 2 \cdot \frac{Precission \cdot Recall}{Precission + Recall}$$
    F1 Score yang tinggi menunjukkan bahwa model memiliki performa yang baik dalam mendeteksi churn dengan meminimalisir kesalahan baik dari sisi false positives maupun false negatives.

4. ROC-AUC

    ROC-AUC mengukur kemampuan model untuk membedakan antara kelas positif dan negatif. ROC (Receiver Operating Characteristic) adalah grafik yang menunjukkan true positive rate (TPR) versus false positive rate (FPR) pada berbagai threshold klasifikasi.
    True Positive Rate (TPR): Juga dikenal sebagai Sensitivitas, dihitung sebagai:
    $$\text{TPR} = \frac{TP}{TP+FP}$$
    dimana TP ada True Positive dan FP adalah False Positive.
    False Positive Rate (FPR): Dihitung sebagai:
    $$\text{FPR} = \frac{FP}{FP+TN}$$
    dimana FP adalah False Positive dan TN adalah True Negative.
    Dalam ROC Curve, FPR ditampilkan pada sumbu X dan TPR pada sumbu Y. Dengan mengubah ambang batas, kita dapat menghitung nilai TPR dan FPR yang berbeda, membentuk kurva.
    AUC adalah ukuran yang menunjukkan seberapa baik model klasifikasi dapat memisahkan dua kelas. Nilai AUC berkisar antara 0 hingga 1:
    - AUC = 1: Model sempurna yang dapat memisahkan kelas positif dan negatif tanpa kesalahan.
    - AUC = 0.5: Model yang tidak lebih baik dari tebakan acak (random).
    - AUC < 0.5: Model yang lebih buruk dari tebakan acak, menunjukkan bahwa model mungkin terbalik dalam klasifikasi.

5. Accuracy

    Akurasi mengukur proporsi prediksi yang benar (positif dan negatif) dari seluruh prediksi yang dilakukan oleh model.
    $$\text{Accuracy} = \frac{TP + TN}{TP+TN+FP+FN}$$
    Akurasi memberikan gambaran umum tentang seberapa baik model dalam membuat prediksi secara keseluruhan. Namun, dalam kasus data yang tidak seimbang, akurasi tidak selalu mencerminkan kinerja model secara tepat.

### Hasil Evaluasi

Berikut adalah tabel hasil evaluasi.
![Evaluation](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/model-result.png?raw=true)
Dan berikut adalah plot model terbaik berdasarkan hasil evaluasi.
![Best Model](https://github.com/Mnjar/bank-churn-rate-analysis/blob/main/images/best_model.png?raw=true)
Setelah menerapkan beberapa model machine learning dan mengevaluasinya dengan metrik-metrik di atas, kesimpulan hasilnya adalah sebagai berikut:

- Precision tertinggi dicapai oleh Gradient Boosting dengan nilai 0.821045, yang menunjukkan bahwa dari seluruh prediksi churn yang dihasilkan oleh model, 82.1% di antaranya adalah benar-benar churn.
- Recall tertinggi juga dicapai oleh Gradient Boosting dengan nilai 0.776733, menunjukkan bahwa model berhasil menangkap 77.7% dari semua nasabah yang sebenarnya churn. Ini menandakan kemampuan model yang baik dalam mendeteksi churn.
- F1 Score terbaik juga diperoleh dari Gradient Boosting, yaitu 0.791447. Ini mengindikasikan keseimbangan yang baik antara Precision dan Recall.
- ROC-AUC model Gradient Boosting mencapai nilai 0.806475, menunjukkan bahwa model memiliki kemampuan yang baik dalam membedakan antara nasabah yang churn dan tidak churn.
Akurasi model Gradient Boosting adalah 0.776733, menunjukkan bahwa 77.7% dari seluruh prediksi model adalah benar, baik untuk kelas positif maupun negatif.

**Kesimpulan**:
Berdasarkan metrik evaluasi yang digunakan, Gradient Boosting terbukti menjadi model yang paling efektif untuk mendeteksi churn dalam proyek ini. Metrik yang dipilih—Precision, Recall, F1 Score, ROC-AUC, dan Akurasi—memberikan gambaran yang komprehensif tentang kinerja model dalam konteks data yang tidak seimbang, sehingga hasilnya dapat diandalkan untuk implementasi strategi retensi pelanggan di masa depan.
