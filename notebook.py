#!/usr/bin/env python
# coding: utf-8

# # Churn Rate Analysis Bank Customer
# ----
# - **Nama:** Muhamad Fajar Faturohman
# - **Email:** fajarftr2605@gmail.com
# - **ID Dicoding:** mnjarrr

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# # Load Dataset

# In[7]:


data = pd.read_csv('../dataset/Churn_Modelling.csv')


# In[8]:


data.shape


# Dataset ini memiliki 14 kolom dengan 10.000 baris

# In[9]:


data.head()


# Drop kolom `RowNumber`, karena ini tidak diperlukan

# In[10]:


data = data.drop('RowNumber', axis=1)


# ## Deskripsi variabel
# 
# - CustomerId: ID unik untuk setiap pelanggan.
# 
# - Surname: Nama belakang pelanggan.
# 
# - Skor kredit pelanggan. Skor ini menggambarkan kelayakan kredit seseorang. Nilai yang lebih tinggi menunjukkan riwayat kredit yang lebih baik, sementara nilai yang lebih rendah menunjukkan risiko kredit yang lebih tinggi.
# 
# - Geography: Negara asal pelanggan.
# 
# - Gender: Jenis kelamin pelanggan
# 
# - Age: Umur pelanggan.
# 
# - Tenure: Lama waktu pelanggan telah menggunakan layanan (dalam tahun).
# 
# - Balance: Saldo akun pelanggan. Ini adalah jumlah uang yang pelanggan miliki di akun mereka. Fitur ini bisa menjadi indikator kekayaan pelanggan atau potensi transaksi.
# 
# - NumOfProducts: Jumlah produk yang digunakan pelanggan. Kolom ini mencerminkan berapa banyak layanan atau produk yang pelanggan gunakan dari perusahaan, yang bisa menunjukkan loyalitas pelanggan.
# 
# - HasCrCard: Status apakah pelanggan memiliki kartu kredit atau tidak (1 = Ya, 0 = Tidak).
# 
# - IsActiveMember: Status apakah pelanggan adalah anggota yang aktif (1 = Aktif, 0 = Tidak Aktif).
# 
# - EstimatedSalary: Estimasi gaji atau penghasilan tahunan pelanggan.
# 
# - Exited: Ini adalah target atau label yang menunjukkan apakah pelanggan telah keluar atau tidak (1 = Keluar, 0 = Tidak Keluar).

# # Explanatory Data Analysis

# ## Basic data understanding

# In[7]:


data.info()


# In[12]:


# Memisahkan categorical, numerical, dan target features
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
target = 'Exited'

print(f'Numerical features: {numerical_features}\n\nCategorical Features: {categorical_features}\n\nTarget features: {target}')


# In[33]:


data[numerical_features].describe()


# Cek apakah terdapat nilai null pada data

# In[16]:


data.isnull().sum()


# Berdasarkan output di atas, tidak terdapat nilai null pada dataset

# ## Univariate Analysis

# In[232]:


churn_or_not_churn = ['Not Churn', 'Churn']
exited_counts = data[target].value_counts().sort_index()
colors = sns.color_palette(['green', 'red'])

# Plotting pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    exited_counts,
    labels=churn_or_not_churn,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Churn vs Not Churn')
plt.show()


# Berdasarkan distribusi di atas, label `Not Churn` lebih banyak dibandingkan label `Churn`. Ini berarti data untuk kolom `Exited` memiliki distribusi yang tidak seimbang. Maka, pada saat pemodelan, resampling techniques seperti SMOTE atau undersampling akan digunakan agar model tidak overfit atau bias.

# ### Histogram

# In[34]:


rows = (len(numerical_features) // 2) + (len(numerical_features) % 2)
cols = 2

# Membuat figure dan axes untuk subplots
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
axes = axes.flatten()  # Mengubah axes menjadi array satu dimensi untuk kemudahan akses

# Membuat histogram untuk setiap feature
for i, feature in enumerate(numerical_features):
    axes[i].hist(data[feature], bins=30, edgecolor='black')
    axes[i].set_title(f'Histogram of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

# Menghapus subplot kosong jika ada
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Mengatur layout dari subplot
plt.tight_layout()
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.subplots_adjust(top=0.9)  # Memberikan ruang untuk title
plt.show()


# ### KDE Plot

# In[35]:


rows = (len(numerical_features) // 2) + (len(numerical_features) % 2)
cols = 2

colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']


fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
axes = axes.flatten()  # Mengubah axes menjadi array satu dimensi untuk kemudahan akses

for n, feature in enumerate(numerical_features):
    sns.kdeplot(data[feature], ax=axes[n], color=colors[n], fill=True)
    axes[n].set_title(f'Distribution of {feature}')
    axes[n].set_xlabel(feature)
    axes[n].set_ylabel('Density')

# Menghapus subplot kosong jika ada
for j in range(n + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Distributions of Numeric Features", fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()


# - Credit score pelanggan berada di range 400 hingga 800 dan miring ke kiri (left skewed).
# 
# - Rentang umur pengguna berada di sekitar 20 hingga 80 tahun, dan miring ke kanan (right skewed). Usia pelanggan dominan berada di rentang 20 hingga 40 tahun.
# 
# - Lama waktu pelanggan menggunakan layanan (Tenure) dari rentang 1 hingga 9 tahun hampir berjumlah sama rata.
# 
# - Produk atau layanan yang digunakan pelanggan paling banyak 1 hingga 2 layanan, hanya sedikit pelanggan yang menggunakan lebih dari 2 layanan.
# 
# - 70.5% pelanggan memiliki kartu kredit, sisanya tidak.
# 
# - Perbedaan proporsi pelanggan yang aktif dan tidak aktif cukup sedikit, hanya berbeda 3% dengan dengan pelanggan aktif yang terbanyak.
# 
# - Distribusi saldo pelanggan menunjukkan dua puncak (bimodal). Ada kelompok pelanggan dengan saldo mendekati nol, dan ada kelompok lain dengan saldo lebih tinggi, berkisar di sekitar 100k hingga 150k. Hal ini menunjukkan bahwa ada dua tipe utama pelanggan: mereka yang cenderung tidak menyimpan saldo di akun mereka dan mereka yang menyimpan saldo dalam jumlah signifikan.
# 
# - Distribusi Estimated Salary tersebar merata di seluruh rentang, dari 0 hingga sekitar 200k. Ini menunjukkan bahwa tidak ada kelompok pendapatan yang dominan di antara pelanggan; nasabah tersebar secara merata di berbagai tingkatan gaji.
# 

# In[36]:


features_to_plot = categorical_features[2:]

fig, axes = plt.subplots(1, len(features_to_plot), figsize=(12, 6))

for i, feature in enumerate(features_to_plot):
    # Menghitung jumlah unique values dari feature
    feature_counts = data[feature].value_counts()
    
    wedges, texts, autotexts = axes[i].pie(
        feature_counts.values,
        labels=feature_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=['green', 'red', 'blue', 'purple', 'orange', 'yellow'],
        wedgeprops={'edgecolor': 'black'}
    )
    
    axes[i].legend(
        wedges,
        feature_counts.index,
        title=f'{feature}',
        loc='upper left',
        bbox_to_anchor=(1, 1)
    )
    
    axes[i].set_title(f'Pie Chart of {feature}')

plt.tight_layout()
plt.show()


# - Sebanyak 70.5% pelanggan memiliki credit card
# - Distribusi member aktif dan tidak aktif hampir seimbang

# In[21]:


for feature in categorical_features[:2]:
    # Menghitung jumlah unique values dari feature
    feature_counts = data[feature].value_counts()
    
    plt.figure(figsize=(6, 6))
    plt.pie(
        feature_counts.values,
        labels=feature_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=['green', 'red', 'yellow', 'purple', 'orange', 'yellow'],
        wedgeprops={'edgecolor': 'black'}
    )
    plt.title(f'Pie Chart of {feature}')
    
    plt.legend(
        feature_counts.index,
        title='Categories',
        loc='upper left',
        bbox_to_anchor=(1, 1)
    )
    
    # Menampilkan pie chart
    plt.tight_layout()
    plt.show()


# - Pelanggan terbanyak berasal dari France yaitu sebesar 50%, diikuti Germany dan Spain.
# - Sebanyak 54.6% pelanggan adalah pria.

# ## Multivariate Analysis

# In[251]:


# cross-tabulation antara Gender dan Churn
gender_churn_ct = pd.crosstab(data['Gender'], data['Exited'])

# heatmap untuk Gender vs Churn
plt.figure(figsize=(8, 6))
sns.heatmap(gender_churn_ct, annot=True, cmap='Blues', fmt='d', cbar=True, 
            xticklabels=['Not Churn', 'Churn'], yticklabels=gender_churn_ct.index.tolist())
plt.title('Heatmap of Gender vs Churn')
plt.show()

# cross-tabulation antara Geography dan Churn
geo_churn_ct = pd.crosstab(data['Geography'], data['Exited'])

# heatmap untuk Geography vs Churn
plt.figure(figsize=(8, 6))
sns.heatmap(geo_churn_ct, annot=True, cmap='Blues', fmt='d', cbar=True, 
            xticklabels=['Not Churn', 'Churn'], yticklabels=geo_churn_ct.index.tolist())
plt.title('Heatmap of Geography vs Churn')
plt.show()


# In[52]:


# Cross-tab untuk Gender dan Churn
gender_churn = pd.crosstab(data['Gender'], data['Exited'], normalize='index') * 100

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].pie(gender_churn.loc['Male'], labels=['Not Churn', 'Churn'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
ax[0].set_title('Male - Churn vs Not Churn')

ax[1].pie(gender_churn.loc['Female'], labels=['Not Churn', 'Churn'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
ax[1].set_title('Female - Churn vs Not Churn')

fig.suptitle('Churn vs Not Churn for Gender', fontsize=16)

plt.tight_layout()
plt.show()


# In[53]:


# Cross-tab untuk Geography dan Churn
geo_churn = pd.crosstab(data['Geography'], data['Exited'], normalize='index') * 100

fig, ax = plt.subplots(1, 3, figsize=(15, 6))

ax[0].pie(geo_churn.loc['France'], labels=['Not Churn', 'Churn'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
ax[0].set_title('France - Churn vs Not Churn')

ax[1].pie(geo_churn.loc['Germany'], labels=['Not Churn', 'Churn'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
ax[1].set_title('Germany - Churn vs Not Churn')

ax[2].pie(geo_churn.loc['Spain'], labels=['Not Churn', 'Churn'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
ax[2].set_title('Spain - Churn vs Not Churn')

fig.suptitle('Churn vs Not Churn for Geography', fontsize=16)

plt.tight_layout()
plt.show()


# Berdasarkan heatmap dan pie chart, churn lebih banyak untuk gender `male` sebanyak 25%, dan negara `Germany` sebanyak 34%

# In[23]:


# Membuat box plot untuk setiap feature numerik
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[feature], color='skyblue')
    plt.title(f"Box Plot for {feature} (Detecting Outliers)")
    plt.xlabel(feature)
    plt.tight_layout()
    plt.show()


# Pada boxplot di atas terdapat beberapa outliers pada fitur numerik. Outlier pada data `Age` akan ditangani karena analisis akan berfokus pada pelanggan dengan usia 20 sampai 60 tahun, dan pelanggan di atas rentang tersebut cukup sedikit, bisa dilihat pada plot [KDE](#kde-plot) di atas. Outlier juga terdapat pada variabel `CreditScore`, dan satu titik outlier terdapat pada variabel `NumOfProducts`.

# In[24]:


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


# In[25]:


data.shape


# In[57]:


sns.pairplot(data[numerical_features + [target]], diag_kind = 'kde',  hue='Exited')
plt.show()


# Berikut kesimpulan berdasarkan pairplot di atas:
# 
# - Variabel `Age` dan `Balance` menunjukkan pola yang lebih jelas dalam memengaruhi churn. Pelanggan yang lebih tua (di atas 50 tahun) dan pelanggan dengan saldo tinggi cenderung lebih berpotensi untuk churn (Exited = 1). Hal ini menunjukkan bahwa usia dan saldo akun merupakan faktor penting yang perlu diperhatikan dalam menganalisis dan memprediksi churn.
# 
# - Variabel seperti `CreditScore`, `Tenure`, dan `EstimatedSalary` tidak menunjukkan pola yang jelas dalam kaitannya dengan churn. Data ini tersebar merata untuk kedua kategori (Exited = 0 dan 1), yang mengindikasikan bahwa variabel-variabel ini mungkin tidak memiliki dampak signifikan dalam menentukan apakah pelanggan akan churn atau tidak.
# 
# - Variabel `NumOfProducts` memperlihatkan bahwa pelanggan dengan satu atau dua produk memiliki distribusi churn yang merata, sedangkan pelanggan dengan tiga produk lebih jarang terlihat churn. Ini mungkin menunjukkan bahwa pelanggan yang lebih beragam dalam penggunaan produk bank cenderung lebih loyal.
# 
# Secara keseluruhan, usia pelanggan dan saldo mereka tampaknya menjadi faktor kunci yang perlu difokuskan dalam strategi retensi pelanggan. Analisis lebih lanjut diperlukan untuk memahami bagaimana mengoptimalkan interaksi dengan segmen pelanggan ini untuk mengurangi tingkat churn.

# In[58]:


# Menghitung matriks korelasi antar fitur numerik
correlation_matrix = data[numerical_features + [target] + categorical_features[2:]].corr()

plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='RdBu_r', 
            vmin=-1, vmax=1,
            center=0, 
            linewidths=0.5,
            cbar_kws={'label': 'Correlation'},
            fmt='.2f')

plt.title("Correlation Matrix for Numerical Features", fontsize=16)
plt.tight_layout()

plt.show()


# Kesimpulan dari correlation matrix di atas adalah:
# Variabel `Age`, `Balance`, `NumOfProducts`, `Tenure`, dan `IsActiveMember` adalah variabel yang menunjukkan korelasi paling kuat dengan churn, meskipun keempatnya masih dalam kategori moderat hingga lemah.
# Variabel lain seperti `CreditScore`, `EstimatedSalary`, dan `HasCrCard` tidak memiliki korelasi yang signifikan dengan churn dalam dataset ini.

# ## Data Preparation
# Variabel yang akan diaplikasikan untuk reduksi dengan PCA yaitu `Age`, `EstimatedSalary`, `CreditScore`, `Balance`, `NumOfProducts`, `Tenure`.

# ### Dimensionality Reduction With PCA 

# In[26]:


# Fungsi untuk plotting variansi kumulatif dan PCA
def plot_pca_variance(X_scaled):
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot variansi kumulatif
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

    # Print variansi kumulatif
    for i, var in enumerate(cumulative_variance):
        print(f"Component {i+1}: {var:.4f} variance explained")


# In[27]:


pca = plot_pca_variance(data[numerical_features])


# Component 1 menjelaskan 54.11% dari variansi, dan Component 2 menjelaskan total 100% (termasuk variansi dari Component 1). Ini menunjukkan bahwa dengan dua komponen pertama, sudah menjelaskan 100% dari variansi data.

# Karena proporsi data `Churn dan Not Churn` pada variabel `Exited` tidak seimbang, teknik oversampling dengan SMOTE akan digunakan untuk menangani masalah ketidakseimbangan kelas pada target variabel.

# In[28]:


# Preprocessor
scaler = StandardScaler()
encoder = OneHotEncoder(drop='first')

# Preprocessing pipeline: Scaling + Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_features),
        ('cat', encoder, categorical_features)
    ]
)

X = data.drop(columns=[target, 'CustomerId', 'Surname'])
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply preprocessing pada train dan test data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Apply SMOTE setelah preprocessing
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Setelah SMOTE, lakukan PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_resampled)
X_test_pca = pca.transform(X_test_preprocessed)


# ## Modeling & Evaluation

# Model yang digunakan adalah:
# - Logistic Regression
# - Random Forest Classifier
# - Gradient Boosting Classifier
# - Support Vector Classifier (SVC)
# - K-Nearest Neighbors Classifier (KNN)
# - XGBoost Classifier
# - MLPClassifier (Multilayer Perceptron)
# 
# Metrik evaluasi yang digunakan adalah sebagai berikut.
# - Precision: Precision mengukur proporsi prediksi positif yang benar dari seluruh prediksi positif yang dilakukan oleh model.
# 
# - Recall: Metrik ini digunakan untuk mengetahui seberapa baik model dalam mendeteksi kelas positif (sangat penting jika Anda menginginkan model yang bagus dalam menangkap positif).
# 
# - F1-Score: Ini adalah rata-rata harmonik dari Precision dan Recall, digunakan untuk menyeimbangkan keduanya, terutama pada dataset yang tidak seimbang.
# 
# - ROC-AUC: Metrik ini mengukur seberapa baik model dapat membedakan antara kelas positif dan negatif secara keseluruhan. AUC (Area Under Curve) dari ROC (Receiver Operating Characteristic) lebih tinggi berarti model lebih baik dalam diskriminasi.
# 
# - Accuracy: Akurasi mengukur proporsi prediksi yang benar (positif dan negatif) dari seluruh prediksi yang dilakukan oleh model.

# In[46]:


# List Model
models = {
    'Logistic Regression': LogisticRegression(C=1, max_iter=100, penalty='l2', solver='saga', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, max_depth=3, n_estimators=200, random_state=42),
    'SVM': SVC(C=1, gamma='scale', kernel='rbf', random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=9, algorithm='auto', weights='uniform'),
    'XGBoost': XGBClassifier(objective='binary:logistic', colsample_bytree=0.8, gamma=1.0, learning_rate=0.1, max_depth=7, n_estimators=200, random_state=42),
    'MLPClassifier': MLPClassifier(max_iter=700, activation='tanh', alpha=0.01, hidden_layer_sizes=(100,), solver='adam', random_state=42)
}

# result DataFrame 
results_df = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'ROC-AUC'])

for name, model in models.items():
    # Train
    model.fit(X_train_pca, y_train_resampled)
    
    # Predict
    y_pred = model.predict(X_test_pca)
    
    # Evaluate
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Memprediksi probabilitas untuk ROC-AUC (untuk model yang mendukung prediksi probabilitas)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_pca)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None
    
    # Extract metrics
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    accuracy = report['accuracy']
    
    # buat dataframe baru yang berisi training result
    new_row = pd.DataFrame({
        'Model': [name],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1_score],
        'Accuracy': [accuracy],
        'ROC-AUC': [roc_auc]
    })
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)


# In[50]:


results_df


# In[54]:


def plot_best_model_comparison(metrics, values, models, title):
    """ Fungsi plot perbandingan model terbaik untuk setiap metrik.

    Args:
        metrics (list): List nama metrik (e.g., ['Recall', 'F1 Score', 'ROC-AUC']).
        values (list): List metric values.
        models (list): List nama model terbaik.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    x = np.arange(len(metrics))
    
    width = 0.4

    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in np.linspace(0, 1, len(metrics))]

    bars = ax.bar(x, values, width, color=colors)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)

    # labels
    def add_labels(bars, values):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.2f}', ha='center', va='bottom')

    add_labels(bars, values)

    legend_labels = [f"{metric}: {model}" for metric, model in zip(metrics, models)]
    ax.legend(bars, legend_labels, title='Best Model for Each Metric')

    plt.tight_layout()
    plt.show()


# In[55]:


# Ambil model terbaik berdasarkan Recall, F1-Score, dan ROC-AUC dari dataset dengan Tenure
best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
best_precission = results_df.loc[results_df['Precision'].idxmax()]
best_recall= results_df.loc[results_df['Recall'].idxmax()]
best_f1_score = results_df.loc[results_df['F1 Score'].idxmax()]
best_roc_auc = results_df.loc[results_df['ROC-AUC'].idxmax()]

# Data untuk plotting
metrics = ['Accuracy', 'precission', 'Recall', 'F1 Score', 'ROC-AUC']
values_with_tenure = [
    best_accuracy['Accuracy'], 
    best_precission['Precision'], 
    best_recall['Recall'], 
    best_f1_score['F1 Score'], 
    best_roc_auc['ROC-AUC']
]

# Nama model terbaik untuk setiap metrik
models_with_tenure = [
    best_accuracy['Model'],
    best_precission['Model'],
    best_recall['Model'], 
    best_f1_score['Model'], 
    best_roc_auc['Model']
]

plot_best_model_comparison(metrics, values_with_tenure, models_with_tenure, title='Best Model')


# Berdasarkan data di atas, model dengan metrik evaluasi terbaik adalah Gradient Boosting. Maka, model Gradient dapat dipilih karena memiliki score evaluasi terbaik karena memiliki skor akurasi terbaik dibanding model lainnya pada semua metrik evaluasi, berikut detailnya:
# 
# - Kinerja Keseluruhan: Dengan nilai tertinggi dalam precision, recall, dan F1 score, Gradient Boosting menunjukkan kemampuannya yang baik dalam mendeteksi kelas positif (nasabah yang churn) sambil mempertahankan tingkat kesalahan yang rendah.
# 
# - Kemampuan Diskriminasi: ROC-AUC yang tinggi menunjukkan bahwa model ini memiliki kemampuan yang sangat baik dalam membedakan antara nasabah yang churn dan tidak churn. Ini sangat penting dalam konteks bisnis, di mana mendeteksi nasabah yang berpotensi keluar dapat mengarah pada strategi retensi yang lebih efektif.
# 
# - Robustness terhadap Data Tidak Seimbang: Gradient Boosting dapat menangani masalah data tidak seimbang dengan lebih baik dibandingkan model lainnya, menjadikannya pilihan yang tepat untuk kasus ini.
# 
# Dengan demikian, Gradient Boosting dipilih sebagai model terbaik untuk digunakan dalam prediksi churn, karena kemampuannya yang superior dalam semua aspek evaluasi dan relevansi terhadap masalah yang dihadapi.
