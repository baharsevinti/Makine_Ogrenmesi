import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report,confusion_matrix,confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
# Dosya yollarını belirleme
txt_file_path = 'data.txt'  # TXT dosya yolu
csv_file_path = 'data.csv'  # CSV dosya yolu

# Sütun adlarını tanımlama
column_names = [
    'Number of times pregnant',
    'Plasma glucose concentration',
    'Diastolic blood pressure',
    'Triceps skinfold thickness',
    '2-Hour serum insulin',
    'Body mass index',
    'Diabetes pedigree function',
    'Age',
    'Class variable'
]

# TXT dosyasını okuma ve DataFrame'e dönüştürme
df = pd.read_csv(txt_file_path, sep='\t', names=column_names)

# DataFrame'i CSV dosyasına yazma
df.to_csv(csv_file_path, index=False)


# CSV dosyasından veri yükleme, sütun isimlerini belirtme
df = pd.read_csv('data.csv', names=column_names,header=0)

# Eksik değerleri (0 olarak kodlanmış) NaN ile değiştirme ve medyan ile doldurma
columns_with_zeros = [
    'Plasma glucose concentration',
    'Diastolic blood pressure',
    'Triceps skinfold thickness',
    '2-Hour serum insulin',
    'Body mass index'
]
for col in columns_with_zeros:
    df[col].replace(0, np.nan, inplace=True)
    df[col].fillna(df[col].median(), inplace=True)

X = df.iloc[:, :-1].values  # Girdi değişkenleri
y = df.iloc[:, -1].values   # Çıktı değişkeni

# Verileri ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veri bölünüyor
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Eğitim setindeki sınıf dengesizliği gideriliyor
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)



feature_names = df.columns[:-1]

# Min-Max Normalizasyonu
min_max_scaler = MinMaxScaler()
X_min_max_scaled = min_max_scaler.fit_transform(X_sm)
df_min_max_scaled = pd.DataFrame(X_min_max_scaled, columns=feature_names)  # Doğru sütun isimlerini kullanma
print("Min-Max Normalizasyonu Sonrası:\n", df_min_max_scaled)



# PCA İÇİN
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sm)
print("PCA sonrası bileşenler:\n", pca.components_)

# LDA İÇİN
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_sm, y_sm)
print("LDA sonrası biileşenleri:\n", lda.scalings_)
# PCA için
pca_features_df = pd.DataFrame(pca.components_.T, index=features, columns=['PCA1', 'PCA2'])
print("PCA için en ayırt edici öznitelikler:\n", pca_features_df.abs().idxmax())

# LDA için
lda_features_df = pd.DataFrame(lda.scalings_, index=features, columns=['LDA1'])
print("LDA için en ayırt edici öznitelikler:\n", lda_features_df.abs().idxmax())
# Veri setini %70 eğitim ve %30 test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Çoklu Doğrusal Regresyon
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Çoklu Doğrusal Regresyon katsayıları
print("Çoklu Doğrusal Regresyon Katsayıları:\n", linear_model.coef_)
print("Çoklu Doğrusal Regresyon Sabiti:", linear_model.intercept_)

# Test veri seti üzerinde tahminler yapma ve performans metrikleri hesaplama
y_pred_linear = linear_model.predict(X_test)
print("Çoklu Doğrusal Regresyon - MSE:", mean_squared_error(y_test, y_pred_linear))
print("Çoklu Doğrusal Regresyon - R2 Skoru:", r2_score(y_test, y_pred_linear))

# Multinominal Lojistik Regresyon modelini eğitme

if feature_names.nunique() > 2:
    logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logistic_model.fit(X_train, y_train)

    # Multinominal Lojistik Regresyon katsayıları
    print("Multinominal Lojistik Regresyon Katsayıları:\n", logistic_model.coef_)
    print("Multinominal Lojistik Regresyon Sabiti:", logistic_model.intercept_)

    # Test veri seti üzerinde tahminler yapma ve performans metrikleri hesaplama
    y_pred_logistic = logistic_model.predict(X_test)
    print("Multinominal Lojistik Regresyon - Doğruluk Skoru:", accuracy_score(y_test, y_pred_logistic))
    print("Multinominal Lojistik Regresyon - Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_logistic))
else:
    print("Hedef değişken yalnızca iki kategori içerdiği için Multinominal Lojistik Regresyon uygulanamaz.")


#  Karar Ağacı modelini eğitme
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Ağaç yapısını görselleştirme
plt.figure(figsize=(20,10))
plot_tree(decision_tree_model, filled=True, feature_names=feature_names, class_names=str(decision_tree_model.classes_))
plt.show()

# Test verisi üzerinde kestirim yapma
y_pred = decision_tree_model.predict(X_test)

# Performans metriklerini hesaplama
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))


# Naive Bayes modelini eğitme
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Test verisi üzerinde kestirim yapma
y_pred = nb_model.predict(X_test)

# Performans metriklerini hesaplama
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("Konfüzyon Matrisi:\n", cm)
print("Doğruluk Skoru:", acc)
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# ROC Eğrisi ve AUC
# Not: ROC ve AUC, ikili sınıflandırma görevleri için geçerlidir.
if len(np.unique(y)) == 2:
    y_probs = nb_model.predict_proba(X_test)[:, 1]  # Sınıf olasılıklarını al
    auc = roc_auc_score(y_test, y_probs)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Naive Bayes (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Eğrisi')
    plt.legend(loc="best")
    plt.show()
