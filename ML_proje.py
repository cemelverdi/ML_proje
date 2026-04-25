import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap

data = load_breast_cancer()

X = data.data
y = data.target

df_X = pd.DataFrame(X, columns=data.feature_names)
df_y = pd.DataFrame(y, columns=["target"])
df = pd.concat([df_X, df_y], axis=1)

print(df.head(5))
print("Satır ve sütun sayısı:", df.shape)
print("\nSütun isimleri:")
print(df.columns)

print("\nVeri tipleri:")
print(df.dtypes)

print("\nEksik değer sayıları:")
print(df.isnull().sum())

#Bu aşamada veri seti başarıyla yüklendi. Özellikler X, hedef değişken
# ise y olarak ayrıldı. Veri tablo formatına dönüştürülerek ilk 5 satır incelendi
# ve veri yapısının doğru şekilde geldiği görüldü.

feature_df = df.drop("target", axis=1)

plt.figure(figsize=(20, 10))

sns.boxplot(data=feature_df)

plt.xticks(rotation=90)
plt.title("Tum Ozellikler Icin Boxplot")
plt.show()

#Boxplot grafikleri incelendiğinde bazı özelliklerde aykırı değer olabilecek gözlemler bulunduğu görülmektedir.
# Bu tür değerler özellikle ölçekleme ve bazı makine öğrenmesi modelleri üzerinde etkili olabilir.

outlier_counts = {}

for col in feature_df.columns:
    Q1 = feature_df[col].quantile(0.25)
    Q3 = feature_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = feature_df[(feature_df[col] < lower_bound) | (feature_df[col] > upper_bound)]
    outlier_counts[col] = outliers.shape[0]

outlier_df = pd.DataFrame(outlier_counts.items(), columns=["Ozellik", "Aykiri Deger Sayisi"])
print(outlier_df)

#IQR yöntemi kullanılarak her özellik için aykırı değer sayısı hesaplandı. Bazı özelliklerde aykırı değer sayısının
# diğerlerine göre daha yüksek olduğu gözlemlendi.
# Bu durum, veri dağılımının her sütunda aynı yapıda olmadığını göstermektedir.

print("Veri tipleri:")
print(df.dtypes)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

print("\nSayisal degisken sayisi:", len(numerical_columns))
print("Kategorik degisken sayisi:", len(categorical_columns))

print("\nSayisal degiskenler:")
print(numerical_columns)

print("\nKategorik degiskenler:")
print(categorical_columns)

#Veri tipi incelemesinde tüm değişkenlerin sayısal yapıda olduğu görülmüştür.
#Kategorik değişken bulunmamaktadır. Bu durum, veri ön işleme ve modelleme aşamalarını kolaylaştırmaktadır

stats_df = pd.DataFrame({
    "Mean": feature_df.mean(),
    "Median": feature_df.median(),
    "Min": feature_df.min(),
    "Max": feature_df.max(),
    "Std": feature_df.std(),
    "Q1": feature_df.quantile(0.25),
    "Q3": feature_df.quantile(0.75)
})

print(stats_df)

#Her özellik için temel istatistiksel ölçüler hesaplanmıştır. Bazı değişkenlerde minimum ve maksimum değerler arasında
# geniş farklar bulunurken, bazı sütunlarda ortalama ve medyan değerlerinin birbirine yakın olduğu görülmüştür.
# Bu durum, değişkenlerin dağılımlarının farklı yapılar gösterdiğini ortaya koymaktadır.

corr_matrix = feature_df.corr()

print(corr_matrix)

plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Korelasyon Matrisi Heatmap")
plt.show()

corr_pairs = corr_matrix.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0]
corr_pairs = corr_pairs.sort_values(ascending=False)

seen = set()
top_pairs = []

for (col1, col2), value in corr_pairs.items():
    pair = tuple(sorted([col1, col2]))
    if pair not in seen:
        seen.add(pair)
        top_pairs.append((col1, col2, value))
    if len(top_pairs) == 3:
        break

top_corr_df = pd.DataFrame(top_pairs, columns=["Ozellik 1", "Ozellik 2", "Korelasyon"])
print(top_corr_df)

#Pearson korelasyon matrisi incelendiğinde bazı özellik çiftleri arasında yüksek düzeyde ilişki olduğu görülmüştür.
#Bu durum, bazı değişkenlerin benzer bilgileri taşıyabileceğini göstermektedir. Heatmap görselleştirmesi ile
#değişkenler arasındaki ilişkiler daha net biçimde gözlemlenmiştir.
#En yüksek korelasyonlu özellik çiftleri, veri setinde bazı değişkenlerin birlikte hareket ettiğini göstermektedir.
#Bu tür güçlü ilişkiler, boyut indirgeme yöntemlerinin neden faydalı olabileceğini desteklemektedir.

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(X_scaled.head())

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)

print("X_train boyutu:", X_train.shape)
print("X_val boyutu:", X_val.shape)
print("X_test boyutu:", X_test.shape)

print("y_train boyutu:", y_train.shape)
print("y_val boyutu:", y_val.shape)
print("y_test boyutu:", y_test.shape)

#Veri seti %70 eğitim, %10 doğrulama ve %20 test olacak şekilde bölünmüştür.
#Sınıf dağılımının korunması amacıyla stratify parametresi kullanılmıştır.

pca_full = PCA()
X_train_pca_full = pca_full.fit_transform(X_train)

explained_variance = pca_full.explained_variance_ratio_

print("Explained variance ratio:")
print(explained_variance)

mean_variance = np.mean(explained_variance)
print("\nExplained variance ratio ortalamasi:", mean_variance)

n_components_selected = np.sum(explained_variance > mean_variance)
print("Secilen PCA bilesen sayisi:", n_components_selected)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel("Bilesen Numarasi")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance Ratio")
plt.grid(True)
plt.show()

#PCA uygulanarak bileşenlerin açıkladığı varyans oranları incelenmiştir. İlk bileşenlerin verinin
#daha büyük bir kısmını temsil ettiği görülmüştür. Ortalama explained variance ratio değerinden
#büyük olan bileşenler seçilerek uygun PCA boyutu belirlenmiştir.

pca = PCA(n_components=n_components_selected)

X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

print("X_train_pca boyutu:", X_train_pca.shape)
print("X_val_pca boyutu:", X_val_pca.shape)
print("X_test_pca boyutu:", X_test_pca.shape)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    cmap='viridis',
    alpha=0.7
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Ilk Iki PCA Bileseni ile Sinif Dagilimi")
plt.colorbar(scatter, label="Sinif")
plt.grid(True)
plt.show()

#Seçilen PCA bileşenleri kullanılarak veri daha düşük boyutlu bir uzaya dönüştürülmüştür.
#İlk iki PCA bileşeni ile çizilen scatter plot incelendiğinde sınıfların belirli ölçüde ayrıştığı gözlemlenebilir.

lda = LinearDiscriminantAnalysis(n_components=1)

X_train_lda = lda.fit_transform(X_train, y_train)
X_val_lda = lda.transform(X_val)
X_test_lda = lda.transform(X_test)

print("X_train_lda boyutu:", X_train_lda.shape)
print("X_val_lda boyutu:", X_val_lda.shape)
print("X_test_lda boyutu:", X_test_lda.shape)

plt.figure(figsize=(10, 6))

for label in sorted(y_train.unique()):
    plt.hist(
        X_train_lda[y_train == label],
        bins=30,
        alpha=0.6,
        label=f"Sinif {label}"
    )

plt.xlabel("LDA 1")
plt.ylabel("Frekans")
plt.title("LDA Bileseni Uzerinde Sinif Dagilimi")
plt.legend()
plt.grid(True)
plt.show()

#LDA yöntemi uygulanarak sınıf ayrımını en iyi yansıtan doğrusal bileşen elde edilmiştir. Ancak kullanılan veri seti
#iki sınıflı olduğu için LDA’da maksimum bileşen sayısı 1 ile sınırlıdır. Bu nedenle ödevde belirtilen 3 bileşen koşulu
#bu veri setinde matematiksel olarak uygulanamamaktadır.

#Ayrıca, Elde edilen LDA bileşeni üzerinde sınıfların belirli ölçüde ayrıştığı gözlemlenmiştir. Bu durum, LDA’nın sınıf
#bilgisi kullanarak ayırıcı bir temsil oluşturabildiğini göstermektedir.

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "GaussianNB": GaussianNB()
}

data_representations = {
    "Ham Veri": (X_train, X_val, X_test),
    "PCA Veri": (X_train_pca, X_val_pca, X_test_pca),
    "LDA Veri": (X_train_lda, X_val_lda, X_test_lda)
}

trained_models = {}

for data_name, (X_tr, X_v, X_te) in data_representations.items():
    for model_name, model in models.items():
        print(f"Egitiliyor: {model_name} - {data_name}")

        model.fit(X_tr, y_train)

        trained_models[(model_name, data_name)] = model

#Beş farklı sınıflandırma algoritması, ham veri, PCA verisi ve LDA verisi üzerinde ayrı ayrı eğitilmiştir.
#Böylece veri temsiline göre model performanslarının karşılaştırılması amaçlanmıştır.

validation_results = []

for data_name, (X_tr, X_v, X_te) in data_representations.items():
    for model_name, model in models.items():
        model.fit(X_tr, y_train)
        y_val_pred = model.predict(X_v)
        y_val_prob = model.predict_proba(X_v)[:, 1]

        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred)
        rec = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_prob)

        validation_results.append({
            "Model": model_name,
            "Veri Temsili": data_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "ROC-AUC": roc_auc
        })


results_df = pd.DataFrame(validation_results)
results_df = results_df.sort_values(by=["F1-score", "ROC-AUC"], ascending=False)

print(results_df)

best_row = results_df.iloc[0]
best_model_name = best_row["Model"]
best_data_name = best_row["Veri Temsili"]

print("\nEn iyi model:")
print(best_row)

#Tüm modeller validation veri seti üzerinde accuracy, precision, recall, F1-score ve ROC-AUC metrikleri ile
#değerlendirilmiştir. Sonuçlar karşılaştırıldığında en iyi modelin Logistic Regression olduğu gözlemlenmiştir.

best_model = models[best_model_name]
X_train_best, X_val_best, X_test_best = data_representations[best_data_name]
best_model.fit(X_train_best, y_train)

y_test_pred = best_model.predict(X_test_best)
y_test_prob = best_model.predict_proba(X_test_best)[:, 1]

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred)
test_rec = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

print("\nTest Sonuclari:")
print("Accuracy:", test_acc)
print("Precision:", test_prec)
print("Recall:", test_rec)
print("F1-score:", test_f1)
print("ROC-AUC:", test_roc_auc)

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen Sinif")
plt.ylabel("Gercek Sinif")
plt.show()

#Confusion matrix incelendiğinde modelin doğru ve yanlış sınıflandırmaları açık biçimde görülmektedir.
#Köşegen üzerindeki yüksek değerler modelin başarılı tahminler yaptığını göstermektedir.

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_value:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

best_model = models[best_model_name]
X_train_best, X_val_best, X_test_best = data_representations[best_data_name]
best_model.fit(X_train_best, y_train)

explainer = shap.Explainer(best_model, X_train_best)
shap_values = explainer(X_test_best)
shap.summary_plot(shap_values, X_test_best)
shap.plots.bar(shap_values)

if best_data_name == "PCA Veri":
    feature_names = [f"PCA_{i+1}" for i in range(X_test_best.shape[1])]
elif best_data_name == "LDA Veri":
    feature_names = [f"LDA_{i+1}" for i in range(X_test_best.shape[1])]
else:
    feature_names = X.columns.tolist()

X_test_best_df = pd.DataFrame(X_test_best, columns=feature_names)
X_train_best_df = pd.DataFrame(X_train_best, columns=feature_names)

explainer = shap.Explainer(best_model, X_train_best_df)
shap_values = explainer(X_test_best_df)

shap.summary_plot(shap_values, X_test_best_df)
shap.plots.bar(shap_values)

#En iyi validation modeli için SHAP analizi uygulanmıştır. Summary plot ve bar plot incelendiğinde model kararlarında
#en etkili olan özellikler belirlenmiştir. Bu sayede modelin yalnızca ne tahmin ettiği değil, tahmini hangi
#değişkenlere dayanarak yaptığı da yorumlanabilmiştir.


pca_model = models[best_model_name]
pca_model.fit(X_train_pca, y_train)

pca_feature_names = [f"PCA_{i+1}" for i in range(X_test_pca.shape[1])]

X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_feature_names)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_feature_names)

pca_explainer = shap.Explainer(pca_model, X_train_pca_df)
pca_shap_values = pca_explainer(X_test_pca_df)

shap.summary_plot(pca_shap_values, X_test_pca_df)
shap.plots.bar(pca_shap_values)

lda_model = models[best_model_name]
lda_model.fit(X_train_lda, y_train)

lda_feature_names = [f"LDA_{i+1}" for i in range(X_test_lda.shape[1])]

X_train_lda_df = pd.DataFrame(X_train_lda, columns=lda_feature_names)
X_test_lda_df = pd.DataFrame(X_test_lda, columns=lda_feature_names)

lda_explainer = shap.Explainer(lda_model, X_train_lda_df)
lda_shap_values = lda_explainer(X_test_lda_df)

shap.summary_plot(lda_shap_values, X_test_lda_df)
shap.plots.bar(lda_shap_values)
