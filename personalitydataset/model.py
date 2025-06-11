import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from holoviews.plotting.bokeh.links import callbacks
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.layers.core import Dropout

#Tüm kolonları terminalde görmek istiyorum.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

veriseti = r"C:\Users\sevva\Desktop\Personality_dataset\personalitydataset\personality_dataset.csv"
df = pd.read_csv(veriseti)
print(df)

#Eksik (Nan) değerlere baktım.
print(df.isnull().sum())
#Sütunların hangi veri tipinde olduğuna bakalım.
print(df.info())

print(df.columns)

#Sayısal sütundaki eksik verileri ortalama ile dolduralım.
df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].mean())
df['Social_event_attendance'] = df['Social_event_attendance'].fillna(df['Social_event_attendance'].mean())
df['Going_outside'] = df['Going_outside'].fillna(df['Going_outside'].mean())
df['Friends_circle_size'] = df['Friends_circle_size'].fillna(df['Friends_circle_size'].mean())
df['Post_frequency'] = df['Post_frequency'].fillna(df['Post_frequency'].mean())

#Kategorik sütunlardaki eksik verileri en çok tekrar edenle dolduralım.
df['Stage_fear'] = df['Stage_fear'].fillna(df['Stage_fear'].mode()[0])
df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna(df['Drained_after_socializing'].mode()[0])

#Eksik (Nan) değerlere baktım.
print(df.isnull().sum())

#One-hot encoding ile kategorik sütunları encode edelim.
df = pd.get_dummies(df, columns=['Stage_fear', 'Drained_after_socializing'], drop_first=True)

#kategorik sütunu label encoder ile encode edelim
le = LabelEncoder()
df['Personality_encoded'] = le.fit_transform(df['Personality'])

#Sütunların hangi veri tipinde olduğuna bakalım.
print(df.info())
sbn.pairplot(df)
plt.show()

#Bağımsız değişkenler
x = df.drop(['Personality'], axis=1)
#Bağımlı değişken
y = df['Personality_encoded']

#eğitim-test veri seti
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#standardizasyon
scr = StandardScaler()
x_train = scr.fit_transform(x_train)
x_test = scr.transform(x_test)

#logistic regression
log = LogisticRegression(random_state=0)
log.fit(x_train,y_train)
y_pred = log.predict(x_test)
print(y_pred)
print(y_test)

#karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print(cm)

#knn-en yakın komşu
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(x_train,y_train)
y_pred2 = knn.predict(x_test)
cm2 = confusion_matrix(y_test,y_pred2)
print("knn")
print(cm2)

#destek vektör regresyonu-support vector regression
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(x_train,y_train)
y_pred3 = svc.predict(x_test)
cm3 = confusion_matrix(y_test,y_pred3)
print('SVC')
print(cm3)

#naive-bayes
from sklearn.naive_bayes import GaussianNB
gau = GaussianNB()
gau.fit(x_train,y_train)
y_pred4 = gau.predict(x_test)
cm4 = confusion_matrix(y_test,y_pred4)
print("GAU")
print(cm4)

#karar ağacı
from sklearn.tree import DecisionTreeClassifier
dec = DecisionTreeClassifier(criterion='gini')
dec.fit(x_train, y_train)
y_pred5 = dec.predict(x_test)
cm5 = confusion_matrix(y_test,y_pred5)
print("DECTREE")
print(cm5)

#random forest
from sklearn.ensemble import RandomForestClassifier
rand= RandomForestClassifier(n_estimators=10,criterion='gini')
rand.fit(x_train,y_train)
y_pred6 = rand.predict(x_test)
cm6 = confusion_matrix(y_test, y_pred6)
print("RFC")
print(cm6)

#k-means uygulayalım
from sklearn.cluster import KMeans, affinity_propagation

kmean = KMeans(n_clusters=3, init='k-means++')
kmean.fit(x)
print(kmean.cluster_centers_)
sonuclar = []
for i in range(1,10):
    kmean = KMeans(n_clusters=i, init='k-means++',random_state=123)
    kmean.fit(x)
    sonuclar.append(kmean.inertia_)

plt.plot(range(1,10),sonuclar)
#plt.show()

etiketler = kmean.labels_
print(etiketler)

#hiyerarşik bölütleme- hierarchical clustering deneyelim.
from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=3,metric='euclidean', linkage='ward')
y_pred7 = agc.fit_predict(x)
print(y_pred7)

plt.scatter(x.iloc[y_pred7 == 0, 0], x.iloc[y_pred7 == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x.iloc[y_pred7 == 1, 0], x.iloc[y_pred7 == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x.iloc[y_pred7 == 2, 0], x.iloc[y_pred7 == 2, 1], s=100, c='green', label='Cluster 3')
plt.title("Kümeler")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.legend()
#plt.show()

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x,method='ward'))
#plt.show()

#yapay sinir ağı oluşturup bakalım deneyelim.
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from sklearn.metrics import classification_report

model = Sequential()
model.add(Input(shape=(x_train.shape[1],)))  # Dinamik ve doğru giriş
model.add(Dropout(0.4))
model.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
model.add(Dropout(0.4))
model.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
model.add(Dropout(0.4))
model.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

earlystopping = EarlyStopping(monitor="val_loss",mode='min',verbose=1,patience=25)

model.fit(x_train,y_train, epochs=30, batch_size=32, validation_split=0.1, callbacks=[earlystopping])
y_pred8 = model.predict(x_test)
y_pred8 = (y_pred8>0.5)

cm8 = confusion_matrix(y_test,y_pred8)
print(cm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred8)
print(f"Doğruluk Oranı: {accuracy * 100:.2f}%")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred8))
#Doğruluk Oranı: 99.90%
#             precision    recall  f1-score   support

#          0       1.00      1.00      1.00       461
#          1       1.00      1.00      1.00       496

#   accuracy                           1.00       957
#  macro avg       1.00      1.00      1.00       957
#weighted avg      1.00      1.00      1.00       957

#y_pred8 int’e çevir
y_pred_int = y_pred8.astype(int).ravel()

cm_nn = confusion_matrix(y_test, y_pred_int)
labels = ['Introvert', 'Extrovert']          # sıralama = LabelEncoder sırası
cm_df = pd.DataFrame(cm_nn, index=labels, columns=labels)

plt.figure(figsize=(5,4))
sbn.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Yapay Sinir Ağı – Confusion Matrix')
plt.ylabel('Gerçek etiket')
plt.xlabel('Tahmin edilen etiket')
plt.tight_layout()
plt.show()

#K-fold ile deneyelim.
from sklearn.model_selection import KFold

def create_model():
    model1 = Sequential()
    model1.add(Dense(12, input_dim=8, activation='relu'))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model1

kf = KFold(n_splits=5, shuffle=True, random_state=1)
accuracies = []

for train_index, test_index in kf.split(x):
    #eğitim test verilerini ayıralım.
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #Yeni model oluşturalım. model1 olsun
    model1 = create_model()
    #Modeli eğitelim.
    model1.fit(x_train, y_train, epochs=10, batch_size=10, verbose=0)
    #doğruluk al model değerlendirelim
    loss, accuracy = model1.evaluate(x_test, y_test, verbose=0)
    scores = []
    # her eğitim sonrasında döngüde
    scores.append(accuracy)
    print(f"Fold doğruluğu: {accuracy*100:.4f}%")

fold_scores = [0.968966, 0.958621]
mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print(f"Ortalama doğruluk: {mean_score:.4f}")
#Fold doğruluğu: 97.7586%
print(f"Standart sapma: {std_score:.4f}")
#Ortalama doğruluk: 0.9638


#modeli kaydedelim.
import pickle

# Kaydetme
with open('model_dosya.pkl', 'wb') as file:
    pickle.dump(model, file)

# Yükleme
with open('model_dosya.pkl', 'rb') as file:
    model = pickle.load(file)
