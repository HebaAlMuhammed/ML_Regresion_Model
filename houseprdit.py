# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from warnings import filterwarnings
from sklearn.neighbors import KNeighborsRegressor
import plotly.express as px
filterwarnings('ignore')

"""""""""""""Veri Ön İşlem """""""""""""""
df = pd.read_csv("/MlRegresionFinal/house.csv")
data= df.copy()
print(data.head())
print(df.head())
print(df.isnull().sum())
print(df.shape)
print(df.info)
df=df.drop(['date','street','statezip'],axis=1) #drop ile dataframe hazırlıyoruz

# kategroi değişkenler numarik değişkene çevirme

df['city']=pd.factorize(df['city'])[0]
df['country']=pd.factorize(df['country'])[0]
print(" new shape of datasets",df.shape)
print(df.head())

b=(df.columns)

"""""""""""""""""""Veri Gösterleşme ve keşif etmek"""""""""""""
sns.set()
cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors','sqft_lot','waterfront','view']
sns.pairplot(df[cols], height = 3.5)
plt.show();
sns.heatmap(df.corr(),annot=True);


bed=[1,2,3,4,5,6]
#Beş airline üzerinden tekrarlayın
for i in bed:
    #  airline alt kümesi
    subset = data[data['bedrooms'] == i]

    # distplot grafiğini çizin
    sns.distplot(subset['price'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = i)
# Plot biçilendirme
plt.legend(prop={'size': 16}, title = 'bedroom')
plt.title('Density Plot with Multiple Airlines')
plt.xlabel('Delay (min)')
plt.ylabel('Density')


#bedroom sayısı göre price nasıl ekileniyor
bedrooms=data['bedrooms'].value_counts()
bedrooms
fig = px.pie(data, values='price', names='bedrooms')
fig.show()


""""""""""""""



#veri kümesinde standart ölçeklendirme
# değerlerin gelecekteki ölçeklendirmesi için standart ölçeklendirmeyi içe aktarma
a=StandardScaler()
df=a.fit_transform(df)
df=pd.DataFrame(df,columns=b)    # pandas ile  dataframe döüştürme
df


x=df.drop('price',axis=1)
y=df[['price']]



#Train Model
# from sklearn importing the train_test_spilt for the dividing the data for training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y)
x_train

print(x)
print(y)


""""************""""K En Yakın Komşu Model ve Tahmini**********"""


#Model
knn_model = KNeighborsRegressor().fit(x_train, y_train)
knn_model

knn_model.n_neighbors #komuşu sayını ulaşmak
y_pred = knn_model.predict(x_test)
np.sqrt(mean_squared_error(y_test , y_pred))  #ortalama kare hatası

#gridSearchCv
knn_parmas = {"n_neighbors" : np.arange(1,30, 1)}
knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn, knn_parmas, cv = 10).fit(x_train, y_train)
knn_cv_model.best_params_

#FinalModel
knn_tuned = KNeighborsRegressor(knn_cv_model.best_params_["n_neighbors"])
y_pred= knn_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))  #0.9201243663699619


"""""""""Destek Vektör Regresyonu Strart"""""""""""
#rbf

svr_model = SVR(kernel="rbf").fit(x_train, y_train)

svr_model
svr_model.predict(x_train)[0:5]
svr_model.predict(x_test)[0:5]
svr_model.intercept_  #sabit sayı


#test
y_pred = svr_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))  #0.7645020197509406

svr_model.score(x_test,y_test)

#linaer
svr_model = SVR(kernel="linear").fit(x_train, y_train)

svr_model
svr_model.predict(x_train)[0:5]
svr_model.predict(x_test)[0:5]
svr_model.intercept_  #sabit sayı
svr_model.coef_       #bütün kat sayılar

#test
y_pred = svr_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))   #0.7611931144304196

svr_model.score(x_test,y_test)

#model tuning
svr_params ={"C": [0.1,0.5,1,3]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv = 5).fit(x_train, y_train)
svr_cv_model.best_params_  #{'C': 1}

svr_cv_model = GridSearchCV(svr_model, svr_params, cv = 5 ,verbose=2, n_jobs=-1).fit(x_train, y_train)

svr_tuned = SVR(kernel="linear", C = 0.5).fit(x_train, y_train)
y_pred = svr_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))  #0.4458237380699369

#gerçek fiyatlar ile tahmin edilen fiyatlar arasındaki ilişkiyi görsel olarak anlamak için
plt.scatter(y_train, y_pred ,'o')
plt.xlabel("Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
""""""""""""""""""



"""""""""yapay Sinir Ağları start..."""""""""

sclar = StandardScaler()
sclar.fit(x_train)
X_train_scaled= sclar.transform(x_train)
X_test_scaled = sclar.transform(x_test)
X_train_scaled
#temel model
mlp_model = MLPRegressor().fit(X_train_scaled ,y_train)
mlp_model
 #tahmin
mlp_model.predict(X_train_scaled)[0:5]
y_pred_scaled_P = mlp_model.predict(X_test_scaled)

#hata oranı hesablama
np.sqrt(mean_squared_error(y_test, y_pred_scaled_P)) #0.49901004790075304

#model tunning
#model optemize
#iki tane gizili katman ve bu katman neron sayısı (10 ve 2)  ve (45 ,45) her alpha değeri değerlendirme yaparız
mlp_params = {"alpha" : [ 0.01, 0.2, 0.001, 0.0001],  "hidden_layer_sizes" : [(10,2) ,(45,45)]}
mlp_cv_model = GridSearchCV(mlp_model , mlp_params, cv=5,verbose= 2, n_jobs= 1).fit(x_train, y_train)

# model perfomansı artırmak için mlp_params içinde en iyi parametreler bulmak
mlp_cv_model.best_params_
#final utn edilmiş model fit etmek
tuned_mlp_model =  MLPRegressor(alpha=0.02 ,hidden_layer_sizes=(10,2)).fit(X_train_scaled, y_train)

tuned_mlp_model.predict(X_test_scaled)
y_pred_scaled_tuned= tuned_mlp_model.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test, y_pred_scaled_tuned))  # 0.45332674161516934






