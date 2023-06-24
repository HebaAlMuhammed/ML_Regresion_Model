# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 12:54:33 2023

@author: bisos
"""
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
df = pd.read_csv("../MLRegresionFinal/car_data.csv")

"""""""VERİ ÖN İŞLEM"""""""
sns.set()
cols = ['Selling_Price', 'Year', 'Present_Price', 'Driven_kms', 'Fuel_Type','Owner','Transmission','Selling_type']
sns.pairplot(df[cols], height = 3.5)
plt.show();
df =df.iloc[:,1:len(df)]
print(df.head())
print(df.shape)
print(df.info())

#ekisik veri varsa kontrol eder
df.isnull().sum()

# kategorik verilerin dağılımının kontrol edilmesi
df.Fuel_Type.value_counts()
df.Selling_type.value_counts()
df.Transmission.value_counts()

#Kategorik Verilerin Kodlanması
df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
df.replace({'Selling_type':{'Dealer':0,'Individual':1}},inplace=True)
df.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

# data frame kategoriik değişkenler düneledikten sonra kontrol etmek
df.head()

""""""




""""veriler göresleştirme ve analiz"""""

#iki hedef değişkenlerimiz arasında ilişki
sns.jointplot(x="Selling_Price", y="Present_Price", data=df, kind="reg")

#bağımsız değişken veri site
X = df.drop(['Car_Name','Selling_Price'],axis=1) #bağımsız değişkenler

print(X)
Y = df['Selling_Price']  #bağımLİ değişken
print(Y)

#Eğitime ve Test verilerini ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

""""""



""""""" Random Forest Regressor START..."""""""
# Random Forest Regressor modelini oluşturun ve eğitin
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # İstediğiniz parametreleri ayarlayabilirsiniz
rf_model.fit(X_train, Y_train)

# Test veri seti üzerinde tahmin yapın
y_pred = rf_model.predict(X_test)

# Tahmin performansını değerlendirin
mse = mean_squared_error(Y_test, y_pred)
mse #  0.2565657364516137
rmse = mse ** 0.5
print("RMSE:", rmse) #RMSE: 0.5065231845153918
""""""




""" DecisionTreeRegressor START...."""
# Decision Tree Regressor modelini oluşturun ve eğitin
dt_model = DecisionTreeRegressor(random_state=42)  # İstediğiniz parametreleri ayarlayabilirsiniz
dt_model.fit(X_train, Y_train)

# Test veri seti üzerinde tahmin yapın
y_pred = dt_model.predict(X_test)

# Tahmin performansını değerlendirin
mse = mean_squared_error(Y_test, y_pred)
mse # 0.623916129032258
rmse = mse ** 0.5
print("RMSE:", rmse) # 0.7898836173970555

""""""


"""""""Lasso Regression model START..."""""""
lass_reg_model = Lasso()

lass_reg_model.fit(X_train,Y_train)


#Model Evaluation

# prediction on Training data
training_data_prediction = lass_reg_model.predict(X_train)
# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)  # R squared Error :  0.842448071824074

#Gerçek fiyatları ve Öngörülen fiyatları görselleştirin
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# prediction on Training data
y_pred = lass_reg_model.predict(X_test)


# R squared Error
error_score = metrics.r2_score(Y_test, y_pred)
print("R squared Error : ", error_score) #R squared Error :  0.8709763132343395

#Visualize the actual prices and Predicted prices

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

lass_reg_model.coef_   #değişken katsayıları
lass_reg_model.intercept_  #sabit sayı


cross_val_score(lass_reg_model , X_train, Y_train, cv = 10, scoring="neg_mean_squared_error")

Lass_Fiyat_prediction = lass_reg_model.predict([[2011, 7.55 ,500 ,1,1,1,1 ]])
print("tahmin edilen fiyat: ",Lass_Fiyat_prediction)  #tahmin edilen fiyat: [3.9867669]
