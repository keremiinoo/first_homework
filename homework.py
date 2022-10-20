import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # test ve train olarak ayıracağımız için
from sklearn.impute import SimpleImputer


veriler = pd.read_csv('/Users/keremaglik/Documents/Machine Learning Bootcamp/Week_1 Homework/water_potability.csv')

# sülfatı veri setinden ayırdım ve eksik verileri mean ile doldurdu
veriler_tamam = veriler.iloc[:, :-1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # eksik olan verileri al yerine o satrın ortalamasını yaz

imputer = imputer.fit(veriler_tamam[:, :-1])
veriler_tamam[:, :-1] = imputer.transform(veriler_tamam[:, :-1])

veriler_tamamdf = pd.DataFrame(data=veriler_tamam, index=range(3276), columns=['veriler_tamam'])
yasdf.to_csv('/Users/keremaglik/Documents/Machine Learning Bootcamp/Week_1 Homework/veriler_tamam.csv', index=False)


# verileri alma
water_data = veriler.iloc[:, 0:9].values
potability = veriler.iloc[:, -1:].values

# ham dataları scale ettik
sc = StandardScaler()
w_sc = sc.fit_transform(water_data)
p_sc = sc.fit_transform(potability)

x_train, x_test, y_train, y_test = train_test_split(water_data, potability, test_size=0.33, random_state=0)

# x verilerini scale ettik
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_train)

# lineer regresyon işlemi yapıldı
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# grafiğini çizdirdik
plt.scatter(water_data, potability, color='red')
plt.plot(x_test, y_pred, color='blue')
plt.show()

svr = SVR(kernel='rbf')
svr.fit(w_sc, p_sc)
y_pred_svr = svr.predict(w_sc)

plt.scatter(w_sc, p_sc, color='pink')
plt.plot(w_sc, y_pred_svr, color='blue')
plt.show()

# kernel değiştirdik poly olarak derecesi 3
svr2 = SVR(kernel='poly', degree=3)
svr2.fit(w_sc, p_sc)
y_pred_svr2 = svr2.predict(w_sc)

plt.scatter(w_sc, p_sc, color='red')
plt.plot(w_sc, y_pred_svr2, color='blue')
plt.show()

# decision tree
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(w_sc, p_sc)
y_pred_dt = dtr.predict(w_sc)

































