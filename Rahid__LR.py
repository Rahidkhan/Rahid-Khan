import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import style
print(plt.style.available)
style.use('fivethirtyeight')#or 'seaborn-whitegrid'
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
test=pd.read_csv(r"C:\Users\onlyp\Documents\SUBLIME_TEXT_SAVES\test.csv")
test.describe()#bofore removing Null values
print(test.shape)
test = test.dropna()#analyze and drop rows/columns with Null values in different ways
print(test.shape)
test.describe()#after removing Null values
clf = LinearRegression()
X = test[['x']]
y = test['y']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=50)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
clf.fit(X_train, y_train)#learning the data
predictions = clf.predict(X_test)
print(predictions)
plt.scatter(y_test, predictions)#plot points individually and not like lineplot() i,e connected
sns.distplot((y_test-predictions),rug=True)# rug plot ON
print(metrics.mean_absolute_error(y_test,predictions))
print(metrics.mean_squared_error(y_test,predictions))
print(np.sqrt(metrics.mean_squared_error(y_test,predictions))