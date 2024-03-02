import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data=pd.read_csv("student-mat.csv",sep=";")
# print(data.head())
data=data[["G1","G2","G3","studytime","failures","absences"]]
# print(data.head()) #first 5 datas
predict="G3"
x= np.array(data.drop(["G3"],1))
y= np.array(data[predict])
# print(x)
# print(y)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.5)
#x and y_train will have datas x and y_test test has 10percent datas as sample datas
linear= linear_model.LinearRegression()
linear.fit(x_train,y_train)
acc=linear.score(x_test,y_test)
print(acc)
# print("coeff ",linear.coef_)
# print("intercept ", linear.intercept_)
predictions= linear.predict(x_test)#predicts final grade(y) for test examples
for i in range(len(predictions)):
    print(predictions[i],x_test[i],y_test[i])# prints predicted final grade for input of features(x) and actual final grade(y)



