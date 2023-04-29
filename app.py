import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st


dataset_path = "https://frenzy86.s3.eu-west-2.amazonaws.com/fav/iris.data"

df = pd.read_csv(dataset_path, header=None)
df.columns = ['sepal length', 'sepal width',
              'petal length', 'petal width', 'class']


X = df.drop(["class"], axis=1)
y = df["class"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=667)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy_score(y_test, y_pred)

fig = plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, '-r')
plt.axis([0, 10, 0, 10])
st.pyplot(fig)
