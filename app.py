import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st



def main():

    dataset_path = "https://frenzy86.s3.eu-west-2.amazonaws.com/fav/iris.data"

    df = pd.read_csv(dataset_path, header=None)
    df.columns = ['sepal length', 'sepal width',
                'petal length', 'petal width', 'class']


    X = df.drop(["class"], axis=1) #asse y = 1 ; asse x = 0
    y = df["class"]


    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=667) 

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred))


    result_df = pd.DataFrame({"y_previste": y_pred ,"y_reali": y_test})
    result_df.reset_index()
    
    length = y_pred.shape[0]
    X = np.linspace(0,length,length)

    fig = plt.figure()
    plt.figure(figsize=(10, 7))
    plt.plot(X_test, y_test, label="valori reali")
    plt.plot(X_test, y_pred, label="valori predetti")
    plt.legend(loc=2);
    st.pyplot(fig)
    


if __name__ == '__main__':
    main()