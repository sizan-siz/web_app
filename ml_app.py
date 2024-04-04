# Import library
import streamlit as st 
import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.header("MACHINE LEARNING APP DEVLOPED BY SIZAN")
st.write("In this app we compare diffrent model")
## Dataset Slider
dataset_name=st.sidebar.selectbox("Select Data_set",('Iris','Breast Cancer','Wine'))
## Model name Slider
classifier_name=st.sidebar.selectbox("Select Model",('KNN','SVM',"Random Forest"))
## Function for Dataset
def dataset(dataset_name):
    data=None
    if dataset_name == 'Iris':
        data=datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    x=data.data
    y=data.target 
    return x,y
df=dataset(dataset_name)
X,y=dataset(dataset_name)

st.write("Shape of Dataset ",X.shape)
st.write("Number of class ",len(np.unique(y)))
def add_parameter_ui(classifier_name):
    params=dict()
    if classifier_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params['C']=C
    elif classifier_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        params["max_depth"]=max_depth
        n_estimeters=st.sidebar.slider("n_estimeters",1,100)
        params["n_estimeters"]=n_estimeters
    return params
params=add_parameter_ui(classifier_name)
def get_classifiar(classifier_name,params):
    clf=None
    if classifier_name=="SVM":
         clf=SVC(C=params['C'])
    elif classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf=clf=RandomForestClassifier(n_estimators=params["n_estimeters"],max_depth=params['max_depth'],random_state=43)
       
    return clf
clf=get_classifiar(classifier_name,params)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
acc=accuracy_score(y_test,y_pred)
st.write("Classifier: ",classifier_name)
st.write("Accuracy: ",acc)

pca=PCA(2)
x_projected=pca.fit_transform(X)
x1=x_projected[:,0]
x2=x_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
# plot show
st.pyplot(fig)
