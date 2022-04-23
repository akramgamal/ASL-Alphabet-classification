import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import os
import cv2
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
train_path=os.listdir('/Users/Zamzam/Desktop/ML 2/ASL_Alphabet_Dataset/data')
print(len(train_path))
x=[]
y=[]
for i in range(len(train_path)):
    pth='/Users/Zamzam/Desktop/ML 2/ASL_Alphabet_Dataset/data/'+train_path[i]
    for j in os.listdir(pth):
        img=cv2.imread(pth+'/'+j)
        img =cv2.resize(img,(100,100))
        x.append(img)
        y.append(i)

x_rgb=[]
for img in x:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_blur = cv2.GaussianBlur(img_rgb, (3, 3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=4, threshold2=100)
    x_rgb.append(edges)
x_gray=[]
for img in x:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=4, threshold2=100)
    x_gray.append(edges)
x_binary=[]
for img in x:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,binary = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY_INV)
    x_binary.append(binary)
x=x_rgb #rgb or gray or binary
print(len(x))
x=np.array(x)
y=np.array(y)
x_updated=x.reshape(len(x),-1)
print(x_updated.shape)
xtrain,xtest,ytrain,ytest=train_test_split(x_updated,y,test_size=.20)
print(xtrain.shape)
xtrain=xtrain/255
xtest=xtest/255
#################################################

lg=LogisticRegression()
lg.fit(xtrain,ytrain)
print("train ",lg.score(xtrain,ytrain)) 
print("test ",lg.score(xtest,ytest)) 
pred=lg.predict(xtest)
print("precision score : ",precision_score(ytest, pred,average="micro"))
print("recall score : ",recall_score(ytest, pred,average="micro"))


###############################################

sv=SVC()
sv.fit(xtrain,ytrain)
print("train ",sv.score(xtrain,ytrain)) 
print("test ",sv.score(xtest,ytest)) 

sv_pred=sv.predict(xtest)
print("precision score : ",precision_score(ytest, sv_pred,average="micro"))
print("recall score : ",recall_score(ytest, sv_pred,average="micro"))

#############################################
dt=DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
print("train ",dt.score(xtrain,ytrain)) 
print("test ",dt.score(xtest,ytest))
dt_pred=dt.predict(xtest)
print("precision score : ",precision_score(ytest, dt_pred,average="micro"))
print("recall score : ",recall_score(ytest, dt_pred,average="micro"))
