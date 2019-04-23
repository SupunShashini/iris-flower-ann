# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'         #To hide warnings

# Importing the dataset
dataset = pd.read_csv('Iris.csv')

#Exploring dataset
print(dataset.head())
print('\n')
print(dataset.info())
print('\n')
print('Value Count:')
print(dataset['Species'].value_counts())
print('\n')

#Data Visualization
fig1 = plt.figure(figsize=(15,10))
sns.countplot(dataset['Species'])
plt.savefig('countplot.png', bbox_inches='tight')

fig2 = plt.figure(figsize=(20,10))
sns.pairplot(dataset.drop('Id',axis=1))
plt.savefig('pairplot.png', bbox_inches='tight')

fig3, ax = plt.subplots(1,2,figsize=(20,10))
sns.scatterplot(x="SepalLengthCm", y="SepalWidthCm",hue="Species",data=dataset,ax=ax[0])
sns.scatterplot(x="PetalLengthCm", y="PetalWidthCm",hue="Species",data=dataset,ax=ax[1])
plt.savefig('scatterplots.png', bbox_inches='tight')

fig4 = plt.figure(figsize=(15,10))
corrmat = dataset.drop('Id',axis=1).corr()
sns.heatmap(corrmat,cmap="PuBuGn")
plt.savefig('corrmatrix_heatmap.png', bbox_inches='tight')

#shuffling the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
from keras.utils import to_categorical
y = to_categorical(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building ANN
import keras
from keras.models import Sequential          #to initialize ANN
from keras.layers import Dense               #to build hidden layers

#Initialising ANN
classifier = Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=4))

#Adding second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#Adding output layer
classifier.add(Dense(units=3,kernel_initializer='uniform',activation='sigmoid'))

#compiling ANN i.e. applying Gradient decsent, here Stochastic GD is adam
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fitting ANN to training set
classifier.fit(X_train,y_train,batch_size=5,epochs=200)

y_pred = classifier.predict(X_test)
y_pred_1d = y_pred.argmax(axis=1)

#confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print('Confusion Matrix:')
print(cm)
print('\n')
cr = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print('Classification Report:')
print(cr)
print('\n')

scores = classifier.evaluate(X_test,y_test)
print('Accuracy: {}'.format(scores[1]*100))