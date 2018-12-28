# Importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data import load_dataset
import scipy
from scipy import ndimage
#Note to self
'''Since we're only doing binary classification, we can use the sigmoid activation function'''

#loading data
X_train, y_train, X_test,y_test,classes=load_dataset()

#PlEASE IGNORE
#Visualizing the datasets
'''
print(train_set_x_orig)
print(train_set_y)
print(classes)
'''
#Printing the image
'''
index = 20
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture.")
'''
'''
#Info about the dataset
print(X_train.shape) #Number of cats in training
print(X_test.shape[0]) #Number of cats in testing
print(X_train.shape[1:3]) #Dimensions
'''
y_train=np.transpose(y_train)
y_test=np.transpose(y_test)
num_px=X_train.shape[1]
#---Done--- Till here that is. Let's go! :D

#Coooool, now vectorization!
#Converting (No., d,d, 3) ((from No. to 3)) to (d*d*3,1) essentially a row matrix(?)
X_train_flat=X_train.reshape((X_train.shape[0], -1))
X_test_flat=X_test.reshape(X_test.shape[0], -1)

#Woah, done with vectorization..NOICE!


#Standardizing the data

X_train=X_train_flat/255
X_test=X_test_flat/255
print(X_train.shape)
print(y_train.shape)

#OKAY, so our data is ready

#Logistic regression let's go!
#We're using scikit-learn here, for a model from scratch, please do visit my github profile. Thank you!

LogRes=LogisticRegression()
LogRes.fit(X_train, y_train)
trained_log_res=LogRes


#Cool, that's Done
#Now, model accuracy
def model_accuracy(trained_model, features, targets):
    accuracy_score=trained_model.score(features,targets)
    return accuracy_score

#Training accuracy
train_accuracy=model_accuracy(trained_log_res, X_train, y_train)
print('Trained accuracy: ', train_accuracy*100, '%', sep='')




#LET'S GET TO TESTING-accuracy
test_accuracy = model_accuracy(trained_log_res, X_test, y_test)
print('Test accuracy: ', test_accuracy*100, '%', sep='')
