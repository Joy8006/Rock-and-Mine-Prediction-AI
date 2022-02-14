import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('Sonar Data.csv',header=None)


# Separating data and labels:
X = sonar_data.drop(columns= 60,axis= 1)
Y = sonar_data[60]


# Separating train and test data from the given dataset!!
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)


# calling the required model
model = LogisticRegression()   


# traing the model with the training data
model.fit(x_train, y_train)  


#  finding the accuracy of the training data
x_train_prediction = model.predict(x_train)
accuracy = accuracy_score(x_train_prediction, y_train)
print("The accuracy of the training data is {} %".format(accuracy*100))


# finding the accuracy of the test data
x_test_prediction = model.predict(x_test)
aaccuracy = accuracy_score(x_test_prediction, y_test)
print("The accuracy of the test data is {} %".format(aaccuracy*100))


# prediction program

input_data = (0.0264,0.0071,0.0342,0.0793,0.1043,0.0783,0.1417,0.1176,0.0453,0.0945,0.1132,0.0840,0.0717,0.1968,0.2633,0.4191,0.5050,0.6711,0.7922,0.8381,0.8759,0.9422,1.0000,0.9931,0.9575,0.8647,0.7215,0.5801,0.4964,0.4886,0.4079,0.2443,0.1768,0.2472,0.3518,0.3762,0.2909,0.2311,0.3168,0.3554,0.3741,0.4443,0.3261,0.1963,0.0864,0.1688,0.1991,0.1217,0.0628,0.0323,0.0253,0.0214,0.0262,0.0177,0.0037,0.0068,0.0121,0.0077,0.0078,0.0066)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped) 


if prediction[0] == 'R':
    print("The object is a Rock")
elif prediction[0] == 'M':
    print("The object is a Mine")     

