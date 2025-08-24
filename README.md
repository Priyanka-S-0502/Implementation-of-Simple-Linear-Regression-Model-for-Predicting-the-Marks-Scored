# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages.
2.Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values. 
 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Priyanka S 
RegisterNumber:212224040255


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#displaying predicted values
y_pred
#displaying the actual values
y_test
print("NAME:PRIYANKA S")
print("REGISTER NO:212224040255")
#graph plot for training data
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("NAME:PRIYANKA S")
print("REGISTER NO:212224040255")
#graph plot for testing data
plt.scatter(x_test,y_test,color="green")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("Hours Vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("NAME:PRIYANKA S")
print("REGISTER NO:212224040255")
#Calculate Mean absolute error (MAE) and Mean squared error (MSE)
mse=mean_squared_error(y_test,y_pred)
print('MSE= ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE= ',mae)

rmse=np.sqrt(mse)
print('RMSE= ',rmse)
```

## Output:
HEAD VALUES

<img width="196" height="270" alt="Screenshot 2025-08-24 181801" src="https://github.com/user-attachments/assets/79261aa9-978d-488a-88df-6a9171891822" />

TAIL VALUES

<img width="218" height="269" alt="Screenshot 2025-08-24 181811" src="https://github.com/user-attachments/assets/16626138-9f4d-41e3-98cf-2c538cd3830f" />

DISPLAYING X VALUE

<img width="330" height="653" alt="Screenshot 2025-08-24 181825" src="https://github.com/user-attachments/assets/ba56e12b-69df-4e8a-ae3a-f70635cb7a74" />

DISPLAYING Y VALUE

<img width="846" height="60" alt="Screenshot 2025-08-24 181838" src="https://github.com/user-attachments/assets/49f29631-fec7-44e0-b97d-7f7de98c812f" />

DISPLAYING PREDICTED VALUE

<img width="783" height="60" alt="Screenshot 2025-08-24 181852" src="https://github.com/user-attachments/assets/043d106f-5d04-4fd9-9705-a1fa1d7bc7da" />

DISPLAYING ACTUAL VALUE

<img width="629" height="44" alt="Screenshot 2025-08-24 181859" src="https://github.com/user-attachments/assets/d35f2757-4ce5-44cb-be77-4cfb47dd09a1" />

TRAINING SET

<img width="864" height="721" alt="Screenshot 2025-08-24 181915" src="https://github.com/user-attachments/assets/3075817f-394c-4cfe-92d3-5478f7a78b5a" />

TEST SET

<img width="897" height="710" alt="Screenshot 2025-08-24 182007" src="https://github.com/user-attachments/assets/223575bb-85cf-4160-b0eb-7f8020c70154" />

ERRORS

<img width="319" height="140" alt="Screenshot 2025-08-24 182018" src="https://github.com/user-attachments/assets/539e521c-f19c-490a-8015-a241f5bc17df" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
