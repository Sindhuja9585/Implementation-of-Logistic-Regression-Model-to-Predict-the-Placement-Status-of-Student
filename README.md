# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SINDHUJA P
RegisterNumber: 212222220047 
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```


## Output:
# Placement data:

![277169578-0d65163e-89bd-4559-a827-b6c984a8b69c](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/d4ebf349-d293-4d86-8ca3-6c98dfe00da4)

# Salary data

![277169604-b50c9d81-07bf-4ea2-bd55-314ae5d7f113](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/6eba41f5-eeab-4f4b-8788-e2ca458526d3)


# Checking the null() function:

![277169635-27949900-92a0-468a-bb12-8dcb01a83ec0](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/cc44c301-cafd-4daf-9f71-e353fd7986b3)

# Data duplicate:

![277169662-9eef8da7-cdf1-4ce6-86f0-9b6c587c6146](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/231e7421-addd-4a94-8a42-02942f661c77)

# Print data:

![277169680-6ae4dffb-18b2-4b82-95dc-6c7f83b8c23d](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/993b924c-8305-4d7f-9e90-fdf6b5b909cb)

# Data-status:

![277169700-1810a3d9-e2d4-4dc5-8c46-0215a94b6f8d](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/1292bda8-0aff-4638-a112-d664bdda7e07)

# Y_prediction array:

![277169740-91beecf1-7253-42fb-a1af-22205d0c25a5](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/99faf58c-cd3f-4d1c-ba43-1e34bc78760a)


# Accuracy value:

![277169758-86e41762-900b-463d-814c-90a1d0e84355](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/15556b79-119b-4a49-bcec-374589ae38eb)

# Confusion array:


![277169800-5ec7c8f5-829c-4042-a9e1-a1c9b0595adb](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/f2c927c3-aa52-4ee5-99bc-fdf93f3f374d)


# Classification Report:


![277169830-2125bc90-ce93-4f7b-93b7-c6bf6786f51b](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/4ea4787e-36ca-4dcd-b2d9-c3646d72c44f)

# Prediction of LR:


![277169929-024f856e-41c9-44aa-874b-c778bf7f1c28](https://github.com/Sindhuja9585/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/122860624/55b4f3c4-5a11-4784-96f2-60f072bedbfe)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
