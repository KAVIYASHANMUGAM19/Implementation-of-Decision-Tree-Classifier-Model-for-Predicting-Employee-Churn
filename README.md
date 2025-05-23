# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset
2. Data Preprocessing
3. Feature and Target Selection
4. Split the Data into Training and Testing Sets
5. Build and Train the Decision Tree Model
6. Make Predictions
7. Evaluate the Model

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S KAVIYA
RegisterNumber: 212223040090
*/

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/9543bc53-6bdc-4f95-b052-ce07491a2bbb)

![image](https://github.com/user-attachments/assets/de96e3fd-f8ca-4883-ba65-d4aa39d8a83d)

![image](https://github.com/user-attachments/assets/84db7ed7-26c1-4cff-9354-1791671dff77)

![image](https://github.com/user-attachments/assets/71424930-b0d4-4fb8-a725-99c065d01802)

![image](https://github.com/user-attachments/assets/56c548b4-525e-49b8-b190-5f3825fb1497)

![image](https://github.com/user-attachments/assets/efe7ae36-6aa6-433a-922c-af1b6ee1eb56)

![image](https://github.com/user-attachments/assets/458b958d-8b49-434d-9a21-1eae029b2c28)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
