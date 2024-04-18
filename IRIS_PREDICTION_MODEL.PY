import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data =pd.read_csv(r"DATASETS\Iris.csv")

#assuming the lastv column contains the target variable and determining them as dependent and independent variables
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
print(x)
print(y)

kf=KFold(n_splits=5,shuffle=True,random_state=42)

fold_number=1
accuracies=[]

for train_index ,test_index in kf.split(x):
    
    print(f"~~~~~~~~~~~~~~~~~~~~~~~FOLD {fold_number}~~~~~~~~~~~~~~~~~~~~~~~")
    x_train,x_test=x[train_index],x[test_index]
    y_train,y_test=y[train_index],y[test_index]
    #initializing and training a logistic regression model
    model=LogisticRegression(max_iter=1000)
    model.fit(x_train,y_train)
    
    #predict on the test set
    y_pred= model.predict(x_test)
    
    #accuracy score of the model
    accuracy=accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"ACCURACY : {accuracy}")
    
    fold_number+=1
    
avg_accuracy=np.mean(accuracies)
print(f"average accuracy of the model on all folds = {avg_accuracy}")
