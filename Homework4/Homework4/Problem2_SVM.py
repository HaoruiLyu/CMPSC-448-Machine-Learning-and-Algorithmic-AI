import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
matplotlib.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")

def get_data(path):           #load data
    data = load_svmlight_file(path)
    return data[0], data[1]

# get training and testing data
train_path='a9a.txt'
x_train, y_train = get_data(train_path)

test_path='a9a.t'
x_test, y_test = get_data(test_path)


### SVM
parms_grid={'kernel':['rbf'],'gamma':[0.01, 0.05, 0.1,1],
           'C':[0.1,1,10,100]}


svm=SVC()
model_svm = GridSearchCV(svm,parms_grid, cv=5,verbose=3)
model_svm.fit(x_train, y_train)

# print best parameters
print(model_svm.best_params_)


# display training results
df=pd.concat([pd.DataFrame(model_svm.cv_results_["params"]),pd.DataFrame(model_svm.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
print(df)
# model evaluation in testing data
svm=model_svm.best_estimator_
y_pred = svm.predict(x_test)
acc = accuracy_score(y_test, y_pred )
print("Accuracy test_error:"  ,1-acc)
# model evaluation in training data
cross_results = cross_val_score(svm,x_train,y_train,cv=5)
mean_acc=np.mean(cross_results)

pred =svm.predict(x_train)
acc = accuracy_score(y_train,pred)

print("Cross_val_error: " ,1-mean_acc)
print("Training error: " ,1-acc)

