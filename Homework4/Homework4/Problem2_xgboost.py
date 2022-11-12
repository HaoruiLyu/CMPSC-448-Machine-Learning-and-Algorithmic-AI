import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
from xgboost import XGBClassifier
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


### xgboost

parms_grid={'max_depth':[4, 6, 8],'learning_rate':[0.01, 0.05, 0.1],
           'missing':[None, 0],'n_estimators':[100, 300,500],
           'reg_lambda':[0.0, 0.05,1.0],'objective':['binary:logistic', 'binary:logitraw', 'binary:hinge']}


xgb=XGBClassifier()
model = GridSearchCV(xgb,parms_grid, cv=5,verbose=3)
model.fit(x_train, y_train)

# print best parameters
print(model.best_params_)


# display training results
df=pd.concat([pd.DataFrame(model.cv_results_["params"]),pd.DataFrame(model.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
print(df)


# model evaluation in testing data
xgb=model.best_estimator_
y_pred = xgb.predict(x_test)
acc = accuracy_score(y_test, y_pred )
print("Accuracy test_error:"  ,1-acc)


# model evaluation in training data
cross_results = cross_val_score(xgb,x_train,y_train,cv=5)
mean_acc=np.mean(cross_results)

pred =xgb.predict(x_train)
acc = accuracy_score(y_train,pred)

print("Cross_val_error: " ,1-mean_acc)
print("Training error: " ,1-acc)
