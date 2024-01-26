import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

#https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
#https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
#https://datagy.io/python-zip-lists/

X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)
print(X)
#enc = OneHotEncoder(handle_unknown='ignore')
#X = enc.fit_transform(X)
try: 
    print(enc.categories_)
except: 
    pass
try: 
    print(X.categories_)
except: 
    pass
print(X)

#original code
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)

#correct class imbalance in training data 
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
x_rus, y_rus = rus.fit_resample(X_train, y_train)

ros = RandomOverSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(X_train, y_train)

smote = SMOTE()

x_smote, y_smote = smote.fit_resample(X_train, y_train)

x_train_datasets = [X_train, x_rus, x_ros, x_smote]
y_train_datasets = [y_train, y_rus, y_ros, y_smote]

balanced_datasets = list(zip(x_train_datasets, y_train_datasets))

def compare_classifiers(X_train, y_train, X_test, y_test): 
    #original code
    clf = RandomForestClassifier(random_state=42)
    clf = clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    print(y_hat)
    print("RF Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))
    
    from autosklearn.classification import AutoSklearnClassifier
    
    automl = AutoSklearnClassifier(time_left_for_this_task=300)
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print(y_hat)
    print("AutoML Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))
    #end original code

x_pos = 0
y_pos = 1

for i in range(0, len(balanced_datasets)): 
    compare_classifiers(balanced_datasets[i][x_pos], balanced_datasets[i][y_pos], X_test, y_test)

#vestigial code
#print(automl.leaderboard())
#print(automl.show_models())
