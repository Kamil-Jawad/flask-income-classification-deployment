# complete understaing of the dataset and features and other model building with validation
# is present in "practice on dataset" folder


#     import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

#     import dataset
df = pd.read_csv('D:\\DataSets\\archive_2\\income_evaluation.csv')

#     remove spaces from first of each column name
df.columns = df.columns.str.replace(' ', '')

#   prepare dataset for train_test_split
X = df[['age', 'fnlwgt', 'education-num', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week']]
X['sex'] = X['sex'].astype("category")
Y = df['income']
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=1, train_size=0.75)

#   labelling the categorical features
labelencoder = LabelEncoder()


X_train['sex'] = labelencoder.fit_transform(X_train['sex'])
X_test['sex'] = labelencoder.transform(X_test['sex'])

y_train = labelencoder.fit_transform(y_train)
y_test = labelencoder.transform(y_test)


#   normalizatio
scaler = MinMaxScaler()

Scaled_X_train = scaler.fit_transform(X_train)
Scaled_X_test = scaler.transform(X_test)

#   build model
svclassifier = SVC(kernel='rbf', C=100.0)
svclassifier.fit(Scaled_X_train, y_train)

# serializing
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(svclassifier, open('model.pkl', 'wb'))
