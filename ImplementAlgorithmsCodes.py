from django.conf import settings
import pandas as pd
import math
# Import modules
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score,confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score

path = settings.MEDIA_ROOT + "\\" + "dataset.csv"
df = pd.read_csv(path)
print(df.columns)
df = df.dropna()
df['Genre'].replace({'Female': 0, 'Male': 1}, inplace=True)
df['TypeEtab'].replace({'Public': 0, 'Private': 1}, inplace=True)
df['Niveau'].replace({'Primary': 1, 'Secondary': 2, 'Tertiary': 3}, inplace=True)
df['RetardSco'].replace({'1 year': 1, '2 years': 2, 'None': 0}, inplace=True)
df['Provenance'].replace({'Rural': 1, 'Suburban': 2, 'Urban': 3}, inplace=True)
df['Handicap'].replace({'Yes': 1, 'No': 0}, inplace=True)
df['Fee_reimbursement'].replace({'Yes': 1, 'No': 0}, inplace=True)
df['Result'].replace({'Continue': 0, 'Discontinue': 1}, inplace=True)

X = df.iloc[:, :-1].values  # indipendent variable
y = df.iloc[:, -1].values  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)


def knnResults():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    precisiom = precision_score(y_pred, y_test)
    recall = recall_score(y_pred,y_test)
    print(f"KNN Results MAE:{acc} RMSE: {precisiom} R2-Score:{recall}")
    # print(self.df.head())
    return {'mae':acc, 'rmse':precisiom, 'r2_knn':recall}


def randomForest():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rcm = confusion_matrix(y_pred,y_test)
    print(rcm)
    acc = accuracy_score(y_pred, y_test)
    precisiom = precision_score(y_pred, y_test)
    recall = recall_score(y_pred,y_test)
    print(f"KNN Results MAE:{acc} RMSE: {precisiom} R2-Score:{recall}")
    # print(self.df.head())
    return {'mae':acc, 'rmse':precisiom, 'r2_knn':recall,'rcm':rcm}


def svmAlgorithm():
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    precisiom = precision_score(y_pred, y_test)
    recall = recall_score(y_pred,y_test)
    print(f"KNN Results MAE:{acc} RMSE: {precisiom} R2-Score:{recall}")
    # print(self.df.head())
    return {'mae':acc, 'rmse':precisiom, 'r2_knn':recall}


# def sgdAlgorithm():
#     from sklearn.linear_model import SGDClassifier
#     sgd = SGDClassifier(max_iter=1000, alpha=0.0001, l1_ratio=0.15, random_state=42)
#     sgd.fit(X_train, y_train)
#     y_pred = sgd.predict(X_test)
#     acc = accuracy_score(y_pred, y_test)
#     precisiom = precision_score(y_pred, y_test)
#     recall = recall_score(y_pred,y_test)
#     print(f"KNN Results MAE:{acc} RMSE: {precisiom} R2-Score:{recall}")
#     # print(self.df.head())
#     return {'mae':acc, 'rmse':precisiom, 'r2_knn':recall}


def corrGraph():
    print(df.corr())
    # Displaying heatmap
    # fig, ax = mp.subplots(figsize=(15, 10))
    # dataplot = sb.heatmap(df.corr(), annot=True, linewidths=.5, ax=ax)
    # mp.savefig("assets/static/images/corr.png")
    # mp.show()

