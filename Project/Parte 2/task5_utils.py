import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
# Decision Tree classifier
from sklearn import tree
from my_utils import format_float

# Metodo che data una matrice di dati ed un array di labels (una per riga della matrice)
# Ritorna il tutto in formato pandas, generando (e facendo shuffle dei dati)
# X_Train, y_train, X_test, y_test, features_names
def database_numpy_2_supervised_data(db_numpy, labels):
    # Genero i nomi per le features
    features_names = ["F"+str(cont) for cont in range(len(db_numpy[0]))]
    # Creo il dataframe pandas e ci inserisco gli esempi e le relative labels
    dataset_pandas = pd.DataFrame()
    for row, label in zip(db_numpy, labels):
        features = {key: value for key, value in zip(features_names, row)}
        features['Label'] = label
        dataset_pandas = dataset_pandas.append(features, ignore_index=True)
    # Splitto, prendendo solo i dati e le labels
    data = dataset_pandas.loc[:, dataset_pandas.columns != 'Label']
    labels = dataset_pandas.loc[:, dataset_pandas.columns == 'Label']
    # Stratify fa si che esca il best rateo fra le classi (cerca di separare manentendo omogeneità nelle distribuzioni)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0, stratify=labels)
    return (X_train, y_train), (X_test, y_test), features_names

# Esegue una ricerca completa e restituisce i valori di accuracy + neighbors in formato pandas
def test_knn_best_K(X_train, y_train, X_test, y_test, max_neighbors=1, weights='uniform'):
    accuracy_data = {
        'neighbors': [],
        'data': []
    }
    # Ciclo per ogni valore di neighbors e vedo come va
    for n_neighbors in range(1, max_neighbors):
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy_data['neighbors'].append(n_neighbors)
        accuracy_data['data'].append(format_float(accuracy_score(y_test, y_pred)))
    # Trasformo in dataframe (cosi plotto)
    accuracy_data_df = pd.DataFrame(columns=["# Neighbors", "Accuracy"])
    for neighbor, accuracy in zip(accuracy_data['neighbors'], accuracy_data['data']):
        accuracy_data_df = accuracy_data_df.append({"# Neighbors": neighbor, "Accuracy": accuracy}, ignore_index=True)
    return accuracy_data_df

def test_tree_best_depth(X_train, y_train, X_test, y_test, max_depth=100):
    accuracy_data = {
        'depth': [],
        'data': []
    }
    # Ciclo per ogni valore di neighbors e vedo come va
    for depth in range(1, max_depth):
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy_data['depth'].append(depth)
        accuracy_data['data'].append(format_float(accuracy_score(y_test, y_pred)))
    # Trasformo in dataframe (cosi plotto)
    accuracy_data_df = pd.DataFrame(columns=["Depth", "Accuracy"])
    for neighbor, accuracy in zip(accuracy_data['depth'], accuracy_data['data']):
        accuracy_data_df = accuracy_data_df.append({"Depth": neighbor, "Accuracy": accuracy}, ignore_index=True)
    return accuracy_data_df