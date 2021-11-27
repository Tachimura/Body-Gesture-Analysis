import numpy as np
import pandas as pd

def database_numpy_2_supervised_data(db_numpy, labels):
    # Eseguo reshape, vedo i vari sensori come se fossero altre features
    db_reshaped = np.transpose(db_numpy,(0,1,2)).reshape(db_numpy.shape[0],-1)
    # Genero i nomi per le reshaped features
    reshaped_features_names = ["F"+str(cont) for cont in range(0, len(db_reshaped[0]))]
    # Creo il dataframe pandas e ci inserisco gli esempi e le relative labels
    dataset_pandas = pd.DataFrame()
    for row, label in zip(db_reshaped, labels):
        features = {key: value for key, value in zip(reshaped_features_names, row)}
        features['Label'] = label['Label']
        dataset_pandas = dataset_pandas.append(pd.DataFrame(features, index=[label['Gesture']]))
    # Splitto in train data e train labels e li ritorno con le reshaped_features_names
    X_train_df = dataset_pandas.loc[:, dataset_pandas.columns != 'Label']
    y_train_df = dataset_pandas.loc[:, dataset_pandas.columns == 'Label']
    return X_train_df, y_train_df, reshaped_features_names

# PROBABILMENTE SARà DA MODIFICARE, TEST_DATA CONTERRà IL PATH DEI DOCUMENTI
# BISOGNERA GENERARE WORDS -> VECTORS -> TRASFORMAZIONE
def prepare_test_dataframe(test_data, reshaped_features_names):
    # Preparo i dati di test trasformandoli in dataframe
    X_test_df = pd.DataFrame()
    for cont, row in zip(range(0, len(test_data)),test_data):
        X_test_df = X_test_df.append(pd.DataFrame({key: value for key, value in zip(reshaped_features_names, row)}, index=['Test'+str(cont)]))
    return X_test_df