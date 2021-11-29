import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from my_utils import format_float
# Fine imports


# Leggo il file di vettori (misure) e lo ritrasformo in un dizionario{nome_documento:"", data:{}} (ridotto xkè salvo solo le info del modello vettoriale di interesse)
def read_gesture_measures_reduced(g_file, path, measure_type, n_sensors=20):
    measure_file_path = path
    if measure_type == 'Modello TF':
        measure_file_path = measure_file_path + "tf_"
    else:
        measure_file_path = measure_file_path + "tfidf_"
    measure_file_path = measure_file_path + "vectors_f" + str(g_file) + ".txt"
    measure_data_dict = {
        'document': g_file,
        'n_sensors': n_sensors,
        'data': {} 
    }
    with open(measure_file_path) as json_file:
        measure_data_dict['data'] = json.load(json_file)
    return measure_data_dict

# trasformo il dizionario (ridotto, vedi metodo sopra) in un array numpy
def metrics_reduced_2_numpy(gesture_metrics_dict, alfabeto):
    n_parole = len(gesture_metrics_dict['data'].values())
    n_sensori = gesture_metrics_dict['n_sensors']
    metrics_numpy = np.zeros(n_parole * n_sensori)

    for cont, parola in zip(range(alfabeto.n_parole), alfabeto.parole):
        for sensor, value in gesture_metrics_dict['data'][parola].items():
            metrics_numpy[cont + (n_parole * int(sensor))] = float(value)
    return metrics_numpy

# Dato il numpy, genera il file corrispettivo in pandas dei dati
def metrics_numpy_2_pandas(metrics_numpy, alfabeto, n_sensors=20, data_to_scale=False):
    df_columns = ["S"+str(val)+ " " + parola for val in range(n_sensors) for parola in alfabeto.parole]
    metrics_df = pd.DataFrame(metrics_numpy, columns=df_columns)
    scaled_metrics_df = metrics_df
    # Scalo per evitare che valori con scala + grande influenzi di più gli altri valori
    if data_to_scale:
        scaled_metrics_df = StandardScaler().fit_transform(metrics_df)
    return scaled_metrics_df

# ---------------- PCA, SVD, LDA GET-FEATURES ----------------
# Dato il dataframe di partenza ed il pca 'fittato', mi tira fuori un dataframe m*k che mi dice lo score di ogni feature verso la feature latente
def get_alg_features_score(alg, columns):
    # Prendo le componenti del pca e le trasfomo in un dataframe (cosi lavoro con pandas ez)
    if hasattr(alg, 'components_'):
        features_score_df = pd.DataFrame(alg.components_)
    else:
        features_score_df = pd.DataFrame(alg.coef_) #lda ha un altro coso
    # Ci assegno le colonne
    features_score_df.columns = columns
    # Trasformo i valori in valori assoluti (mi interessa solo sapere se c'è o meno una relazione, non il suo tipo)
    # (valori prossimi allo 0, indicano poca importanza, valori prossimi a -1 o a 1 indicano una alta relazione positiva o negativa)
    features_score_df = features_score_df.apply(np.abs)
    # Traspongo, cosi ho per ogni parola, la relazione con le varie componenti
    features_score_df = features_score_df.transpose()
    # Per dare un nome sensato alle varie colonne, prima mi prendo quante componenti latenti ho trovato
    num_pcs = features_score_df.shape[1]
    # Mi genero i nomi delle colonne come PCi, i=1,...,K e rinomino le colonne
    new_columns = [f'PC{i}' for i in range(1, num_pcs + 1)]
    features_score_df.columns = new_columns
    return features_score_df


def save_alg(alg, fitted_alg, features_score_df, show_intermediate_data=False):
    # Unisco i punteggi al nome del documento
    gesture_features_score_result = {
        'fitted_alg': fitted_alg,
        'scores': {}
    }
    # Salvo le informazioni in un dizionario scores
    for column in features_score_df:
        gesture_features_score_result['scores'][column] = {}
        column_scores = features_score_df[column]
        for word, score in zip(column_scores.index, column_scores):
            gesture_features_score_result['scores'][column][word] = format_float(score, n_floats=4)

    # Se voglio mostrare i dati intermedi / raffigurazione grafica
    if show_intermediate_data:
        # Show df
        display(features_score_df)
        ## PC1 top 10 important features
        pc1_top_10_features = features_score_df['PC1'].sort_values(ascending = False)[:10]
        print(), print(f'PC1 top 10 features are: \n')
        display( [word + ":     " + str(format_float(val, n_floats=6)) for word, val in zip(pc1_top_10_features.index, pc1_top_10_features)] )
    # Fine show risultati

    # np.cumsum mostra la varianza totale rappresentata dalle varie componenti (mostra quante ne bastano per 100% o meno), possiamo poi plottarlo
    return gesture_features_score_result, np.cumsum(alg.explained_variance_ratio_)

# ---------------- PCA ----------------
# Dato i dati in versione pandas, ed il nome del gesto (nome documento), esegue il PRINCIPAL COMPONENT ANALYSIS
# Ritorna per ogni componente latente, lo score di ogni feature. (con show_intermediate_data=True ritorna come secondo parametro pure un plot)
def metrics_numpy_2_PCA(metrics_df, show_intermediate_data=False):
    pca = PCA(len(metrics_df.index)) # Se non passo nulla, come K lui usa il minimo fra features e n. dati (nel nostro caso, min XXXX e 93 -> 93)
    fitted_pca = pca.fit_transform(metrics_df)
    features_score_df = get_alg_features_score(pca, metrics_df.columns)
    return save_alg(pca, fitted_pca, features_score_df, show_intermediate_data=show_intermediate_data)
    
# ---------------- SVD ----------------

# Dato i dati in versione pandas, ed il nome del gesto (nome documento), esegue il PRINCIPAL COMPONENT ANALYSIS
# Ritorna per ogni componente latente, lo score di ogni feature. (con show_intermediate_data=True ritorna come secondo parametro pure un plot)
def metrics_numpy_2_SVD(metrics_df, show_intermediate_data=False):
    svd = TruncatedSVD(n_components = len(metrics_df.index) - 1) # mi tengo in modo analogo a quanto fatto con le altre (qua richiede -1)
    fitted_svd = svd.fit_transform(metrics_df)
    features_score_df = get_alg_features_score(svd, metrics_df.columns)
    return save_alg(svd, fitted_svd, features_score_df, show_intermediate_data=show_intermediate_data)

# ---------------- LDA ----------------
# Dato i dati in versione pandas, ed il nome del gesto (nome documento), esegue il LINEAR DISCRIMINANT ANALYSIS
# Ritorna per ogni componente latente, lo score di ogni feature. (con show_intermediate_data=True ritorna come secondo parametro pure un plot)
def metrics_numpy_2_LDA(metrics_df, labels, show_intermediate_data=False):
    #Fitting the LDA class
    lda = LDA(n_components = len(set(labels)) - 1) # n components non deve essere maggiore tra min(n.data, n_classes-1)
    fitted_lda = lda.fit_transform(metrics_df.to_numpy(), y=labels)
    features_score_df = get_alg_features_score(lda, metrics_df.columns)
    return save_alg(lda, fitted_lda, features_score_df,  show_intermediate_data=show_intermediate_data)
# -------------------------------------

# Dato una lista di np.cumsum (vedi risultato sopra), prepara i dati in formato dataframe pandas da plottare
def prepare_alg_variances_to_plot(data_to_plot):
    dtypes = np.dtype(
        [
            ("document", str),
            ("component", int),
            ("variance", float)
        ]
    )
    data_df = pd.DataFrame(np.empty(0, dtype=dtypes))
    for doc_values in data_to_plot:
        doc_name, values = doc_values
        cont = 1
        for value in values:
            data_df = data_df.append({"document": doc_name, "component": cont, "variance": value}, ignore_index=True)
            cont+=1
    return data_df


# Dato un dizionario del tipo {'document':XXX, scores:{'C1':{'word1':score1, ..., 'wordm':scorem}, ..., 'Cn':{...}}}
# Ritorna in forma contratta le top K features latenti con le parole ordinate in ordine decrescente di score (quella con score + alto prima)
def get_top_k_latent_features(metrics_PCA, k):
    latent_features = metrics_PCA['scores']
    ordered_data = []
    cont = 0
    for latent_feature, data in latent_features.items():
        if cont >= k:
            break
        cont += 1
        latent_feature_data = {}
        word_scores_ordered = []
        for word, score in data.items():
            word_scores_ordered.insert(0, (word, score))

        latent_feature_data[latent_feature] = list(sorted(word_scores_ordered, key = lambda x: x[1], reverse=True))
        ordered_data.append(latent_feature_data)
    return ordered_data