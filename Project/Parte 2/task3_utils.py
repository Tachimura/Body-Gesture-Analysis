import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD

from task1_utils import get_alg_features_score
from my_utils import format_float
# Fine imports

def generate_gestures_similarity_matrix(dataset_gesture, k_features_to_use):
    matrix = np.zeros((len(dataset_gesture), len(dataset_gesture)))
    for row, gesture_np in zip(range(0, len(dataset_gesture)), dataset_gesture):
        ds_components_differences = dataset_gesture[:, :k_features_to_use] - gesture_np[:k_features_to_use]
        # - Posso normalizzare la distanza usando un certo peso
        #weights = np.array([1, 1, 1, ..])
        #ds_gesture_differences *= weights
        ds_gesture_differences = np.abs(ds_components_differences).sum(axis=1)
        normalized_gesture_differences = ds_gesture_differences / ds_gesture_differences.max()
        ds_similarities = 1 - normalized_gesture_differences
        for column, similarity in zip(range(row, len(ds_similarities)), ds_similarities[row:]):
            matrix[row][column] = similarity
            matrix[column][row] = similarity
    return matrix

def gestures_similarity_matrix_2_pandas(similarity_matrix, gestures_names):
    return pd.DataFrame(data=similarity_matrix, index=gestures_names, columns=gestures_names)

def generate_gesture_gesture_svd(gestures_sim_matrix_pd, n_components=60):
    svd = TruncatedSVD(n_components=n_components) # mi tengo in modo analogo a quanto fatto con le altre
    fitted_svd = svd.fit_transform(gestures_sim_matrix_pd)
    features_score_df = get_alg_features_score(svd, gestures_sim_matrix_pd.columns)
    return save_alg(svd, features_score_df)

def save_alg(alg, features_score_df, show_intermediate_data=True):
    gesture_features_score_result = {}
    # Salvo le informazioni in un dizionario scores
    for column in features_score_df:
        gesture_features_score_result[column] = {}
        column_scores = features_score_df[column]
        for word, score in zip(column_scores.index, column_scores):
            gesture_features_score_result[column][word] = format_float(score, n_floats=4)

    # np.cumsum mostra la varianza totale rappresentata dalle varie componenti (mostra quante ne bastano per 100% o meno), possiamo poi plottarlo
    return gesture_features_score_result, np.cumsum(alg.explained_variance_ratio_)

# Dato un dizionario del tipo {component: {'gesture_file1': score1, 'gesture_file2':score2,...,'gesture_filen':scoren}}
# Ritorna in forma contratta le top K gestures latenti con i gesti ordinate in ordine decrescente di score (quella con score + alto prima)
def get_top_p_latent_gestures(latent_gestures, k):
    ordered_data = []
    cont = 0
    for component, data in latent_gestures.items():
        if cont >= k:
            break
        cont += 1
        component_gesture_scores = [(gesture, score) for gesture, score in data.items()]
        ordered_component_gesture_scores = {}
        ordered_component_gesture_scores[component] = list(sorted(component_gesture_scores, key = lambda x: x[1], reverse=True))
        ordered_data.append(ordered_component_gesture_scores)
    #
    return ordered_data