import numpy as np
import pandas as pd

from task1_utils import read_gesture_measures_reduced, metrics_reduced_2_numpy, metrics_numpy_2_PCA
# Fine imports
    

def find_k_most_similar_dp(alg_database, gesture_np, query_settings):
    k_top_gestures_to_return = query_settings['k_gestures_return'] # Quante gestures tornare come risultato
    # Eseguo dot product
    dot_product = np.dot(alg_database['data'], gesture_np)  / (np.linalg.norm(alg_database['data']) * np.linalg.norm(gesture_np))
    ds_gestures_differences = np.zeros((dot_product.shape[0], 2))
    for row, elem in enumerate(dot_product):
        ds_gestures_differences[row][0] = elem
    # Ritorno in modo ordinato
    return sorted_gestures_2_pandas(generate_ordered_differences(ds_gestures_differences, alg_database['gestures'])[:k_top_gestures_to_return])

def find_k_most_similar_tksem(alg_database, gesture_np, query_settings):
    k_features_to_use = query_settings['k_latent_features'] # Quante features latenti usare
    k_top_gestures_to_return = query_settings['k_gestures_return'] # Quante gestures tornare come risultato
    # Differenza data come la sottrazione
    ds_gestures_differences = alg_database['data'][:, :k_features_to_use] - gesture_np[:k_features_to_use]
    # Ritorno in modo ordinato
    return sorted_gestures_2_pandas(generate_ordered_similarity(ds_gestures_differences, alg_database['gestures'])[:k_top_gestures_to_return])

######################################################################################
######################################## TODO ########################################
######################################################################################
def find_k_most_similar_ed(alg_database, gesture_np, query_settings):
    k_top_gestures_to_return = query_settings['k_gestures_return'] # Quante gestures tornare come risultato

    # Differenza data come la sottrazione (QUA DA MODIFICARE)
    ds_gestures_differences = alg_database['data'][:, :k_features_to_use] - gesture_np[:k_features_to_use]

    # Ritorno in modo ordinato
    return sorted_gestures_2_pandas(generate_ordered_similarity(ds_gestures_differences, alg_database['gestures'])[:k_top_gestures_to_return])
def find_k_most_similar_dtw(alg_database, gesture_np, query_settings):
    k_top_gestures_to_return = query_settings['k_gestures_return'] # Quante gestures tornare come risultato

    # Differenza data come la sottrazione (QUA DA MODIFICARE)
    ds_gestures_differences = alg_database['data'][:, :k_features_to_use] - gesture_np[:k_features_to_use]
    
    # Ritorno in modo ordinato
    return sorted_gestures_2_pandas(generate_ordered_similarity(ds_gestures_differences, alg_database['gestures'])[:k_top_gestures_to_return])
#######################################################################################
####################################### END TODO ######################################
#######################################################################################

def generate_ordered_similarity(ds_differences, db_gestures):
    ds_similarities = 1 - generate_sub_differences(ds_differences)
    gesture_similarities = [(gesture, similarity) for gesture, similarity in zip(db_gestures, ds_similarities)]
    return sorted(gesture_similarities, key = lambda x: x[1], reverse=True) # Ordiniamo per il valore di differenza (file, differenza)
    
def generate_ordered_differences(ds_differences, db_gestures):
    ds_similarities = generate_sub_differences(ds_differences)
    gesture_similarities = [(gesture, similarity) for gesture, similarity in zip(db_gestures, ds_similarities)]
    return sorted(gesture_similarities, key = lambda x: x[1], reverse=True) # Ordiniamo per il valore di differenza (file, differenza)

def generate_sub_differences(ds_differences):
    # Sommo la differenza di tutte le feature latenti del mio gesto
    ds_gesture_differences = np.abs(ds_differences).sum(axis=1)
    # - Posso normalizzare la distanza usando un certo peso
    #weights = np.array([1, 1, 1, ..])
    #ds_gesture_differences *= weights
    # Dato che voglio somiglianze e non distanze, trovo il valore max, normalizzo e poi faccio 1 - values
    return ds_gesture_differences / ds_gesture_differences.max()


def sorted_gestures_2_pandas(sorted_gestures):
    dtypes = np.dtype(
        [
            ("gesture", str),
            ("similarity", float)
        ]
    )
    sorted_gestures_df = pd.DataFrame(np.empty(0, dtype=dtypes))
    for (gesture, similarity) in sorted_gestures:
        sorted_gestures_df = sorted_gestures_df.append({"gesture": gesture, "similarity": similarity}, ignore_index=True)
    return sorted_gestures_df

######################################################################################
######################################## TODO ########################################
######################################################################################
def read_words_data_ed_dtw(gesture_files, data_components, alfabeto, path_words_directory):
    my_data = {
        'ed': dict(),
        'dtw': dict()
    }
    # TODO LEGGERE FILE WORD DA JSON (LO TIRA FUORI COME DIZIONARIO)
    return my_data
#######################################################################################
####################################### END TODO ######################################
#######################################################################################