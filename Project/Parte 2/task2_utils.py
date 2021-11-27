import numpy as np
import pandas as pd

from task1_utils import read_gesture_measures_reduced, metrics_reduced_2_numpy, metrics_numpy_2_PCA
# Fine imports

#Genera il database sotto forma di matrice
def generate_alg_database(alg_metrics):
    alg_database = generate_alg_db_numpy_3d(alg_metrics)
    alg_gesture_files = []
    # logica di riempire il database con i valori
    riga = -1
    for alg_metric in alg_metrics:
        riga += 1
        alg_database[riga] = alg_metrics_2_numpy_3d(alg_metric)
        alg_gesture_files.append(alg_metric['document'])
    #
    database = {
        'gestures': alg_gesture_files,
        'np_database': alg_database
    }
    return database

# 3D
def generate_alg_db_numpy_3d(alg_metrics):
    n_gesti = len(alg_metrics)
    n_components = len(alg_metrics[0]['scores'])
    n_parole = len(alg_metrics[0]['scores']['PC1'])
    return np.zeros((n_gesti, n_components, n_parole))

def alg_metrics_2_numpy_3d(alg_metric):
    n_components = len(alg_metric['scores'])
    n_parole = len(alg_metric['scores']['PC1'])
    alg_numpy = np.zeros((n_components, n_parole))
    riga = 0
    for pc_component, scores in alg_metric['scores'].items():
        colonna = 0
        for word, score in scores.items():
            alg_numpy[riga][colonna] = score
            colonna += 1
        riga += 1
    return alg_numpy
##############

def find_k_most_similar(alg_database, gesture_np, query_settings):
    k_features_to_use = query_settings['k_latent_features'] # Quante features latenti usare
    k_top_gestures_to_return = query_settings['k_gestures_return'] # Quante gestures tornare come risultato
    # Differenza data come la sottrazione (qua sarà da modificare)
    ds_sensors_differences = alg_database['np_database'][:, :k_features_to_use, :] - gesture_np[:k_features_to_use, :]
    # Ritorno in modo ordinato
    return sorted_gestures_2_pandas(generate_ordered_similarity(ds_sensors_differences, alg_database['gestures'])[:k_top_gestures_to_return])

def generate_ordered_similarity(ds_sensors_differences, db_gestures):
    # - Prima sum su axis 2 per avere 1 valore per ogni feature latente x ogni gesto
    ds_gesture_differences = np.abs(ds_sensors_differences).sum(axis=2)
    # - Posso normalizzare la distanza usando un certo peso
    #weights = np.array([1, 1, 1, ..])
    #ds_gesture_differences *= weights
    # - Seconda somma per avere 1 valore per ogni gesto
    ds_gesture_differences = ds_gesture_differences.sum(axis=1)
    normalized_gesture_differences = ds_gesture_differences / ds_gesture_differences.max()
    ds_similarities = 1 - normalized_gesture_differences

    gesture_similarities = [(gesture, similarity) for gesture, similarity in zip(db_gestures, ds_similarities)]
    return sorted(gesture_similarities, key = lambda x: x[1], reverse=True) # Ordiniamo per il valore di differenza (file, differenza)
    
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