import numpy as np
import pandas as pd
import json
from my_utils import format_float, matrix_vector_euclidean_distance
from task1_utils import read_gesture_measures_reduced, metrics_reduced_2_numpy, metrics_numpy_2_PCA
# Fine imports
    

def find_k_most_similar_dp(alg_database, gesture_np, query_settings):
    k_top_gestures_to_return = query_settings['k_gestures_return'] # Quante gestures tornare come risultato
    # Eseguo dot product
    dot_product = np.dot(alg_database['data'], gesture_np)  / (np.linalg.norm(alg_database['data']) * np.linalg.norm(gesture_np))
    # Allargo aggiungendo una colonna di zeri, giusto x poter sfruttare il metodo di ritorno (che richiede una matrice e non un vettore)
    # Passandoci lo zero, il metodo facendone la somma non fa cambiare il risultato
    ds_gestures_differences = np.zeros((dot_product.shape[0], 2))
    for row, elem in enumerate(dot_product):
        ds_gestures_differences[row][0] = elem
    # Ritorno in modo ordinato
    return sorted_gestures_2_pandas(generate_ordered_differences(ds_gestures_differences, alg_database['gestures'])[:k_top_gestures_to_return])
#
def find_k_most_similar_tksem(alg_database, gesture_np, query_settings):
    k_features_to_use = query_settings['k_latent_features'] # Quante features latenti usare
    k_top_gestures_to_return = query_settings['k_gestures_return'] # Quante gestures tornare come risultato
    #NEW
    euclidean_distance = matrix_vector_euclidean_distance(alg_database['data'][:, :k_features_to_use], gesture_np[:k_features_to_use])
    ds_gestures_differences = np.zeros((len(euclidean_distance), 2))
    for row, elem in enumerate(euclidean_distance):
        ds_gestures_differences[row][0] = elem
    #OLD
    #ds_gestures_differences =  alg_database['data'][:, :k_features_to_use] - gesture_np[:k_features_to_use]
    # Ritorno in modo ordinato
    return sorted_gestures_2_pandas(generate_ordered_similarity(ds_gestures_differences, alg_database['gestures'])[:k_top_gestures_to_return])
#
def find_k_most_similar_ed(alg_database, gesture_np, query_settings):
    k_top_gestures_to_return = query_settings['k_gestures_return'] # Quante gestures tornare come risultato
    # Differenza data come la sottrazione
    ds_gestures_differences = alg_database['data'] - gesture_np
    # Ritorno in modo ordinato
    return sorted_gestures_2_pandas(generate_ordered_similarity(ds_gestures_differences, alg_database['gestures'])[:k_top_gestures_to_return])
#
def find_k_most_similar_dtw(alg_database, gesture_np, query_settings):
    k_top_gestures_to_return = query_settings['k_gestures_return'] # Quante gestures tornare come risultato
    # Differenza data come la matrice di dtw (percorso minimo)
    dtw_matrixes = dtw_matrix(alg_database['data'], gesture_np)
    ds_gestures_differences = np.zeros((len(dtw_matrixes), 2))
    for row, elem in enumerate(dtw_matrixes):
        ds_gestures_differences[row][0] = elem
    # Ritorno in modo ordinato
    return sorted_gestures_2_pandas(generate_ordered_similarity(ds_gestures_differences, alg_database['gestures'])[:k_top_gestures_to_return])


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


#Provo a fare una sola lettura e amen (per ora singola, poi prendo tutto)
def read_values_from_file_words(path):
    my_dict = {
        'ed' : {},
        'dtw' : {}
    }
    #leggo il file del gesto (.wrd)
    with open(path) as f:
        data = json.load(f)
    # Leggo i valori della dimensione quantizzata dell'intervallo e del simbolo dell'intervallo
    for coordinata, elem in data.items():  # X,Y,Z,W
        coordinata_ed_dict = {}
        coordinata_dtw_dict = {}
        for sensor, dati_utili in elem.items(): #0,1,2,3,...
            ed_sensor = []
            dtw_sensor = []
            for windows, dati_da_prendere in dati_utili.items(): #mean,std, w_0,...
                #a noi interessano solo i dati dentro alle finestre
                if(windows.startswith("w")):
                    dtw_sensor.append(dati_da_prendere[0]) #dtw
                    ed_sensor.append(dati_da_prendere[1]) #edit distance - ed
            coordinata_dtw_dict[sensor] = dtw_sensor
            coordinata_ed_dict[sensor] = ed_sensor
        my_dict['dtw'][coordinata] = coordinata_dtw_dict
        my_dict['ed'][coordinata] = coordinata_ed_dict
    # Ora genero la matrice dei valori di DTW ed ED per ogni istante temporale
    # DTW
    n_sensors = len(my_dict['dtw'][next(iter(my_dict['dtw'].keys()))].keys())
    n_windows = len(next(iter(my_dict['dtw'][next(iter(my_dict['dtw'].keys()))].values())))
    #faccio la media dei valori
    media = 0.0
    dtw_mean = np.zeros(n_windows)
    for coordinata, sensori in my_dict['dtw'].items(): # X,Y,Z,W
        for sensore, valori in sensori.items(): # 0, 1, 2..., 19
            for index, value in enumerate(valori): # [xxx, yyy, zzzz, kkk, ...]
                dtw_mean[index] += value
    # Media pesata dei sensori
    dtw_mean /= n_sensors
    # ED
    ed_mean = np.zeros(n_windows)
    for coordinata, sensori in my_dict['ed'].items(): # X,Y,Z,W
        for sensore, valori in sensori.items(): # 0, 1, 2..., 19
            for index, value in enumerate(valori): # ['-0.2882;-0.2882', '-0.2882;-0.2882'] -> [-0.57, -0.57]
                arr_values = value.split(';')
                # Una parola ha come valore la somma dei valori suoi simboli (in questo modo -1,-1 che come dato è max distante da 1,1 anche qua lo è)
                # Quindi somiglianza nei dati è anche somiglianza nelle distanze
                for arr_value in arr_values:
                    ed_mean[index] += float(arr_value)
    # Media pesata dei sensori
    ed_mean /= n_sensors
    return {'ed': ed_mean, 'dtw': dtw_mean}


def generate_matrix_ed_dtw(words_paths):
    ed = []
    dtw = []
    for word_file in words_paths:
        word_metrics = read_values_from_file_words(word_file)
        ed.append(word_metrics['ed'])
        dtw.append(word_metrics['dtw'])
    ed = normalize_edit_distance(ed)
    return {'ed':ed, 'dtw': dtw}

def normalize_edit_distance(ed_list):
    ed_normalized = []
    max_length = 0
    for gesture_ed in ed_list:
        if len(gesture_ed) > max_length:
            max_length = len(gesture_ed)
    for gesture_ed in ed_list:
        norm_gesture_ed = []
        # Se la lista è più piccola della + lunga, vedo quante copie dovro fare dell'ultimo elemento
        el_to_insert = max_length - len(gesture_ed)
        # Copiamo la lista
        norm_gesture_ed.extend(gesture_ed)
        cont = 0
        # Copio l'ultimo elemento tante volte quanto è definito da el_to_insert
        while cont < el_to_insert:
            norm_gesture_ed.append(gesture_ed[len(gesture_ed)-1])
            el_to_insert -= 1
        ed_normalized.append([format_float(elem, n_floats=6) for elem in norm_gesture_ed])
    return np.array(ed_normalized)

# Metodo che calcola la DTW distance dato una matrice (array di array di dati) ed un array di dati
def dtw_matrix(matrix_np, array_np):
    # Banalmente eseguo dtw_array per ogni array della matrice rispetto all'array di confronto
    return [dtw_array(row, array_np) for row in matrix_np]

# Metodo che calcola la DTW distance fra due array (numpy x performance, altrimenti anche liste vanno bene)
def dtw_array(array1_np, array2_np):
    # Mi prendo le lunghezze dei 2 array per creare la matrice (+1 xkè al primo giro faccio sempre l'inserzione di entrambi gli elementi (i primi elementi))
    # Banalmente in questo modo la prima colonna e la prima riga rimarranno sempre tutte +inf (a parte la cella 0,0 che avrà valore 0)
    len_a1, len_a2 = len(array1_np) + 1, len(array2_np) + 1
    matrix_cost = np.zeros((len_a1, len_a2))
    # Inizializzo l'intera matrice ad infinito (a parte il primo punto (come spiegato prima))
    for row in range(len_a1):
        for column in range(len_a2):
            matrix_cost[row, column] = np.inf
    matrix_cost[0, 0] = 0
    # Aggiorno la matrice, calcolando per ogni possibile inserzione, rimozione, vado avanti
    for row in range(1, len_a1):
        for column in range(1, len_a2):
            # Prendo il costo come la differenza fra i valori
            cost = abs(array1_np[row-1] - array2_np[column-1])
            # Cerco il best percorso fra i 3 percorribili (inserzione, delezione o move) (per arrivare al punto attuale)
            best_min_cost = np.min([matrix_cost[row-1, column], matrix_cost[row, column-1], matrix_cost[row-1, column-1]])
            # Per andare al prossimo punto il costo è dato dal best percorso fin ora, più il costo delle differenze fra i 2 valori
            matrix_cost[row, column] = cost + best_min_cost
    # Ritorno l'ultimo elemento in fondo della matrice che indicherà il costo totale del cammino
    return matrix_cost[len_a1 - 1, len_a2 - 1]