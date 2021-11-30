# Per quantizzazione gaussiana
from scipy import stats
import scipy
#
import numpy as np
from numpy import random
import pandas as pd
import json
import math
from my_utils import format_float, quanticize_interval, Alfabeto
# Fine imports

task0_stats = {
    "window_size" : 5,      # Dimensione finestra
    "window_shift" : 3,    # Dimensione shift della finestra
    "r": 2,                 # Risoluzione (2*R = # tagli nell'intervallo)
    "interval_min": -1,     # Valore minimo dell'intervallo normalizzato
    "interval_max": +1      # Valore massimo dell'intervallo normalizzato
}


##################################################
#################### TASK0  A ####################
##################################################

# Estrae la parola da un array di valori (estraendone i simboli uno per volta e poi concatenandoli)
def extract_word(array, symbols):
    # Inizio dalla parola vuota
    word = ""
    for element in array:
        # Per ogni elemento nell'array estraggo il simbolo (centro) che equivale a quell'elemento
        word += ";" + extract_symbol(element, symbols)
    # Ritorno la parola formata dalla concatenazione dei simboli
    return word[1:] #tolgo il primo carattere perchè conterrà il ;

# Estrae il simbolo da un valore (guardando i range dentro i simboli dell'alfabeto)
def extract_symbol(element, symbols):
    element_symbol = 0
    for symbol, (off1,  off2) in symbols.items():
        min_value = min(off1, off2)
        max_value = max(off1, off2)
        if element >= min_value and element <= max_value:
            element_symbol = symbol
            break
    if element_symbol == 0:
        print("Elemento non in bucket:", element)
    return str(element_symbol)

##########################
###  METODO EURISTICO  ###
##########################
# Nel nostro caso, se la finestra non ha abbastanza valori, la completo copiando l'ultimo elemento letto finchè è piena
def extract_padded_window(starting_array, array_window, start_position, window_size):
        padded_window = np.zeros(window_size)
        last_elem = 0
        cont = 0
        while cont < window_size:
            # Se posso prendere dall'array
            if cont < len(array_window):
                last_elem = array_window[cont]
                padded_window[cont] = last_elem
            else: # Aggiungo padding sempre dell'ultimo elemento
                padded_window[cont] = last_elem
            cont += 1
        return padded_window
##########################

# Estrae una finestra nel vettore, se ci sono meno valori, richiama un metodo che gestisce questo (definito proprio qua sopra)
def extract_window(array, start_position, window_size):
    window = array[start_position : (start_position + window_size)]
    if len(window) != window_size: #Se non è size corretto, ritorno la versione con padding
        return extract_padded_window(array, window, start_position, window_size)
    return window

# Associa dei valori alle corrispettive parole dell'alfabeto
def associate_data_to_symbols(data_row, alfabeto, window_options):
    window_size, window_shift = window_options
    data_to_symbols = []
    current_position = 0
    while current_position < len(data_row):
        # IDX => istante_temporale
        idx = current_position
        # Estraggo una finestra
        window = extract_window(data_row, current_position, window_size)
        avgQ = window.mean()
        # Converto gli elementi nella finestra nella parola dell'alfabeto (WINQ)
        winQ = extract_word(window, alfabeto.simboli)
        data_to_symbols.append((idx, (avgQ, winQ)))
        # Sposto la posizione della finestra
        current_position += window_shift
    return data_to_symbols

# WRAPPER DA RICHIAMARE
# Permette di prendere le parole di un file di gesti (.csv) e ritorna il dict che lo rappresenta
def gesture_2_word(gesture_file_name, components, alfabeto, input_directory, options):
    # Leggo i file
    gesture_dict = {}
    for component in components:
        gesture_df = pd.read_csv(input_directory + component + "/" + gesture_file_name, header=None)
        stds = gesture_df.std(axis=1) # Calcolo std di tutte le righe
        means = gesture_df.mean(axis=1) # Calcolo mean di tutte le righe
        normalized_df = normalize_data(gesture_df, options["interval_min"], options["interval_max"])
        gesture_dict[str(component)] = {}
        for sensor in range(0, len(gesture_df.index)):
            gesture_dict[str(component)][str(sensor)] = {}
            gesture_dict[str(component)][str(sensor)]["mean"] = format_float(means.iloc[sensor])
            gesture_dict[str(component)][str(sensor)]["std"] = format_float(stds.iloc[sensor])
            sensor_infos = associate_data_to_symbols(normalized_df.iloc[sensor].to_numpy(), alfabeto, (options["window_size"], options["window_shift"]))
            for (finestra, (avgQ, winQ)) in sensor_infos:
                gesture_dict[str(component)][str(sensor)]["w_"+str(finestra)] = (format_float(avgQ), winQ)
    return gesture_dict

# WRAPPER DA RICHIAMARE
# Salvataggio su file del dict contenente le words di un file gesture
def write_word_as_json(gesture_word_dict, gesture_file_name, output_directory):
    output_path = output_directory + gesture_file_name.split(".")[0] + ".wrd"
    with open(output_path, "w") as output_file:
        json.dump(gesture_word_dict, output_file, indent=4)

# Normalizza i dati del dataframe in range [min_val, max_val]
def normalize_data(data, min_val, max_val):
    normalized_df = data.copy()
    min_x = normalized_df.min().min()
    max_x = normalized_df.max().max()
    # normalize x in (a,b) = (b - a) * (x-min_x / max_x-min_x) + a
    [normalized_df[col].update((max_val - min_val) * ((normalized_df[col] - min_x) / (max_x - min_x)) + min_val) for col in normalized_df.columns]
    return normalized_df

# WRAPPER DA RICHIAMARE
## ROBA X GENERARE LE PAROLE DELL'ALFABETO PRENDENDO LE PAROLE NELLE FINESTRE DEI SENSORI
def generate_alphabet_words(data, alfabeto, window_options):
    window_size, window_shift = window_options
    for row in data.to_numpy():
        current_position = 0
        while current_position < len(row):
            # Estraggo una finestra
            window = extract_window(row, current_position, window_size)
            # Converto gli elementi nella finestra nella parola dell'alfabeto (WIN)
            win = extract_word(window, alfabeto.simboli)
            alfabeto.addParola(win)
            # Sposto la posizione della finestra
            current_position += window_shift

# WRAPPER DA RICHIAMARE
# genera l'alfabeto in base ai file di gesture (.csv) passati in ingresso
def generate_alphabet(gesture_files, components, input_directory, options):
    # Mi prendo tutti i documenti
    documents = []
    for gesture_file_name in gesture_files:
        for component in components:
            document = input_directory + component + "/" + gesture_file_name
            documents.append(document)
    alfabeto = quanticize_interval(options) # Genero i simboli dell'alfabeto
    # Mi genero le parole prendendo solo quelle presenti nel dataset
    for document in documents:
        data = pd.read_csv(document, header=None)
        # 1 - Prendo il file di gesti normalizzato
        normalized_data = normalize_data(data, options["interval_min"], options["interval_max"])
        # Estraggo le parole e le inserisco nell'alfabeto
        generate_alphabet_words(normalized_data, alfabeto, (options["window_size"], options["window_shift"]))
    return alfabeto

# WRAPPER DA RICHIAMARE
# Genera un dizionario con tuple  (Component_name, SensorID, winQ) ->  Contatore
def generate_dizionario_parole_from_gesturewordsdict(gesture_words_dictionary):
    dizionario_gesture = {} # Contiene key: (Component_name, SensorID, winQ) - Value: Contatore
    for component, component_infos in gesture_words_dictionary.items():
        for sensor, sensor_infos in component_infos.items():
            for key, value in sensor_infos.items():
                if key != "mean" and key != "std":
                    _, parola = value # prendo solo la parola e non il valore della media della finestra
                    tripla = (component, sensor, parola)
                    if tripla in dizionario_gesture: # Se è già presente, incremento di 1 il contatore
                        dizionario_gesture[tripla] += 1
                    else:                            # Altrimenti, ci aggiungo la voce
                        dizionario_gesture[tripla] = 1
    return dizionario_gesture


##################################################
#################### TASK0  B ####################
##################################################

class DatasetWordsPreProcessing:
    def __init__(self, n_documents, alfabeto):
        self.n_documents = n_documents
        self.words_4_document = {} #key -> word value -> lista dei documenti in cui la parola compare
        for parola in alfabeto.parole:
            self.words_4_document[parola] = set()

    def generate_IDF(self, gesture_preprocess_unit):
        for key, _ in gesture_preprocess_unit.IDF.items():
            gesture_preprocess_unit.IDF[key] = math.log((self.n_documents + 2) / (len(self.words_4_document[key]) + 1))

class GestureWordsPreProcessing:
    def __init__(self, document_name, alfabeto):
        self.document_name = document_name
        self.IDF = {} #key -> word value -> log(n_documents/#n_documents che hanno word)
        for parola in alfabeto.parole:
            self.IDF[parola] = 0

# WRAPPER DA RICHIAMARE
# Eseguo pre-processing, calcolo IDF e frequenze
def words_preprocessing(dataset_preprocess_unit, gesto_dict, nome_documento):
    # Aggiungo il sensore nel dict di parole (che è poi un set, quindi mi indica solo quali documenti hanno la parola)
    for (_, _, word), cont in gesto_dict.items():
        dataset_preprocess_unit.words_4_document[word].add(nome_documento)

# Classe che contiene le metriche TF e TF-IDF
class GestureMetrics:
    def __init__(self, document_name, n_sensors, alfabeto):
        self.document = document_name
        self.n_sensors = n_sensors
        self.TF = {} #key -> word value -> (frequenza parola / n parole nel sensore)
        self.TFIDF = {} #key -> word value -> log(n_sensors/#n_sensors che hanno word)
        for parola in alfabeto.parole:
            self.TF[parola] = {}
            self.TFIDF[parola] = {}
            for sensor in range(0, n_sensors):
                sensor = str(sensor)
                self.TF[parola][sensor] = 0
                self.TFIDF[parola][sensor] = 0
        #

# Calcola le frequenze delle parole in ogni sensore (facendo la somma delle occorrenze nelle N componenti)
def calculate_words_frequencies(word_file, n_sensors, alfabeto):
    words_frequencies = {}
    n_words_4_sensor = 0
    for parola in alfabeto.parole:
        words_frequencies[parola] = {}
        for sensor in range(0, n_sensors):
            sensor = str(sensor)
            words_frequencies[parola][sensor] = 0
    #
    word_file_dict = None
    with open(word_file) as json_file:
        word_file_dict = json.load(json_file)
    # Conto frequenza delle parole
    for component, component_values in word_file_dict.items():
        for sensor, sensor_values in component_values.items():
            for key, value in sensor_values.items():
                if key == 'mean' or key == 'std':
                    continue
                _, word = value
                words_frequencies[word][sensor] += 1

    # Conto quante parole compaiono in ogni riga
    n_words_4_sensor = len(next(iter(next(iter(word_file_dict.values())).values()))) - 2 # -2 perchè c'è anche mean ed std
    return words_frequencies, n_words_4_sensor

# WRAPPER DA RICHIAMARE
# Prende un file di parole (.wrd) e ne genera le metriche TF E TF-IDF ritornando un oggetto di tipo GestureMetrics
def words_2_metrics(dataset_preprocess_unit, word_file, data_components, preprocess_unit, n_sensors, alfabeto):
    gesture_metrics = GestureMetrics(word_file, n_sensors, alfabeto)
    words_frequencies, words_4_sensors = calculate_words_frequencies(word_file, n_sensors, alfabeto)
    dataset_preprocess_unit.generate_IDF(preprocess_unit) # Genero IDF (e aggiorno dizionario nella preprocess_unit)
    words_4_sensors *= len(data_components) # Aggiorno tenendo conto che ho più componenti (asse X,Y,Z,W)
    for word, sensors in words_frequencies.items():
        for sensor, cont in sensors.items():
            sensor = str(sensor)
            tf = 0
            tfidf = 0
            if cont != 0:
                tf = format_float(cont / words_4_sensors)
                tfidf = format_float(tf * preprocess_unit.IDF[word])
            gesture_metrics.TF[word][sensor] = tf
            gesture_metrics.TFIDF[word][sensor] = tfidf
    return gesture_metrics

# WRAPPER DA RICHIAMARE
# Salva i vettori (metriche tf e tf-idf) del dict di metriche di un file .wrd dentro due file chiamati tf_vectors_tX e tfidf_vectors_fX
def write_metrics_as_json(gesture_metrics_dict, words_file_name, output_directory):
    # Salvo vettore TF
    output_path_tf = output_directory + "tf_vectors_f" + words_file_name.split(".")[0] + ".txt"
    with open(output_path_tf, "w") as output_file:
        json.dump(gesture_metrics_dict.TF, output_file, indent=4)
    # SALVO VETTORE TF-IDF    
    output_path_tfidf = output_directory + "tfidf_vectors_f" + words_file_name.split(".")[0] + ".txt"
    with open(output_path_tfidf, "w") as output_file:
        json.dump(gesture_metrics_dict.TFIDF, output_file, indent=4)

# WRAPPER DA RICHIAMARE
# Metodo per mostrare graficamente il contenuto di una unità di Gesture Metrics
def show_gesture_metrics(gesture_metrics):
    print("File gesto:", gesture_metrics.document)
    print("Numero sensori:", gesture_metrics.n_sensors)
    
    # PRINT TF
    cont = 0
    # Conto quanti diversi da 0
    for _, sensors in gesture_metrics.TF.items():
        sensor_has_a_value_non0 = False
        for sensor, value in sensors.items():
            if value != 0:
                sensor_has_a_value_non0 = True
                break
        if sensor_has_a_value_non0:
            cont += 1
    print("TF: (" + str(cont) + "}")
    for word, sensors in gesture_metrics.TF.items():
        sensor_has_a_value_non0 = False
        for _, value in sensors.items():
            if value != 0:
                sensor_has_a_value_non0 = True
                break
        if sensor_has_a_value_non0:
            print("word:",word, "tf:", sensors)
        
    # PRINT TF-IDF
    cont = 0
    # Conto quanti diversi da 0
    for _, sensors in gesture_metrics.TFIDF.items():
        sensor_has_a_value_non0 = False
        for sensor, value in sensors.items():
            if value != 0:
                sensor_has_a_value_non0 = True
                break
        if sensor_has_a_value_non0:
            cont += 1
    print("TF-IDF: (" + str(cont) + "}")
    for word, sensors in gesture_metrics.TFIDF.items():
        sensor_has_a_value_non0 = False
        for _, value in sensors.items():
            if value != 0:
                sensor_has_a_value_non0 = True
                break
        if sensor_has_a_value_non0:
            print("word:",word, "tf-idf:", sensors)



# Wrapper esterno
def gesture_words_preprocessing(preprocessing_settings):
    # Mi estraggo le stats richieste
    dizionario_gesti_parole = preprocessing_settings['dizionario_gesti_parole']
    data_components = preprocessing_settings['data_components']
    words_files = preprocessing_settings['words_files']
    n_sensori = preprocessing_settings['n_sensori']
    alfabeto = preprocessing_settings['alfabeto']
    path_words_directory = preprocessing_settings['path_words_directory']
    path_vectors_directory = preprocessing_settings['path_vectors_directory']

    gestures_metrics = []
    dataset_preprocess_unit = DatasetWordsPreProcessing(len(words_files), alfabeto)
    for words_file in words_files:
        words_preprocessing(dataset_preprocess_unit, dizionario_gesti_parole[words_file.split(".")[0]], words_file)

    for words_file in words_files:
        preprocess_unit = GestureWordsPreProcessing(words_file, alfabeto) # mi creo preprocess_unit x questo file
        gesture_metrics = words_2_metrics(dataset_preprocess_unit, path_words_directory + words_file, data_components, preprocess_unit, n_sensori, alfabeto)
        gestures_metrics.append(gesture_metrics)
        # Salvo su file il dizionario completo
        write_metrics_as_json(gesture_metrics, words_file, path_vectors_directory)

    return gestures_metrics