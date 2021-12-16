import pandas as pd
import numpy as np
import json
    
# METODI DI GESTIONE DEI SIMBOLI E DELLE PAROLE DELL'ALFABETO

# Ritorna il centro in cui cade questo valore
def extract_word(array, symbols):
    # Inizio dalla parola vuota
    word = ""
    for element in array:
        # Per ogni elemento nell'array estraggo il simbolo (centro) che equivale a quell'elemento
        word += ";" + extract_symbol(element, symbols)
    # Ritorno la parola formata dalla concatenazione dei simboli
    return word[1:] #tolgo il primo carattere perchè conterrà il ;

def extract_symbol(element, symbols):
    element_symbol = 0
    for symbol, (off1,  off2) in symbols.items():
        min_value = min(off1, off2)
        max_value = max(off1, off2)
        if element >= min_value and element <= max_value:
            element_symbol = symbol
            break
    return str(element_symbol)

##########################
##########################
###  METODO EURISTICO  ###
##########################
##########################
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
##########################

def extract_window(array, start_position, window_size):
    window = array[start_position : (start_position + window_size)]
    if len(window) != window_size: #Se non è size corretto, ritorno la versione con padding
        return extract_padded_window(array, window, start_position, window_size)
    return window

# Wrapper da richiamare
def associate_data_to_symbols(file_name, data, alfabeto, window_options, add_to_alfabet=True):
    window_size, window_shift = window_options
    sensor_id = 0
    data_to_symbols = []
    for row in data.to_numpy():
        current_position = 0
        while current_position < len(row):
            # IDX => file_name, sensor_id, istante_temporale
            idx = (file_name, sensor_id, current_position)
            # Estraggo una finestra
            window = extract_window(row, current_position, window_size)
            # Converto gli elementi nella finestra nella parola dell'alfabeto (WIN)
            win = extract_word(window, alfabeto.simboli)
            data_to_symbols.insert(0, (idx, win))
            # Sposto la posizione della finestra
            current_position += window_shift
        # Cambia riga -> cambia sensore
        sensor_id +=1
    # Faccio si che sia ordinato in base alle righe lette (insert in 0 fa si che le ultime siano le prime e bisogna invertire)
    data_to_symbols.reverse()
    return data_to_symbols

#############################################################################
#############################################################################
def normalize_data(data, min_val, max_val):
    normalized_df = data.copy()
    min_x = normalized_df.min().min()
    max_x = normalized_df.max().max()
    # normalize x in (a,b) = (b - a) * (x-min_x / max_x-min_x) + a
    [normalized_df[col].update((max_val - min_val) * ((normalized_df[col] - min_x) / (max_x - min_x)) + min_val) for col in normalized_df.columns]
    return normalized_df

def gesture_2_words(gesture_file_path, words_output_path, alfabeto, options):
    ## Prendo il nome del file
    file_splits = gesture_file_path.split("/") # splitto per il / dato che ho path assoluto
    file_name = file_splits[len(file_splits) - 1] # mi prendo l'ultimo elemento che è proprio il nome del file

    # 1 - Prendo il file di gesti normalizzato
    normalized_data = gesture_2_normalized(gesture_file_path, (options["min"], options["max"]))
    #
    # 2 - Prendo le parole del file normalizzato
    words = normalized_2_words(normalized_data, file_name, alfabeto, options)
    #
    write_output_gesturewords(words_output_path, file_name, words)

def gesture_2_normalized(gesture_file_path, normalize_vals):
    normalize_min_val, normalize_max_val = normalize_vals
    # Leggo il file di gesti
    gesture_data = pd.read_csv(gesture_file_path, header=None)
    # Normalizzo il file di gesti
    return normalize_data(gesture_data, normalize_min_val, normalize_max_val)

def normalized_2_words(normalized_data, file_name, alfabeto, options):
    # Mi calcolo le parole presenti nel file
    return associate_data_to_symbols(file_name, normalized_data, alfabeto, (options["w_size"], options["w_shift"]))

def write_output_gesturewords(output_path, file_name, words):
    words_path = output_path + file_name.split(".")[0] + ".wrd"
    words_datafile = open(words_path, "w")
    cont = 0
    for word in words:
        cont += 1
        # Se sono ultima riga evito di scrivere "\n"
        if cont == len(words):
            words_datafile.write(str(word))
        else:
            words_datafile.write(str(word) + "\n")
    words_datafile.close()


#############################################################################
#############################################################################
## FUNZIONI X VETTORIZZARE LE PAROLE
class GestureDataMetrics:
    def __init__(self, alfabeto, n_sensors):
        self.document_path = ""
        self.n_words = {}
        self.words_frequency = {}
        self.TF = {}
        self.TFIDF = {}
        self.TFIDF2 = {}
        # INIT STRUTTURE
        for sensor in range(0, n_sensors):
            self.n_words[str(sensor)] = 0
        for word in alfabeto.parole:
            self.words_frequency[word] = {}
            self.TF[word] = {}
            self.TFIDF[word] = {}
            self.TFIDF2[word] = {}
            for sensor in range(0, n_sensors):
                self.words_frequency[word][str(sensor)] = 0
                self.TF[word][str(sensor)] = 0
                self.TFIDF[word][str(sensor)] = 0
                self.TFIDF2[word][str(sensor)] = 0

    def setDocumentPath(self, document_path):
        self.document_path = document_path
    
    def addFrequency(self, word, sensor):
        self.n_words[sensor] += 1
        self.words_frequency[word][sensor] += 1

    def getWordsInSensor(self, sensor):
        return self.n_words[str(sensor)]

    def getTF(self, word, sensor):
        return self.TF[word][sensor]

    def getTFIDF(self, word, sensor):
        return self.TFIDF[word][sensor]

    def getTFIDF2(self, word, sensor):
        return self.TFIDF2[word][sensor]
        
    def updateTF(self, word, sensor, value):
        self.TF[word][sensor] = value

    def updateTFIDF(self, word, sensor, value):
        self.TFIDF[word][sensor] = value

    def updateTFIDF2(self, word, sensor, value):
        self.TFIDF2[word][sensor] = value

# Legge file di parole
def read_words_file(word_file):
    word_data = pd.read_table(word_file, header=None)[0]
    data_readable = []
    for row in word_data:
        # Dai dati tolgo parentesi, e caratteri speciali e trimmo
        x = row.replace("(", "").replace(")", ""). replace("'", "").split(",")
        info = x[0:3]
        info[0] = info[0].strip()
        info[1] = info[1].strip()
        info[2] = info[2].strip()
        word = x[3].strip()
        # Ritrasformo i dati in senso forma sensata (idx, word)
        idx = (info[0], info[1], info[2])
        data_readable.insert(0, (idx, word))
    data_readable.reverse()
    return data_readable

def words_2_vector(word_file, preprocess_unit, vector_output_path, alfabeto):
    vectorized_data = get_words_frequencies(word_file, preprocess_unit, alfabeto)
    write_vectorized_data(vectorized_data, vector_output_path)
    return vectorized_data
    
def get_words_frequencies(word_file_path, preprocess_unit, alfabeto):
    # Calcolo frequenza e parole presenti nel documento
    document_data = GestureDataMetrics(alfabeto, preprocess_unit.n_sensors)
    document_data.setDocumentPath(word_file_path)
    
    # Mi leggo il file e salvo tutte le informazioni
    for (_, sensor, _), word in read_words_file(word_file_path):
        document_data.addFrequency(word, sensor)

    # Calcolo TF per ogni parola definito:  frequenza parola nel documento / # parole nel documento
    for key, s_values in document_data.words_frequency.items():
        for sensor, value in s_values.items():
            document_data.updateTF(key, sensor, float("{:0.5f}".format(value / document_data.getWordsInSensor(sensor))))
            # Dentro preprocess_unit ci sono presenti già gli idf, quindi prendo da li
            document_data.updateTFIDF(key, sensor, float("{:0.5f}".format(document_data.getTF(key, sensor) * preprocess_unit.idfs[word])))
            document_data.updateTFIDF2(key, sensor, float("{:0.5f}".format(document_data.getTF(key, sensor) * preprocess_unit.idfs2[word])))

    return document_data

def write_vectorized_data(vectorized_data, output_directory_path):
    output_path = output_directory_path + vectorized_data.document_path.split("/")[3].split(".")[0] + "_vectors.txt"
    vectorized_data_file = open(output_path, "w")
    cont = 0
    for key, s_values in vectorized_data.words_frequency.items():
        cont += 1
        word = {}
        word["word"] = key
        word["sensors"] = {}
        for sensor, value in s_values.items():
            sensor_dict = []
            sensor_dict.insert(0, vectorized_data.getTF(key, sensor))
            sensor_dict.insert(1, vectorized_data.getTFIDF(key, sensor))
            sensor_dict.insert(2, vectorized_data.getTFIDF2(key, sensor))
            word['sensors'][sensor] = sensor_dict
        if cont == len(vectorized_data.words_frequency.items()):
            vectorized_data_file.write(str(word))
        else:
            vectorized_data_file.write(str(word) + "\n")
    vectorized_data_file.close()

def show_vectorized_data(vectorized_data):
    print("Vectorized data path:")
    print(vectorized_data.document_path)
    print("N words 4 sensors")
    print(vectorized_data.n_words)
    print("Words Frequency")
    print(vectorized_data.words_frequency)
    print("TF Data:")
    print(vectorized_data.TF)
    print("TF-IDF Data:")
    print(vectorized_data.TFIDF)
    print("TF-IDF2 Data:")
    print(vectorized_data.TFIDF2)

## VECTORIZED METRICS TO NUMPY
def metrics_2_numpy(metrics_data):
    # Conto numero di sensori (in realtà è 20, ma per rendere + generico)
    n_sensors = len(metrics_data[0]["sensors"].items())
    # Conto numero di parole (non ho passato l'alfabeto)
    n_words = len(metrics_data)
    numpy_data = np.zeros((n_sensors, n_words), dtype=float)
    cont = 1
    for data_row in metrics_data:
        for sensor, value in data_row["sensors"].items():
            # non so xkè ma tira fuori i dati al contrario, quindi x averli come li ho ordinati nel file vectors
            # devo girare, facendo n_words-cont (con cont che parte da 1 se no va fuori array)
            numpy_data[int(sensor)][n_words-cont] = float(value)
        cont += 1
    return numpy_data

## ESTRAZIONI DELLE METRICHE DAL FILE DEI VETTORI
def vectorpath_2_metrics(vector_data_path, metric):
    vector_data = pd.read_table(vector_data_path, header=None)[0]
    results = []
    for row in vector_data:
        result = {}
        row_dict = json.loads(row.replace("'", "\""))
        result["metric"] = metric
        result["word"] = row_dict["word"]
        sensors = row_dict['sensors']
        result['sensors'] = {}
        for sensor, metrics in sensors.items():
            if metric == "Value TF":
                result['sensors'][sensor] = metrics[0]
            elif metric == "Value TF-IDF":
                result['sensors'][sensor] = metrics[1]
            else: # "Value TF_IDF2"
                result['sensors'][sensor] = metrics[2]
        results.insert(0, result)
    return results

# Permette di salvare le misure dei vettori di gesti del nostro database di gesti (quelli su cui abbiamo creato alfabeto e tutto)
class MyDatabase:
    def __init__(self, database_path, gesture_vectors, metric):
        self.path = database_path
        self.metric = metric
        self.gesture_vectors = {}
        for gesture_vector in gesture_vectors:
            gesture_metrics = vectorpath_2_metrics(gesture_vector, metric)
            self.gesture_vectors[gesture_vector] = metrics_2_numpy(gesture_metrics)

def find_k_most_similar(data, dataset, k):
    similarities = []
    a_flat = np.hstack(data)
    # Calcolo similarities fra file di data e gli altri dati
    for gesture_file, metric_numpy in dataset.gesture_vectors.items():
        b_flat = np.hstack(metric_numpy)
        my_sim = my_cosine_similarity(a_flat, b_flat)
        similarities.insert(0, (gesture_file, my_sim))
    # Ordino in modo decrescente per il valore di cosine_similarity, e prendo i primi k elementi
    return sorted(similarities, key = lambda x: x[1], reverse=True)[:k]

def my_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))