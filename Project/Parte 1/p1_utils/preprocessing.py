import pandas as pd
import math

class P1DatabasePreprocessing:
    def __init__(self, n_documents, alfabeto):
        self.n_documents = n_documents
        self.words_4_document = {} #key -> word value -> lista dei documenti in cui la parola compare
        for parola in alfabeto.parole:
            self.words_4_document[parola] = set()

# Unità di preprocessing, ci aiuta a calcolare informazioni su idfs
class P1PreProcessing:
    def __init__(self, document_name, n_sensors, alfabeto):
        self.document = document_name
        self.n_sensors = n_sensors
        self.words_4_sensors = {} #key -> word value -> lista dei sensori in cui la parola compare
        self.idfs = {} #key -> word value -> log(n_documenti/#n_documenti che hanno word)
        self.idfs2 = {} #key -> word value -> log(n_sensori/#n_sensori che hanno word)
        for elem in alfabeto.parole:
            self.words_4_sensors[elem] = set()
            self.idfs[elem] = 0
            self.idfs2[elem] = 0

# Metodo per mostrare graficamente il contenuto di una unità di preprocessing
def show_preprocessing_unit(unit):
    print("Nome gesto:", unit.document)
    print("Numero documenti:", unit.n_sensors)
    cont = 0
    # Conto quanti diversi da 0
    for el, value in unit.words_4_sensors.items():
        if len(value) == 0:
            continue
        cont += 1

    print("words_4_sensors: {" + str(cont) + "}")
    for el, value in unit.words_4_sensors.items():
        if len(value) == 0:
            continue
        print("word:",el, "sensors:", value)
    
    cont = 0
    # Conto quanti diversi da 0
    for el, value in unit.idfs.items():
        if value == 0:
            continue
        cont += 1
    print("IDFS: (" + str(cont) + "}")
    for el, value in unit.idfs.items():
        if value == 0:
            continue
        print("word:",el, "idfs:", value)

    cont = 0
    # Conto quanti diversi da 0
    for el, value in unit.idfs2.items():
        if value == 0:
            continue
        cont += 1
    print("IDFS2: (" + str(cont) + "}")
    for el, value in unit.idfs2.items():
        if value == 0:
            continue
        print("word:",el, "idfs2:", value)

# Metodo wrapper per far partire il preprocessing di un insieme di file di words
def preprocessing_task2(words_files, alfabeto):
    database_pp_unit = P1DatabasePreprocessing(len(words_files), alfabeto)
    task_2_preprocess = []
    for words_file in words_files:
        preprocess_unit = preprocessing_task2_singlefile(database_pp_unit, words_file, alfabeto)
        task_2_preprocess.insert(0, preprocess_unit)

    for preprocess_unit in task_2_preprocess:
        preprocessing_subtask_idfs(database_pp_unit, preprocess_unit, alfabeto)
    return task_2_preprocess
    
# Metodo wrapper per far partire il preprocessing di un singolo file di words
def preprocessing_task2_singlefile(database_pp_unit, words_file, alfabeto):
    preprocess_unit = P1PreProcessing(words_file, 20, alfabeto)
    # Eseguo task di preprocess 
    preprocessing_subtask_words(database_pp_unit, preprocess_unit, words_file)
    preprocessing_subtask_idfs2(preprocess_unit, alfabeto)
    return preprocess_unit

# Leggo la riga di ogni singolo sensore, e per ogni sensore raccolgo i dati (un sensore = un documento)
def preprocessing_subtask_words(database_pp_unit, preprocess_unit, file_path):
    data = pd.read_table(file_path, header=None)[0]
    for row in data:
        # recupero la parola
        x = row.replace("(", "").replace(")", ""). replace("'", "").split(",")
        info = x[0:3]
        info[1] = info[1].strip() # sensore
        word = x[3].strip()
        preprocess_unit.words_4_sensors[word].add(info[1])
        database_pp_unit.words_4_document[word].add(file_path)

# L'insieme dei gesti è il dataset, il singolo gesto è il documento
# Calcolo idfs come numero di gesti (documenti) / numero di gesti (documenti) che hanno almeno una occorrenza della parola
def preprocessing_subtask_idfs(database_pp_unit, prep_unit, alfabeto):
    for word in alfabeto.parole:
        # 'aggiungiamo' due documenti: uno non contiene nessuna parola e l'altro le contiene tutte
        # Questo permette di evitare divisioni per 0 (se nessun sensore ha una parola) e x avere risultati + smooth (con correzione di laplace)
        prep_unit.idfs[word] = float("{:0.5f}".format(math.log((database_pp_unit.n_documents + 2) / (len(database_pp_unit.words_4_document[word]) + 1)))) 

# Il gesto è il dataset, i sensori sono i documenti
# Calcolo idfs2 come numero di sensori (documenti) / numero di sensori (documenti) che hanno almeno una occorrenza della parola
def preprocessing_subtask_idfs2(prep_unit, alfabeto):
    for word in alfabeto.parole:
        # 'aggiungiamo' due sensori: uno non contiene nessuna parola e l'altro le contiene tutte
        # Questo permette di evitare divisioni per 0 (se nessun sensore ha una parola) e x avere risultati + smooth (con correzione di laplace)
        prep_unit.idfs2[word] = float("{:0.5f}".format(math.log((prep_unit.n_sensors + 2) / (len(prep_unit.words_4_sensors[word]) + 1)))) 