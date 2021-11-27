from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd
from random import randrange
import math
# Fine imports

# Metodo che collega le labels ottenute dal clustering con le gestures, nella coppia (gesture, label)
def cluster_labels_2_pandas(cluster_labels, gestures):
    point_cluster_labels = []
    for label, gesture in zip(cluster_labels, gestures):
        data = {
            'Gesture': gesture,
            'Label': label
        }
        point_cluster_labels.insert(0, data)
    # manteniamo ordine del pandas df 
    point_cluster_labels.reverse()
    return point_cluster_labels

# Metodo che trasforma le top_p_gestures in un array numpy
def top_p_gestures_2_numpy(gestures, columns):
    # Mi prendo il numero di documenti (faccio così anche se so che sono 60)
    _, (_, values) = next(enumerate(gestures[0].items()))
    rows = len(values)
    # Mi creo una matrice con tante righe (quanti sono le gesture) e tante colonne quante sono le semantiche latenti (inizializzo a 0)
    numpy_top_p_gestures = np.zeros((rows, columns))
    # Leggo tutti i dizionari nell'array dei dizionari
    for column, gesture in zip(range(columns), gestures):
        # Mi tiro fuori i valori da questi dizionari (array di coppie (gesto, valore))
        _, (_, values) = next(enumerate(gesture.items()))
        # Ciclo per ognuno di queste coppie
        for row, value in values:
            # Inserisco il valore nel gesto, nella posizione dell'attuale semantica latente (parte da 1 - 2... +importante a meno importante)
            # -1 perchè i gesti partono da 1, qua lavoriamo ad indice che parte da 0
            numpy_top_p_gestures[int(row) - 1][column] = value
    # Ritorno in formato numpy
    return numpy_top_p_gestures
    
##################################################
#################### TASK4  A ####################
##################################################

# Partiziona lo spazio in regioni, le inizializza inserendo punti random nei vari gruppi
# Eseguendo una ricerca greedy prova a ridurre la dimensione degli spazi utilizzati dai vari gruppi
# Esegue X iterazioni, nella quale cerca se almeno un elemento può essere spostato in un qualche altro gruppo (può avere gruppi vuoti e può portare a soluzioni sub-ottimali)
class MyPartitioner:
    def __init__(self, n_groups, max_iterations=10):
        self.n_groups = n_groups
        self.max_iterations = max_iterations
        self.is_fitted = False
        self.data_dimension = 0 # Dimensionalità dei dati
        self.groups_space_dimension = 0 # 'Spazio' utilizzato dai gruppi (dipende dai valori delle coordinate)
        self.groups = {} # Gruppi dei dati
        for i in range(n_groups):
            # Ogni gruppo ha una dimensione, alcune condizioni (usate x sapere se cercare un elemento al suo interno o meno) ed i suoi elementi
            self.groups[i] = {
                'size': 0,
                'conditions': [],
                'elements': []
            }

    # Metodo che imposta le condizioni x un raggruppamento dato e ci fornisce la 'dimensione' totale utilizzata da questo
    def generate_groups_metrics(self, groups):
        # Total dimension è la dimensione totale dei gruppi, vogliamo minimizzarla
        total_dimension = 0
        # Per ogni gruppo mi calcolo la sua dimensione
        for _, group in groups.items():
            dimensions = []
            group_size = 1 # Inizializzo a 1, che è elemento neutro moltiplicazione
            # Mi cerco i punti minimi e massimi per ogni dimensione dei dati del gruppo
            for index in range(self.data_dimension):
                g_min = math.inf
                g_max = -math.inf
                for group_point in group['elements']:
                    if group_point[index] > g_max:
                        g_max = group_point[index]
                    if group_point[index] < g_min:
                        g_min = group_point[index]
                # Inserisco la coppia nella lista contenente i punti min e max di ogni dimensione
                dimensions.append((g_min, g_max))
                # Calcolo la dimensione del gruppo come dimensione lato * ogni altra dimensione dei lati
                if g_max != g_min: # se sono uguali è xkè ho solo 1 punto e allora lascio come dimensione 1
                    # 1 + serve a far si che se il gruppo è vuoto occupa 0, se ha un elemento occupa 1, se di più, indifferentemente dalla scala occupano più di 1
                    group_size *= (1 + np.abs(g_max - g_min))
            # Le condizioni x rientrare in questo gruppo dipendono dai vertici della figura che rappresenta il gruppo
            group['conditions'] = dimensions
            # Ora posso calcolarmi la dimensione della figura che rappresenta i punti
            if len(group['elements']) < 1: # Se ho zero punti, lo spazio occupato è 0
                group_size = 0
            group['size'] = group_size
            total_dimension += group_size
        return total_dimension


    def fit(self, data):
        # Se non abbiamo dati, mi fermo
        if len(data) == 0:
            print("MyPartitioner -> fit - Size error, no data points given")
            return
        self.data_dimension = len(data[0]) # Imposto la dimensione dei dati
        # Inizializzo assegnando ogni dato ad un gruppo a caso
        for elem in data:
            elem = list(elem)
            random_index = randrange(0, self.n_groups)
            self.groups[random_index]['elements'].append(elem)

        # Calcolo le metriche per il raggruppamento di partenza
        self.groups_space_dimension = self.generate_groups_metrics(self.groups)

        # Parte iterativa, cerco di minimizzare l'area dei gruppi
        for iteration in range(self.max_iterations):
            # Creo un nuovo raggruppamento a partire da quello attuale
            groups = dict(self.groups)
            # Mi ricordo se un elemento è stato swappato in questa iterazione
            no_one_swapped = True
            # Cerco se c'è un elemento che può essere spostato in un altro gruppo
            for data_point in data:
                data_point = list(data_point)
                starting_group_index = 0
                # 1 - Cerco in che gruppo fa parte e lo rimuovo da esso
                for index, group in groups.items():
                    if data_point in group['elements']:
                        group['elements'].remove(data_point)
                        starting_group_index = index
                        break # Esco dal ciclo di ricerca del data_point
                # 2 - Provo ad inserire il punto in un gruppo che mi riduce la dimensione del raggruppamento totale (se presente)
                for index, group in groups.items():
                    # Aggiungo il punto al gruppo e vedo come cambia la dimensione totale
                    group['elements'].append(data_point)
                    space_dimension = self.generate_groups_metrics(groups)
                    # Se è meglio di prima, aggiorno lo stato dei gruppi e passo all'iterazione successiva
                    if space_dimension < self.groups_space_dimension:
                        self.groups = dict(groups)
                        self.groups_space_dimension = space_dimension
                        no_one_swapped = False
                        break
                    # Se è peggio o uguale a prima, faccio rollback
                    else:
                        group['elements'].remove(data_point)
                # 3 - Se ho spostato il punto, passo alla prossima iterazione (sposto al max 1 punto ad iterazione)
                if not no_one_swapped:
                    break
                else: # Altrimenti ritorno allo stato di prima (questo punto stava bene nel gruppo iniziale) e cerco con gli altri punti rimanenti
                    groups[starting_group_index]['elements'].append(data_point)
            # 4 - Se nessun punto nell'iterazione attuale è stato spostato, mi fermo xkè sono in un minimo (locale/globale)
            if no_one_swapped:
                break
        #

        # Per sicurezza eseguo aggiornamento metriche e condizioni x il raggruppamento finale generato
        self.groups_space_dimension = self.generate_groups_metrics(self.groups)
        # Impostiamo is_fitted a true, così possiamo usarlo x cercare elementi nei gruppi
        self.is_fitted = True
        return self

    # Metodo helper, mi dice se un punto PUò far parte di un dato gruppo (se soddisfa le condizioni di appartenenza al gruppo)
    def can_contain(self, data_point, group):
        might_contain = True
        # Cerco fra tutte le condizioni, e se ne trovo anche solo una che non soddisfo allora il gruppo SICURAMENTE non mi conterrà
        for data_point_value, condition in zip(data_point, group['conditions']):
            min_cond, max_cond = condition
            if data_point_value < min_cond or data_point_value > max_cond:
                might_contain = False
                break
        # Ritornare true NON significa che appartengo al gruppo (ma solo che è possibile che appartengo al gruppo)
        return might_contain

    # Cerco un elemento dentro i gruppi
    def predict(self, data):
        if not self.is_fitted:
            print("MyPartitioner -> predict - Cannot predict if not fitted")
            return None
        groups_predicted = []
        # Controllo che tutti i dati da 'predirre' abbiano la dimensione corretta 
        for data_point in data:
            if len(data_point) != self.data_dimension:
                print("MyPartitioner -> predict - One or more data points has size:", len(data_point), "expected size:", self.data_dimension)
                return None
            data_point = list(data_point)
            # Cerco (possibilmente) in tutti i gruppi
            group_index = None
            for index, group in self.groups.items():
                # Cerco nel gruppo solo se il dato soddisfa le condizioni del gruppo, se no sicuramente non è al suo interno
                if self.can_contain(data_point, group):
                    # Se l'elemento è presente finisco di ricercare (questo punto), passo ai successivi
                    for element in group['elements']:
                        if data_point == element:
                            group_index = index
                            break
                    # Come prima, se group_index è != None vuol dire che ho trovato il gruppo, e anche se è possibile, non mi interessa tornare quali altri gruppi mi contengono
                    if group_index != None:
                        break
            # Aggiungo la predizione al punto (None se il punto non è presente in alcun gruppo)
            groups_predicted.append((data_point, group_index))
        # Ritorno la lista di predizioni in formato [(punto, gruppo), ...]
        return groups_predicted

##################################################
#################### TASK4  B ####################
##################################################
# Semplice k-means
class SimpleKMeans:
    # k numero di clusters voluto
    # min_tolerance è il cambiamento minimo da una iterazione alla prossima che basta x fermarmi (0 - 1 in percentuale)
    # max_iterations è il numero massimo di iterazioni eseguite
    def __init__(self, k=5, min_tolerance=0.1, max_iterations=10):
        self.k = k
        self.min_tolerance = min_tolerance
        self.max_iterations = max_iterations
        self.centroids = {}
        self.is_fitted = False

    # Metodo che esegue il fit, generando i vari centroidi a partire dai dati
    def fit(self, data):
        # Se data ha meno punti del numero di cluster voluti ritorniamo
        if len(data) < self.k:
            print("SimpleKMeans -> fit - Size error, expecting", self.k, "clusters, but data has only", len(data), "points")
            return
        # 1 - SCELGO K PUNTI COME CENTROIDI INIZIALI
        random_indices = np.random.choice(len(data), size=self.k, replace=False) # replace=False serve x evitare doppioni
        # Reinizializzo i centroidi (in caso facessi più fit con lo stesso oggetto)
        # Inizializzo i centroidi ai k random elementi scelti
        self.centroids = {}
        for index, data_point in zip(range(self.k), data[random_indices, :]):
            self.centroids[index] = data_point

        # Itero un numero max di volte
        for iteration in range(self.max_iterations):
            # Mi Creo la lista di clusters di questa iterazione, ognuno inizializzato a lista vuota
            current_clusters = {}
            for i in range(self.k):
                current_clusters[i] = []

            # Ciclo una volta su ogni dato
            for data_point in data:
                # 2 - CALCOLO LA DISTANZA EUCLIDEA FRA IL PUNTO ED I CENTROIDI DEI CLUSTERS
                distances = [np.linalg.norm(data_point - centroid) for _, centroid in self.centroids.items()]
                # 3- ASSEGNO IL PUNTO AL CLUSTER PIù VICINO
                # Cerco qual'è il cluster con distanza minima
                nearest_cluster_index = distances.index(min(distances))
                # Inserisco l'elemento nel cluster più vicino
                current_clusters[nearest_cluster_index].append(data_point)
            
            # 4 - GENERO I NUOVI CENTROIDI COME PUNTO MEDIO DEI NUOVI CLUSTERS
            # Mi salvo i vecchi centroidi x vedere differenza in questa iterazione
            old_centroids = dict(self.centroids)
            # Aggiorno i centroidi attuali facendo una media delle posizioni degli elementi nei clusters
            for index, cluster in current_clusters.items():
                self.centroids[index] = np.average(cluster, axis=0)

            # 5 - CALCOLO LA DIFFERENZA FRA I NUOVI E I VECCHI CLUSTERS, MI FERMO SE è TOLLERABILE
            # Calcolo la differenza come distanza fra il centroide nuovo e quello vecchio (in percentuale), faccio somma x ogni cluster
            difference = 0
            for (_, new_centroid), (_, old_centroid) in zip(self.centroids.items(), old_centroids.items()):
                difference += np.abs(np.sum((new_centroid-old_centroid) / (old_centroid + 0.001))) # la somma è xkè ogni tanto da divisione x zero
            # Se la differenza è minore della tolleranza allora posso fermarmi
            if difference <= self.min_tolerance:
                break
        # Una volta finito il fit, posso essere usato per predizioni
        self.is_fitted = True
        return

    # Metodo che permette di predire un dato per volta
    def predict(self, data):
        # Se non sono stato fittato do errore
        if not self.is_fitted:
            print("SimpleKMeans -> predict - Cannot predict if not fitted")
            return
        classification = []
        for data_point in data:
            # Calcolo le distanze fra il punto ed i centroidi
            distances = [np.linalg.norm(data_point - centroid) for _, centroid in self.centroids.items()]
            # Prendo il cluster più vicino (in questo caso la label corrisponde all'indice del cluster stesso)
            classification.append(distances.index(min(distances)))
        return classification

##################################################
#################### TASK4  C ####################
##################################################

# Metodo che esegue il clustering spettrale sfruttando la matrice di laplace
def spectral_clustering(similarity_matrix, n_clusters, offset=0):
    # Since we have a similarity matrix we don't need to generate the laplacian matrix (it is generated by the SpectralClustering method)
    # Otherwise we would have to calculate like this: LM = D - S
    # Where LM is the Laplacian Matrix, S is the similarity matrix and D is the Diagonal matrix
    # Affinity = 'precomputed' utilizza i dati come se fossero una matrice di similarità e crea la matrice di laplace di conseguenza
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize', random_state=0).fit(similarity_matrix)
    return clustering.labels_ + offset

# Metodo che esegue clustering per vari valori di n_clusters (usao per eseguire una analisi)
def spectral_clustering_analyze(similarity_matrix, min_clusters=1, max_clusters=10):
    results = []
    label_off = 0
    min_clusters = max(1, min_clusters)
    max_cluster = min(max_clusters, similarity_matrix.shape[0])
    for n_clusters in range(min_clusters, max_clusters):
        clusters = spectral_clustering(similarity_matrix, n_clusters, offset=label_off)
        label_off = max(clusters) + 1
        data = {
            "clusters": n_clusters,
            "labels": clusters.tolist()
        }
        results.insert(0, data)
    results.reverse()
    return results

# Trasforma i risultati ottenuti dall'analisi in risultati pandas
def spectral_analisys_results_2_pandas(spectral_results, df):
    # Extract only the labels
    spectral_labels = [val['labels'] for val in spectral_results]
    # Insert into dataframe using for columns the gesture name, and for rows the n_cluster used in that clustering session
    spectral_data_df = pd.DataFrame(spectral_labels, columns=df.columns, index=range(1, len(spectral_labels) + 1))
    # Return Each row, the gesture label with column n_clusters
    return spectral_data_df.transpose()

# Splitta i dati ottenuti dall'analisi spettrale nelle 3 componenti per il sankey plot (source, target, value)
def spectral_analisys_results_pandas_2_split(spectral_results):
    sources, targets, values = [], [], []
    data = {}
    # Get time #els change in clusters
    for column in range (1, len(spectral_results.columns)):
        row = spectral_results[column]
        row_next = spectral_results[column + 1]
        # Check element with n_clusters and his behaviour with n_clusters+1
        for val, val_next in zip(row, row_next):
            if (val, val_next) in data:
                data[(val, val_next)] += 1
            else:
                data[(val, val_next)] = 1
    # Traduce the data dictionary into his 3 elems 
    for (source, target), value in data.items():
        sources.insert(0, source)
        targets.insert(0, target)
        values.insert(0, value)
    # Reverse to keep the time order
    sources.reverse()
    targets.reverse()
    values.reverse()
    return sources, targets, values