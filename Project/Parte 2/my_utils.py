import os
import glob
# Per quantizzazione gaussiana
from scipy import stats
import scipy
from numpy import random
from numpy.linalg import norm
# Fine imports

interval_utils_stats = {
    "mu": 0,
    "sigma": 0.25
}

# Metodo che calcola la distanza euclidea fra ogni riga di una matrice ed un vettore
def matrix_vector_euclidean_distance(matrix, vector):
    return norm(matrix - vector, axis=1, ord=2)

# Formatta il numero mettendo al massimo n_floats numeri dopo la virgola
def format_float(number, n_floats=5):
    format_query = "{:0."+str(n_floats)+"f}"
    return float(format_query.format(number))

# Classe che permette di gestire semplicemente un alfabeto e di memorizzarlo in modo efficiente
class Alfabeto:
    def __init__(self):
        self.n_simboli = 0
        self.n_parole = 0
        self.simboli = dict()
        self.parole = set()
    
    def setSimboli(self, simboli):
        self.n_simboli = len(simboli)
        self.simboli = simboli
    
    def addParola(self, parola):
        if parola not in self.parole:
            self.n_parole += 1
            self.parole.add(parola)

# generate Gaussian function
def gauss(x):
    return stats.norm.pdf(x, interval_utils_stats["mu"], interval_utils_stats["sigma"])

# Quantizza e ritorna i simboli dell'alfabeto sotto forma di dizionario -> {centro(parola): (min_val, max_val)}
# r indica il numero di simboli
def quanticize_interval(options):
    dim = (options["interval_max"] - options["interval_min"]) / (2 * float(options["r"])) # 2 è la dimensione -1 -> +1, ma devo dividere per 2r, quindi rimane 1/r
    guassian = random.normal(loc=interval_utils_stats["mu"], scale=interval_utils_stats["sigma"])
    results = [] # Simboli
    normalize_gauss, _ = scipy.integrate.quad(gauss, options["interval_min"], options["interval_max"]) #valore, errore stimato
    min_val = interval_utils_stats["mu"] # parto dal punto centrale da cui eseguo il conto
    # Mi calcolo la lunghezza degli intervalli gaussiani usando le 2*r suddivisioni dell'intervallo base
    # Ottimizzazione, a sx sarà uguale che a dx, (ma invertito di segno) faccio solo verso destra
    for _ in range(options['r']):
        next_val = min_val + dim
        # Gaussian Integral between current and next all divided by integral between -1 and 1 (to normalize result) and multiplied by 2
        result, _ = scipy.integrate.quad(gauss, min_val, next_val) # valore, errore stimato
        normalized_result = 2 * result / normalize_gauss
        results.append((min_val, next_val, normalized_result))
        min_val = next_val
    # A partire dalle lunghezze degli intervalli, mi costruisco il centro di questi facendo punto_iniziale_intervallo + lunghezza / 2
    # X evitare scritte enormi, taglio dopo la 3 cifra decimale
    alfabeto = Alfabeto()
    simboli_alfabeto = {} # Conterrà l'alfabeto dei simboli (simboli = centro degli intervalli gaussiani)
    current = interval_utils_stats["mu"] # parto dal punto centrale da cui eseguo il conto
    lim = len(results)
    cont = 1
    for min_r, max_r, length_r in results:
        min_r = format_float(min_r)
        max_r = format_float(max_r)
        length_r = format_float(length_r)
        middle = format_float(current + length_r / 2)
        if cont == lim: # se sono al limite forzo il valore x evitare problemi
            simboli_alfabeto[str(middle)] = (format_float(current), format_float(options["interval_max"]))
            # Dato che ho calcolato solo i valori x i numeri positivi (quelli negativi sono uguali ma con il -)
            # Copio il centro appena trovato ma con le coordinate opposte
            simboli_alfabeto[str(-middle)] = (format_float(-current), format_float(options["interval_min"]))
        else:
            simboli_alfabeto[str(middle)] = (format_float(current), format_float(current + length_r))
            # Dato che ho calcolato solo i valori x i numeri positivi (quelli negativi sono uguali ma con il -)
            # Copio il centro appena trovato ma con le coordinate opposte
            simboli_alfabeto[str(-middle)] = (format_float(-current), format_float(-(current + length_r)))
        current += length_r
        cont += 1
    alfabeto.setSimboli(simboli_alfabeto)
    return alfabeto

# Ritorna l'ultimo elemento del path (il nome del file, così so come si chiamano)
def get_files_name(path, ext_filter="*"):
    return [file_path.split("/")[-1] for file_path in sorted(filter(os.path.isfile, glob.glob(path + "*." + ext_filter)))]