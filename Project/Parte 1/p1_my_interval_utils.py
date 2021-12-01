# Per quantizzazione gaussiana
from scipy import stats
import scipy
from numpy import random
from p1_my_utils import Alfabeto # Contiene roba di utility nostra
from p1_gesture_utils import gesture_2_normalized
from p1_gesture_utils import extract_window
from p1_gesture_utils import extract_word

interval_utils_stats = {
    "mu": 0,
    "sigma": 0.25
}

# generate Gaussian function
def gauss(x):
    return stats.norm.pdf(x, interval_utils_stats["mu"], interval_utils_stats["sigma"])

# Quantizza e ritorna i simboli dell'alfabeto sotto forma di dizionario -> {centro(parola): (min_val, max_val)}
# r indica il numero di simboli
def quanticize_interval(options):
    dim = (options["max"] - options["min"]) / (2 * float(options["r"])) # 2 è la dimensione -1 -> +1, ma devo dividere per 2r, quindi rimane 1/r
    guassian = random.normal(loc=options["mu"], scale=options["sigma"])
    results = [] # Simboli
    normalize_gauss, _ = scipy.integrate.quad(gauss, options["min"], options["max"]) #valore, errore stimato
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
        min_r = float("{:0.5f}".format(min_r))
        max_r = float("{:0.5f}".format(max_r))
        length_r = float("{:0.5f}".format(length_r))
        middle = current + length_r / 2
        middle = float("{:0.5f}".format(middle))
        if cont == lim: # se sono al limite forzo il valore x evitare problemi
            simboli_alfabeto[str(middle)] = (current, options["max"])
            # Dato che ho calcolato solo i valori x i numeri positivi (quelli negativi sono uguali ma con il -)
            # Copio il centro appena trovato ma con le coordinate opposte
            simboli_alfabeto[str(-middle)] = (float(-current), options["min"])
        else:
            simboli_alfabeto[str(middle)] = (current, current + length_r)
            # Dato che ho calcolato solo i valori x i numeri positivi (quelli negativi sono uguali ma con il -)
            # Copio il centro appena trovato ma con le coordinate opposte
            simboli_alfabeto[str(-middle)] = (float(-current), -(current + length_r))
        current += length_r
        cont += 1
    alfabeto.setSimboli(simboli_alfabeto)
    return alfabeto

## ROBA X GENERARE LE PAROLE DELL'ALFABETO (IN BASE A COSA SI SCEGLIE POI, SARà DA SPOSTARE IN MY_INTERVAL_UTILS)
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

# WRAPPER DA RICHIAMARE ALL'ESTERNO
def generate_alphabet(documents, options):
    alfabeto = quanticize_interval(options) # Genero i simboli dell'alfabeto
    # Mi genero le parole prendendo solo quelle presenti nel dataset
    for document in documents:
        # 1 - Prendo il file di gesti normalizzato
        normalized_data = gesture_2_normalized(document, (options["min"], options["max"]))
        # Estraggo le parole e le inserisco nell'alfabeto
        generate_alphabet_words(normalized_data, alfabeto, (options["w_size"], options["w_shift"]))
    return alfabeto