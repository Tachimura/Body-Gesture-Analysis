import sys
import matplotlib.pylab as plt

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
    
# Permette di stampare una barra di completamento
def drawProgressBar(percent, barLen = 20):
    # percent float from 0 to 1. 
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()

def show_heatmap(numpy_data, axes_title, fig, ax, y_min_val=0, y_max_val=1, color_map='gray'):
    myfigmap = ax.pcolor(numpy_data, cmap=color_map, vmin=y_min_val, vmax=y_max_val)
    fig.colorbar(myfigmap, ax=ax)
    # Imposto i ticks degli assi a soli numeri interi
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.xaxis.get_major_locator().set_params(integer=True)
    # label sugli assi
    ax.set_xlabel('words')
    ax.set_ylabel('sensors')
    ax.set_title(axes_title)