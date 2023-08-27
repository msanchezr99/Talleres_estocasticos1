from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_dB(n: int, Ts: float=1 , d: int =1, random_state: Optional[int] = None) -> np.ndarray:
    """
    Recibe:
    n: número deseado de incrementos.
    Ts: varianza=tiempo de muestreo (t-s). Por defecto =1
    d: número de procesos independientes buscadas. Por defecto =1
    Optional random_state to reproduce results.
    Retorna:
    dW: Una matriz de dimensiones dxn, cada una de las d filas será un vector de n muestreos de 
    una normal, es decir, una matriz cuyas filas son incrementos de un MBEU
    """
    np.random.seed(random_state)
    sample=np.random.normal(0.0, Ts**(1/2), (d,n-1))#Ts es la varianza, d-1 porque al final agrego condición inicial
    B_0=np.zeros((d,1)) #Fila de ceros. Cada proceso (columna) se inicializa en 0
    dB=np.concatenate((B_0,sample),axis=1) #Agrego los estados iniciales como primer vector columna
    return dB

def get_B(n: int, Ts: float=1, d:int = 1,random_state: Optional[int] = None) -> np.ndarray:
    """
    Recibe:
    n: tamaño de muestras deseadas.
    d: número de trayectorias muestreadas a intervalos Ts unidades de tiempo.
    Retorna:
    Matriz dxn con cada fila de la matriz como el vector de sumas acumuladas de los incrementos.
    """
    dB = get_dB(n,Ts=Ts,d=d,random_state=random_state)  #Cambiar por dB y B
    
    return np.cumsum(dB,axis=1) #Cum sum retorna un vector y cada entrada se convierte en la suma acumulada

def quadratic_variation(B):
    """Devuelve la matriz con las variaciones cuadráticas de cada fila (mov browniano) de W"""
    return np.cumsum(np.power(np.diff(B,axis=1,prepend=0),2),axis=1)




###################3
###########Editar funciones, cambiar por las del notebook (comentadas) que tienen agregado poder crear más de un solo vector
def get_dW(T: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Sample T times from a normal distribution,
    to simulate discrete increments (dW) of a Brownian Motion.
    Optional random_state to reproduce results.
    """
    np.random.seed(random_state)
    sample=np.random.normal(0.0, 1.0, T-1)
    dW=np.insert(sample,0,0)
    return dW

def get_W(T: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Simula un movimiento Browuniano sampleado en unidades de tiempo.
    Returns the cumulative sum
    """
    dW = get_dW(T, random_state)
    return np.cumsum(dW) #Cum sum retorna un vector y cada entrada se convierte en la suma acumulada



#Con correlación:

def _get_correlated_dW(dW: np.ndarray, rho: float, random_state: Optional[int] = None) -> np.ndarray:
    """
    Sample correlated discrete Brownian increments to given increments dW.
    """
    dW2 = get_dW(len(dW), random_state=random_state)  # generate Brownian icrements.
    if np.array_equal(dW2, dW):
        # dW cannot be equal to dW2.
        raise ValueError("Brownian Increment error, try choosing different random state.")
    return rho * dW + np.sqrt(1 - rho ** 2) * dW2

#Matricial
#Métodos auxiliares para matriz:

def _get_random_state_i(random_state: Optional[int], i: int) -> Optional[int]:
    """Add i to random_state is is int, else return None."""
    return random_state if random_state is None else random_state + i


def _get_corr_ref_dW(
    dWs: list[np.ndarray], i: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Choose randomly a process (dW) the from the
    already generated processes (dWs).
    """
    random_proc_idx = rng.choice(i)
    return dWs[random_proc_idx]

def get_corr_dW_matrix(
    T: int,
    n_procs: int,
    rho: Optional[float] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Matriz con n_procs Brownianos dW, cada uno compuesto por T muestras de incrementos discretos.
    Cada columna será un proceso, por lo que resulta una matriz (T, n_procs).
    rho es la correlación con la que se simularán los procesos a partir de alguno de los anteriormente simulados (proceso elegido por la función _get)

    """
    rng = np.random.default_rng(random_state)
    dWs: list[np.ndarray] = []
    for i in range(n_procs):
        random_state_i = _get_random_state_i(random_state, i)
        if i == 0 or rho is None:
            dW_i = get_dW(T, random_state=random_state_i)
        else:
            dW_corr_ref = _get_corr_ref_dW(dWs, i, rng)
            dW_i = _get_correlated_dW(dW_corr_ref, rho, random_state_i)
        dWs.append(dW_i)
    return np.asarray(dWs).T
    

