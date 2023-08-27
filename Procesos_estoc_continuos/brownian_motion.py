from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_dB(n: int, Ts: float=1 , random_state: Optional[int] = None) -> np.ndarray:
    """
    Recibe:
    n: número deseado de incrementos.
    Ts: varianza=tiempo de muestreo (t-s). Por defecto =1
    Optional random_state to reproduce results.
    Retorna:
    dW: Vector con n incrementos de un MBEU
    """
    np.random.seed(random_state)
    sample=np.random.normal(0.0, Ts**(1/2), size=n-1)#Ts es la varianza, n-1 porque agrego luego condición inicial
    dB=np.insert(sample,0,0) #Agrego los estados iniciales como primer vector columna
    return dB

def get_B(n: int, Ts: float=1,random_state: Optional[int] = None) -> np.ndarray:
    """
    Recibe:
    n: tamaño de muestras deseadas.
    d: número de trayectorias muestreadas a intervalos Ts unidades de tiempo.
    Retorna:
    Matriz dxn con cada fila de la matriz como el vector de sumas acumuladas de los incrementos.
    """
    dB = get_dB(n,Ts=Ts,random_state=random_state)  #Cambiar por dB y B
    
    return np.cumsum(dB) #Cum sum retorna un vector y cada entrada se convierte en la suma acumulada

def quadratic_variation(B):
    """Devuelve la matriz (o el vector en caso de dimensión 1) con las variaciones cuadráticas de cada fila (mov browniano) de W"""
    if B.ndim==1:
        qv=np.cumsum(np.power(np.diff(B,prepend=0),2))
    else: 
        qv= np.cumsum(np.power(np.diff(B,axis=1,prepend=0),2),axis=1)
    return qv


#Métodos auxiliares para matriz
def _get_correlated_dB(dB: np.ndarray, rho: float, Ts=1, random_state: Optional[int] = None) -> np.ndarray:
    """
    Recibe:
    Incrementos de browniano.
    rho: coeficiente de correlación.
    Ts: paso de muestreo.
    Retorna:
    d vectores de incrementos de procesos brownianos correlacionado.
    Sample correlated discrete Brownian increments to given increments dW.
    """
    dB2 = get_dB(len(dB), Ts=Ts, random_state=random_state)  # genera las d listas de incrementos.
    if np.array_equal(dB2, dB):
        # dB no puede ser igual a dB2.
        raise ValueError("Brownian Increment error, try choosing different random state.")
    return rho * dB + np.sqrt(1 - rho ** 2) * dB2

def _vary_random_state_i(random_state: Optional[int], i: int) -> Optional[int]:
    """Add i to random_state if is int, else return None."""
    return random_state if random_state is None else random_state + i

def _get_previous_dB(
    dWs: list[np.ndarray], i: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Elige un proceso dB ya generado.
    """
    random_proc_idx = rng.choice(i)
    return dWs[random_proc_idx]

def get_dB_matrix(
    n: int,
    d: int,
    Ts:Optional[int]=1,
    rho: Optional[float] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Matriz con d Brownianos dB, cada uno compuesto por n muestras de incrementos discretos.
    Cada fila será un proceso, por lo que resulta una matriz (dxn).
    Puede incluir correlación rho con la que se simularán los procesos a partir de alguno de los anteriormente simulados 
    (proceso elegido por la función _get)
    Retorna:

    """
    rng = np.random.default_rng(random_state) #Generador
    dBs: list[np.ndarray] = []
    for i in range(d):
        random_state_i = _vary_random_state_i(random_state, i)
        if i == 0 or rho is None:
            dB_i = get_dB(n, Ts=Ts,random_state=random_state_i)
        else:
            dB_previous = _get_previous_dB(dBs, i, rng)
            dB_i = _get_correlated_dB(dB_previous, rho, Ts=Ts,random_state=random_state_i)
        dBs.append(dB_i)
    return np.asarray(dBs) #d vectores fila de incrementos.

def get_B_matrix(dBs):
    return np.cumsum(dBs,axis=1)#Por filas