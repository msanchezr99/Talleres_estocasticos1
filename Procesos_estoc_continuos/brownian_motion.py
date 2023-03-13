from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    

