from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
Convenciones:
Aunque para plotear python asume columnas como cada trayectoria. Las matrices de los métodos están formadas
por trayectorias en cada fila. Matrices B de tamaño (d x n) = (n_proc x tamaño_muestr)

El proceso de simulación en general será elegir un tiempo máximo T, un número de subintervalos s 
lo que define dt=T/s e invocar las funciones get con n=s+1 y dt. Aunque dt será 1 por default, puede 
especificarse dt y n lo que condiciona T=n*dt.

Por como está implementado (func get_dB), cada entrada de una trayectoria B (de get_B o get_B_matrix)
B[i] es igual al proceso en el tiempo i*dt: B_{i+dt}.
Cada fila (B[i,:]) es instancia del proceso estocástico, cada columna (B[:,j]) es instancia
de la variable aleatoria B_{j*dt}.
"""



#########Movimiento Browniano individual
def get_dB(n: int, dt: float=1) -> np.ndarray:
    """
    Recibe:
    n: número deseado de incrementos.
    dt: varianza=tiempo de muestreo (t-s). Por defecto =1
    Optional random_state to reproduce results.
    Retorna:
    dW: Vector con n incrementos de un MBEU
    """
    sample=np.random.normal(0.0, dt**(1/2), size=n-1)#dt es la varianza, n-1 porque agrego luego condición inicial
    dB=np.insert(sample,0,0) #Agrego los estados iniciales como primer vector columna
    return dB

def get_B(n: int, dt: float=1) -> np.ndarray:
    """
    Recibe:
    n: tamaño de muestras deseadas.
    dt: unidades de tiempo.
    Retorna:
    Vector trayectoria de n entrada: sumas acumuladas de los incrementos.
    """
    dB = get_dB(n,dt=dt)   #Cambiar por dB y B
    
    return np.cumsum(dB) #Cum sum retorna un vector y cada entrada se convierte en la suma acumulada
###################

############## Simulación en matriz
#Métodos auxiliares para matriz
def _get_correlated_dB(dB: np.ndarray, rho: float, dt:Optional[float]=1) -> np.ndarray:
    """
    Recibe:
    Incrementos de browniano.
    rho: coeficiente de correlación.
    dt: paso de muestreo.
    Retorna:
    Vector de incrementos de procesos browniano correlacionado al dB recibido..
    """
    dB2 = get_dB(len(dB), dt=dt)  # genera las d listas de incrementos.
    if np.array_equal(dB2, dB):
        # dB no puede ser igual a dB2.
        raise ValueError("Brownian Increment error, try choosing different random state.")
    return rho * dB + np.sqrt(1 - rho ** 2) * dB2

def get_B_matrix(
    n: int,
    d: int,
    dt:Optional[float]=1,
    rho: Optional[float] = None
) -> np.ndarray:
    """
    Recibe:
    n: tamaño deseado por trayectoria.
    d: número de trayectorias.
    dt (opc): tamaño de paso, default =1.
    rho (opc): correlación (no es par a par, sino corr con la que se simularán los procesos a partir de alguno de los anteriormente simulados )
    Retorna: 
    Matriz (d x n): d trayectorias brownianas, cada una de tamaño n (fila es un prceso).
    
    """
    dBs: list[np.ndarray] = []
    for i in range(d):
        if i == 0 or rho is None:
            dB_i = get_dB(n, dt=dt)
        else:
            dB_previous = dBs[np.random.choice(i)]#Elige una fila (una trayectoria)
            dB_i = _get_correlated_dB(dB_previous, rho, dt=dt)
        dBs.append(dB_i)
    return np.cumsum(np.asarray(dBs),axis=1) #d vectores fila, se suma por filas.


####################################################
#Evaluación de funciones teóricas de esperanza, varianza y covarianza de variables aleatorias B_t.

def mbeu_theoret_mu_cov(tiempos:np.array)->tuple[np.array]:
    """
    Recibe:
    tiempos: vector de tiempos. ("Verdaderos" índices de los B en las trayectorias: los tiempos en los que evaluamos el proceso)
    Retorna:
    Evaluación de función teórica de esperanza y varianza
    """
    s_v,t_v=np.meshgrid(tiempos,tiempos)
    mu_teor=np.zeros(len(tiempos))
    cov_teor=np.minimum(s_v,t_v)
    return mu_teor, cov_teor
    


#######################################################################################
# Movimientos asociados
#######################################################################################
######### Browniano Bridge
def get_Bridge_matrix(n:int,d:int=1)->np.array:
    """
    Proceso Bridge entre 0 y 1. Al recibir n, se induce dt=1/(n-1)
    Recibe:
    n: longitud trayectorias (condiciona dt)
    d: # de trayectorias
    (Usa time=np.linspace(0,1,n) como vector de tiempos)
    Devuelve:
    Matriz con cada fila un proceso browniano Bridge.
    Como T=1 en este proceso, se induce dt=1/n
    Relación índice en arreglo y tiempo: B[i]=B_{i*dt}
    """
    dt=1/(n-1)
    B=get_B_matrix(n,d,dt=dt)
    time=np.linspace(0,1,n)
    tB_1=np.array([time[i]*B[:,-1] for i in range(n)]).T #B[:,-1] es el vector con cada trayectoria en su t final (asumo 1) Se transpone porque cada elemento de la lista debe ser columna
    return B-tB_1

#Propiedades teóricas 

def bridge_theoret_mu_cov(tiempos:np.array)->np.array:
    """
    Recibe:
    tiempos: vector de tiempos. ("Verdaderos" índices de los B en las trayectorias: los tiempos en los que evaluamos el proceso)
    Retorna:
    Evaluación de función teórica de esperanza y varianza
    """
    s_v,t_v=np.meshgrid(tiempos,tiempos)
    mu_teor=np.zeros(len(tiempos))
    cov_teor=np.minimum(s_v,t_v)-np.multiply(s_v,t_v)
    return mu_teor, cov_teor

################################# Ruido blanco
def get_w_noise_matrix(n:int, d:int, h: int=1, dt:float=1)->np.array:
    """
    Recibe:
    n: tamaño por trayectoria.
    d: número de trayectorias
    h: (h<n) incremento temporal en número de pasos dt. (cada uno de tamaño h*dt)
    Retorna:
    Matriz (d x n) con 
    """
    h_dt=h*dt
    B=get_B_matrix(n+h,d,dt=dt)
    if h>=n:
        raise Exception("h>n")
    White_noise=(B[:,:-h]-B[:,h:])/(h_dt)
    return White_noise


#Propiedades teóricas
def w_noise_theoret_mu_cov(tiempos:np.array,h:float=1,dt=1)->tuple[np.array]:
    h_dt=h*dt
    s_v,t_v=np.meshgrid(tiempos,tiempos)
    mu_teor=np.zeros(len(tiempos))
    cov_teor=(1/h_dt**2)*(np.minimum(s_v+h_dt,t_v+h_dt)-np.minimum(s_v+h_dt,t_v))
    return mu_teor,cov_teor

########## Movimiento drift


#Propiedades teóricas

########## Movimiento browniano geométrico

#Propiedades teóricas

##################

###Obtención de estadísticos de matriz de datos (d x n) tomando cada columna (tiempo t) como variable:
def empiric_mu_cov(B:np.array):
    """
    Recibe: 
    B: matriz (d x n) de d trayectorias de dimensión n. Los datos se asumen simulados en [0,T] con n+1 puntos (dt=T/n)
    Retorna: 
    mu_hat_t: vector de función esperanza estimada (func de t) y
    cov_hat_st: matriz que estima función de covarianza (func de s y t). 
    En cada caso, mu_hat_t[i] será la estimación de E(W_{i*dt}) mientras que 
    cov_hat_st[i,j] será estimación de Cov(W_{i*dt},W_{j*dt}) para el dt elegido al momento de simular.
    """
    if B.ndim==1:
        raise Exception("Argument not a matrix")
    else: 
        d,n=B.shape
        mu_hat_t=np.mean(B,axis=0)
        cov_hat_st=np.cov(B,rowvar=False) #Puedo obtener la estimación de varianzas con np.diag(cov)
    return mu_hat_t, cov_hat_st

#################################### error, tomar diferencias y variación cuadrática
def error(matr_teor,matr_hat):
    return np.absolute(matr_teor-matr_hat)

def dif_B(B:np.array)->np.array:
    """Obtener vector dB o matriz de vectores dB a partir del vector o matriz B"""
    if B.ndim==1:
        dB=np.diff(B,prepend=0)
    else: 
        dB=np.diff(B,prepend=0,axis=1)
    return dB

def quadratic_variation(B:np.array):
    """Devuelve la matriz (o el vector en caso de dimensión 1) con las variaciones cuadráticas de cada fila (mov browniano) de W"""
    if B.ndim==1:
        qv=np.cumsum(np.power(np.diff(B,prepend=0),2))
    else: 
        qv= np.cumsum(np.power(np.diff(B,prepend=0,axis=1),2),axis=1)
    return qv