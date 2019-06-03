# Nombre: Carlos Iván Santillán Téllez
# Fecha: 3 Mayo 2019
# Desc: Práctica 1, Series de Tiempo

## 7, 8, 9, 12, 14 tienen tendencia

#######################################################################################################################
######  PROCEDIMIENTO ------------------------------------------------------------------------------------------------
#######################################################################################################################

# Se considerará Xt = Tt + St + Wt donde Tt será tendencia, St estacionalidad y Wt ruido blanco
#   1. Graficar Xt y ACF(Xt)
#       1.1 Buscar componente de Tt
#   2. En Xt, estimar Tt y obtener Xt - Tt
#   3. Graficar Xt - Tt y ACF(Xt - Tt)
#       3.1 Buscar componente St en ACF(Xt - Tt)
#   4. Estimar St en Xt - Tt y obtener Xt - St (NO Xt - Tt - St)
#   5. Graficar Xt - St y ACF(Xt - Tt)
#       5.1 Buscar componente Tt en Xt - St
#   6. Estimar Tt y obtener Xt - St - Tt
#   7. Graficar Xt - St - Tt y ACF(Xt- St - Tt)
#       7.1 Buscar componente de Wt
#   8. Realizar Prueba de Ljung-Box a Xt - St - Tt
#   9. Concluir si se ajusta al modelo clásico o no


#######################################################################################################################
#------ I. LIBRERÍAS --------------------------------------------------------------------------------------------------
#######################################################################################################################

import numpy as np                  # vectores
import pandas as pd                 # DataFrames
import scipy as sp                  # Cómputo
from scipy import stats             # Estadística
import matplotlib.pyplot as plt     # Gráficas
import matplotlib
import math                         # Aritmética
import statsmodels as sm            # Regresión

#######################################################################################################################
#----- II. IMPORTE DE DATOS -------------------------------------------------------------------------------------------
#######################################################################################################################

## Importamos archivo con series
series = pd.read_csv("C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Series_Tiempo_1.csv")
print(series)

#######################################################################################################################
#   III. DEFINICIÓN DE FUNCIONES
#######################################################################################################################

## Definimos las funciones de Autocorrelación
### Necesitamos pasar el vecor con xt y el valor de S
### Antes se pasaba dataFrame
def gamma_acf (vectorDatos, S):         # vectorDatos = series['nombre_columna_con_serie']
    n = len(vectorDatos)
    x_prom = np.average(vectorDatos)
    sum_corr = np.zeros(n)
    for i in range(0, n - 1 - S, 1):
        sum_corr[i] = (vectorDatos[i] - x_prom) * (vectorDatos[i + S] - x_prom)
    total_sum = np.sum(sum_corr)
    frac = 1 / n
    gamma_res = frac * total_sum
    print(gamma_res)
    return gamma_res

### Antes, se pasaba dataFrame
def acf_fun (vectorDatos):
    n = len(vectorDatos)
    rho_t = np.zeros(n)
    rho_numerator = np.zeros(n)
    for i in range(0, n-1, 1):
        rho_numerator[i] = gamma_acf(vectorDatos, i)
    rho_t = rho_numerator/gamma_acf(vectorDatos, 0)
    return rho_t

######################################################################################################################
## Definimos función para limpiar DataFrame
# FUNCIONA #
def prepara_datos(dataFrame):
    serie_datos = dataFrame[['xt.19']]          ### Las 20 series están en xt, xt.1, ..., xt.19
    serie_datos = serie_datos.dropna()         ### Nos deshacemos de los NA
    serie_datos['t'] = np.arange(len(serie_datos))     ### Añadimos columna con números consecutivos
    serie_datos['t'] = serie_datos['t']+1
    serie_datos['xt'] = serie_datos['xt.19']     ### Por practicidad, añadimos de nuevo xt.n pero nombrada xt
    serie_datos = pd.DataFrame(serie_datos)
    print(serie_datos)
    return serie_datos

#######################################################################################################################
## Definimos funciones para obtener gráficas
# FUNCIONA #
def grafica_Xt (vectorXt):
    plt.plot(vectorXt)      ### La entrada debe ser el vector Xt de nuestro DataFrame
    plt.title('Gráfica de Xt')
    plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/Serie_20.png')
    plt.show()              ### Mostramos gráfico
    return

def grafica_ACF (vectorXt):
    plt.plot(acf_fun(vectorXt))  ### La entrada es el vector Xt
    plt.title('ACF de Xt')
    plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/ACF_20.png')
    plt.show()                  ### Mostramos gráfico
    return

#######################################################################################################################
## Definimos función para obtener las matrices X y Y
# FUNCIONA #
def obtiene_matrices (dataFrame):
    dataFrame['Intercepto'] = 1                         ### Por practicidad, añadimos vector de 1s al DataFrame original
    matriz_indep = dataFrame[['Intercepto', 't']]       ### Nos quedamos ahora con el intercepto y el Tiempo
    matriz_indep['t'] = matriz_indep['t'] - 1           ### Restamos 1 al Tiempo para obtener vector iniciado en 0
    var_dep = dataFrame['xt']                             ### Nuestra variable dependiente es xt, nos quedamos con ese vector solamente
    ### Transformamos los DataFrames (Pandas) anteriores a matrices (NumPy)
    X = matriz_indep.values         ### Deberían llamarse X e Y, para no alterar los DataFrames
    Y = var_dep.values
    print(X)
    print(Y)
    return X, Y

#######################################################################################################################
## Definimos función para Método de Mínimos Cuadrados
### Recordemos que b = (X'X)^-1 X' Y
def funcion_OLS (matrizX, matrizY):
    X_Transpuesta = np.transpose(matrizX)               ### Obtenemos la transpuesta de X
    X_prod = np.dot(X_Transpuesta, matrizX)             ### Obtenemos la multiplicación de X' con X
    prod_inverso = np.linalg.inv(X_prod)                ### Obtenemos la Inversa del producto
    prod_tot = np.dot(prod_inverso, X_Transpuesta)      ### Obtenemos producto de la inversa con X'
    a = np.dot(prod_tot, matrizY)                             ### Obtenemos el producto con Y
    a0 = a[0]
    a1 = a[1]
    print(a)
    print(a0)
    print(a1)
    return a, a0, a1

#######################################################################################################################
#   IV. ACF y Datos
#######################################################################################################################

## Trabajamos con la primera serie (alternamos entre xt, xt.1, xt.2, etc.)
### Las 20 series están en xt, xt.1, ..., xt.19
serie_datos = series[['xt.19']]
#print(serie_1)

### Nos deshacemos de los NA
serie_datos = serie_datos.dropna()
#print(serie_1)

### Añadimos columna con números consecutivos
serie_datos['t'] = np.arange(len(serie_datos))
serie_datos['t'] = serie_datos['t'] + 1
#### Por practicidad, añadimos la columna xt.n de nuevo, pero la llamamos xt
serie_datos['xt'] = serie_datos['xt.19']

print(serie_datos)

### Obtenemos gráfica de Xt
plt.plot(serie_datos['xt'])
plt.title('Xt')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/Serie_20.png')
plt.show()

### Obtenemos gráfica de ACF
plt.plot(acf_fun(serie_datos['xt']))
plt.title('ACF de Xt')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/ACF_20.png')
plt.show()

#######################################################################################################################
#   V. TENDENCIA
#######################################################################################################################

### Necesitamos ahora una matriz conformada por el intercepto (vector de 1) y el tiempo

### Por practicidad, añadimos vector de 1 a nuestro data frame
serie_datos['Intercepto'] = 1

### Nos quedamos ahora con el intercepto y el Tiempo
matriz_indep = serie_datos[['Intercepto', 't']]

### Restamos 1 al Tiempo para obtener vector iniciado en 0
matriz_indep['t'] = matriz_indep['t']-1

### Nuestra variable dependiente es xt, nos quedamos con ese vector solamente
var_dep = serie_datos['xt']

### Transformamos los DataFrames (Pandas) anteriores a matrices (NumPy)
### Deberían llamarse X e Y, para no alterar los DataFrames
X = matriz_indep.values
Y = var_dep.values

print(X)
print(Y)

### Usamos el Método de Mínimos Cuadrados para estimar a0 y a1
### Recordemos que b = (X'X)^-1 X' Y

### Obtenemos la transpuesta de X
X_Transpuesta = np.transpose(X)

### Obtenemos la multiplicación de X' con X
X_prod = np.dot(X_Transpuesta, X)

### Obtenemos la Inversa del producto
prod_inverso = np.linalg.inv(X_prod)

### Obtenemos producto de la inversa con X'
prod_tot = np.dot(prod_inverso, X_Transpuesta)

### Obtenemos el producto con Y
a = np.dot(prod_tot, Y)

a0 = a[0]
a1 = a[1]

print(a)
print(a0)
print(a1)

### Añadimos a nuestra serie un vector con T_t
serie_datos['T_t'] = a0+a1*matriz_indep['t']

### Añadimos a nuestra serie_datos un vector con Xt - Tt
serie_datos['Xt-Tt'] = serie_datos['xt'] - serie_datos['T_t']

print(serie_datos)

### Graficamos la diferencia de Xt - Tt
plt.plot(serie_datos['Xt-Tt'])
plt.title('Xt - Tt')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/Xt_Tt_20.png')
plt.show()

### Graficamos la ACF de Xt - Tt
plt.plot(acf_fun(serie_datos['Xt-Tt']))
plt.title('ACF de Xt - Tt')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/ACF_Xt_Tt_20.png')
plt.show()


#######################################################################################################################
#   VI. ESTACIONALIDAD
#######################################################################################################################

### Determinamos periodo de estacionalidad
delta = int(input('Ingrese el valor observado del periodo de estacionalidad (delta)'))
print('Delta es: ' + str(delta))

### Declaramos la cota superior
n_1 = len(serie_datos) - 1
print('El valor de n-1 es: ' + str(n_1))

### Buscamos el máximo k (entero) tal que (k+1)delta <= n-1
### De modo que k = (n-1)/delta - 1

#### En el caso de la Serie 20, con delta = 12, k nos queda de 7.25, pero con delta = 11, k = 8

k = (n_1/delta) -1
k = math.ceil(k)
print('El valor máximo de k es:' + str(k))

frac_s_gorro = 1/(k+1)

### Creamos nuestro vector de residuos
residuos = np.zeros(n_1+1)
for i in range(0, n_1+1):
    residuos[i] = i % delta
### Añadimos nuestro vector de residuos al DataFrame
serie_datos['residuos'] = residuos

print(serie_datos)

### Nos quedamos ahora con el los residuos y Xt-Tt
matriz_residuos = serie_datos[['residuos', 'Xt-Tt']]
vector_residuos = matriz_residuos.groupby('residuos').mean()
print(vector_residuos)

### Obtenemos el promedio de dichos residuos
promedio_residuos = np.mean(vector_residuos['Xt-Tt'])
print(promedio_residuos)

### Restamos a S' el promedio de S'
S_i = np.zeros(len(vector_residuos))
for i in range(0, len(S_i)):
    S_i[i] = vector_residuos['Xt-Tt'][i] - promedio_residuos

print(S_i)

### Creamos el vector St a partir de replicar k veces el Si
S_t = np.zeros(n_1+1)
S_t = np.tile(S_i, k)
print(S_t)

### Añadimos el vector St a nuestro DataFrame
### A S_t le faltan algunos valores para tener el mismo tamaño que la serie
### Cuando n = 100 y delta = 12, nos faltan 4 valores
S_t = np.append(S_t, S_t[0])
S_t = np.append(S_t, S_t[1])
S_t = np.append(S_t, S_t[2])
S_t = np.append(S_t, S_t[3])
serie_datos['St'] = S_t

#########################################
#   Quitamos St de Xt                   #
#########################################

### Construimos el vector Xt-St
serie_datos['Xt-St'] = serie_datos['xt'] - serie_datos['St']
print(serie_datos)

### Observamos gráficas de Xt-St y su correspondiente ACF
### Graficamos la diferencia de Xt - St
plt.plot(serie_datos['Xt-St'])
plt.title('Xt - St')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/Xt_St_20.png')
plt.show()

### Graficamos la ACF de Xt - St
plt.plot(acf_fun(serie_datos['Xt-St']))
plt.title('ACF de Xt - St')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/ACF_Xt_St_20.png')
plt.show()

#######################################################################################################################
#   V. 2a TENDENCIA
#######################################################################################################################

### Usamos la matriz de la primera estimación de la tendencia
### Nuestra variable dependiente es Xt - St, nos quedamos con ese vector solamente
var_dep_2 = serie_datos['Xt-St']

### Transformamos el DataFrame (Pandas) anterior a matriz (NumPy)
Y2 = var_dep_2.values

print(X)
print(Y2)

### Obtenemos el producto con Y
a = np.dot(prod_tot, Y2)

a0 = a[0]
a1 = a[1]

print(a)
print(a0)
print(a1)

### Añadimos a nuestra serie un vector con T_t verdadera
serie_datos['T_t_verdadera'] = a0+a1*matriz_indep['t']

### Añadimos a nuestra serie_datos un vector con Xt - St - Tt
serie_datos['Xt-St-Tt'] = serie_datos['Xt-St'] - serie_datos['T_t_verdadera']

print(serie_datos)

### Graficamos la diferencia de Xt - St - Tt
plt.plot(serie_datos['Xt-St-Tt'])
plt.title('Xt - St - Tt')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/Xt_St_Tt_20.png')
plt.show()

### Graficamos la ACF de Xt - St - Tt
plt.plot(acf_fun(serie_datos['Xt-St-Tt']))
plt.title('ACF de Xt - St - Tt')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Prueba_Final_Graficas/ACF_Xt_St_Tt_20.png')
plt.show()

#######################################################################################################################
# RUIDO BLANCO
#######################################################################################################################

### Fijamos nuestra m
m = math.floor(np.log(len(serie_datos['xt']))) + 1
print('Nuestra m es de: ' + str(m))

n = n_1 +1
print('Tenemos un total de ' + str(n) + ' datos')

## Prueba de Ljung-Box
### Con n = 100, nuestra m = 5, de modo que a lo más ese sería el número de Q´s a calcular
### Primero necesitamos la ACF de Xt-Tt-St
rho_ph = np.zeros(len(serie_datos))
rho_ph = acf_fun(serie_datos['Xt-St-Tt'])

#print(rho_ph)

rho_ph_cuadrado = rho_ph**2
#print(rho_ph_cuadrado)

### Construimos las Q´s
Q_suma1 = rho_ph_cuadrado[1] / (n-1)
Q_suma2 = Q_suma1 + (rho_ph_cuadrado[2] / (n-2))
Q_suma3 = Q_suma2 + (rho_ph_cuadrado[3] / (n-3))
Q_suma4 = Q_suma3 + (rho_ph_cuadrado[4] / (n-4))
Q_suma5 = Q_suma4 + (rho_ph_cuadrado[5] / (n-5))

Q1 = n * (n+2) * Q_suma1
Q2 = n * (n+2) * Q_suma2
Q3 = n * (n+2) * Q_suma3
Q4 = n * (n+2) * Q_suma4
Q5 = n * (n+2) * Q_suma5

print(Q1, Q2, Q3, Q4, Q5)

### Obtenemos valores de Ji-cuadrada inversa
#ji_cuad_1 = chi2.ppf(0.95, df=1)
#ji_cuad_2 = chi2.ppf(0.95, df=2)
#ji_cuad_3 = chi2.ppf(0.95, df=3)
#ji_cuad_4 = chi2.ppf(0.95, df=4)
#ji_cuad_5 = chi2.ppf(0.95, df=5)

#print(ji_cuad_1, ji_cuad_2, ji_cuad_3, ji_cuad_4, ji_cuad_5)

### Obtenemos p-values
p_value_1 = 1 - stats.chi2.cdf(Q1, 1)
p_value_2 = 1 - stats.chi2.cdf(Q2, 2)
p_value_3 = 1 - stats.chi2.cdf(Q3, 3)
p_value_4 = 1 - stats.chi2.cdf(Q4, 4)
p_value_5 = 1 - stats.chi2.cdf(Q5, 5)

print(p_value_1, p_value_2, p_value_3, p_value_4, p_value_5)

### Construimos un vector para los rechazos
rechazos = np.chararray(m, itemsize=15)
if p_value_1 < 0.05:
    rechazos[0] = 'Rechaza H0'
else:
    rechazos[0] = 'No rechazar H0'

if p_value_2 < 0.05:
    rechazos[1] = 'Rechaza H0'
else:
    rechazos[1] = 'No rechazar H0'

if p_value_3 < 0.05:
    rechazos[2] = 'Rechaza H0'
else:
    rechazos[2] = 'No rechazar H0'

if p_value_4 < 0.05:
    rechazos[3] = 'Rechaza H0'
else:
    rechazos[3] = 'No rechazar H0'

if p_value_5 < 0.05:
    rechazos[4] = 'Rechaza H0'
else:
    rechazos[4] = 'No rechazar H0'

print(rechazos)