# Nombre: Carlos Iván Santillán Téllez
# Fecha: 3 Mayo 2019
# Desc: Práctica 1, Series de Tiempo

## 7, 8, 9, 12, 14 tienen tendencia

## Importamos librerías necesarias
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import math
import statsmodels as sm


## Importamos archivo con series
series = pd.read_csv("C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Series_Tiempo_1.csv")
#print(series)

## Definimos las funciones de Autocorrelación
### Necesitamos pasar el vecor con xt y el valor de S
### Antes se pasaba dataFrame
def gamma_acf (vectorDatos, S):
    n = len(vectorDatos)
    x_prom = np.average(vectorDatos)
    sum_corr = []
    for i in range(0, n - 1 - S, 1):
        sum_corr.append(
            (vectorDatos[i] - np.average(vectorDatos)) * (vectorDatos[i + S] - np.average(vectorDatos)))
    total_sum = sum(sum_corr)
    frac = 1 / n
    gamma_res = frac*total_sum
    #print(gamma_res)
    return gamma_res

### Antes, se pasaba dataFrame
def acf_fun (vectorDatos):
    rho_t = []
    rho_numerator = []
    n = len(vectorDatos)
    for i in range(0, n-1, 1):
        rho_numerator.append(gamma_acf(vectorDatos, i))
    rho_t = rho_numerator/gamma_acf(vectorDatos, 0)
    return rho_t

######################################################################################################################
## Definimos función para limpiar DataFrame
def prepara_datos(dataFrame):
    serie_datos = dataFrame[['xt.1']]          ### Las 20 series están en xt, xt.1, ..., xt.19
    serie_datos = serie_datos.dropna()      ### Nos deshacemos de los NA
    serie_datos['t'] = np.arrange(len(serie_datos))     ### Añadimos columna con números consecutivos
    serie_datos['t'] = serie_datos['t']+1
    serie_datos['xt'] = serie_datos['xt.1']     ### Por practicidad, añadimos de nuevo xt.n pero nombrada xt
    print(serie_datos)
    return serie_datos

#######################################################################################################################
## Definimos funciones para obtener gráficas
def grafica_Xt (vectorXt):
    plt.plot(vectorXt)      ### La entrada debe ser el vector Xt de nuestro DataFrame
    plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/Serie_20.png')
    plt.show()              ### Mostramos gráfico
    return

def grafica_ACF (vector):
    plt.plot(acf_fun(serie_1))  ### La entrada es el vector rho
    plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/ACF_20.png')
    plt.show()                  ### Mostramos gráfico
    return

def grafica_St (vectorSt):
    plt.plot(vectorSt)      ### La entrada sería el vector 'diferencia'
    plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/ACF_Tt_10.png')
    plt.show()
    return

#######################################################################################################################
## Definimos función para obtener las matrices X y Y
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
## Definimos funciones para obtener St y las demás componentes del modelo
def componente_St (dataFrame):
    dataFrame['T_t'] = a0 + a1 * matriz_indep['T']                ### Añadimos a nuestra serie un vector con T_t
    dataFrame['diferencia'] = dataFrame['xt'] - dataFrame['T_t']      ### Añadimos a nuestra serie_1 un vector con Xt - Tt
    print(dataFrame)
    return dataFrame

#######################################################################################################################
# ACF Y DATOS
#######################################################################################################################
## Trabajamos con la primera serie (alternamos entre xt, xt.1, xt.2, etc.)
### Las 20 series están en xt, xt.1, ..., xt.19
serie_1 = series[['xt.19']]
#print(serie_1)

### Nos deshacemos de los NA
serie_1 = serie_1.dropna()
#print(serie_1)

### Añadimos columna con números consecutivos
serie_1['T'] = np.arange(len(serie_1))
serie_1['T'] = serie_1['T'] +1
#### Por practicidad, añadimos la columna xt.n de nuevo, pero la llamamos xt
serie_1['xt'] = serie_1['xt.19']

print(serie_1)

### Obtenemos gráfica de Xt
#plt.plot(serie_1['xt'])
#plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/Serie_20.png')
#plt.show()

### Obtenemos gráfica de ACF
#plt.plot(acf_fun(serie_1))
#plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/ACF_20.png')
#plt.show()

#######################################################################################################################
# TENDENCIA
#######################################################################################################################

### Necesitamos ahora una matriz conformada por el intercepto (vector de 1) y el tiempo

### Por practicidad, añadimos vector de 1 a nuestro data frame
serie_1['Intercepto'] = 1

### Nos quedamos ahora con el intercepto y el Tiempo
matriz_indep = serie_1[['Intercepto', 'T']]

### Restamos 1 al Tiempo para obtener vector iniciado en 0
matriz_indep['T'] = matriz_indep['T']-1

### Nuestra variable dependiente es xt, nos quedamos con ese vector solamente
var_dep = serie_1['xt']

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
serie_1['T_t'] = a0+a1*matriz_indep['T']

### Añadimos a nuestra serie_1 un vector con Xt - Tt
serie_1['Xt-Tt'] = serie_1['xt'] - serie_1['T_t']

print(serie_1)

### Graficamos la diferencia de Xt - Tt
plt.plot(serie_1['Xt-Tt'])
plt.title('Xt - Tt')
#plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/Xt_Tt_20.png')
plt.show()

### Graficamos la ACF de Xt - Tt
plt.plot(acf_fun(serie_1['Xt-Tt']))
plt.title('ACF de Xt - Tt')
#plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/ACF_Xt_Tt_20.png')
plt.show()

### Nuestros resultados son:
#### Para Serie 7: a0 = -40.18811004     a1 = 1.31696879

#######################################################################################################################
# ESTACIONALIDAD
#######################################################################################################################

### Determinamos periodo de estacionalidad
delta = int(input('Ingrese el valor observado del periodo de estacionalidad'))
print('Delta es: ' + str(delta))

### Declaramos la cota superior
n_1 = len(serie_1) - 1
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
serie_1['residuos'] = residuos

print(serie_1)

### Nos quedamos ahora con el los residuos y Xt-Tt
matriz_residuos = serie_1[['residuos', 'Xt-Tt']]
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
serie_1['St'] = S_t

### Construimos el vector Xt-Tt-St
serie_1['Xt-Tt-St'] = serie_1['Xt-Tt'] - serie_1['St']
print(serie_1)

### Observamos gráficas de Xt-Tt-St y su correspondiente ACF
### Graficamos la diferencia de Xt - Tt - St
plt.plot(serie_1['Xt-Tt-St'])
plt.title('Xt - Tt - St')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/Xt_Tt_St_20.png')
plt.show()

### Graficamos la ACF de Xt - Tt - St
plt.plot(acf_fun(serie_1['Xt-Tt-St']))
plt.title('ACF de Xt - Tt - St')
plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/ACF_Xt_Tt_St_20.png')
plt.show()

#######################################################################################################################
# RUIDO BLANCO
#######################################################################################################################

### Fijamos nuestra m
m = math.floor(np.log(len(serie_1['xt']))) + 1
print('Nuestra m es de: ' + str(m))

n = n_1 +1
print('Tenemos un total de ' + str(n) + ' datos')

## Prueba de Ljung-Box
### Con n = 100, nuestra m = 5, de modo que a lo más ese sería el número de Q´s a calcular
### Primero necesitamos la ACF de Xt-Tt-St
rho_ph = np.zeros(len(serie_1))
rho_ph = acf_fun(serie_1['Xt-Tt-St'])

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

### Recordamos que si los resultados son "NO RECHAZAR" entonces concluimos que es Ruido Blanco


### Definimos términos para calcular Q(m)
#vector_suma_Q = np.zeros(m+1)
#for i in range(0, m):
#    vector_suma_Q[0] =  ((gamma_acf(serie_1['Xt-Tt-St'], 0) / gamma_acf(serie_1['Xt-Tt-St'], 0))**2) * (1/(n))
#    vector_suma_Q[i+1] = vector_suma_Q[i] + ((gamma_acf(serie_1['Xt-Tt-St'], i+1) /
#                                              gamma_acf(serie_1['Xt-Tt-St'], 0))**2) * (1/(n - (i+1)))

#print(vector_suma_Q)

#Q = np.zeros(m+1)
#Q = n*(n+2)*vector_suma_Q
#print(Q)

#suma_Q = np.sum(vector_suma_Q)

#Q_m



#s_gorro = np.zeros(delta)
#vector_parcial = np.zeros(k+1)

### Primero, definimos cada vector conformado por x_{delta*j +i}
### Necesitaremos delta vectores parciales
#vector_parcial_1 = np.zeros(k+1)
#vector_parcial_2 = np.zeros(k+1)
#vector_parcial_3 = np.zeros(k+1)
#vector_parcial_4 = np.zeros(k+1)
#vector_parcial_5 = np.zeros(k+1)
#vector_parcial_6 = np.zeros(k+1)
#vector_parcial_7 = np.zeros(k+1)
#vector_parcial_8 = np.zeros(k+1)
#vector_parcial_9 = np.zeros(k+1)
#vector_parcial_10 = np.zeros(k+1)
#vector_parcial_11 = np.zeros(k+1)

### Llenaremos los vectores parciales
#for j in range(0, k+1):
 #   vector_parcial_1[j] = serie_1['Xt-Tt'][(delta * j) + 0]
  #  vector_parcial_2[j] = serie_1['Xt-Tt'][(delta * j) + 1]
   # vector_parcial_3[j] = serie_1['Xt-Tt'][(delta * j) + 2]
    #vector_parcial_4[j] = serie_1['Xt-Tt'][(delta * j) + 3]
    #vector_parcial_5[j] = serie_1['Xt-Tt'][(delta * j) + 4]
    #vector_parcial_6[j] = serie_1['Xt-Tt'][(delta * j) + 5]
    #vector_parcial_7[j] = serie_1['Xt-Tt'][(delta * j) + 6]
    #vector_parcial_8[j] = serie_1['Xt-Tt'][(delta * j) + 7]
    #vector_parcial_9[j] = serie_1['Xt-Tt'][(delta * j) + 8]
    #vector_parcial_10[j] = serie_1['Xt-Tt'][(delta * j) + 9]
    #vector_parcial_11[j] = serie_1['Xt-Tt'][(delta * j) + 10]

#print(vector_parcial_1)
#print(vector_parcial_2)
#print(vector_parcial_3)
#print(vector_parcial_4)
#print(vector_parcial_5)
#print(vector_parcial_6)
#print(vector_parcial_7)
#print(vector_parcial_8)
#print(vector_parcial_9)
#print(vector_parcial_10)
#print(vector_parcial_11)

### En segundo lugar, necesitamos un vector
### formado por las sumas de los elementos de los vectores parciales
#s_gorro_suma_1 = np.sum(vector_parcial_1)
#s_gorro_suma_2 = np.sum(vector_parcial_2)
#s_gorro_suma_3 = np.sum(vector_parcial_3)
#s_gorro_suma_4 = np.sum(vector_parcial_4)
#s_gorro_suma_5 = np.sum(vector_parcial_5)
#s_gorro_suma_6 = np.sum(vector_parcial_6)
#s_gorro_suma_7 = np.sum(vector_parcial_7)
#s_gorro_suma_8 = np.sum(vector_parcial_8)
#s_gorro_suma_9 = np.sum(vector_parcial_9)
#s_gorro_suma_10 = np.sum(vector_parcial_10)
#s_gorro_suma_11 = np.sum(vector_parcial_11)

#print(s_gorro_suma_1)
#print(s_gorro_suma_2)
#print(s_gorro_suma_3)
#print(s_gorro_suma_4)
#print(s_gorro_suma_5)
#print(s_gorro_suma_6)
#print(s_gorro_suma_7)
#print(s_gorro_suma_8)
#print(s_gorro_suma_9)
#print(s_gorro_suma_10)
#print(s_gorro_suma_11)

### Formamos nuestro vector S' con las entradas de las sumas entre k+1
#s_gorro[0] = frac_s_gorro * s_gorro_suma_1
#s_gorro[1] = frac_s_gorro * s_gorro_suma_2
#s_gorro[2] = frac_s_gorro * s_gorro_suma_3
#s_gorro[3] = frac_s_gorro * s_gorro_suma_4
#s_gorro[4] = frac_s_gorro * s_gorro_suma_5
#s_gorro[5] = frac_s_gorro * s_gorro_suma_6
#s_gorro[6] = frac_s_gorro * s_gorro_suma_7
#s_gorro[7] = frac_s_gorro * s_gorro_suma_8
#s_gorro[8] = frac_s_gorro * s_gorro_suma_9
#s_gorro[9] = frac_s_gorro * s_gorro_suma_10
#s_gorro[10] = frac_s_gorro * s_gorro_suma_11

#print(s_gorro)

### Necesitamos calcular Si
### Primero calcularemos el promedio de las S'i
#s_gorro_promedio = np.mean(s_gorro)

### Ahora calculamos las entradas de Si
#S_i = np.zeros(delta)
#for i in range(0, delta):
 #   S_i[i] = s_gorro[i] - s_gorro_promedio

#print(S_i)

### Ahora necesitamos calcular los residuos para t = 0, 1, ..., n-1
#residuos = np.zeros(n_1+1)
#for i in range(0, n_1+1):
 #   residuos[i] = i % delta

#print(residuos)

#S_t = np.zeros(n_1+1)       ### con n_1 +1 sale bien el vector

### Repetimos nuestro vector Si para formar St
#S_t = np.tile(S_i, k)

#print(S_t)
#print(len(S_t))
#print(S_t[98])      ### última entrada en S_t

### Necesitamos una entrada extra, acorde a nuestros residuos, corresponde al primer elemento de S_t
#S_t = np.append(S_t, S_t[0])
#print(S_t[99])      ### Nuevo último elemento de S_t

### Creamos el vector Xt-Tt-St
#Xt_Tt_St = np.zeros(n_1+1)
#Xt_Tt_St = serie_1['Xt-Tt'] - S_t
#serie_1['Xt-Tt-St'] = Xt_Tt_St

#print(serie_1)

### Observamos gráficas de Xt-Tt-St y su correspondiente ACF
### Graficamos la diferencia de Xt - Tt - St
#plt.plot(serie_1['Xt-Tt-St'])
#plt.title('Xt - Tt - St')
#plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/Xt_Tt_St_20.png')
#plt.show()

### Graficamos la ACF de Xt - Tt - St
#plt.plot(acf_fun(serie_1['Xt-Tt-St']))
#plt.title('ACF de Xt - Tt - St')
#plt.savefig('C:/Users/PC/Desktop/ULSA/6 Semestre/Series de Tiempo/Tarea_1_Series/ACF_Xt_Tt_St_20.png')
#plt.show()














#for i in range(0, delta):
 #   for j in range(0, k+1):     ### Para que aparezca todo el vector, se requiere k+1 en la suma
  #     s_gorro[i] = np.sum(serie_1['Xt-Tt'][(delta*j)+i])

#print(s_gorro)
