# Nombre: Carlos Iván Santillán Téllez
# Fecha: 3 Mayo 2019
# Desc: Práctica 1, Series de Tiempo

## 7, 8, 9, 12, 14 tienen tendencia

## Importamos librerías necesarias
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib

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

### Determinamos periodo de estacionalidad
delta = int(input('Ingrese el valor observado del periodo de estacionalidad'))
print('El periodo de estacionalidad es: ' + str(delta))

### Declaramos la cota superior
n_1 = len(serie_1) - 1
print('El valor de n-1 es: ' + str(n_1))

### Buscamos el máximo k (entero) tal que (k+1)delta <= n-1
### De modo que k = (n-1)/delta - 1

#### En el caso de la Serie 20, con delta = 12, k nos queda de 7.25, pero con delta = 11, k = 8

k = (n_1/delta) -1
k = int(k)
print('El valor máximo de k es:' + str(k))

frac_s_gorro = 1/(k+1)

s_gorro = []
vector_parcial = []

for j in range(0, k+1):     ### Para que aparezca todo el vector, se requiere k+1 en la suma
    vector_parcial.append(serie_1['Xt-Tt'][(delta*j)+i])


print(vector_parcial)
print(s_gorro)
#res_suma_parcial = sum(suma_parcial)
#print(res_suma_parcial)

#for i in range(0, delta-1, 1):
 #   for j in range(0, k, 1):
  #      s_gorro.append(sum(serie_1['xt'][(delta*j)+i]))

#print(s_gorro)


