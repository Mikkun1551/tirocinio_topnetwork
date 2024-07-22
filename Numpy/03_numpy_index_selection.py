import numpy as np


# L'indice degli array funziona in modo molto simile alle liste di python
my_arr = np.arange(0, 11)
print(my_arr[5])
print(my_arr[1:5])
print(my_arr[2:])
# Ricorda le operazioni sulle liste si applicano a tutti gli elementi e
# vanno salvati come varibile i cambiamenti altrimenti non rimangono
new_arr = my_arr + 100
print(new_arr)
new_arr / 2
print(new_arr)
# Puoi anche elevare alle potenze
pot_arr = my_arr ** 2
print(pot_arr)

# Posso anche salvare una parte dell'array in un altro array con uno slice
slice_of_arr = my_arr[0:6]
print(slice_of_arr)
# Si può anche selezionare gli elementi dell'array e modificarli insieme
slice_of_arr[:] = 99
print(slice_of_arr)
# Il punto dello slice è che i cambiamenti che avvengono in esso
# si manifestano anche sull'array d'origine
print(my_arr)
# Se voglio una copia indipendente serve usare copy
copy_arr = my_arr.copy()
copy_arr[:] = 1000
print(copy_arr)
print(my_arr)

# INDEXING SU UN ARRAY A DUE DIMENSIONI
# Per capire bene la struttura degli array a due dimensioni immaginateli come
# una tabella con righe e colonne, le dimensioni sono le righe e le colonne
# la divisione degli elementi.
# Creo un array a due dimensioni
arr_2d = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
print(arr_2d.shape)
print(arr_2d)
# Se voglio gli elementi di una dimensione seleziono la dimensione
# partendo da indice 0
print(arr_2d[1])
# Se voglio un elemento specifico di una dimensione metto un altro index
# con parentesi quadre accanto
print(arr_2d[1][1])
# Posso anche farlo in una singola quadra separato da una virgola
print(arr_2d[2, 2])
# Se voglio le dimensioni in un range
print(arr_2d[:2])
# Se voglio un range di elementi in un range di dimensioni
print(arr_2d[:2, 1:])

# SELEZIONE CONDIZIONALE
# Creiamo un nuovo array
my_arr = np.arange(1, 11)
# Posso restituire l'array in formato bool degli elementi
# che soddisfano una condizione
bool_arr = my_arr > 4
print(bool_arr)
# Preso l'array bool posso restituire i valori con lo stesso indice
# dei valori true del bool array
print(my_arr[bool_arr])
# Oppure posso direttamente passare una condizione come indice
# e ricavare solo gli elementi che soddisfano al condizione
print(my_arr[my_arr <= 5])