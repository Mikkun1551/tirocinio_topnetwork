import numpy as np


# Una lista normale
mylist = [1, 2, 3]
print(type(mylist))

# Trasformare la lista in un array con numpy
arr_mylist = np.array(mylist)
# Devo inizializzare una nuova varibile se voglio l'array visto che non va a
# modificare la lista normale
print(arr_mylist)
print(type(arr_mylist))

# Inizializzo una nested list
my_list_2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
my_matrix = np.array(my_list_2)
# L'array risulta avere più dimensioni, calcolati in base
# a quanti array contiene all'interno
print(my_matrix)

# C'è un metodo per vedere la forma di un array
print(my_matrix.shape)
# In base all'array il risultato cambia, in questo caso il risultato
# è (3, 3) il primo sta a indicare che l'array ha 3 dimensioni,
# ossia contiene 3 array, e il secondo indica che
# ciascuno contiene 3 elementi, se il numero di elementi non è uguale
# per tutti gli array verrà sollevato un errore ValueError

# Per creare un array usando numpy
my_arr = np.arange(0, 10)
print(my_arr)
# Crearà un array con numeri da 0 (incluso) a 10 (escluso),
# se inserito un terzo numero verranno saltati i numeri desiderati
my_arr = np.arange(0, 10, 2)
print(my_arr)

# Per creare un array di soli 0 di tipo float
my_arr = np.zeros(5)
# Si crea un array di zeri tanti quanti specificati
print(my_arr)

# Se voglio un array di zeri con più dimensioni
my_arr = np.zeros((5, 10))
# Passi come variabile una tupla con prima le dimensioni
# e poi quanti zeri si desidera
print(my_arr)

# Se si vuole un array di uno il metodo è diverso
# ma funziona allo stesso modo
my_arr = np.ones(5)
print(my_arr)

# Come per gli zeri se si passa un operatore fuori si può modificare
# il contenuto degli array, queste operazioni in una lista normale
# hanno un risultato diverso
my_arr = np.ones(5) + 4
print(my_arr)
my_arr = np.ones(5) - 4
print(my_arr)
my_arr = np.ones(5) * 10
print(my_arr)
my_arr = np.ones(5) / 10
print(my_arr)

# E' possibile creare un array di numeri specificando da quale iniziare
# e quale finire (incluso) e quanti numeri equamente divisi avere tra di loro
my_arr = np.linspace(1, 10, 19)
print(my_arr)

# Data la lunghezza di un array è possibile creare una sequenza di 0 e 1
# dove ogni array avrà l'1 in una posizione diversa dagli altri e sarà
# circondato da zeri
my_arr = np.eye(5)
print(my_arr)
