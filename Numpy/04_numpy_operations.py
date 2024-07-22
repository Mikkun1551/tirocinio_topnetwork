import numpy as np


# Con l'uso delle parentesi è possibile fare più operazioni su un array
my_arr = np.arange(0, 11)
print((my_arr * 10) / 2)
# Posso anche sommare gli elementi di un array con quelli di un altro
print(my_arr + my_arr)
# Con numpy alcune operazioni che produrrebbero un errore non
# crasheranno il programma ma solleverà un avviso e metterà un
# placeholder dove ci sta il problema, per esempio divisioni per 0
# daranno inf (infinito) mentre dividendo 0 restituisce nan
print(1 / my_arr)
print(my_arr / my_arr)

# Posso fare anche la radice quadrata degli elementi in un array
print(np.sqrt(my_arr))
# Il logaritmo
print(np.log(my_arr))
# Il seno trigonometrico
print(np.sin(my_arr))
# La somma di tutti gli elementi
print(np.sum(my_arr))
# La media
print(np.mean(my_arr))

# Queste operazioni possono anche essere specifiche delle righe e colonne
# di array a più dimensioni
arr_2d = np.array([[1, 2, 3, 4,], [5, 6, 7, 8], [9, 10, 11, 12]])
print(arr_2d.shape)
# Per esempio se voglio la somma di solo le colonne,
# il primo argomento di sum è l'asse di riferimento,
# 0 è l'asse verticale
print(arr_2d.sum(0))
# 1 è l'asse orizontale
print(arr_2d.sum(1))