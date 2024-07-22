import numpy as np
import pandas as pd


labels = ['a', 'b', 'c']
mylist = [10, 20, 30]
arr = np.array(mylist)
d = {'a': 10, 'b': 20, 'c': 30}
# Con Series ottengo le informazioni riguardo il dato inserito
print(pd.Series(data=mylist))
print(pd.Series(arr))
# Posso specificare anche un indice oltre al dato per nominare manualmente
# il nome dell'indice invece dei classici numeri da 0 in poi
print(pd.Series(arr, index=labels))
# È possibile passare a pandas liste con dati di tipo diversi all'interno
print(pd.Series(data=[10, 'a', 4.4]))
print(pd.Series(data=['d', 'a', 'e']))
# Rinominando gli indici è possibile usare le series
# di pandas in modo simile ai dizionari di python
ser1 = pd.Series([1, 2, 3, 4], index=['USA', 'Germany', 'USSR', 'Japan'])
print(ser1)
print(ser1['Japan'])
# Se alcuni dati coincidono è possibile sommare due series insieme,
# nel caso di dati non coicidenti verrà restituito NaN
ser2 = pd.Series([1, 4, 5, 6], index=['USA', 'Germany', 'Italy', 'Japan'])
print(ser2)
# NaN risulta perchè con italy si va a sommare un numero
# dell'altra series che non esiste, invece nei casi di
# dati che si sono sommati i valori sono diventati float,
# non è un errore ma un retaggio di python 2 a causa di una
# possibile perdita di dati se non si convertivano in float
print(ser1 + ser2)