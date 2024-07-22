import numpy as np
import pandas as pd
from numpy.random import randn


np.random.seed(101)
rand_mat = randn(5, 4)
print(rand_mat)
# I dataframe sono un unione di più pandas series in uno solo, è anche
# un modo per visualizzare array a più dimensioni in modo tabellato
df = pd.DataFrame(data=rand_mat, index='A B C D E'.split(), columns='W X Y Z'.split())
# Un modo rapido di passare un indice personale è anche scriverlo
# direttamente nel metodo del dataframe come una singola stringa
# e separare ogni indice da uno spazio per poi splittarlo
print(df)
# Per ricavare i dati da un dataframe inserire il nome della COLONNA
# interessata sopratutto se le hai rinominate manualmente
# Il risultato sarà come vedere una series di pandas
print(df['W'])
# Per avere più colonne posso passare una lista
print(df[['W', 'Y']])
# È possibile riferirsi alle colonne come faresti in SQL
# anche se è consigliato usare i metodi specifici
print(df.W)

# Per aggiungere una nuova colonna al dataframe dandogli un valore
df['NEW'] = df['W'] + df['Y']
print(df)
# Per rimuoverla va specificato l'asse
df.drop('NEW', axis=1)
print(df)
# Però così non verrà rimosso in modo permanente
# va aggiunto il parametro inplace che richiede un booleano
df.drop('NEW', axis=1, inplace=True)
print(df)
# Invece per rimuovere le righe l'asse è diverso (0) e per rendere
# la rimozione permanente ricorda inplace

# Considera che nonostante uno possa cambiare il nome dell'indice
# delle righe puoi comunque riferirti a esse con il loro numero,
# cambia solo il metodo da utilizzare
# Con il nome
print(df.loc['A'])
# Usando l'indice
print(df.iloc[0])
# Ovviamente è possibile visualizzare più righe contemporaneamente,
# ricorda di passare le righe come una lista
print(df.loc[['A', 'E']])
print(df.iloc[[0, 3]])
# Se vogliamo degli elementi incrociando le colonne e le righe
# dovrai inserire le colonne e righe interessate come una lista
# dentro un lista ciascuna indipendente dall'altra
print(df.loc[['A', 'B']][['Y','Z']])