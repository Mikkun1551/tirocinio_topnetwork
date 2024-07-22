import numpy as np
import pandas as pd
from numpy.random import randn


np.random.seed(101)
rand_mat = randn(5, 4)
df = pd.DataFrame(data=rand_mat, index='A B C D E'.split(), columns='W X Y Z'.split())

# SELEZIONE CONDIZIONALE
# Funziona in modo simile a numpy e gli array
df_bool = df > 0
print(df_bool)
# E posso passare il dataframe di booleani nel mio dataframe,
# posso anche passare direttamente la condizione al dataframe.
# I valori false diventeranno NaN
print(df[df_bool])
# Posso anche ottenere i risultati di una condizione per una colonna
print(df['W'] > 0)
# anche in questo caso direttamente nel dataframe
print(df[df['W'] > 0])
# e specificando anche quale colonna mi interessa particolarmente
print(df[df['W'] > 0]['Y'])
# o un valore specifico
print(df[df['W'] > 0]['Y'].loc['A'])
# Per ricapitolare abbiamo preso il valore del dataframe:
# - dove i valori di una colonna sono maggiori a zero
# - abbiamo preso una singola colonna ignorando le righe dove la
# condizione precedente non si applica
# - infine abbiamo preso il valore di una singola riga di quella colonna
# Possiamo anche inserire due condizioni
cond1 = df['W'] > 0
cond2 = df['Y'] > 1
# Però la seguente condizione darà errore
# print(df[cond1 and cond2])
# Questo perchè and e or non sono ideati per le series di pandas
# staresti passando tecnicamente una lista di true e false
# Quindi per usare una cosa del genere in pandas:
# usi & per l'and, mentre | per l'or, simile a java
print(df[cond1 & cond2])
# Come al solito, le condizioni si possono scrivere direttamente
# nel dataframe, in questo caso ricorda le parentesi
print(df[(df['W'] > 0) & (df['Y'] > 1)])

# INDEXING
# Se voglio ripristinare i numeri degli indici dopo averli rinominati
# ricorda l'inplace come gli altri metodi se vuoi che sia permanente
df.reset_index()
print(df)
# Creiamo una colonna
new_ind = 'CA NY WY OR CO'.split()
df['States'] = new_ind
print(df)
# Anzi usiamola come indice
df.set_index('States', inplace=True)
# In questo modo rimuoverai la colonna
print(df)
# Se vuoi in qualche modo conservare l'indice precedente
# devi prima fare reset_index come sopra

# Possiamo ricavare informazioni sul nostro dataframe
df.info()
# Per solo i types
print(df.dtypes)
# Per informazioni dettagliate dei dati inseriti
# più che altro se il df ha dati numerici
print(df.describe())

# Ci sono metodi per avere un conteggio dei valori
ser_w = df['W'] > 0
print(ser_w.value_counts())
# Con i booleani è anche possibile sommarli come 1 e 0
print(sum(ser_w))
# Quanti valori hai
print(len(ser_w))