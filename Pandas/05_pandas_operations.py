import pandas as pd


df = pd.DataFrame({'col1': [1, 2, 3, 4],
                   'col2': [444, 555, 666, 444],
                   'col3': ['abc', 'def', 'ghi', 'xyz']})
print(df)
# Per vedere i valori non ripetuti (unique) di una colonna
print(df['col2'].unique())
# Per sapere quanti valori unique ci sono in una colonna
print(df['col2'].nunique())
# Per visualizzare tutti i valori unique e quante volte esistono
print(df['col2'].value_counts())

# Vediamo un esempio dove col1 > col2 & col2 == 444
newdf = df[(df['col1'] > 2) & (df['col2'] == 444)]
print(newdf)
# Creiamo una funzione da usare su ogni colonna
def times_two(number):
    return number * 2
print(times_two(4))
# Applichiamolo alla colonna del dataframe
print(df['col1'].apply(times_two))
# Ovviamente posso usare i risultati per crearci una nuova colonna
df['New'] = df['col1'].apply(times_two)
print(df)
# Per rimuovere una colonna
del df['New']
print(df)
# Per farlo con metodi di pandas usa drop() ma ricorda
# di scegliere l'asse e settare l'inplace

# Per avere l'indice delle colonne in una lista
print(df.columns)
# per l'indice invece
print(df.index)

# Se vogliamo filtrare per una determinata condizione
print(df.sort_values('col2'))
# Di base sarà ordinata in ordine crescente, per decrescente
print(df.sort_values('col2', ascending=False))
# anche gli indici saranno affetti dal metodo
