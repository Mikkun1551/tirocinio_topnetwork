import pandas as pd


data = {'Company': ['GOOG', 'GOOG', 'MSTF', 'MSTF', 'FB', 'FB'],
        'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
        'Sales': [200, 120, 340, 124, 243, 350]}
df = pd.DataFrame(data)
print(df)
# Usiamo il metodo groupby
print(df.groupby('Company'))
# Per visualizzare il dataframe serve fare qualche method e non solo groupby
# proviamo a vedere la media di vendite e ordinando per company
print(df.groupby('Company')['Sales'].mean())
# La colonna passata come groupby sarà il nome dell'indice del nuovo dataframe
# Possiamo fare anche il groupby insieme al describe
print(df.groupby('Company')['Sales'].describe())
# Si può anche invertire la visualizzazione delle colonne e righe
print(df.groupby('Company')['Sales'].describe().transpose())
